"""NFL Sagarin staging fetch utilities (append-only + week snapshot).

Purpose & scope:
    Fetch the latest NFL Sagarin ratings, append them to a staging ledger,
    materialize the current build week's snapshot, and upsert the master
    table so downstream builders can read consistent power ratings.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (C, E, F, I)

Invariants:
    * Staging writes are append-only (no truncation).
    * Week snapshots are rewritten atomically via tmp->replace.
    * Latest selection is based on staging `fetch_ts`, not provider week labels.

Side effects:
    * Appends to `staging/sagarin_latest/nfl/<season>.jsonl`.
    * Rewrites `out/nfl/{season}_week{week}/sagarin_nfl_{season}_wk{week}.{csv,jsonl}`.
    * Rewrites `out/master/sagarin_nfl_master.csv` with de-duplicated rows.

Do not:
    * Depend on provider-reported week numbers.
    * Alter public schema or filenames consumed by the UI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.current_week_service import get_current_week
from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir, getenv
from src.fetch_sagarin_week_nfl import (
    SAGARIN_URL,
    extract_table_week,
    fetch_html,
    parse_hfa,
    parse_page_stamp,
    parse_table_lines,
    read_local_html,
    records_to_dataframe,
    strip_html,
)

OUT_ROOT = ensure_out_dir()
STAGING_DIR = OUT_ROOT / "staging" / "sagarin_latest" / "nfl"
MASTER_PATH = OUT_ROOT / "master" / "sagarin_nfl_master.csv"
LEAGUE = "NFL"

WEEKLY_TEMPLATE = "sagarin_nfl_{season}_wk{week}"
MASTER_COLUMNS = [
    "league",
    "season",
    "week",
    "team_norm",
    "team_raw",
    "pr",
    "rank",
    "sos",
    "sos_rank",
]
MASTER_KEY = ["league", "season", "week", "team_norm"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_staging(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_staging_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def _select_latest_by_team(records: Iterable[Mapping[str, object]]) -> Tuple[List[dict], Optional[str]]:
    latest: Dict[str, dict] = {}
    latest_ts: Optional[datetime] = None
    for record in records:
        team = record.get("team_norm")
        fetch_ts = record.get("fetch_ts")
        if not isinstance(team, str) or not team:
            continue
        try:
            candidate_ts = datetime.fromisoformat(str(fetch_ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            candidate_ts = datetime.min.replace(tzinfo=timezone.utc)
        current = latest.get(team)
        if current is None or candidate_ts >= datetime.fromisoformat(current["fetch_ts"].replace("Z", "+00:00")).astimezone(
            timezone.utc
        ):
            latest[team] = dict(record)
        if latest_ts is None or candidate_ts >= latest_ts:
            latest_ts = candidate_ts
    ordered = sorted(latest.values(), key=lambda rec: int(rec.get("rank") or rec.get("pr_rank") or 0))
    iso_ts = latest_ts.isoformat().replace("+00:00", "Z") if latest_ts else None
    return ordered, iso_ts


def _normalize_weekly_snapshot(records: Sequence[Mapping[str, object]], week: int) -> pd.DataFrame:
    columns = [
        "season",
        "week",
        "team_raw",
        "team_norm",
        "team",
        "pr",
        "pr_rank",
        "sos",
        "sos_rank",
        "hfa",
        "source_url",
        "fetched_at",
        "page_stamp",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records).copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df["week"] = week
    df["team"] = df["team_norm"]
    df["pr_rank"] = df.get("rank", df.get("pr_rank"))
    df = df[columns].copy()
    return df.sort_values("pr_rank").reset_index(drop=True)


def _load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(MASTER_PATH)
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[MASTER_COLUMNS].copy()


def _upsert_master(weekly_df: pd.DataFrame) -> Tuple[int, int]:
    master_df = _load_master()
    before = len(master_df)
    payload = weekly_df.rename(columns={"pr_rank": "rank"}).copy()
    payload["league"] = LEAGUE
    payload = payload[MASTER_COLUMNS].copy()
    combined = pd.concat([master_df, payload], ignore_index=True)
    combined = combined.drop_duplicates(MASTER_KEY, keep="last")
    combined = combined.sort_values(["season", "week", "team_norm"]).reset_index(drop=True)
    write_atomic_csv(MASTER_PATH, combined)
    after = len(combined)
    return before, after


def run_nfl_sagarin_staging(
    *,
    local_html: Optional[Path] = None,
) -> Dict[str, object]:
    """Fetch, stage, and snapshot NFL Sagarin ratings for the current week."""
    if getenv("SAGARIN_STAGING_ENABLE", "1").strip().lower() in {"0", "false", "off", "disabled"}:
        print("Sagarin(NFL): staging disabled")
        return {
            "latest_fetch_ts": None,
            "teams_selected": 0,
            "wrote_master_rows": 0,
            "master_total": len(_load_master()),
        }

    try:
        html = read_local_html(local_html) if local_html else fetch_html()
        source_url = str(Path(local_html).resolve()) if local_html else SAGARIN_URL
    except Exception as exc:
        print(f"WARNING: NFL Sagarin fetch failed ({exc})")
        return {
            "latest_fetch_ts": None,
            "teams_selected": 0,
            "wrote_master_rows": 0,
            "master_total": len(_load_master()),
        }

    stripped = strip_html(html)
    page_season, _page_week, header_line = extract_table_week(stripped)
    records = parse_table_lines(stripped)
    if not records:
        print("WARNING: NFL Sagarin parser returned zero rows; skipping staging")
        return {
            "latest_fetch_ts": None,
            "teams_selected": 0,
            "wrote_master_rows": 0,
            "master_total": len(_load_master()),
        }

    hfa = parse_hfa(stripped)
    page_stamp = header_line or parse_page_stamp(stripped.splitlines())
    season = int(page_season or datetime.now(timezone.utc).year)
    season = max(season, datetime.now(timezone.utc).year)  # ensure current season at minimum
    fetch_ts = _now_iso()

    df = records_to_dataframe(
        records,
        season,
        0,  # placeholder week; actual selection occurs via current week service
        hfa,
        source_url,
        page_stamp,
    )

    staging_rows = []
    for row in df.itertuples(index=False):
        staging_rows.append(
            {
                "league": LEAGUE,
                "season": season,
                "team_norm": getattr(row, "team_norm"),
                "team_raw": getattr(row, "team_raw"),
                "pr": float(getattr(row, "pr", 0.0)),
                "rank": int(getattr(row, "pr_rank", 0)),
                "sos": None if pd.isna(getattr(row, "sos")) else float(getattr(row, "sos")),
                "sos_rank": None if pd.isna(getattr(row, "sos_rank")) else int(getattr(row, "sos_rank")),
                "hfa": None if pd.isna(getattr(row, "hfa")) else float(getattr(row, "hfa")),
                "source_url": source_url,
                "page_stamp": page_stamp,
                "fetch_ts": fetch_ts,
                "fetched_at": getattr(row, "fetched_at"),
            }
        )

    staging_path = STAGING_DIR / f"{season}.jsonl"
    _append_staging(staging_path, staging_rows)

    season_from_state, week_from_state, _ = get_current_week("NFL")
    staging_records = _load_staging_records(staging_path)
    latest_records, latest_ts = _select_latest_by_team(staging_records)

    weekly_df = _normalize_weekly_snapshot(latest_records, week_from_state)
    base_dir = OUT_ROOT / "nfl" / f"{season_from_state}_week{week_from_state}"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / WEEKLY_TEMPLATE.format(season=season_from_state, week=week_from_state)
    write_atomic_csv(base_path.with_suffix(".csv"), weekly_df)
    write_atomic_jsonl(base_path.with_suffix(".jsonl"), weekly_df.where(pd.notna(weekly_df), None).to_dict(orient="records"))

    before, after = _upsert_master(weekly_df)

    print(
        f"Sagarin(NFL): latest_fetch_ts={latest_ts or fetch_ts} "
        f"teams={len(weekly_df)} wrote_master_rows={len(weekly_df)} "
        f"selected_for_build={len(weekly_df)}"
    )

    return {
        "latest_fetch_ts": latest_ts or fetch_ts,
        "teams_selected": len(weekly_df),
        "wrote_master_rows": len(weekly_df),
        "master_total": after,
    }


__all__ = ["run_nfl_sagarin_staging"]

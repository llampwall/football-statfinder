"""CFB Sagarin staging fetch and selection utilities.

Purpose & scope:
    Fetch the latest College Football Sagarin ratings, append them to an
    append-only staging ledger, materialize the current week's snapshot, and
    update the master table that downstream builders consume.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (C, E, F, I)

Invariants:
    * All timestamps are recorded in UTC ISO8601 (Z suffix).
    * Staging writes are append-only; master snapshots use atomic rewrites.
    * Weekly outputs keep the legacy schema (CSV + JSONL under ``out/cfb``).

Side effects:
    * Appends to ``staging/sagarin_latest/cfb/<season>.jsonl``.
    * Rewrites ``out/master/sagarin_cfb_master.csv`` with deduplicated rows.
    * Rewrites ``out/cfb/<season>_week<week>/sagarin_cfb_<season>_wk<week>.*``.

Do not:
    * Mutate or truncate prior staging entries.
    * Trust the provider's "Week #"; the caller provides the authoritative week.

Log contract:
    * Callers print a single summary line:
      ``Sagarin(CFB): latest_fetch_ts=<ts> teams=<n> wrote_master_rows=<k> selected_for_build=<n>``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir
from src.fetch_sagarin_week_cfb import (
    SAGARIN_URL,
    acceptance_summary,
    extract_table_week,
    fetch_html,
    parse_hfa,
    parse_page_stamp,
    parse_table_lines,
    read_local_html,
    records_to_dataframe,
    strip_html,
    validate_records,
)

LEAGUE = "CFB"
OUT_ROOT = ensure_out_dir()
STAGING_DIR = OUT_ROOT / "staging" / "sagarin_latest" / "cfb"
MASTER_PATH = OUT_ROOT / "master" / "sagarin_cfb_master.csv"

WEEKLY_TEMPLATE = "sagarin_cfb_{season}_wk{week}"
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


def _parse_iso8601(value: Optional[str]) -> datetime:
    """Parse ISO8601 timestamps (Z allowed) into aware UTC datetimes."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _append_staging(season: int, rows: Sequence[Mapping[str, object]]) -> Path:
    """Append normalized records to the season staging JSONL."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = STAGING_DIR / f"{season}.jsonl"
    with staging_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return staging_path


def _load_staging_records(season: int) -> List[dict]:
    """Load all staging records for the season."""
    staging_path = STAGING_DIR / f"{season}.jsonl"
    if not staging_path.exists():
        return []
    records: List[dict] = []
    with staging_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if int(record.get("season") or season) != season:
                continue
            records.append(record)
    return records


def _records_from_dataframe(
    df: pd.DataFrame,
    season: int,
    week: int,
    page_stamp: Optional[str],
) -> List[dict]:
    """Serialize parsed DataFrame rows into staging-friendly dictionaries."""
    rows: List[dict] = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "season": season,
                "week": week,
                "team_norm": getattr(row, "team_norm"),
                "team_raw": getattr(row, "team_raw"),
                "pr": float(getattr(row, "pr")),
                "rank": int(getattr(row, "pr_rank")),
                "sos": None if pd.isna(getattr(row, "sos")) else float(getattr(row, "sos")),
                "sos_rank": None if pd.isna(getattr(row, "sos_rank")) else int(getattr(row, "sos_rank")),
                "hfa": None if pd.isna(getattr(row, "hfa")) else float(getattr(row, "hfa")),
                "source_url": getattr(row, "source_url", SAGARIN_URL),
                "page_stamp": getattr(row, "page_stamp", page_stamp),
                "fetch_ts": getattr(row, "fetched_at"),
            }
        )
    return rows


def _select_latest_by_team(records: Iterable[Mapping[str, object]]) -> Tuple[List[dict], Optional[str]]:
    """Select the latest record per team_norm."""
    latest: Dict[str, dict] = {}
    latest_ts: Optional[datetime] = None
    for record in records:
        team = record.get("team_norm")
        fetch_ts = _parse_iso8601(record.get("fetch_ts"))
        if not team:
            continue
        current = latest.get(team)
        if current is None or fetch_ts >= _parse_iso8601(current.get("fetch_ts")):
            latest[team] = dict(record)
        if latest_ts is None or fetch_ts >= latest_ts:
            latest_ts = fetch_ts
    ordered = sorted(latest.values(), key=lambda item: int(item.get("rank") or 0))
    latest_iso = latest_ts.isoformat().replace("+00:00", "Z") if latest_ts else None
    return ordered, latest_iso


def _normalize_weekly_snapshot(
    records: Sequence[Mapping[str, object]],
    season: int,
    week: int,
) -> pd.DataFrame:
    """Convert latest records into the weekly DataFrame schema."""
    if not records:
        return pd.DataFrame(columns=[
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
        ])

    df = pd.DataFrame(records).copy()
    df["season"] = season
    df["week"] = week
    df["team"] = df["team_norm"]
    df["pr_rank"] = df["rank"].astype(int)
    df["sos_rank"] = df.get("sos_rank")
    df["hfa"] = df["hfa"]
    df["source_url"] = df.get("source_url", SAGARIN_URL)
    df["fetched_at"] = df.get("fetch_ts")
    df["page_stamp"] = df.get("page_stamp")
    df = df[
        [
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
    ].copy()
    df = df.sort_values("pr_rank").reset_index(drop=True)
    return df


def _load_master() -> pd.DataFrame:
    """Load the existing Sagarin master sheet."""
    if not MASTER_PATH.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(MASTER_PATH)
    missing = [col for col in MASTER_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = None
    return df[MASTER_COLUMNS].copy()


def _upsert_master(season: int, week: int, weekly_df: pd.DataFrame) -> Tuple[int, int]:
    """Upsert the week's rows into the master CSV."""
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


def run_cfb_sagarin_staging(
    season: int,
    week: int,
    *,
    local_html: Optional[Path] = None,
) -> Dict[str, object]:
    """Fetch, stage, and materialize Sagarin data for the requested week.

    Args:
        season: Target season identifier.
        week: Target week identifier.
        local_html: Optional path to saved HTML for offline parsing.

    Returns:
        Summary dictionary with staging paths, counts, and latest fetch metadata.

    Raises:
        RuntimeError: When parsing or validation fails.
    """
    try:
        if local_html:
            html = read_local_html(local_html)
            source_url = str(local_html.resolve())
        else:
            html = fetch_html()
            source_url = SAGARIN_URL
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Sagarin page: {exc}") from exc

    stripped = strip_html(html)
    page_season, page_week, header_line = extract_table_week(stripped)
    if page_season and page_season != season:
        print(
            f"WARNING: Sagarin page season differs (page={page_season}, requested={season}); "
            "using requested season for outputs."
        )
    if page_week and page_week != week:
        print(
            f"WARNING: Sagarin page week differs (page={page_week}, requested={week}); "
            "using requested week for outputs."
        )
    records = parse_table_lines(stripped)
    if not records:
        raise RuntimeError("Sagarin parser returned zero rows.")

    hfa = parse_hfa(stripped)
    page_stamp = header_line or parse_page_stamp(stripped.splitlines())
    df, unmapped = records_to_dataframe(
        records,
        season,
        week,
        hfa,
        source_url,
        page_stamp,
    )
    errors = validate_records(df, unmapped)
    if errors:
        raise RuntimeError(f"Sagarin validation failed: {'; '.join(errors)}")

    acceptance_summary(df, hfa, (page_season, page_week))
    staging_rows = _records_from_dataframe(df, season, week, page_stamp)
    staging_path = _append_staging(season, staging_rows)

    latest_records = _load_staging_records(season)
    selected_records, latest_ts = _select_latest_by_team(latest_records or staging_rows)
    weekly_df = _normalize_weekly_snapshot(selected_records, season, week)
    if weekly_df.empty:
        raise RuntimeError("No Sagarin records available after staging selection.")

    base_dir = OUT_ROOT / "cfb" / f"{season}_week{week}"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / WEEKLY_TEMPLATE.format(season=season, week=week)
    csv_path = base_path.with_suffix(".csv")
    jsonl_path = base_path.with_suffix(".jsonl")
    write_atomic_csv(csv_path, weekly_df)
    json_records = weekly_df.where(pd.notna(weekly_df), None).to_dict(orient="records")
    write_atomic_jsonl(jsonl_path, json_records)

    before, after = _upsert_master(season, week, weekly_df)

    summary = {
        "season": season,
        "week": week,
        "latest_fetch_ts": latest_ts or weekly_df["fetched_at"].iloc[-1],
        "teams_selected": len(weekly_df),
        "wrote_master_rows": len(weekly_df),
        "master_total": after,
        "staging_path": staging_path,
        "weekly_csv": csv_path,
        "weekly_jsonl": jsonl_path,
        "page_stamp": page_stamp,
        "hfa": hfa,
        "source_url": source_url,
    }
    return summary


__all__ = ["run_cfb_sagarin_staging"]

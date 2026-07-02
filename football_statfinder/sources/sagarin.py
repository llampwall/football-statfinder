"""One Sagarin staging engine for both leagues.

Port of the generation-2 staging fetchers ``src/ratings/sagarin_nfl_fetch.py``
and ``src/ratings/sagarin_cfb_fetch.py`` (~80% identical twins), using
:mod:`football_statfinder.sources.sagarin_parsers` as the parser library.
Model preserved: append-only per-season staging ledger, latest-per-team
selection by ``fetch_ts``, atomic weekly snapshot, master CSV upsert.

Legacy behavior deliberately changed (REBUILD.md bug patterns 14/15/16):

* Raw HTML is archived to ``paths.sagarin_raw_html_dir`` BEFORE parsing
  (best-effort), so a parser regression never loses the evidence.
* Validation runs ALWAYS for both leagues using the League gates
  (``sagarin_expected_teams`` exact / ``sagarin_team_range`` bounds, plus the
  legacy rank-coverage and 2-decimal-precision gates). On failure the engine
  writes a receipt artifact and raises — the legacy NFL soft-fail (return a
  zero-count summary, stage nothing, stay green) is gone.
* ``season``/``week`` are passed in by the caller (the current-week service)
  and stamp both paths and content. Never derived as
  ``max(page_season, calendar_year)`` (bug 14), and the mismatch warnings say
  what actually happens: requested values win (bug 15 — the legacy NFL
  warnings claimed page values were used when they were not).
* Fetch failures raise instead of returning a zero-count summary.
* Ledger reads go through the counted-skip JSONL reader (bug 10).

Kept: the spoofed browser User-Agent, the append-only (non-atomic) ledger
append, the weekly snapshot filename ``sagarin_{league}_{season}_wk{week}``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from .. import paths
from ..common.io_atomic import write_atomic_csv, write_atomic_json, write_atomic_jsonl, write_atomic_text
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League
from .sagarin_parsers import USER_AGENT, RatedTeam, decode_bytes, get_parser, parse_hfa, strip_html

logger = logging.getLogger(__name__)

WEEKLY_TEMPLATE = "sagarin_{league}_{season}_wk{week}"

WEEKLY_COLUMNS = [
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

FetchHtml = Callable[[str], str]


class SagarinValidationError(RuntimeError):
    """Raised when a parse fails the League acceptance gates (receipt written)."""


@dataclass(frozen=True)
class SagarinStagingResult:
    league: str
    season: int
    week: int
    page_season: Optional[int]
    page_week: Optional[int]
    teams_parsed: int
    teams_selected: int
    master_before: int
    master_after: int
    latest_fetch_ts: Optional[str]
    hfa: Optional[float]
    page_stamp: Optional[str]
    source_url: str
    raw_html_path: Optional[Path]
    staging_path: Path
    weekly_csv: Path
    weekly_jsonl: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sagarin_html(url: str) -> str:
    """Fetch a Sagarin page with the legacy spoofed browser User-Agent."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return decode_bytes(
        resp.content,
        (resp.apparent_encoding or resp.encoding, "cp1252", "latin-1", "utf-8"),
    )


def read_local_html(path: Path) -> str:
    """Read a saved Sagarin page (offline parsing), legacy encoding fallbacks."""
    return decode_bytes(Path(path).read_bytes(), ("utf-8", "cp1252", "latin-1"))


def _archive_raw_html(league: League, season: int, week: int, html: str) -> Optional[Path]:
    """Archive the raw page BEFORE parsing; best-effort, never raises."""
    if not html:
        return None
    stem = Path(urlparse(league.sagarin_url).path).stem or f"sagarin_{league.code}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = paths.sagarin_raw_html_dir(league.code) / f"{stem}_{season}_wk{week}_{timestamp}.html"
    try:
        write_atomic_text(target, html)
        return target
    except Exception as exc:  # pragma: no cover - archival best-effort
        logger.warning(
            "failed to archive Sagarin HTML for %s %s week %s: %s", league.display, season, week, exc
        )
        return None


def _rated_to_dataframe(
    league: League,
    rated: Sequence[RatedTeam],
    season: int,
    week: int,
    hfa: Optional[float],
    source_url: str,
    page_stamp: Optional[str],
    fetched_at: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the weekly-schema DataFrame, stamping the CALLER's season/week."""
    rows: List[dict] = []
    unmapped: List[str] = []
    for rec in rated:
        team_norm = league.normalize_display(rec.team_raw)
        if not team_norm:
            unmapped.append(rec.team_raw)
            team_norm = rec.team_raw
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team_raw": rec.team_raw,
                "team_norm": team_norm,
                "team": team_norm,
                "pr": round(rec.pr, 2),
                "pr_rank": int(rec.pr_rank),
                "sos": None if rec.sos is None else round(rec.sos, 2),
                "sos_rank": rec.sos_rank,
                "hfa": hfa,
                "source_url": source_url,
                "fetched_at": fetched_at,
                "page_stamp": page_stamp,
            }
        )
    df = pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
    if not df.empty:
        df = df.sort_values("pr_rank").reset_index(drop=True)
    return df, unmapped


def validate_ratings(league: League, df: pd.DataFrame, unmapped: Sequence[str]) -> List[str]:
    """League-gated acceptance checks; returns human-readable errors (pure).

    Unifies the legacy NFL and CFB validators: count gate from the League
    (exact ``sagarin_expected_teams`` or ``sagarin_team_range`` bounds),
    contiguous 1..N rank coverage, 2-decimal PR precision, unique non-blank
    normalized names, no unmapped teams, 2-decimal SoS precision. The legacy
    NFL SoS coverage/rank-bound gate applies only to exact-count leagues
    (CFB SoS ranks legitimately exceed the FBS row count).
    """
    errors: List[str] = []
    count = len(df)

    if league.sagarin_expected_teams is not None:
        if count != league.sagarin_expected_teams:
            errors.append(f"Count != {league.sagarin_expected_teams} (found {count})")
    elif league.sagarin_team_range is not None:
        low, high = league.sagarin_team_range
        if not (low <= count <= high):
            errors.append(f"Count {count} outside expected range [{low}, {high}]")
    else:  # pragma: no cover - every league defines one gate
        errors.append("league defines no Sagarin count gate")

    if df.empty:
        errors.append("no rows parsed")
        return errors

    ranks = df["pr_rank"].tolist()
    expected = set(range(1, count + 1))
    missing = expected.difference(ranks)
    duplicates = sorted({r for r in ranks if ranks.count(r) > 1})
    if missing or duplicates:
        errors.append(f"Ranks invalid; missing={sorted(missing)} duplicates={duplicates}")

    if not df["pr"].apply(lambda v: pd.notna(v) and abs(v * 100 - round(v * 100)) < 1e-6).all():
        errors.append("PR values missing or not two-decimal precision")

    if df["team_norm"].nunique() != count:
        errors.append(f"team_norm unique count {df['team_norm'].nunique()} != total rows {count}")
    if df["team_norm"].isna().any() or (df["team_norm"].astype(str).str.strip() == "").any():
        errors.append("team_norm contains blank values")
    if unmapped:
        errors.append(f"Unmapped teams: {sorted(set(unmapped))}")

    sos_nonnull = df["sos"].dropna()
    if not sos_nonnull.empty:
        precision_ok = sos_nonnull.apply(lambda v: abs(v * 100 - round(v * 100)) < 1e-6).all()
        if not precision_ok:
            errors.append("sos values not two-decimal precision")
        if league.sagarin_expected_teams is not None:
            coverage = len(sos_nonnull) / count
            ranks_valid = (
                df[df["sos"].notna()]["sos_rank"]
                .apply(lambda v: pd.notna(v) and 1 <= int(v) <= count)
                .all()
            )
            if coverage < 0.9 or not ranks_valid:
                errors.append(
                    f"SoS coverage check failed (coverage={coverage:.2%}, ranks_valid={ranks_valid})"
                )
    return errors


def _write_failure_receipt(
    league: League,
    season: int,
    week: int,
    stripped_text: str,
    receipt: Mapping[str, object],
) -> Tuple[Path, Path]:
    """Persist the failure receipt + stripped page under the staging dir."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = paths.sagarin_staging_dir(league.code) / f"failure_{season}_wk{week}_{timestamp}"
    receipt_path = base.with_suffix(".receipt.json")
    raw_path = base.with_suffix(".raw.txt")
    write_atomic_json(receipt_path, dict(receipt))
    write_atomic_text(raw_path, stripped_text)
    return receipt_path, raw_path


def _append_staging(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Append-only ledger write (legacy semantics: plain append, no rewrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _parse_iso8601(value: object) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _select_latest_by_team(
    records: Iterable[Mapping[str, object]],
) -> Tuple[List[dict], Optional[str]]:
    """Latest record per team_norm by ``fetch_ts`` (legacy selection)."""
    latest: Dict[str, dict] = {}
    latest_ts: Optional[datetime] = None
    for record in records:
        team = record.get("team_norm")
        if not isinstance(team, str) or not team:
            continue
        fetch_ts = _parse_iso8601(record.get("fetch_ts"))
        current = latest.get(team)
        if current is None or fetch_ts >= _parse_iso8601(current.get("fetch_ts")):
            latest[team] = dict(record)
        if latest_ts is None or fetch_ts >= latest_ts:
            latest_ts = fetch_ts
    ordered = sorted(latest.values(), key=lambda item: int(item.get("rank") or 0))
    latest_iso = latest_ts.isoformat().replace("+00:00", "Z") if latest_ts else None
    return ordered, latest_iso


def _weekly_snapshot(records: Sequence[Mapping[str, object]], season: int, week: int) -> pd.DataFrame:
    """Ledger records -> weekly snapshot frame stamped with caller season/week."""
    if not records:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    df = pd.DataFrame(records).copy()
    df["season"] = int(season)
    df["week"] = int(week)
    df["team"] = df["team_norm"]
    df["pr_rank"] = df["rank"].astype(int)
    df["fetched_at"] = df.get("fetch_ts")
    for col in WEEKLY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[WEEKLY_COLUMNS].copy()
    return df.sort_values("pr_rank").reset_index(drop=True)


def _load_master(league: League) -> pd.DataFrame:
    master_csv = paths.sagarin_master_csv(league.code)
    if not master_csv.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(master_csv)
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[MASTER_COLUMNS].copy()


def _upsert_master(league: League, weekly_df: pd.DataFrame) -> Tuple[int, int]:
    """The ONE master upsert path (legacy had two writers on one CSV)."""
    master_df = _load_master(league)
    before = len(master_df)
    payload = weekly_df.rename(columns={"pr_rank": "rank"}).copy()
    payload["league"] = league.display
    payload = payload[MASTER_COLUMNS].copy()
    if master_df.empty:
        combined = payload.copy()
    else:
        combined = pd.concat([master_df, payload], ignore_index=True)
    combined = combined.drop_duplicates(MASTER_KEY, keep="last")
    combined = combined.sort_values(["season", "week", "team_norm"], kind="mergesort").reset_index(drop=True)
    write_atomic_csv(paths.sagarin_master_csv(league.code), combined)
    return before, len(combined)


def run_sagarin_staging(
    league: League,
    season: int,
    week: int,
    settings: Settings,
    *,
    local_html: Optional[Path] = None,
    fetch_html_fn: Optional[FetchHtml] = None,
) -> SagarinStagingResult:
    """Fetch, validate, stage, snapshot, and upsert Sagarin ratings.

    ``season``/``week`` come from the caller (current-week service) and stamp
    both paths and content. ``fetch_html_fn(url) -> html`` overrides the
    network fetch for tests. ``settings`` is accepted for the uniform stage
    signature; Sagarin needs no secrets today.

    Raises on fetch failure, zero-row parse, or gate failure (with a receipt
    artifact under the staging dir) — never silently stages a partial parse.
    """
    del settings  # uniform stage signature; no Sagarin config yet

    if local_html is not None:
        html = read_local_html(local_html)
        source_url = str(Path(local_html).resolve())
    else:
        fetcher = fetch_html_fn or fetch_sagarin_html
        try:
            html = fetcher(league.sagarin_url)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Sagarin page for {league.display}: {exc}") from exc
        source_url = league.sagarin_url

    # Archive before parsing so parser regressions never lose the page.
    raw_html_path = _archive_raw_html(league, season, week, html)

    stripped = strip_html(html)
    spec = get_parser(league.code)
    page_season, page_week, header_line = spec.extract_table_week(stripped)
    if page_season and page_season != season:
        logger.warning(
            "%s Sagarin page reports season %s but requested %s; outputs keep requested season",
            league.display,
            page_season,
            season,
        )
    if page_week and page_week != week:
        logger.warning(
            "%s Sagarin page reports week %s but requested %s; outputs keep requested week",
            league.display,
            page_week,
            week,
        )

    rated = spec.rated_teams(stripped)
    hfa = parse_hfa(stripped)
    page_stamp = header_line or spec.parse_page_stamp(stripped.splitlines())
    fetch_ts = _now_iso()

    df, unmapped = _rated_to_dataframe(
        league, rated, season, week, hfa, source_url, page_stamp, fetch_ts
    )
    errors = validate_ratings(league, df, unmapped)
    if errors:
        receipt = {
            "league": league.display,
            "season": season,
            "week": week,
            "page_season": page_season,
            "page_week": page_week,
            "source_url": source_url,
            "row_count": len(df),
            "unmapped_team_raw": sorted(set(unmapped)),
            "errors": errors,
            "raw_html_path": str(raw_html_path) if raw_html_path else None,
        }
        receipt_path, _ = _write_failure_receipt(league, season, week, stripped, receipt)
        raise SagarinValidationError(
            f"{league.display} Sagarin acceptance failed ({'; '.join(errors)}); receipt: {receipt_path}"
        )

    staging_rows = [
        {
            "league": league.display,
            "season": int(season),
            "week": int(week),
            "team_norm": row.team_norm,
            "team_raw": row.team_raw,
            "pr": float(row.pr),
            "rank": int(row.pr_rank),
            "sos": None if pd.isna(row.sos) else float(row.sos),
            "sos_rank": None if pd.isna(row.sos_rank) else int(row.sos_rank),
            "hfa": None if hfa is None else float(hfa),
            "source_url": source_url,
            "page_stamp": page_stamp,
            "fetch_ts": fetch_ts,
        }
        for row in df.itertuples(index=False)
    ]
    staging_path = paths.sagarin_staging_dir(league.code) / f"{int(season)}.jsonl"
    _append_staging(staging_path, staging_rows)

    ledger = read_jsonl(staging_path)
    if ledger.skipped:
        logger.warning("%s Sagarin ledger %s: %d bad line(s) skipped", league.display, staging_path, ledger.skipped)
    season_records = [r for r in ledger.rows if int(r.get("season") or season) == int(season)]
    selected, latest_ts = _select_latest_by_team(season_records or staging_rows)

    weekly_df = _weekly_snapshot(selected, season, week)
    if weekly_df.empty:
        raise RuntimeError(f"{league.display}: no Sagarin records available after staging selection")

    base = paths.week_dir(league.code, season, week, create=True) / WEEKLY_TEMPLATE.format(
        league=league.code, season=int(season), week=int(week)
    )
    weekly_csv = base.with_suffix(".csv")
    weekly_jsonl = base.with_suffix(".jsonl")
    write_atomic_csv(weekly_csv, weekly_df)
    write_atomic_jsonl(weekly_jsonl, weekly_df.where(pd.notna(weekly_df), None).to_dict(orient="records"))

    master_before, master_after = _upsert_master(league, weekly_df)

    logger.info(
        "Sagarin(%s): season=%s week=%s latest_fetch_ts=%s teams=%d master=%d->%d",
        league.display,
        season,
        week,
        latest_ts or fetch_ts,
        len(weekly_df),
        master_before,
        master_after,
    )
    return SagarinStagingResult(
        league=league.display,
        season=int(season),
        week=int(week),
        page_season=page_season,
        page_week=page_week,
        teams_parsed=len(rated),
        teams_selected=len(weekly_df),
        master_before=master_before,
        master_after=master_after,
        latest_fetch_ts=latest_ts or fetch_ts,
        hfa=hfa,
        page_stamp=page_stamp,
        source_url=source_url,
        raw_html_path=raw_html_path,
        staging_path=staging_path,
        weekly_csv=weekly_csv,
        weekly_jsonl=weekly_jsonl,
    )


__all__ = [
    "MASTER_COLUMNS",
    "MASTER_KEY",
    "SagarinStagingResult",
    "SagarinValidationError",
    "WEEKLY_COLUMNS",
    "fetch_sagarin_html",
    "read_local_html",
    "run_sagarin_staging",
    "validate_ratings",
]

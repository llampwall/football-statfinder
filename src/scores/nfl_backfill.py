"""NFL scores backfill helpers (prior weeks before ATS).

Purpose & scope:
    Populate missing final scores (and W-L-T strings) for recent NFL weeks
    using the schedule master so downstream ATS calculations operate on
    finalized data.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (C, E, F, H)

Invariants:
    * Only prior weeks (bounded by BACKFILL_WEEKS) are mutated.
    * JSONL/CSV rewrites are atomic (tmp write → replace).
    * Schedule master is treated as the authoritative score source.

Side effects:
    * Rewrites `out/nfl/{season}_week{w}/games_week_{season}_{w}.jsonl` and
      the matching CSV when updates occur.

Do not:
    * Touch the current week or future weeks.
    * Overwrite existing scores when schedule master lacks finals.
"""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.backfill_merge import merge_games_week, summarize_preservation
from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir, getenv
from src.common.team_names import TEAM_ABBR_TO_FULL, team_merge_key
from src.odds.nfl_promote_week import promote_week_odds
from src.schedule_master import load_master as load_nfl_schedule_master

OUT_ROOT = ensure_out_dir()
NFL_ROOT = OUT_ROOT

FULL_TO_ABBR = {full: abbr for abbr, full in TEAM_ABBR_TO_FULL.items()}
KEY_TO_ABBR = {team_merge_key(full): abbr for abbr, full in TEAM_ABBR_TO_FULL.items()}


def _week_path(season: int, week: int) -> Tuple[Path, Path]:
    week_dir = NFL_ROOT / f"{season}_week{week}"
    json_path = week_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = week_dir / f"games_week_{season}_{week}.csv"
    return json_path, csv_path


def _build_game_key(row: Mapping[str, object]) -> Optional[str]:
    kickoff = row.get("kickoff_iso_utc") or row.get("kickoff_iso")
    if not isinstance(kickoff, str):
        return None
    try:
        dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    except Exception:
        return None
    away_full = row.get("away_team_norm") or row.get("away_team")
    home_full = row.get("home_team_norm") or row.get("home_team")
    away_key = row.get("away_team_key")
    home_key = row.get("home_team_key")
    away_abbr = _normalize_abbr(away_full, away_key)
    home_abbr = _normalize_abbr(home_full, home_key)
    if not away_abbr or not home_abbr:
        return None
    return f"{dt.strftime('%Y%m%d_%H%M')}_{away_abbr.lower()}_{home_abbr.lower()}"


def _normalize_abbr(name: Optional[str], key: Optional[str]) -> Optional[str]:
    if isinstance(name, str):
        abbr = FULL_TO_ABBR.get(name)
        if abbr:
            return abbr
    if isinstance(key, str):
        abbr = KEY_TO_ABBR.get(key)
        if abbr:
            return abbr
    if isinstance(name, str):
        merge = team_merge_key(name)
        return KEY_TO_ABBR.get(merge)
    return None


def _build_score_and_record_maps(season: int) -> Tuple[Dict[str, Tuple[int, int]], Dict[Tuple[str, str], str]]:
    schedule_df = load_nfl_schedule_master()
    if schedule_df.empty:
        return {}, {}
    schedule_df["season"] = pd.to_numeric(schedule_df["season"], errors="coerce")
    schedule_df = schedule_df[schedule_df["season"] == season].copy()
    if schedule_df.empty:
        return {}, {}
    schedule_df["home_score"] = pd.to_numeric(schedule_df.get("home_score"), errors="coerce")
    schedule_df["away_score"] = pd.to_numeric(schedule_df.get("away_score"), errors="coerce")
    schedule_df["kickoff_iso_utc"] = schedule_df.get("kickoff_iso_utc")
    schedule_df["game_key"] = schedule_df.apply(_build_game_key, axis=1)
    schedule_df = schedule_df.dropna(subset=["game_key"])

    score_lookup: Dict[str, Tuple[int, int]] = {}
    for row in schedule_df.itertuples(index=False):
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        score_lookup[row.game_key] = (int(row.home_score), int(row.away_score))

    schedule_df["kickoff_dt"] = pd.to_datetime(schedule_df["kickoff_iso_utc"], errors="coerce", utc=True)
    schedule_df = schedule_df.sort_values(["kickoff_dt", "game_key"])

    records_after_game: Dict[Tuple[str, str], str] = {}
    team_records: Dict[str, Dict[str, int]] = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})

    for row in schedule_df.itertuples(index=False):
        game_key = row.game_key
        scores = score_lookup.get(game_key)
        if not scores:
            continue
        home_score, away_score = scores
        home_abbr = _normalize_abbr(row.home_team_norm, row.home_team_key) or row.home_team_key
        away_abbr = _normalize_abbr(row.away_team_norm, row.away_team_key) or row.away_team_key
        if not home_abbr or not away_abbr:
            continue
        home_abbr = str(home_abbr).upper()
        away_abbr = str(away_abbr).upper()
        if home_score > away_score:
            team_records[home_abbr]["w"] += 1
            team_records[away_abbr]["l"] += 1
        elif home_score < away_score:
            team_records[home_abbr]["l"] += 1
            team_records[away_abbr]["w"] += 1
        else:
            team_records[home_abbr]["t"] += 1
            team_records[away_abbr]["t"] += 1
        for abbr, side in ((home_abbr.lower(), "home"), (away_abbr.lower(), "away")):
            rec = team_records[abbr]
            record_str = f"{rec['w']}-{rec['l']}"
            if rec["t"] > 0:
                record_str += f"-{rec['t']}"
            records_after_game[(game_key, side)] = record_str

    return score_lookup, records_after_game


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _load_week_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _align_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = [col for col in columns if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + remainder)

def _needs_odds_repair(rows: list[dict]) -> bool:
    """Return True if odds/rvo look nuked on any row."""
    for r in rows or []:
        if (
            r.get("spread_favored_team") in (None, "",)
            and r.get("rating_vs_odds") in (None, "",)
        ):
            return True
    return False


def backfill_nfl_scores(season: int, week: int) -> Dict[str, object]:
    """Backfill missing scores for recent NFL weeks."""
    if getenv("SCORES_BACKFILL_ENABLE", "1").strip().lower() in {"0", "false", "off", "disabled"}:
        return {"weeks": [], "updated": 0, "skipped": 0}

    include_weeks = int(getenv("BACKFILL_WEEKS", "2") or "2")
    if include_weeks <= 0 or week <= 1:
        return {"weeks": [], "updated": 0, "skipped": 0}

    weeks = sorted([w for w in range(week - include_weeks, week) if w >= 1])
    if not weeks:
        return {"weeks": [], "updated": 0, "skipped": 0}

    score_lookup, record_lookup = _build_score_and_record_maps(season)

    updated_total = 0
    skipped_total = 0
    preserved_odds_total = 0
    preserved_rvo_total = 0
    files_rewritten = 0
    promote_prev_enabled = _is_truthy(getenv("BACKFILL_PROMOTE_PREV", "0"))

    for target_week in weeks:
        json_path, csv_path = _week_path(season, target_week)
        existing_rows = _load_week_rows(json_path)
        if not existing_rows:
            continue
        incoming_rows = deepcopy(existing_rows)
        csv_df = pd.read_csv(csv_path) if csv_path.exists() else None
        file_changed = False
        row_updates = 0
        for row in incoming_rows:
            game_key = row.get("game_key")
            if not isinstance(game_key, str):
                continue
            scores = score_lookup.get(game_key)
            if not scores:
                if _is_missing(row.get("home_score")) or _is_missing(row.get("away_score")):
                    skipped_total += 1
                continue
            home_score, away_score = scores

            row_changed = False
            if row.get("home_score") != home_score or row.get("away_score") != away_score:
                row["home_score"] = home_score
                row["away_score"] = away_score
                row_changed = True
            home_record = record_lookup.get((game_key, "home"))
            away_record = record_lookup.get((game_key, "away"))
            if home_record and row.get("home_su") != home_record:
                row["home_su"] = home_record
                row_changed = True
            if away_record and row.get("away_su") != away_record:
                row["away_su"] = away_record
                row_changed = True

            schedule_row = row.get("raw_sources", {}).get("schedule_row")
            if isinstance(schedule_row, dict):
                if schedule_row.get("home_score") != home_score:
                    schedule_row["home_score"] = home_score
                    row_changed = True
                if schedule_row.get("away_score") != away_score:
                    schedule_row["away_score"] = away_score
                    row_changed = True

            if row_changed:
                file_changed = True
                updated_total += 1
                row_updates += 1
            needs_repair = _needs_odds_repair(existing_rows)

        if file_changed or (promote_prev_enabled and target_week < week and needs_repair):
            merged_rows = merge_games_week(existing_rows, incoming_rows)

            preservation = summarize_preservation(existing_rows, merged_rows)
            preserved_odds_total += preservation["preserved_odds"]
            preserved_rvo_total += preservation["preserved_rvo"]

            final_rows = merged_rows

            if promote_prev_enabled and target_week < week:
                promoted_rows = deepcopy(merged_rows)
                promote_stats = promote_week_odds(promoted_rows, season, target_week)
                promoted = promote_stats.get("promoted_games", 0)
                final_rows = promoted_rows
                print(
                    f"ODDS_REPROMOTE(NFL): week={season}-{target_week} "
                    f"promoted={promoted} source=staging/odds_pinned"
                )

                if not file_changed and needs_repair:
                    print(
                        f"BACKFILL_REPAIR(NFL): week={season}-{target_week} "
                        f"reason=odds_rvo_missing"
                    )

            write_atomic_jsonl(json_path, final_rows)
            final_df = pd.DataFrame(final_rows)
            if csv_df is not None:
                final_df = _align_columns(final_df, list(csv_df.columns))
            write_atomic_csv(csv_path, final_df)
            files_rewritten += 1

            if file_changed:
                print(
                    f"BACKFILL_MERGE(NFL): week={season}-{target_week} "
                    f"updated_scores={row_updates} preserved_odds={preservation['preserved_odds']} "
                    f"preserved_rvo={preservation['preserved_rvo']}"
                )

    return {
        "weeks": [f"W{w}" for w in weeks],
        "updated": updated_total,
        "skipped": skipped_total,
        "files_rewritten": files_rewritten,
        "preserved_odds": preserved_odds_total,
        "preserved_rvo": preserved_rvo_total,
    }


__all__ = ["backfill_nfl_scores"]

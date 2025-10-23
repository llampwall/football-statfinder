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
import math
from collections import Counter, defaultdict
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.backfill_merge import merge_games_week, summarize_preservation
from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl, write_atomic_json
from src.common.io_utils import ensure_out_dir, getenv
from src.common.team_names import TEAM_ABBR_TO_FULL, team_merge_key
from src.odds.nfl_promote_week import promote_week_odds
from src.odds.ats_compute import compute_ats, resolve_closing_spread, is_blank as ats_is_blank
from src.schedule_master import load_master as load_nfl_schedule_master
from src.odds.ats_compute import compute_ats, resolve_closing_spread, is_blank as atsats_is_blank

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


def _rebuild_nfl_game_view(season: int, week: int) -> None:
    cmd = [sys.executable, "-m", "src.gameview_build", "--season", str(season), "--week", str(week)]
    subprocess.run(cmd, check=False)

def _needs_odds_repair(rows: list[dict]) -> bool:
    """Return True if odds/rvo look nuked on any row."""
    for r in rows or []:
        if (
            r.get("spread_favored_team") in (None, "")
            and r.get("rating_vs_odds") in (None, "")
        ):
            return True
    return False


def _is_finite_number(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_sidecar(sidecar_dir: Path, game_key: str, cache: Dict[str, dict]) -> Optional[dict]:
    if game_key in cache:
        return cache[game_key]
    path = sidecar_dir / f"{game_key}.json"
    if not path.exists():
        cache[game_key] = None  # cache miss to avoid repeat IO
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        cache[game_key] = None
        return None
    cache[game_key] = {"path": path, "data": payload, "dirty": False}
    return cache[game_key]


def _update_sidecar_entry(entries: Iterable[dict], season: int, week: int, *, ats: Optional[str], margin: Optional[float]) -> bool:
    updated = False
    for entry in entries or []:
        entry_season = entry.get("season")
        entry_week = entry.get("week")
        try:
            entry_season = int(entry_season)
            entry_week = int(entry_week)
        except Exception:
            continue
        if entry_season != season or entry_week != week:
            continue
        if ats and atsats_is_blank(entry.get("ats")):
            entry["ats"] = ats
            updated = True
        if margin is not None:
            current_margin = entry.get("to_margin")
            if not _is_finite_number(current_margin):
                entry["to_margin"] = round(float(margin), 2)
                updated = True
        break
    return updated




def _sidecar_needs_ats(entries: Iterable[dict], season: int, week: int) -> bool:
    for entry in entries or []:
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except Exception:
            continue
        if entry_season != season or entry_week != week:
            continue
        if ats_is_blank(entry.get("ats")):
            return True
        if not _is_finite_number(entry.get("to_margin")):
            return True
        return False
    return True
def _compute_team_ats(entries: Iterable[dict], season: int, thru_week: int) -> Tuple[str, Optional[float]]:
    wins = losses = pushes = 0
    margins: List[float] = []
    for entry in entries or []:
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except Exception:
            continue
        if entry_season != season or entry_week > thru_week:
            continue
        result = (entry.get("ats") or "").strip().upper()
        if result == "W":
            wins += 1
        elif result == "L":
            losses += 1
        elif result == "P":
            pushes += 1
        margin = entry.get("to_margin")
        if _is_finite_number(margin):
            margins.append(float(margin))
    record = f"{wins}-{losses}-{pushes}"
    avg_margin = sum(margins) / len(margins) if margins else None
    return record, avg_margin


def _game_needs_ats(row: Mapping[str, Any]) -> bool:
    if ats_is_blank(row.get("home_ats")) or ats_is_blank(row.get("away_ats")):
        return True
    if not _is_finite_number(row.get("home_to_margin_pg")) or not _is_finite_number(row.get("away_to_margin_pg")):
        return True
    return False


def _apply_ats_backfill(
    league: str,
    season: int,
    week: int,
    rows: List[dict],
    sidecar_dir: Path,
) -> Tuple[int, Counter, bool]:
    sidecar_cache: Dict[str, Optional[dict]] = {}
    source_counts: Counter = Counter()
    games_fixed = 0
    rows_changed = False

    for row in rows:
        game_key = row.get("game_key")
        if not isinstance(game_key, str):
            continue
        row_needs = _game_needs_ats(row)
        sidecar_entry = _load_sidecar(sidecar_dir, game_key, sidecar_cache)
        sidecar_needs = False
        if sidecar_entry:
            data = sidecar_entry["data"]
            sidecar_needs = _sidecar_needs_ats(data.get("home_ytd"), season, week) or _sidecar_needs_ats(data.get("away_ytd"), season, week)
        if not row_needs and not sidecar_needs:
            continue
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        try:
            home_score_int = int(home_score)
            away_score_int = int(away_score)
        except Exception:
            continue
        closing = resolve_closing_spread(league, season, row)
        if not closing:
            continue
        ats_payload = compute_ats(home_score_int, away_score_int, closing.get("favored_team"), closing.get("spread"))
        if not ats_payload:
            continue

        game_updated = False

        if sidecar_entry:
            data = sidecar_entry["data"]
            home_changed = _update_sidecar_entry(
                data.get("home_ytd"), season, week, ats=ats_payload["home_ats"], margin=ats_payload["to_margin_home"]
            )
            away_changed = _update_sidecar_entry(
                data.get("away_ytd"), season, week, ats=ats_payload["away_ats"], margin=ats_payload["to_margin_away"]
            )
            if home_changed or away_changed:
                sidecar_entry["dirty"] = True
                game_updated = True

            home_record, home_avg = _compute_team_ats(data.get("home_ytd"), season, week)
            away_record, away_avg = _compute_team_ats(data.get("away_ytd"), season, week)

            if home_record and row.get("home_ats") != home_record:
                row["home_ats"] = home_record
                game_updated = True
            if away_record and row.get("away_ats") != away_record:
                row["away_ats"] = away_record
                game_updated = True
            if home_avg is not None:
                avg_val = round(float(home_avg), 2)
                current = row.get("home_to_margin_pg")
                if not _is_finite_number(current) or abs(float(current) - avg_val) > 1e-6:
                    row["home_to_margin_pg"] = avg_val
                    game_updated = True
            if away_avg is not None:
                avg_val = round(float(away_avg), 2)
                current = row.get("away_to_margin_pg")
                if not _is_finite_number(current) or abs(float(current) - avg_val) > 1e-6:
                    row["away_to_margin_pg"] = avg_val
                    game_updated = True
        else:
            if ats_is_blank(row.get("home_ats")):
                result = ats_payload["home_ats"]
                if result == "W":
                    row["home_ats"] = "1-0-0"
                elif result == "L":
                    row["home_ats"] = "0-1-0"
                else:
                    row["home_ats"] = "0-0-1"
                game_updated = True
            if ats_is_blank(row.get("away_ats")):
                result = ats_payload["away_ats"]
                if result == "W":
                    row["away_ats"] = "1-0-0"
                elif result == "L":
                    row["away_ats"] = "0-1-0"
                else:
                    row["away_ats"] = "0-0-1"
                game_updated = True
            if not _is_finite_number(row.get("home_to_margin_pg")):
                row["home_to_margin_pg"] = round(ats_payload["to_margin_home"], 2)
                game_updated = True
            if not _is_finite_number(row.get("away_to_margin_pg")):
                row["away_to_margin_pg"] = round(ats_payload["to_margin_away"], 2)
                game_updated = True

        if game_updated:
            row.setdefault("raw_sources", {})["closing_spread"] = {
                "source": closing.get("source"),
                "book": closing.get("book"),
                "spread": closing.get("spread"),
                "favored_team": closing.get("favored_team"),
                "fetched_ts": closing.get("fetched_ts"),
            }
            source_counts[closing.get("source", "unknown")] += 1
            games_fixed += 1
            rows_changed = True

    for entry in sidecar_cache.values():
        if entry and entry.get("dirty"):
            write_atomic_json(entry["path"], entry["data"])

    return games_fixed, source_counts, rows_changed


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
    total_ats_fixed = 0
    ats_source_counts: Counter = Counter()
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
        needs_repair = _needs_odds_repair(existing_rows)
        
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

        merged_rows = merge_games_week(existing_rows, incoming_rows)
        preservation = summarize_preservation(existing_rows, merged_rows)

        final_rows = merged_rows
        rebuild_game_view = False
        promoted = 0

        if promote_prev_enabled and target_week < week:
            promoted_rows = deepcopy(merged_rows)
            promote_stats = promote_week_odds(promoted_rows, season, target_week)
            promoted = promote_stats.get("promoted_games", 0)
            final_rows = promoted_rows
            if promoted > 0:
                rebuild_game_view = True
                file_changed = True
            print(
                f"ODDS_REPROMOTE(NFL): week={season}-{target_week} "
                f"promoted={promoted} source=staging/odds_pinned"
            )

            if not file_changed and needs_repair:
                print(
                    f"BACKFILL_REPAIR(NFL): week={season}-{target_week} "
                    f"reason=odds_rvo_missing"
                )

        sidecar_dir = NFL_ROOT / f"{season}_week{target_week}" / "game_schedules"
        ats_games_fixed, ats_counts, ats_rows_changed = _apply_ats_backfill(
            "nfl", season, target_week, final_rows, sidecar_dir
        )
        total_ats_fixed += ats_games_fixed
        ats_source_counts.update(ats_counts)
        if ats_rows_changed:
            file_changed = True
        source_str = (
            f"pinned:{ats_counts.get('pinned', 0)},"
            f"snapshot:{ats_counts.get('snapshot', 0)},"
            f"history:{ats_counts.get('history', 0)}"
        )
        print(
            f"ATS_BACKFILL(NFL): week={season}-{target_week} "
            f"games_fixed={ats_games_fixed} source_counts={source_str}"
        )
        if ats_games_fixed > 0:
            rebuild_game_view = True

        if file_changed or promoted > 0:
            preserved_odds_total += preservation["preserved_odds"]
            preserved_rvo_total += preservation["preserved_rvo"]

            write_atomic_jsonl(json_path, final_rows)
            final_df = pd.DataFrame(final_rows)
            if csv_df is not None:
                final_df = _align_columns(final_df, list(csv_df.columns))
            write_atomic_csv(csv_path, final_df)
            files_rewritten += 1

            if row_updates > 0:
                print(
                    f"BACKFILL_MERGE(NFL): week={season}-{target_week} "
                    f"updated_scores={row_updates} preserved_odds={preservation['preserved_odds']} "
                    f"preserved_rvo={preservation['preserved_rvo']}"
                )

            if rebuild_game_view:
                _rebuild_nfl_game_view(season, target_week)
    return {
        "weeks": [f"W{w}" for w in weeks],
        "updated": updated_total,
        "skipped": skipped_total,
        "files_rewritten": files_rewritten,
        "preserved_odds": preserved_odds_total,
        "preserved_rvo": preserved_rvo_total,
        "ats_fixed": total_ats_fixed,
        "ats_sources": dict(ats_source_counts),
    }


__all__ = ["backfill_nfl_scores"]
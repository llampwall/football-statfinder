"""CFB scores backfill helpers (prior weeks before ATS).

Purpose & scope:
    Load recent College Football week files, hydrate missing final scores from
    the schedule master, and rewrite the week JSONL/CSV outputs atomically so
    the ATS step operates on complete historical data.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (D, E, F, I)

Invariants:
    * Only past weeks are touched; current week is never mutated.
    * Weekly files are rewritten atomically when updates occur.
    * Schedule master remains the source of truth for authoritative finals.

Side effects:
    * Rewrites ``out/cfb/{season}_week{w}/games_week_{season}_{w}.jsonl`` and
      matching CSV when scores are backfilled.

Do not:
    * Attempt to infer scores when the schedule master lacks both totals.
    * Modify weeks beyond the configured lookback window.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.backfill_merge import merge_games_week, summarize_preservation
from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl, write_atomic_json
from src.common.io_utils import ensure_out_dir, getenv
from src.odds.cfb_promote_week import promote_week_odds
from src.odds.ats_compute import compute_ats, resolve_closing_spread, is_blank as ats_is_blank
from src.schedule_master_cfb import load_master

OUT_ROOT = ensure_out_dir()
CFB_ROOT = OUT_ROOT / "cfb"


def _team_token(value: Optional[str]) -> str:
    text = (value or "").lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    parts = [part for part in token.split("_") if part]
    return "_".join(parts)


def _build_game_key_from_row(row: pd.Series) -> Optional[str]:
    kickoff = row.get("kickoff_iso_utc")
    if not isinstance(kickoff, str) or "T" not in kickoff:
        return None
    try:
        dt = pd.to_datetime(kickoff, utc=True)
    except Exception:
        return None
    if pd.isna(dt):
        return None
    away_token = _team_token(row.get("away_team_norm"))
    home_token = _team_token(row.get("home_team_norm"))
    yyyymmdd = dt.strftime("%Y%m%d")
    hhmm = dt.strftime("%H%M")
    return f"{yyyymmdd}_{hhmm}_{away_token}_{home_token}"


def _load_week_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def _is_missing_score(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, str):
        return not value.strip()
    return False


def _build_score_lookup(season: int) -> Dict[str, Tuple[int, int]]:
    master_df = load_master()
    if master_df.empty:
        return {}
    season_df = master_df[master_df["season"] == season].copy()
    if season_df.empty:
        return {}
    season_df["game_key"] = season_df.apply(_build_game_key_from_row, axis=1)
    season_df = season_df.dropna(subset=["game_key", "home_score", "away_score"])
    lookup: Dict[str, Tuple[int, int]] = {}
    for row in season_df.itertuples(index=False):
        game_key = getattr(row, "game_key", None)
        if not isinstance(game_key, str):
            continue
        home_score = getattr(row, "home_score", None)
        away_score = getattr(row, "away_score", None)
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        lookup[game_key] = (int(round(float(home_score))), int(round(float(away_score))))
    return lookup


def _derive_weeks_to_backfill(current_week: int, include_weeks_back: int) -> List[int]:
    weeks: List[int] = []
    for offset in range(1, include_weeks_back + 1):
        target = current_week - offset
        if target >= 1:
            weeks.append(target)
    return sorted(weeks)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _align_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = [col for col in columns if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + remainder)


def _is_finite_number(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_sidecar(sidecar_dir: Path, game_key: str, cache: Dict[str, Optional[dict]]) -> Optional[dict]:
    if game_key in cache:
        return cache[game_key]
    path = sidecar_dir / f"{game_key}.json"
    if not path.exists():
        cache[game_key] = None
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
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except Exception:
            continue
        if entry_season != season or entry_week != week:
            continue
        if ats and ats_is_blank(entry.get("ats")):
            entry["ats"] = ats
            updated = True
        if margin is not None:
            current_margin = entry.get("to_margin")
            if not _is_finite_number(current_margin):
                entry["to_margin"] = round(float(margin), 2)
                updated = True
        break
    return updated


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


def _rebuild_cfb_game_view(season: int, week: int) -> None:
    cmd = [sys.executable, "-m", "src.gameview_build_cfb", "--season", str(season), "--week", str(week)]
    subprocess.run(cmd, check=False)

def _needs_odds_repair(rows: list[dict]) -> bool:
    """Return True if any row looks like odds/rvo were nuked."""
    for r in rows or []:
        # pick a couple of canonical fields that must not be None/blank
        if (
            r.get("spread_favored_team") in (None, "",)
            and r.get("rating_vs_odds") in (None, "",)
        ):
            return True
    return False


def backfill_cfb_scores(
    season: int,
    week: int,
    *,
    include_weeks_back: Optional[int] = None,
) -> Dict[str, object]:
    """Backfill missing scores for prior weeks using the schedule master.

    Args:
        season: Target season identifier.
        week: Current week (used to bound the lookback).
        include_weeks_back: Optional override for how many weeks to scan.

    Returns:
        Dict summarising the operation with keys: ``weeks`` (list of ints),
        ``updated`` (count), ``skipped`` (missing finals), ``files_rewritten``.
    """
    if include_weeks_back is None:
        include_weeks_back = int(getenv("BACKFILL_WEEKS", "2") or "2")
    if include_weeks_back <= 0 or week <= 1:
        return {"weeks": [], "updated": 0, "skipped": 0, "files_rewritten": 0}

    weeks_to_scan = _derive_weeks_to_backfill(week, include_weeks_back)
    if not weeks_to_scan:
        return {"weeks": [], "updated": 0, "skipped": 0, "files_rewritten": 0}

    score_lookup = _build_score_lookup(season)
    updated = 0
    skipped = 0
    files_rewritten = 0
    preserved_odds_total = 0
    preserved_rvo_total = 0
    total_ats_fixed = 0
    ats_source_counts: Counter = Counter()
    promote_prev_enabled = _is_truthy(getenv("BACKFILL_PROMOTE_PREV", "0"))

    for prior_week in weeks_to_scan:
        week_dir = CFB_ROOT / f"{season}_week{prior_week}"
        json_path = week_dir / f"games_week_{season}_{prior_week}.jsonl"
        csv_path = week_dir / f"games_week_{season}_{prior_week}.csv"
        existing_rows = _load_week_json(json_path)
        if not existing_rows:
            continue

        incoming_rows = deepcopy(existing_rows)
        csv_df = pd.read_csv(csv_path) if csv_path.exists() else None
        row_updates = 0
        week_changed = False
        needs_repair = _needs_odds_repair(existing_rows)

        for idx, row in enumerate(incoming_rows):
            game_key = row.get("game_key")
            if not isinstance(game_key, str):
                continue
            home_score_present = not _is_missing_score(row.get("home_score"))
            away_score_present = not _is_missing_score(row.get("away_score"))
            if home_score_present and away_score_present:
                continue
            scores = score_lookup.get(game_key)
            if not scores:
                if not home_score_present or not away_score_present:
                    skipped += 1
                continue

            incoming_rows[idx]["home_score"], incoming_rows[idx]["away_score"] = scores
            week_changed = True
            updated += 1
            row_updates += 1

        merged_rows = merge_games_week(existing_rows, incoming_rows)
        preservation = summarize_preservation(existing_rows, merged_rows)

        final_rows = merged_rows
        rebuild_game_view = False
        promoted_games = 0

        if promote_prev_enabled and prior_week < week:
            promoted_rows = deepcopy(merged_rows)
            promote_stats = promote_week_odds(promoted_rows, season, prior_week)
            promoted_games = promote_stats.get("promoted_games", 0)
            final_rows = promoted_rows
            if promoted_games > 0:
                rebuild_game_view = True
                week_changed = True
            print(
                f"ODDS_REPROMOTE(CFB): week={season}-{prior_week} "
                f"promoted={promoted_games} source=staging/odds_pinned"
            )

            if not week_changed and needs_repair:
                print(
                    f"BACKFILL_REPAIR(CFB): week={season}-{prior_week} "
                    f"reason=odds_rvo_missing"
                )

        sidecar_dir = week_dir / "game_schedules"
        ats_games_fixed, ats_counts, ats_rows_changed = _apply_ats_backfill(
            "cfb", season, prior_week, final_rows, sidecar_dir
        )
        total_ats_fixed += ats_games_fixed
        ats_source_counts.update(ats_counts)
        if ats_rows_changed:
            week_changed = True
        source_str = (
            f"pinned:{ats_counts.get('pinned', 0)},"
            f"snapshot:{ats_counts.get('snapshot', 0)},"
            f"history:{ats_counts.get('history', 0)}"
        )
        print(
            f"ATS_BACKFILL(CFB): week={season}-{prior_week} "
            f"games_fixed={ats_games_fixed} source_counts={source_str}"
        )
        if ats_games_fixed > 0:
            rebuild_game_view = True

        if week_changed or promoted_games > 0:
            final_df = pd.DataFrame(final_rows)
            if csv_df is not None:
                final_df = _align_columns(final_df, list(csv_df.columns))

            if not final_rows or final_df.empty:
                print(
                    f"Scores(CFB): week={season}-{prior_week} "
                    "already up to date; 0 rows written"
                )
                continue

            preserved_odds_total += preservation["preserved_odds"]
            preserved_rvo_total += preservation["preserved_rvo"]

            write_atomic_jsonl(json_path, final_rows)
            write_atomic_csv(csv_path, final_df)
            files_rewritten += 1

            if week_changed:
                print(
                    f"BACKFILL_MERGE(CFB): week={season}-{prior_week} "
                    f"updated_scores={row_updates} "
                    f"preserved_odds={preservation['preserved_odds']} "
                    f"preserved_rvo={preservation['preserved_rvo']}"
                )

            if rebuild_game_view:
                _rebuild_cfb_game_view(season, prior_week)


    if updated == 0 and files_rewritten == 0:
        print("Scores(CFB): current week already up to date; 0 rows written")

    return {
        "weeks": weeks_to_scan,
        "updated": updated,
        "skipped": skipped,
        "files_rewritten": files_rewritten,
        "preserved_odds": preserved_odds_total,
        "preserved_rvo": preserved_rvo_total,
        "ats_fixed": total_ats_fixed,
        "ats_sources": dict(ats_source_counts),
    }


__all__ = ["backfill_cfb_scores"]

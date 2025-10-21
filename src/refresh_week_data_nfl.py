"""NFL refresh wrapper with odds staging & promotion (legacy builder + staging hook).

Purpose & scope:
    Preserve the legacy NFL weekly refresh behaviour while adding
    append-only odds staging (raw ingest + schedule pinning) immediately
    after the legacy pipeline completes.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B1–B2, F, H, I)

Notes:
    * CLI remains identical to `src.refresh_week_data` (season/week required).
    * The wrapper logs CurrentWeek(NFL) (read-only), invokes the legacy
      builder, then runs staging and emits a single diagnostics line plus
      a notify summary.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from src.common.current_week_service import get_current_week
from src.common.io_utils import ensure_out_dir
from src.odds.nfl_ingest import ingest_nfl_odds_raw
from src.odds.nfl_pin_to_schedule import pin_nfl_odds
from src.odds.nfl_promote_week import (
    promote_week_odds,
    read_week_json,
    write_week_outputs,
    diff_game_rows,
)
from src.ratings.sagarin_nfl_fetch import run_nfl_sagarin_staging
from src.scores.nfl_backfill import backfill_nfl_scores
from src.ats.nfl_ats import build_team_ats, apply_ats_to_week


def _extract_args(argv: list[str]) -> Tuple[int, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    known, _ = parser.parse_known_args(argv)
    if known.season is None or known.week is None:
        raise SystemExit("NFL refresh requires --season and --week arguments.")
    return int(known.season), int(known.week)


def _log_current_week_readonly() -> None:
    season, week, ts = get_current_week("NFL")
    print(f"CurrentWeek(NFL)={season} W{week} computed_at={ts} (readonly)")


def _run_odds_staging(season: int, week: int) -> Dict[str, Any]:
    ingest_result = ingest_nfl_odds_raw()
    raw_records = ingest_result.get("records", []) or []

    day_window = int(os.getenv("ODDS_PIN_DAY_WINDOW", "3") or "3")
    max_delta_hours = float(os.getenv("ODDS_PIN_MAX_KICKOFF_DELTA_HOURS", "36") or "36")
    role_swap_enabled = os.getenv("ODDS_ROLE_SWAP_TOLERANCE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "disabled",
    }

    pin_result = pin_nfl_odds(
        raw_records,
        day_window=day_window,
        max_delta_hours=max_delta_hours,
        role_swap_tolerance=role_swap_enabled,
    )
    counts = pin_result["counts"]
    markets_snapshot = counts.get("markets", {})
    books_snapshot = counts.get("books", {})
    log_line = (
        "NFL ODDS STAGING: "
        f"raw={counts.get('raw', 0)} "
        f"pinned={counts.get('pinned', 0)} "
        f"unmatched={counts.get('unmatched', 0)} "
        f"candidate_sets_zero={counts.get('candidate_sets_zero', 0)} "
        f"candidate_sets_multi={counts.get('candidate_sets_multi', 0)} "
        f"markets={markets_snapshot} "
        f"books={books_snapshot}"
    )
    if counts.get("raw", 0) > 0 and counts.get("pinned", 0) == 0:
        log_line += ' hint="provider ahead or schedule mismatch"'
    print(log_line)
    return counts


def main() -> None:
    season, week = _extract_args(sys.argv[1:])
    _log_current_week_readonly()

    from src.refresh_week_data import main as legacy_main

    exit_code = legacy_main()
    if exit_code not in (None, 0):
        raise SystemExit(exit_code)

    staging_counts = _run_odds_staging(season, week)
    run_nfl_sagarin_staging()

    backfill_summary = backfill_nfl_scores(season, week)
    weeks_label = "[" + ",".join(backfill_summary.get("weeks", [])) + "]"
    print(
        f"Scores(NFL): weeks={weeks_label} updated={backfill_summary.get('updated', 0)} "
        f"skipped={backfill_summary.get('skipped', 0)}"
    )

    team_ats = build_team_ats(season, week)
    ats_rows_updated = apply_ats_to_week(season, week, team_ats)
    print(f"ATS(NFL): teams={apply_ats_to_week.teams_in_week} rows_updated={ats_rows_updated}")

    promotion_info = None
    legacy_mismatch = 0
    promoted_total = 0
    rows_count = 0

    out_root = ensure_out_dir()
    week_dir = out_root / f"{season}_week{week}"
    json_path = week_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = week_dir / f"games_week_{season}_{week}.csv"

    if os.getenv("ODDS_PROMOTION_ENABLE", "1").strip().lower() not in {"0", "false", "off", "disabled"}:
        legacy_rows = read_week_json(json_path)
        rows = copy.deepcopy(legacy_rows)
        policy = os.getenv("ODDS_SELECT_POLICY", "latest_by_fetch_ts") or "latest_by_fetch_ts"
        promotion_info = promote_week_odds(rows, season, week, policy=policy)
        promoted_total = promotion_info.get("promoted_games", 0)
        rows_count = len(rows)
        if promoted_total > 0:
            write_week_outputs(rows, season, week)
        legacy_flag = os.getenv("ODDS_LEGACY_JOIN_ENABLE", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "disabled",
        }
        if legacy_flag:
            diff_summary = diff_game_rows(rows, legacy_rows)
            legacy_mismatch = diff_summary.get("mismatched", 0)
        log_line = (
            f"NFL ODDS PROMOTION: week={season}-{week} promoted={promoted_total} "
            f"by_market={promotion_info.get('by_market', {})} "
            f"by_book={promotion_info.get('by_book', {})} "
            f"legacy_mismatch={legacy_mismatch}"
        )
        if promoted_total == 0 and promotion_info.get("other_week_records", 0):
            log_line += ' hint="provider ahead; staged"'
        print(log_line)
    else:
        print("NFL ODDS PROMOTION: disabled via ODDS_PROMOTION_ENABLE")
        rows_count = len(read_week_json(json_path))

    print(
        f"NOTIFY: NFL refresh complete week={season}-{week} rows={rows_count} "
        f"odds_promoted={promoted_total}."
    )


if __name__ == "__main__":
    main()

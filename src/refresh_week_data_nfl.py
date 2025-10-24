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
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple
from pathlib import Path

from src.common.current_week_service import get_current_week
from src.common.io_atomic import write_atomic_json
from src.common.io_utils import ensure_out_dir, getenv
from src.odds.nfl_ingest import ingest_nfl_odds_raw
from src.odds.nfl_pin_to_schedule import pin_nfl_odds
from src.odds.nfl_promote_week import (
    promote_week_odds,
    read_week_json,
    write_week_outputs,
    diff_game_rows,
)
from src.odds.ats_compute import resolve_closing_spread, compute_ats
from src.ratings.sagarin_nfl_fetch import run_nfl_sagarin_staging
from src.scores.nfl_backfill import backfill_nfl_scores, _update_sidecar_entry
from src.ats.nfl_ats import build_team_ats, apply_ats_to_week


def _run_odds_staging(season: int, week: int) -> Dict[str, Any]:
    ingest_result = ingest_nfl_odds_raw()
    raw_records = ingest_result.get("records", []) or []

    day_window = int(getenv("ODDS_PIN_DAY_WINDOW", "3") or "3")
    max_delta_hours = float(getenv("ODDS_PIN_MAX_KICKOFF_DELTA_HOURS", "36") or "36")
    role_swap_enabled = getenv("ODDS_ROLE_SWAP_TOLERANCE", "1").strip().lower() not in {
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


_BLANK_SENTINELS = {None, "", "\u2014", "-"}

def _merge_only(old_value: Any, new_value: Any) -> Any:
    return new_value if old_value in _BLANK_SENTINELS else old_value

def _to_records_if_df(obj):
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records"), "df"
    except Exception:
        pass
    return obj, "list"


def _load_sidecar_map(sidecar_dir) -> Dict[str, Dict[str, Any]]:
    side_map: Dict[str, Dict[str, Any]] = {}
    if not sidecar_dir or not sidecar_dir.exists():
        return side_map
    for path in sidecar_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        game_key = payload.get("game_key") or path.stem
        if not isinstance(game_key, str) or not game_key:
            continue
        side_map[game_key] = {"data": payload, "path": path, "dirty": False}
    return side_map


# --- replace the old backfill_ats_for_week with this version in BOTH NFL/CFB refreshers ---

def backfill_ats_for_week(
    LEAGUE: str,
    season: int,
    week: int,
    games: List[Dict[str, Any]],
    sidecar_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Sidecars-first fix:
      - If sidecar YTD row is blank for this week, fill it.
        • Prefer values already on the game row (fast path).
        • Otherwise resolve closing spread (history→pinned→snapshot) and compute ATS.
      - Merge-only everywhere (no overwrites of non-blank).
      - WRITE THE SIDECAR IMMEDIATELY when we change it (no outer flag dependency).
    """

    counts = Counter()
    games_fixed = 0

    for g in games or []:
        game_key = g.get("game_key")
        entry = sidecar_map.get(game_key) or {}
        sc = entry.get("data") or {}
        home_arr = sc.get("home_ytd") or []
        away_arr = sc.get("away_ytd") or []

        # Values on the game row (may already be present)
        g_home_ats = g.get("home_ats")
        g_away_ats = g.get("away_ats")
        g_tm_home = g.get("to_margin_home")
        g_tm_away = g.get("to_margin_away")

        # If the sidecar YTD entries for this week are blank, we need values.
        need_home = need_away = True  # let _update_sidecar_entry decide but we still route logic

        # Fast path: if game already has ATS, use it to patch sidecars.
        have_game_ats = bool(g_home_ats or g_away_ats)
        if not have_game_ats:
            # Only compute if we actually need to populate sidecars (merge-only means
            # _update_sidecar_entry will no-op if the sidecar’s already filled).
            res = resolve_closing_spread(LEAGUE, season, g)
            if res:
                out = compute_ats(
                    int(g.get("home_score") or 0),
                    int(g.get("away_score") or 0),
                    res["favored_team"],
                    float(res["spread"]),
                )
                # merge onto the game row
                g["home_ats"] = _merge_only(g_home_ats, out["home_ats"])
                g["away_ats"] = _merge_only(g_away_ats, out["away_ats"])
                g["to_margin_home"] = _merge_only(g_tm_home, out["to_margin_home"])
                g["to_margin_away"] = _merge_only(g_tm_away, out["to_margin_away"])

                # refresh locals
                g_home_ats = g.get("home_ats")
                g_away_ats = g.get("away_ats")
                g_tm_home = g.get("to_margin_home")
                g_tm_away = g.get("to_margin_away")

                counts[res["source"]] += 1
                games_fixed += 1
            else:
                # No source available; skip silently
                counts["unresolved"] += 1

        # --- Patch sidecars (merge-only); if anything changed, write immediately ---
        changed = False
        try:
            if _update_sidecar_entry(home_arr, season, week, ats=g_home_ats, to_margin=g_tm_home):
                changed = True
            if _update_sidecar_entry(away_arr, season, week, ats=g_away_ats, to_margin=g_tm_away):
                changed = True
        except NameError:
            # If helper lives in the league module, import there; keeping this guard avoids hard-crash.
            pass

        if changed:
            # Persist now — don’t depend on outer layers.
            path = entry.get("path")
            if path:
                write_atomic_json(Path(path), sc)
            entry["dirty"] = True  # keep signal for any outer bookkeeping

    return games, games_fixed, {k: int(v) for k, v in counts.items()}




def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    args, remaining = parser.parse_known_args(sys.argv[1:])

    if (args.season is None) != (args.week is None):
        raise SystemExit("NFL refresh: provide both --season and --week or neither.")

    current_week_info = None
    current_week_error = None
    try:
        current_week_info = get_current_week("NFL")
    except Exception as exc:
        current_week_error = exc

    if args.season is None:
        if current_week_info is None:
            raise SystemExit(f"Global Week Service unavailable: {current_week_error}")
        season, week, current_ts = current_week_info
        args_label = "auto"
    else:
        season = int(args.season)
        week = int(args.week)
        current_ts = current_week_info[2] if current_week_info else "unknown"
        args_label = "manual"

    if current_week_info:
        cur_season, cur_week, cur_ts = current_week_info
        print(f"CurrentWeek(NFL)={cur_season} W{cur_week} computed_at={cur_ts}; args={args_label}")
    else:
        print(f"CurrentWeek(NFL)=unavailable; args={args_label}")

    from src.refresh_week_data import main as legacy_main

    original_argv = sys.argv[:]
    legacy_argv = [original_argv[0], "--season", str(season), "--week", str(week), *remaining]
    sys.argv = legacy_argv
    try:
        exit_code = legacy_main()
    finally:
        sys.argv = original_argv
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

    out_root = ensure_out_dir()
    week_dir = out_root / f"{season}_week{week}"
    json_path = week_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = week_dir / f"games_week_{season}_{week}.csv"
    sidecar_dir = week_dir / "game_schedules"

    current_rows = read_week_json(json_path)
    sidecar_map = _load_sidecar_map(sidecar_dir)
    current_rows, ats_games_fixed, ats_counts = backfill_ats_for_week(
        "NFL",
        season,
        week,
        current_rows,
        sidecar_map,
    )
    rows_for_write = current_rows if isinstance(current_rows, list) else current_rows.to_dict("records")  # type: ignore[attr-defined]
    if ats_games_fixed > 0:
        write_week_outputs(rows_for_write, season, week)
    for entry in sidecar_map.values():
        if entry.get("dirty"):
            write_atomic_json(entry["path"], entry["data"])

    team_ats = build_team_ats(season, week)
    ats_rows_updated = apply_ats_to_week(season, week, team_ats)
    print(f"ATS(NFL): teams={apply_ats_to_week.teams_in_week} rows_updated={ats_rows_updated}")

    promotion_info = None
    legacy_mismatch = 0
    promoted_total = 0

    if getenv("ODDS_PROMOTION_ENABLE", "1").strip().lower() not in {"0", "false", "off", "disabled"}:
        legacy_rows = read_week_json(json_path)
        rows = copy.deepcopy(legacy_rows)
        policy = getenv("ODDS_SELECT_POLICY", "latest_by_fetch_ts") or "latest_by_fetch_ts"
        promotion_info = promote_week_odds(rows, season, week, policy=policy)
        promoted_total = promotion_info.get("promoted_games", 0)
        if promoted_total > 0:
            write_week_outputs(rows, season, week)
        legacy_flag = getenv("ODDS_LEGACY_JOIN_ENABLE", "0").strip().lower() not in {
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

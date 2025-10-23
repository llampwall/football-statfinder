"""Weekly orchestrator for refreshing NFL outputs in one command.

Purpose:
    Coordinate the game view build, league metrics table, and odds snapshot.
Inputs:
    Season/week arguments plus optional env keys (THE_ODDS_API_KEY).
Outputs:
    - /out/games_week_{season}_{week}.jsonl and .csv
    - /out/league_metrics_{season}_{week}.csv
    - /out/odds_{season}_wk{week}.jsonl (if API key available)
Source(s) of truth:
    nflverse schedules & stats; The Odds API for pricing when credentials exist.
Example:
    python -m src.refresh_week_data --season 2025 --week 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.io_utils import read_env, week_out_dir, write_jsonl, write_csv
from src.common.team_names import team_merge_key
from src.fetch_games import filter_week_reg, load_games, get_schedule_df
from src.fetch_week_odds_nfl import fetch_odds_theoddsapi
from src.fetch_sagarin_week_nfl import fetch_sagarin_week
from src.gameview_build import build_gameview
from src.fetch_year_to_date_stats import generate_league_metrics
from src.sagarin_master import append_week, load_master as load_sagarin_master
from src.schedule_master import ensure_weeks_present, load_master as load_schedule_master
from src.build_team_timelines import build_timelines


def _is_int_in_range(value, lo: int = 1, hi: int = 32) -> bool:
    try:
        ivalue = int(value)
        return lo <= ivalue <= hi
    except Exception:
        return False


def _count_non_null(series: Optional[pd.Series]) -> int:
    if series is None:
        return 0
    return int(series.notna().sum())


def _nan_inf_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return 0
    arr = numeric.to_numpy(dtype="float64", copy=False)
    return int(np.isinf(arr).sum() + np.isnan(arr).sum())


def refresh_week(season: int, week: int, bookmaker: str = "Pinnacle", build_timelines_flag: bool = True) -> dict:
    out_dir = week_out_dir(season, week)
    schedule = load_games(season)
    week_games = filter_week_reg(schedule, season, week)
    expected = int(week_games.shape[0])

    league_df = generate_league_metrics(season, week)
    league_path = out_dir / f"league_metrics_{season}_{week}.csv"
    write_csv(league_df, league_path)

    sagarin_result = fetch_sagarin_week(season, week)
    before_master, after_master = append_week(sagarin_result["csv_path"], league="NFL")

    gameview_result = build_gameview(
        season=season,
        week=week,
        sagarin_path=str(sagarin_result["csv_path"]),
    )

    env = read_env(["THE_ODDS_API_KEY"])
    odds_records = []
    odds_path: Optional[Path] = None
    api_key = env.get("THE_ODDS_API_KEY")
    if api_key:
        odds_records = fetch_odds_theoddsapi(season, week, api_key, book_pref=bookmaker)
        odds_path = out_dir / f"odds_{season}_wk{week}.jsonl"
        write_jsonl(odds_records, odds_path)
    else:
        print("Skipped odds fetch: THE_ODDS_API_KEY not set.")

    summary = {
        "schedule_count": expected,
        "gameview_count": gameview_result["count"],
        "league_rows": len(league_df),
        "sagarin_count": sagarin_result["count"],
        "sagarin_csv": sagarin_result["csv_path"],
        "odds_count": len(odds_records),
        "league_path": league_path,
        "gameview_jsonl": gameview_result["jsonl"],
        "gameview_csv": gameview_result["csv"],
        "odds_path": odds_path,
    }

    print("=== WEEKLY REFRESH ===")
    print(f"Season {season} Week {week}")
    print(f"Outputs directory: {out_dir}")
    print(f"Schedule games (REG): {expected}")
    print(f"Game View records:     {gameview_result['count']} ({gameview_result['jsonl']})")
    print(f"League metrics rows:   {len(league_df)} ({league_path})")
    print(f"Sagarin records:       {sagarin_result['count']} ({sagarin_result['csv_path']})")
    print(f"Sagarin master upsert: {after_master - before_master:+d} (rows now {after_master})")
    if sagarin_result.get("page_week") and sagarin_result["page_week"] != week:
        print(
            f"Sagarin page reports week {sagarin_result['page_week']} while request was week {week}."
        )
    if odds_path:
        print(f"Odds records:          {len(odds_records)} ({odds_path})")
    else:
        print("Odds records:          skipped (no API key)")

    if build_timelines_flag:
        pr_master = load_sagarin_master()
        if pr_master.empty:
            print("Game schedules sidecars: skipped (Sagarin master empty)")
        else:
            ensure_weeks_present([season, season - 1])
            sched_master = load_schedule_master()
            mask = (
                (sched_master["league"].astype(str).str.upper() == "NFL")
                & (sched_master["game_type"] == "REG")
                & (sched_master["season"].isin([season, season - 1]))
            )
            schedule_rows = int(mask.sum())
            print(
                f"Schedule master rows (NFL REG) for seasons {season-1},{season}: {schedule_rows}"
            )
            games_df = pd.read_json(gameview_result["jsonl"], lines=True)
            side_map, side_stats = build_timelines(season, week, pr_master, out_dir, games_df)
            print(
                f"Sidecars written: {len(side_map)}/{len(games_df)} -> {out_dir / 'game_schedules'}"
            )
            first_two = list(side_stats.items())[:2]
            for game_key, info in first_two:
                if game_key == "_missing":
                    continue
                print(
                    f"game={game_key} home_ytd={info['home_ytd_len']} away_ytd={info['away_ytd_len']} "
                    f"home_prev={info['home_prev_len']} away_prev={info['away_prev_len']}"
                )
            missing_schedule_games = side_stats.get("_missing", [])
            if missing_schedule_games:
                raise RuntimeError(f"Schedule master missing rows for games: {missing_schedule_games}")
            rank_counts = side_stats.get("_rank_fix_counts", {})
            print(
                "SIDEcar_RANKS(NFL): "
                f"week={season}-{week} "
                f"fixed_pr={rank_counts.get('pr', 0)} "
                f"fixed_opp_pr={rank_counts.get('opp_pr', 0)} "
                f"fixed_sos={rank_counts.get('sos', 0)} "
                f"fixed_opp_sos={rank_counts.get('opp_sos', 0)}"
            )
            filtered_stats = {
                k: v for k, v in side_stats.items() if k not in {"_missing", "_rank_fix_counts"}
            }
            team_expected = sum(info["team_expected_pr"] for info in filtered_stats.values())
            team_present = sum(info["team_present_pr"] for info in filtered_stats.values())
            opp_expected = sum(info["opp_expected_pr"] for info in filtered_stats.values())
            opp_present = sum(info["opp_present_pr"] for info in filtered_stats.values())
            print(f"team PR join missing: {team_expected - team_present}")
            print(f"opp PR join missing: {opp_expected - opp_present}")
            if team_expected > 0 and (team_present / team_expected) < 0.8:
                raise RuntimeError("Team PR coverage below 80% for Sagarin joins")
            if opp_expected > 0 and (opp_present / opp_expected) < 0.8:
                raise RuntimeError("Opponent PR coverage below 80% for Sagarin joins")
    else:
        print("Game schedules sidecars: skipped (--nogames)")

    try:
        games_df = pd.read_json(gameview_result["jsonl"], lines=True)
    except Exception as exc:
        print(f"[diagnostics] unable to load game view JSON ({exc}); skipping extra checks")
    else:
        lm_keys = set()
        if not league_df.empty and "Team" in league_df.columns:
            lm_keys = set(league_df["Team"].map(team_merge_key).dropna())
        team_keys = set()
        for col in ("home_team_raw", "home_team_norm", "away_team_raw", "away_team_norm"):
            if col in games_df.columns:
                mapped = games_df[col].dropna().map(team_merge_key)
                team_keys.update(mapped.dropna().tolist())
        lm_total = len(lm_keys) if lm_keys else 32
        lm_join_coverage = len(team_keys & lm_keys) if lm_keys else 0

        total_slots = len(games_df) * 2
        ry_pg_non_null = _count_non_null(games_df.get("home_ry_pg")) + _count_non_null(games_df.get("away_ry_pg"))
        ry_allowed_non_null = _count_non_null(games_df.get("home_ry_allowed_pg")) + _count_non_null(games_df.get("away_ry_allowed_pg"))
        rush_def_rank_non_null = _count_non_null(games_df.get("home_rush_def_rank")) + _count_non_null(games_df.get("away_rush_def_rank"))

        rank_fields = [
            "home_rush_def_rank",
            "home_pass_def_rank",
            "home_tot_def_rank",
            "away_rush_def_rank",
            "away_pass_def_rank",
            "away_tot_def_rank",
            "home_rush_rank",
            "home_pass_rank",
            "home_tot_off_rank",
            "away_rush_rank",
            "away_pass_rank",
            "away_tot_off_rank",
        ]
        rank_values = [val for field in rank_fields for val in games_df.get(field, pd.Series(dtype=float)).dropna()]
        rank_range_ok = all(_is_int_in_range(val) for val in rank_values)

        spreads = games_df.get("spread_home_relative")
        if spreads is not None:
            spreads_present = spreads.notna()
            favored_present = (
                games_df.get("favored_side")
                .where(spreads_present, other=None)
                .notna()
                if "favored_side" in games_df.columns
                else pd.Series([False] * len(games_df), index=games_df.index)
            )
            favored_spread_present = (
                games_df.get("spread_favored_team")
                .where(spreads_present, other=None)
                .notna()
                if "spread_favored_team" in games_df.columns
                else pd.Series([False] * len(games_df), index=games_df.index)
            )
            favored_diff_present = (
                games_df.get("rating_diff_favored_team")
                .where(spreads_present, other=None)
                .notna()
                if "rating_diff_favored_team" in games_df.columns
                else pd.Series([False] * len(games_df), index=games_df.index)
            )
            favored_rvo_present = (
                games_df.get("rating_vs_odds_favored_team")
                .where(spreads_present, other=None)
                .notna()
                if "rating_vs_odds_favored_team" in games_df.columns
                else pd.Series([False] * len(games_df), index=games_df.index)
            )
            favored_total = int(spreads_present.sum())
            favored_count = int((favored_present & favored_spread_present & favored_diff_present & favored_rvo_present).sum())
        else:
            favored_total = 0
            favored_count = 0

        nan_count = _nan_inf_count(games_df)

        print(f"LM join coverage: {lm_join_coverage}/{lm_total}")
        print(
            f"S2D coverage: ry_pg {ry_pg_non_null}/{total_slots}; "
            f"ry_allowed_pg {ry_allowed_non_null}/{total_slots}; "
            f"rush_def_rank {rush_def_rank_non_null}/{total_slots}"
        )
        print(f"Rank fields in range: {'PASS' if rank_range_ok else 'FAIL'}")
        print(f"Favorite metrics coverage: {favored_count}/{favored_total}")
        print(f"NaN/Inf fields: {nan_count}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh weekly NFL outputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--book", type=str, default="Pinnacle", help="Preferred bookmaker for odds.")
    parser.add_argument("--nogames", action="store_true", help="Skip building per-game timelines")
    args = parser.parse_args()
    refresh_week(args.season, args.week, bookmaker=args.book, build_timelines_flag=not args.nogames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

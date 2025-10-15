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
from src.fetch_games import filter_week_reg, load_games
from src.fetch_week_odds_nfl import fetch_odds_theoddsapi
from src.fetch_sagarin_week_nfl import fetch_sagarin_week
from src.gameview_build import build_gameview
from src.fetch_year_to_date_stats import generate_league_metrics
from src.sagarin_master import append_week


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


def refresh_week(season: int, week: int, bookmaker: str = "Pinnacle") -> dict:
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
    args = parser.parse_args()
    refresh_week(args.season, args.week, bookmaker=args.book)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

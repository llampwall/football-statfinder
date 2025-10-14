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

from src.common.io_utils import ensure_out_dir, read_env, write_jsonl, write_csv
from src.fetch_games import filter_week_reg, load_games
from src.fetch_week_odds_nfl import fetch_odds_theoddsapi
from src.gameview_build import build_gameview
from src.fetch_year_to_date_stats import generate_league_metrics


def refresh_week(season: int, week: int, bookmaker: str = "Pinnacle") -> dict:
    out_dir = ensure_out_dir()
    schedule = load_games(season)
    week_games = filter_week_reg(schedule, season, week)
    expected = int(week_games.shape[0])

    league_df = generate_league_metrics(season, week)
    league_path = out_dir / f"league_metrics_{season}_{week}.csv"
    write_csv(league_df, league_path)

    gameview_result = build_gameview(season=season, week=week)

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
        "odds_count": len(odds_records),
        "league_path": league_path,
        "gameview_jsonl": gameview_result["jsonl"],
        "gameview_csv": gameview_result["csv"],
        "odds_path": odds_path,
    }

    print("=== WEEKLY REFRESH ===")
    print(f"Season {season} Week {week}")
    print(f"Schedule games (REG): {expected}")
    print(f"Game View records:     {gameview_result['count']} ({gameview_result['jsonl']})")
    print(f"League metrics rows:   {len(league_df)} ({league_path})")
    if odds_path:
        print(f"Odds records:          {len(odds_records)} ({odds_path})")
    else:
        print("Odds records:          skipped (no API key)")
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

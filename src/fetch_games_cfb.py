"""Stub CFB games fetcher that writes shapeful empty outputs.

This mirrors the NFL fetcher entry-point so downstream consumers have
paths and schemas ready before the real ETL is implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_jsonl

GAMES_FIELDS: List[str] = [
    "season",
    "week",
    "game_key",
    "kickoff_iso_utc",
    "snapshot_at",
    "is_closing",
    "league",
    "home_team_raw",
    "home_team_norm",
    "away_team_raw",
    "away_team_norm",
    "home_pr",
    "away_pr",
    "home_sos",
    "away_sos",
    "home_sos_rank",
    "away_sos_rank",
    "spread_home_relative",
    "total",
    "favored_side",
    "spread_favored_team",
    "rating_diff",
    "rating_vs_odds",
    "rating_diff_favored_team",
    "rating_vs_odds_favored_team",
    "home_pf_pg",
    "home_pa_pg",
    "away_pf_pg",
    "away_pa_pg",
    "home_ry_pg",
    "home_py_pg",
    "home_ty_pg",
    "away_ry_pg",
    "away_py_pg",
    "away_ty_pg",
    "home_to_margin_pg",
    "away_to_margin_pg",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_empty_games(season: int, week: int) -> Path:
    out_dir = cfb_week_dir(season, week)
    jsonl_path = out_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = out_dir / f"games_week_{season}_{week}.csv"

    write_jsonl([], jsonl_path)
    empty_df = pd.DataFrame(columns=GAMES_FIELDS)
    empty_df.to_csv(csv_path, index=False)
    return jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB games fetcher (empty outputs).")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_games_cfb season={args.season} week={args.week}")
    jsonl_path = write_empty_games(args.season, args.week)
    print(f"PASS: wrote shapeful empty games to {jsonl_path}")
    print("Info: future implementation will source CFB schedules & odds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

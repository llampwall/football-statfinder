"""Stub CFB schedule fetcher that writes a header-only placeholder.

The real NFL pipeline owns the `games_week_*` artifacts via the Game View
builder. This module intentionally limits itself to a schedule CSV so the
ownership model stays identical for the CFB pathway.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir

SCHEDULE_COLUMNS: List[str] = [
    "season",
    "week",
    "game_key",
    "kickoff_iso_utc",
    "home_team",
    "away_team",
    "home_team_norm",
    "away_team_norm",
    "venue",
    "network",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_schedule_placeholder(season: int, week: int) -> Path:
    out_dir = cfb_week_dir(season, week)
    schedule_path = out_dir / f"schedule_{season}_{week}.csv"
    df = pd.DataFrame(columns=SCHEDULE_COLUMNS)
    df.to_csv(schedule_path, index=False)
    return schedule_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB schedule fetcher (header-only output).")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_games_cfb season={args.season} week={args.week}")
    schedule_path = write_schedule_placeholder(args.season, args.week)
    print(f"PASS: wrote schedule placeholder to {schedule_path}")
    print("Info: future implementation will source real CFB schedules and populate this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

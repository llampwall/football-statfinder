"""Stub CFB year-to-date metrics writer.

Produces an empty CSV with the same headers as the NFL league metrics file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv

LEAGUE_COLUMNS: List[str] = [
    "Team",
    "RY(O)",
    "R(O)_RY",
    "PY(O)",
    "R(O)_PY",
    "TY(O)",
    "R(O)_TY",
    "RY(D)",
    "R(D)_RY",
    "PY(D)",
    "R(D)_PY",
    "TY(D)",
    "R(D)_TY",
    "TO",
    "PF",
    "PA",
    "SU",
    "ATS",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_empty_league_metrics(season: int, week: int) -> Path:
    out_dir = cfb_week_dir(season, week)
    csv_path = out_dir / f"league_metrics_{season}_{week}.csv"
    df = pd.DataFrame(columns=LEAGUE_COLUMNS)
    write_csv(df, csv_path)
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB league metrics writer.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_year_to_date_stats_cfb season={args.season} week={args.week}")
    csv_path = write_empty_league_metrics(args.season, args.week)
    print(f"PASS: wrote shapeful empty league metrics to {csv_path}")
    print("Info: future implementation will source CFB team-week stats and turnover data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

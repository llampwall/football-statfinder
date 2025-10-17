"""Stub CFB Sagarin snapshot writer.

Creates empty CSV/JSONL artifacts with the same schema as the NFL module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl

DEFAULT_COLUMNS: List[str] = [
    "season",
    "week",
    "team",
    "team_norm",
    "team_raw",
    "pr",
    "pr_rank",
    "sos",
    "sos_rank",
    "hfa",
    "source_url",
    "fetched_at",
    "page_stamp",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def infer_columns(season: int, week: int) -> List[str]:
    nfl_dir = ensure_out_dir() / f"{season}_week{week}"
    nfl_csv = nfl_dir / f"sagarin_nfl_{season}_wk{week}.csv"
    if nfl_csv.exists():
        nfl_cols = pd.read_csv(nfl_csv, nrows=0).columns.tolist()
        if nfl_cols:
            return nfl_cols
    return DEFAULT_COLUMNS.copy()


def write_empty_sagarin(season: int, week: int) -> tuple[Path, Path]:
    out_dir = cfb_week_dir(season, week)
    base = out_dir / f"sagarin_cfb_{season}_wk{week}"
    csv_path = base.with_suffix(".csv")
    jsonl_path = base.with_suffix(".jsonl")

    columns = infer_columns(season, week)
    df = pd.DataFrame(columns=columns)
    write_csv(df, csv_path)
    write_jsonl([], jsonl_path)
    return csv_path, jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB Sagarin exporter.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_sagarin_week_cfb season={args.season} week={args.week}")
    csv_path, jsonl_path = write_empty_sagarin(args.season, args.week)
    print(f"PASS: wrote shapeful empty Sagarin CSV to {csv_path}")
    print(f"PASS: wrote shapeful empty Sagarin JSONL to {jsonl_path}")
    print("Info: future implementation will scrape Jeff Sagarin's CFB ratings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

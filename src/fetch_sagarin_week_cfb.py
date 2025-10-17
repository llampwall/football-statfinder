"""Stub CFB Sagarin snapshot writer.

Creates empty CSV/JSONL artifacts with the same schema as the NFL module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl

SAGARIN_COLUMNS: List[str] = [
    "league",
    "season",
    "week",
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


def write_empty_sagarin(season: int, week: int) -> tuple[Path, Path]:
    out_dir = cfb_week_dir(season, week)
    base = out_dir / f"sagarin_cfb_{season}_wk{week}"
    csv_path = base.with_suffix(".csv")
    jsonl_path = base.with_suffix(".jsonl")

    df = pd.DataFrame(columns=SAGARIN_COLUMNS)
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

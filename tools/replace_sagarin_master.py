"""Utility to replace the Sagarin master table from a cleaned CSV."""

import argparse
from pathlib import Path

from src.sagarin_master import load_master, replace_master_from_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace Sagarin master table from CSV")
    parser.add_argument(
        "--csv",
        default="data/SAGARIN_WEEKLY_HISTORICAL_NFL.csv",
        help="Path to the cleaned historical Sagarin CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    count = replace_master_from_csv(csv_path)
    print(f"Replaced master with {count} rows from {csv_path}.")

    preview = load_master().head()
    print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

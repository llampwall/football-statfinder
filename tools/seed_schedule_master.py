"""Seed the NFL schedule master from a CSV."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schedule_master import load_master, upsert_from_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the schedule master from CSV")
    parser.add_argument(
        "--csv",
        default="data/NFL_SCHEDULE_SEED.csv",
        help="Path to the schedule seed CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    delta = upsert_from_seed(csv_path)
    master = load_master()
    print(f"Schedule master upsert: {delta:+d} (rows now {len(master)})")
    if not master.empty:
        print(master.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

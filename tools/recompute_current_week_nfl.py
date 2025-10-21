"""CLI helper to recompute the current week state file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.common.current_week_service import get_current_week


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute current week (read-only service).")
    parser.add_argument("--league", default="NFL", help="League code (default: NFL).")
    args = parser.parse_args()

    league = (args.league or "NFL").upper()
    season, week, computed_at = get_current_week(league, persist=True)
    print(f"CurrentWeek({league})={season} W{week} computed_at={computed_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

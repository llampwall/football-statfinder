"""Weekly orchestrator for refreshing stubbed CFB outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

OUT_ROOT = Path(__file__).resolve().parents[1] / "out"


def run_module(module: str, season: int, week: int) -> None:
    cmd = [sys.executable, "-m", module, "--season", str(season), "--week", str(week)]
    subprocess.run(cmd, check=True)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        # subtract header if present
        rows = sum(1 for _ in handle)
    return max(rows - 1, 0)


def build_summary(season: int, week: int) -> Dict[str, int]:
    base = OUT_ROOT / "cfb" / f"{season}_week{week}"
    summary = {
        "games_jsonl": count_jsonl(base / f"games_week_{season}_{week}.jsonl"),
        "games_csv_rows": count_csv_rows(base / f"games_week_{season}_{week}.csv"),
        "league_rows": count_csv_rows(base / f"league_metrics_{season}_{week}.csv"),
        "odds_records": count_jsonl(base / f"odds_{season}_wk{week}.jsonl"),
        "sagarin_rows": count_csv_rows(base / f"sagarin_cfb_{season}_wk{week}.csv"),
        "sidecars": len(list((base / "game_schedules").glob("*.json"))) if (base / "game_schedules").exists() else 0,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh stubbed CFB weekly outputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print("=== CFB WEEKLY REFRESH (stub) ===")
    print(f"Season {args.season} Week {args.week}")

    steps = [
        "src.fetch_games_cfb",
        "src.fetch_year_to_date_stats_cfb",
        "src.fetch_week_odds_cfb",
        "src.fetch_sagarin_week_cfb",
        "src.build_team_timelines_cfb",
        "src.gameview_build_cfb",
    ]

    for module in steps:
        print(f"\n>>> Running {module} …")
        run_module(module, args.season, args.week)

    summary = build_summary(args.season, args.week)
    print("\n=== SUMMARY ===")
    print(f"Games JSONL records : {summary['games_jsonl']}")
    print(f"Games CSV rows      : {summary['games_csv_rows']}")
    print(f"League metric rows  : {summary['league_rows']}")
    print(f"Odds records        : {summary['odds_records']}")
    print(f"Sagarin rows        : {summary['sagarin_rows']}")
    print(f"Sidecar files       : {summary['sidecars']}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

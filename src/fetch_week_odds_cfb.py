"""Stub CFB odds fetcher that writes an empty JSONL snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io_utils import ensure_out_dir, write_jsonl


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_empty_odds(season: int, week: int) -> Path:
    out_dir = cfb_week_dir(season, week)
    odds_path = out_dir / f"odds_{season}_wk{week}.jsonl"
    write_jsonl([], odds_path)
    return odds_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB odds fetcher.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_week_odds_cfb season={args.season} week={args.week}")
    odds_path = write_empty_odds(args.season, args.week)
    print(f"PASS: wrote shapeful empty odds snapshot to {odds_path}")
    print("Info: future implementation will integrate a CFB odds feed or schedule fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

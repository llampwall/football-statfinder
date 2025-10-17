"""Stub CFB games fetcher that only prepares the weekly directory.

The NFL pipeline writes the `games_week_*` artifacts via the Game View
builder, so this module simply mirrors the directory creation without
producing any additional files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io_utils import ensure_out_dir


def ensure_week_dirs(season: int, week: int) -> Path:
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "game_schedules").mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB games fetcher (no-op placeholder).")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: fetch_games_cfb season={args.season} week={args.week}")
    out_dir = ensure_week_dirs(args.season, args.week)
    print(f"PASS: ensured CFB week directory exists at {out_dir}")
    print("Info: future implementation will download real CFB schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

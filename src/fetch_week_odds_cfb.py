"""Stub CFB odds fetcher that writes an empty JSONL snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import json

from src.common.io_utils import ensure_out_dir, write_jsonl

DEFAULT_KEYS: List[str] = [
    "source",
    "league",
    "season",
    "week",
    "kickoff_iso",
    "home_team_raw",
    "away_team_raw",
    "home_team_norm",
    "away_team_norm",
    "rotation_home",
    "rotation_away",
    "spread_home_relative",
    "total",
    "moneyline_home",
    "moneyline_away",
    "market_scope",
    "is_closing",
    "book_label",
    "snapshot_at",
    "raw_payload",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def infer_keys(season: int, week: int) -> List[str]:
    nfl_path = ensure_out_dir() / f"{season}_week{week}" / f"odds_{season}_wk{week}.jsonl"
    if nfl_path.exists():
        keys: set[str] = set()
        with nfl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.update(record.keys())
                if keys:
                    break
        if keys:
            return list(keys)
    return DEFAULT_KEYS.copy()


def write_empty_odds(season: int, week: int) -> Path:
    out_dir = cfb_week_dir(season, week)
    odds_path = out_dir / f"odds_{season}_wk{week}.jsonl"
    keys = infer_keys(season, week)
    records = []
    if keys:
        placeholder = {key: None for key in keys}
        if "is_closing" in placeholder:
            placeholder["is_closing"] = False
        if "league" in placeholder:
            placeholder["league"] = "CFB"
        records.append(placeholder)
    write_jsonl(records, odds_path)
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

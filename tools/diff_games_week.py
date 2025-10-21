"""Diff helper for games_week JSONL outputs (league-agnostic).

Purpose & scope:
    Compare two games_week JSONL files while ignoring volatile timestamp
    fields so odds promotion can be verified without legacy noise.

Ignored fields:
    * Top-level ``snapshot_at``
    * ``raw_sources.odds_row.markets.*.fetch_ts``

Usage:
    python -m tools.diff_games_week before.jsonl after.jsonl [--limit 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

IGNORED_TOP_LEVEL = {"snapshot_at"}


def _load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            key = record.get("game_key")
            if isinstance(key, str) and key:
                records[key] = record
    return records


def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: v for k, v in row.items() if k not in IGNORED_TOP_LEVEL}
    raw_sources = cleaned.get("raw_sources")
    if isinstance(raw_sources, dict):
        odds_row = raw_sources.get("odds_row")
        if isinstance(odds_row, dict):
            markets = odds_row.get("markets")
            if isinstance(markets, dict):
                sanitized_markets: Dict[str, Any] = {}
                for market, payload in markets.items():
                    if not isinstance(payload, dict):
                        sanitized_markets[market] = payload
                        continue
                    sanitized_payload = dict(payload)
                    sanitized_payload.pop("fetch_ts", None)
                    sanitized_markets[market] = sanitized_payload
                new_odds = dict(odds_row)
                new_odds["markets"] = sanitized_markets
                new_sources = dict(raw_sources)
                new_sources["odds_row"] = new_odds
                cleaned["raw_sources"] = new_sources
    return cleaned


def _compare(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]]:
    only_before = len(set(before) - set(after))
    only_after = len(set(after) - set(before))
    mismatches: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for key in set(before) & set(after):
        if _sanitize(before[key]) != _sanitize(after[key]):
            mismatches[key] = (before[key], after[key])
    return only_before, only_after, mismatches


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff games_week JSONL outputs.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows_before = _load_rows(args.before)
    rows_after = _load_rows(args.after)
    only_before, only_after, mismatches = _compare(rows_before, rows_after)
    print(f"Rows only in {args.before}: {only_before}")
    print(f"Rows only in {args.after}: {only_after}")
    print(f"Mismatched rows      : {len(mismatches)}")
    if mismatches and args.limit > 0:
        print("\nSample mismatches:")
        for key in sorted(mismatches.keys())[: args.limit]:
            before_row, after_row = mismatches[key]
            print(f"- {key}:")
            print(f"  before={_sanitize(before_row)}")
            print(f"  after ={_sanitize(after_row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diff helper for CFB games_week JSONL outputs.

Purpose & scope:
    Quickly compare two Game View JSONL files while ignoring timestamp-style
    fields so we can confirm the odds-promotion step does not introduce
    unexpected schema differences.

Usage:
    python -m tools.diff_games_week path/to/a.jsonl path/to/b.jsonl

Ignored fields:
    * Top-level ``snapshot_at`` (varies per promotion run).
    * ``raw_sources.odds_row.markets.*.fetch_ts`` (book fetch timestamps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

IGNORED_TOP_LEVEL = {"snapshot_at"}


def _load_game_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a games_week JSONL file into a keyed dictionary."""
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("game_key")
            if not isinstance(key, str):
                continue
            rows[key] = record
    return rows


def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of a row with volatile fields removed."""
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
                odds_row = dict(odds_row)
                odds_row["markets"] = sanitized_markets
                raw_sources = dict(raw_sources)
                raw_sources["odds_row"] = odds_row
                cleaned["raw_sources"] = raw_sources
    return cleaned


def _compare(
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """Compare two row maps and return discrepancy details."""
    only_a = len(set(rows_a) - set(rows_b))
    only_b = len(set(rows_b) - set(rows_a))

    mismatches: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for key in set(rows_a) & set(rows_b):
        if _sanitize(rows_a[key]) != _sanitize(rows_b[key]):
            mismatches[key] = (rows_a[key], rows_b[key])

    return only_a, only_b, mismatches


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Diff games_week JSONL outputs.")
    parser.add_argument("before", type=Path, help="Path to the baseline games_week JSONL file.")
    parser.add_argument("after", type=Path, help="Path to the promoted games_week JSONL file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of mismatched rows to display (default: 5).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    rows_before = _load_game_rows(args.before)
    rows_after = _load_game_rows(args.after)
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

"""NFL odds promotion helpers (staging -> week outputs).

Purpose & scope:
    Promote staged/pinned NFL odds snapshots into the current week's
    Game View rows so public outputs reflect the freshest bookmaker
    lines without relying on legacy joins.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B3, E, F, H, I)

Invariants:
    * Only pinned rows whose ``week`` matches the target week are eligible.
    * Selection policy is deterministic (latest ``fetch_ts`` wins).
    * Promotions mutate in-memory rows; callers decide whether to persist.

Side effects:
    * None directly; callers handle JSONL/CSV rewrites when needed.

Do not:
    * Modify staging artifacts or legacy odds files.
    * Broaden selection policies without explicit spec updates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir

OUT_ROOT = ensure_out_dir()
PINNED_DIR = OUT_ROOT / "staging" / "odds_pinned" / "nfl"


def _parse_fetch_ts(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _load_pinned(season: int) -> List[dict]:
    path = PINNED_DIR / f"{season}.jsonl"
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def _select_latest(records: Iterable[dict]) -> Dict[Tuple[str, str, str], dict]:
    latest: Dict[Tuple[str, str, str], dict] = {}
    for record in records:
        key = (record.get("game_key"), record.get("market"), record.get("book"))
        if None in key:
            continue
        candidate_ts = _parse_fetch_ts(record.get("fetch_ts"))
        existing = latest.get(key)
        if existing is None or candidate_ts > _parse_fetch_ts(existing.get("fetch_ts")):
            latest[key] = record
    return latest


def _choose_best(records: Sequence[dict]) -> Optional[dict]:
    if not records:
        return None
    return max(
        records,
        key=lambda rec: (_parse_fetch_ts(rec.get("fetch_ts")), rec.get("book") or ""),
    )


def _merge_line(row: MutableMapping[str, Any], record: dict, line: Mapping[str, Any]) -> None:
    market = record.get("market")
    if market == "spreads":
        row["spread_home_relative"] = line.get("spread_home_relative")
        row["favored_side"] = line.get("favored_side")
        row["spread_favored_team"] = line.get("spread_favored_team")
    elif market == "totals":
        row["total"] = line.get("total_points")
    elif market == "h2h":
        row["moneyline_home"] = line.get("moneyline_home")
        row["moneyline_away"] = line.get("moneyline_away")


def promote_week_odds(
    rows: List[MutableMapping[str, Any]],
    season: int,
    week: int,
    *,
    policy: str = "latest_by_fetch_ts",
) -> Dict[str, Any]:
    """Promote pinned NFL odds into the provided in-memory rows."""
    if policy != "latest_by_fetch_ts":
        raise ValueError(f"Unsupported odds selection policy: {policy}")
    if not rows:
        return {
            "promoted_games": 0,
            "used_records": 0,
            "available_records": 0,
            "season_records": 0,
            "current_week_records": 0,
            "other_week_records": 0,
            "by_market": {},
            "by_book": {},
        }

    game_lookup: Dict[str, MutableMapping[str, Any]] = {}
    eligible_keys: set[str] = set()
    for row in rows:
        key = row.get("game_key")
        if not key:
            continue
        game_lookup[key] = row
        if row.get("season") == season and row.get("week") == week:
            eligible_keys.add(key)

    pinned_records = _load_pinned(season)
    week_records = [rec for rec in pinned_records if rec.get("week") == week]
    other_week_records = len(pinned_records) - len(week_records)
    relevant_records = [rec for rec in week_records if rec.get("game_key") in eligible_keys]
    latest_map = _select_latest(relevant_records)

    per_game: Dict[str, Dict[str, List[dict]]] = {}
    for (game_key, market, _), record in latest_map.items():
        per_game.setdefault(game_key, {}).setdefault(market, []).append(record)

    by_market: Dict[str, int] = {}
    by_book: Dict[str, int] = {}
    promoted_games: set[str] = set()

    for game_key, market_records in per_game.items():
        row = game_lookup.get(game_key)
        if not row:
            continue
        odds_payload = {
            "source": "staging",
            "season": season,
            "week": week,
            "markets": {},
        }
        primary_record: Optional[dict] = None
        for market, records in market_records.items():
            best = _choose_best(records)
            if not best:
                continue
            line = best.get("line") or {}
            _merge_line(row, best, line)
            odds_payload["markets"][market] = {
                "book": best.get("book"),
                "fetch_ts": best.get("fetch_ts"),
                "line": line,
            }
            by_market[market] = by_market.get(market, 0) + 1
            book_key = best.get("book") or "unknown"
            by_book[book_key] = by_book.get(book_key, 0) + 1
            if primary_record is None or _parse_fetch_ts(best.get("fetch_ts")) > _parse_fetch_ts(
                primary_record.get("fetch_ts") if primary_record else None
            ):
                primary_record = best
        if odds_payload["markets"]:
            promoted_games.add(game_key)
            row.setdefault("raw_sources", {})["odds_row"] = odds_payload
            if primary_record:
                row["odds_source"] = primary_record.get("book")
                row["snapshot_at"] = primary_record.get("fetch_ts")
                row["is_closing"] = False

    return {
        "promoted_games": len(promoted_games),
        "used_records": len(latest_map),
        "available_records": len(relevant_records),
        "season_records": len(pinned_records),
        "current_week_records": len(relevant_records),
        "other_week_records": other_week_records,
        "by_market": by_market,
        "by_book": by_book,
    }


def write_week_outputs(
    rows: Sequence[Mapping[str, Any]],
    season: int,
    week: int,
) -> Tuple[Path, Path]:
    """Rewrite games_week outputs atomically with the provided rows."""
    base_dir = OUT_ROOT / f"{season}_week{week}"
    json_path = base_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = base_dir / f"games_week_{season}_{week}.csv"
    write_atomic_jsonl(json_path, rows)
    write_atomic_csv(csv_path, pd.DataFrame(rows))
    return json_path, csv_path


def read_week_json(path: Path) -> List[dict]:
    """Read a games_week JSONL file into memory."""
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def diff_game_rows(
    promoted: Sequence[Mapping[str, Any]],
    legacy: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = (
        "spread_home_relative",
        "favored_side",
        "spread_favored_team",
        "total",
        "moneyline_home",
        "moneyline_away",
    ),
) -> Dict[str, int]:
    """Compare promoted vs legacy rows focusing on odds-related fields."""

    def _build_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple]:
        mapping: Dict[str, Tuple] = {}
        for row in rows:
            key = row.get("game_key")
            if not isinstance(key, str):
                continue
            mapping[key] = tuple(row.get(field) for field in fields)
        return mapping

    promoted_map = _build_map(promoted)
    legacy_map = _build_map(legacy)
    promoted_only = len(set(promoted_map) - set(legacy_map))
    legacy_only = len(set(legacy_map) - set(promoted_map))
    mismatched = 0
    both_equal = 0
    for key in set(promoted_map) & set(legacy_map):
        if promoted_map[key] == legacy_map[key]:
            both_equal += 1
        else:
            mismatched += 1
    return {
        "promoted_only": promoted_only,
        "legacy_only": legacy_only,
        "both_equal": both_equal,
        "mismatched": mismatched,
    }


__all__ = ["promote_week_odds", "write_week_outputs", "read_week_json", "diff_game_rows"]

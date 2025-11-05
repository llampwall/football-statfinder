
"""CFB odds promotion helpers (staging -> week outputs).

Purpose & scope:
    Promote College Football odds that were previously pinned to schedule
    games into the in-memory Game View rows so the current week's files can
    surface bookmaker lines without re-running legacy joins.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B3, E, F, H, I)

Invariants:
    * Selection policy is deterministic (latest ``fetch_ts`` by (game_key, market, book)).
    * UTC timestamps are required for comparisons and copied verbatim.
    * Promotions mutate in-memory rows; callers must persist updated files.

Side effects:
    * No direct I/O writes; callers decide whether to rewrite week outputs.

Do not:
    * Do not alter pinned staging artifacts in-place.
    * Do not widen selection policy without an explicit spec change.

Log contract:
    * Promotion metrics returned here are logged by the orchestrator as:
      ``CFB ODDS PROMOTION: week=<season>-<week> promoted=<n> ...``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from src.common.io_utils import ensure_out_dir
from src.fetch_week_odds_cfb import parse_utc
from src.odds.ats_compute import pick_latest_before

OUT_ROOT = ensure_out_dir()
PINNED_DIR = OUT_ROOT / "staging" / "odds_pinned" / "cfb"

LINE_FIELDS = (
    "spread_home_relative",
    "favored_side",
    "spread_favored_team",
    "total",
    "moneyline_home",
    "moneyline_away",
)
_WARNED_POLICIES: set[str] = set()


def _parse_fetch_ts(value: Optional[str]) -> datetime:
    """Parse ISO timestamps into aware UTC datetimes.

    Args:
        value: ISO8601 timestamp (Z-suffixed) or None.

    Returns:
        ``datetime`` anchored to UTC. ``datetime.min`` is returned when parsing fails.
    """
    dt = parse_utc(value)
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _load_pinned_for_season(season: int) -> Iterable[dict]:
    """Yield pinned staging rows for the requested season.

    Args:
        season: Season identifier (e.g., 2025).

    Returns:
        Iterable of JSON-decoded dictionaries. Invalid lines are skipped.
    """
    path = PINNED_DIR / f"{season}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _select_latest_by_fetch(records: Iterable[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Return the freshest record for each (game_key, market, book).

    Args:
        records: Iterable of pinned staging rows.

    Returns:
        Mapping keyed by ``(game_key, market, book)`` pointing at the latest record.
    """
    latest: Dict[Tuple[str, str, str], dict] = {}
    for record in records:
        key = (record.get("game_key"), record.get("market"), record.get("book"))
        if None in key:
            continue
        candidate_ts = _parse_fetch_ts(record.get("fetch_ts"))
        existing = latest.get(key)
        if not existing or candidate_ts > _parse_fetch_ts(existing.get("fetch_ts")):
            latest[key] = record
    return latest


def _choose_best(records: Sequence[dict]) -> Optional[dict]:
    """Select the record with the newest fetch timestamp.

    Args:
        records: Candidate records for a single ``(game_key, market)``.

    Returns:
        The record whose ``fetch_ts`` is the latest, breaking ties by book label.
    """
    if not records:
        return None
    return max(
        records,
        key=lambda rec: (_parse_fetch_ts(rec.get("fetch_ts")), rec.get("book") or ""),
)


def _parse_kickoff(value: Optional[str]) -> Optional[datetime]:
    dt = parse_utc(value)
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc)
    return None


def _choose_by_policy(
    records: Sequence[dict],
    policy: str,
    kickoff: Optional[datetime],
) -> Tuple[Optional[dict], bool]:
    if not records:
        return None, False
    if policy == "closing_pre_kickoff" and kickoff:
        chosen = pick_latest_before(records, kickoff)
        if chosen:
            return chosen, True
    fallback = _choose_best(records)
    return fallback, False


def _merge_line(row: MutableMapping[str, Any], record: dict, line: Mapping[str, Any]) -> None:
    """Apply odds line values to a ``games_week`` row.

    Args:
        row: Mutable row dictionary mutated in-place.
        record: Pinned record supplying metadata (market, book).
        line: Nested odds payload emitted by pinning.
    """
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
    """Promote pinned odds into the provided in-memory rows.

    Args:
        rows: ``games_week`` row dictionaries (mutated in-place).
        season: Target season (e.g., 2025).
        week: Target week number.
        policy: Selection policy (``latest_by_fetch_ts`` or ``closing_pre_kickoff``).

    Returns:
        Dict summarizing promotion results with counts by market and book.
    """
    policy = (policy or "latest_by_fetch_ts").strip() or "latest_by_fetch_ts"
    supported = {"latest_by_fetch_ts", "closing_pre_kickoff"}
    if policy not in supported:
        if policy not in _WARNED_POLICIES:
            print(
                f"WARNING: Unsupported odds selection policy '{policy}'; "
                "falling back to latest_by_fetch_ts"
            )
            _WARNED_POLICIES.add(policy)
        policy = "latest_by_fetch_ts"

    if not rows:
        return {
            "promoted_games": 0,
            "used_records": 0,
            "available_records": 0,
            "season_records": 0,
            "by_market": {},
            "by_book": {},
            "current_week_records": 0,
            "other_week_records": 0,
        }

    game_lookup: Dict[str, MutableMapping[str, Any]] = {}
    eligible_keys: set[str] = set()
    for row in rows:
        key = row.get("game_key")
        if not key:
            continue
        game_lookup[key] = row
        row_week = row.get("week")
        row_season = row.get("season")
        if row_week == week and row_season == season:
            eligible_keys.add(key)

    pinned_records = list(_load_pinned_for_season(season))
    season_records = len(pinned_records)
    relevant_records = [rec for rec in pinned_records if rec.get("game_key") in eligible_keys]
    current_week_records = len(relevant_records)
    other_week_records = season_records - current_week_records
    latest_map = _select_latest_by_fetch(pinned_records)

    per_game_market: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    used_record_total = 0
    for (game_key, market, _), record in latest_map.items():
        if game_key not in eligible_keys:
            continue
        per_game_market[game_key][market].append(record)
        used_record_total += 1

    by_market: Counter[str] = Counter()
    by_book: Counter[str] = Counter()
    promoted_games: set[str] = set()

    for game_key, market_records in per_game_market.items():
        row = game_lookup.get(game_key)
        if not row:
            continue
        kickoff_iso = (
            row.get("kickoff_iso_utc")
            or row.get("kickoff_iso")
            or row.get("kickoff_utc")
        )
        kickoff_dt = _parse_kickoff(kickoff_iso)
        if kickoff_dt is None:
            sample_records = next(iter(market_records.values()), [])
            if sample_records:
                kickoff_dt = _parse_kickoff(sample_records[0].get("kickoff_utc"))
        odds_payload: Dict[str, Any] = {
            "source": "staging",
            "season": season,
            "week": week,
            "markets": {},
        }
        primary_record: Optional[dict] = None
        closing_selected = False
        for market, records in market_records.items():
            best, used_closing = _choose_by_policy(records, policy, kickoff_dt)
            if not best:
                continue
            line = best.get("line") or {}
            _merge_line(row, best, line)
            odds_payload["markets"][market] = {
                "book": best.get("book"),
                "fetch_ts": best.get("fetch_ts"),
                "line": line,
            }
            by_market[market] += 1
            by_book[best.get("book") or "unknown"] += 1
            if primary_record is None or _parse_fetch_ts(best.get("fetch_ts")) > _parse_fetch_ts(primary_record.get("fetch_ts")):
                primary_record = best
            if used_closing and market == "spreads":
                closing_selected = True
        if odds_payload["markets"]:
            promoted_games.add(game_key)
            row.setdefault("raw_sources", {})["odds_row"] = odds_payload
            if primary_record:
                row["odds_source"] = primary_record.get("book")
                row["snapshot_at"] = primary_record.get("fetch_ts")
                if policy == "closing_pre_kickoff":
                    row["is_closing"] = closing_selected
                else:
                    row["is_closing"] = False

    return {
        "promoted_games": len(promoted_games),
        "used_records": used_record_total,
        "available_records": len(relevant_records),
        "season_records": season_records,
        "current_week_records": current_week_records,
        "other_week_records": other_week_records,
        "by_market": dict(by_market),
        "by_book": dict(by_book),
    }


def diff_game_rows(
    promoted: Sequence[Mapping[str, Any]],
    legacy: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = LINE_FIELDS,
) -> Dict[str, int]:
    """Compare promoted vs legacy rows focusing on odds fields.

    Args:
        promoted: Row set after promotion.
        legacy: Row set before promotion.
        fields: Ordered odds fields to compare.

    Returns:
        Dict summarising mismatches between the two views.
    """

    def _build_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple]:
        output: Dict[str, Tuple] = {}
        for row in rows:
            key = row.get("game_key")
            if not key:
                continue
            payload = tuple(row.get(field) for field in fields)
            output[key] = payload
        return output

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


__all__ = ["promote_week_odds", "diff_game_rows"]

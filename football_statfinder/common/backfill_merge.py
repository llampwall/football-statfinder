"""Helpers for merging backfilled games data while preserving odds/ratings."""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ODDS_PRESERVE_FIELDS: Tuple[str, ...] = (
    "spread_home_relative",
    "spread_favored_team",
    "favored_side",
    "total",
    "moneyline_home",
    "moneyline_away",
    "odds_source",
    "snapshot_at",
    "is_closing",
)

RATING_EXACT_FIELDS: Tuple[str, ...] = (
    "rating_diff",
    "rating_vs_odds",
    "rating_diff_favored_team",
    "rating_vs_odds_favored_team",
    "hfa",
)

RATING_PREFIXES: Tuple[str, ...] = (
    "home_pr",
    "away_pr",
    "home_sos",
    "away_sos",
)

ODDS_SIGNAL_FIELDS: Tuple[str, ...] = (
    "spread_home_relative",
    "spread_favored_team",
    "total",
    "moneyline_home",
    "moneyline_away",
    "favored_side",
    "odds_source",
)


def merge_games_week(
    existing_rows: Sequence[Mapping[str, Any]],
    incoming_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge incoming week rows with an existing snapshot keyed by ``game_key``.

    Existing odds and rating-related display fields are preserved whenever they
    already carry values, while incoming rows supply updated scores, records,
    and other backward-looking fields.
    """

    existing_by_key: Dict[str, Mapping[str, Any]] = {}
    for row in existing_rows:
        key = row.get("game_key")
        if isinstance(key, str):
            existing_by_key[key] = row

    merged_rows: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    for incoming in incoming_rows:
        key = incoming.get("game_key")
        if isinstance(key, str) and key in existing_by_key:
            merged_rows.append(_merge_row(existing_by_key[key], incoming))
            seen_keys.add(key)
        else:
            merged_rows.append(deepcopy(dict(incoming)))
            if isinstance(key, str):
                seen_keys.add(key)

    for key, existing in existing_by_key.items():
        if key not in seen_keys:
            merged_rows.append(deepcopy(dict(existing)))

    return merged_rows


def summarize_preservation(
    existing_rows: Sequence[Mapping[str, Any]],
    merged_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Summarise preserved odds and rating-vs-odds counts post-merge."""

    existing_by_key: Dict[str, Mapping[str, Any]] = {}
    for row in existing_rows:
        key = row.get("game_key")
        if isinstance(key, str):
            existing_by_key[key] = row

    preserved_odds = 0
    preserved_rvo = 0

    for row in merged_rows:
        key = row.get("game_key")
        if not isinstance(key, str):
            continue
        existing = existing_by_key.get(key)
        if not existing:
            continue
        if _preserved_fields(existing, row, ODDS_SIGNAL_FIELDS):
            preserved_odds += 1
        if _preserved_fields(existing, row, ("rating_vs_odds",)):
            preserved_rvo += 1

    return {
        "preserved_odds": preserved_odds,
        "preserved_rvo": preserved_rvo,
    }


def _merge_row(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(dict(incoming))

    for field, value in existing.items():
        if field == "raw_sources":
            merged[field] = _merge_raw_sources(existing.get(field), incoming.get(field))
            continue
        if _should_preserve_field(field, value, incoming):
            merged[field] = deepcopy(value)
        elif field not in incoming:
            merged[field] = deepcopy(value)

    return merged


def _should_preserve_field(field: str, value: Any, incoming: Mapping[str, Any]) -> bool:
    if field in ODDS_PRESERVE_FIELDS or field in RATING_EXACT_FIELDS:
        if _has_value(value):
            return True
        return field not in incoming
    for prefix in RATING_PREFIXES:
        if field.startswith(prefix):
            if _has_value(value):
                return True
            return field not in incoming
    return False


def _merge_raw_sources(
    existing: Any,
    incoming: Any,
) -> Any:
    if not isinstance(existing, MutableMapping):
        return deepcopy(incoming) if isinstance(incoming, MutableMapping) else deepcopy(existing)
    if not isinstance(incoming, MutableMapping):
        return deepcopy(existing)

    merged: Dict[str, Any] = deepcopy(dict(incoming))

    for key, value in existing.items():
        if key.startswith("odds"):
            if not _has_value(merged.get(key)):
                merged[key] = deepcopy(value)
            continue
        if key not in merged:
            merged[key] = deepcopy(value)

    return merged


def _preserved_fields(
    existing: Mapping[str, Any],
    merged: Mapping[str, Any],
    fields: Iterable[str],
) -> bool:
    seen = False
    for field in fields:
        if not _has_value(existing.get(field)):
            continue
        seen = True
        if merged.get(field) != existing.get(field):
            return False
    return seen


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


__all__ = [
    "merge_games_week",
    "summarize_preservation",
    "ODDS_PRESERVE_FIELDS",
    "RATING_EXACT_FIELDS",
    "RATING_PREFIXES",
    "ODDS_SIGNAL_FIELDS",
]

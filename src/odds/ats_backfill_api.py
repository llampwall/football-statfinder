"""
ATS backfill primitives (helpers only; no I/O).

Purpose:
    Provide pure helpers to compute ATS outputs and select closing spreads via the Odds API.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - Functions are idempotent and side-effect free (no file or network writes).
    - Inputs and outputs remain UTC-aware where timestamps apply.
    - Merge-only semantics are enforced by callers; helpers never mutate passed objects.
Side effects:
    - None. Pure computation or HTTP reads via odds_api_client.
Do not:
    - Perform file I/O or orchestrator wiring (reserved for later tasks).
Log contract:
    - None; higher-level orchestrators handle provenance logging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.common.io_utils import ensure_out_dir
from src.odds.odds_api_client import find_event_id, get_current_spread, get_historical_spread


def compute_ats(home_score: int, away_score: int, favored: str, spread: float) -> Optional[Dict[str, Any]]:
    """Compute ATS outcomes and margin deltas for a completed game.

    Args:
        home_score: Home team final score.
        away_score: Away team final score.
        favored: Which side was favored ('HOME', 'AWAY', 'PICK').
        spread: Absolute spread number (non-negative).

    Returns:
        Dictionary containing ATS letters and margin values, or None if inputs are invalid.
    """
    try:
        spread_val = float(spread)
        home_val = int(home_score)
        away_val = int(away_score)
    except Exception:
        return None

    if favored == "HOME":
        home_line = -spread_val
    elif favored == "AWAY":
        home_line = spread_val
    else:
        home_line = 0.0

    margin_home = (home_val - away_val) + home_line
    margin_away = -margin_home

    def _ats(value: float) -> str:
        return "W" if value > 0 else ("L" if value < 0 else "P")

    return {
        "home_ats": _ats(margin_home),
        "away_ats": _ats(margin_away),
        "to_margin_home": float(margin_home),
        "to_margin_away": float(margin_away),
    }


def load_pinned_event_index(league: str, season: int) -> Dict[str, str]:
    """Return game_key -> event_id mapping from the pinned spreads index."""
    root = ensure_out_dir() / "staging" / "odds_pinned" / league.lower()
    path = root / f"{season}.jsonl"
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return mapping

    for line in text.splitlines():
        entry = line.strip()
        if not entry:
            continue
        try:
            record = json.loads(entry)
        except json.JSONDecodeError:
            continue
        if record.get("market") != "spreads":
            continue
        game_key = record.get("game_key")
        event_id = (record.get("raw_event") or {}).get("event_id")
        if isinstance(game_key, str) and isinstance(event_id, str):
            mapping[game_key] = event_id
    return mapping


def _normalize_kickoff(kickoff: Optional[datetime], kickoff_iso: Optional[str]) -> Optional[str]:
    if kickoff is not None:
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        else:
            kickoff = kickoff.astimezone(timezone.utc)
        return kickoff.isoformat()
    return kickoff_iso


def select_closing_spread(
    league: str,
    event_id: Optional[str] = None,
    kickoff_iso: Optional[str] = None,
    home_name: Optional[str] = None,
    away_name: Optional[str] = None,
    *,
    season: Optional[int] = None,
    kickoff: Optional[datetime] = None,
    game_key: Optional[str] = None,
    pinned_index: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """Resolve a closing spread snapshot using historical odds first, then current odds.

    Args:
        league: League identifier ('nfl' or 'cfb').
        event_id: Pre-resolved event identifier (optional).
        kickoff_iso: Kickoff ISO8601 timestamp (UTC).
        home_name: Home team schedule label.
        away_name: Away team schedule label.
        season: Season year (used when lazily loading the pinned map).
        kickoff: Kickoff datetime (UTC expected); overrides kickoff_iso when provided.
        game_key: Schedule game_key for pinned map lookup.
        pinned_index: Optional preloaded pinned event map.

    Returns:
        Tuple of (spread_payload_or_none, resolver, resolved_event_id).
    """
    kickoff_value = _normalize_kickoff(kickoff, kickoff_iso)
    kickoff_dt = kickoff
    if kickoff_dt is None and kickoff_value:
        try:
            kickoff_dt = datetime.fromisoformat(kickoff_value.replace("Z", "+00:00"))
        except Exception:
            kickoff_dt = None

    if pinned_index is None and season is not None:
        pinned_index = load_pinned_event_index(league, season)

    resolved_event_id = event_id if isinstance(event_id, str) and event_id.strip() else None
    resolver_used = "pinned" if resolved_event_id else "failed"

    if not resolved_event_id and pinned_index and game_key:
        candidate = pinned_index.get(game_key)
        if candidate:
            resolved_event_id = candidate
            resolver_used = "pinned"

    if not resolved_event_id and kickoff_dt and home_name is not None and away_name is not None:
        candidate = find_event_id(league, kickoff_dt, home_name, away_name)
        if candidate:
            resolved_event_id = candidate
            resolver_used = "events"

    if not resolved_event_id:
        return None, "failed", None

    if not kickoff_value:
        kickoff_value = kickoff_dt.isoformat() if kickoff_dt else None
    if not kickoff_value:
        return None, resolver_used, resolved_event_id

    home_label = home_name or ""
    away_label = away_name or ""

    historical = get_historical_spread(league, resolved_event_id, kickoff_value, home_label, away_label)
    if historical:
        historical["source"] = "history"
        historical["resolver"] = resolver_used
        historical["event_id"] = resolved_event_id
        return historical, resolver_used, resolved_event_id

    current = get_current_spread(league, resolved_event_id, kickoff_value, home_label, away_label)
    if current:
        current["source"] = "current"
        current["resolver"] = resolver_used
        current["event_id"] = resolved_event_id
        return current, resolver_used, resolved_event_id

    return None, resolver_used, resolved_event_id


def resolve_event_id(league: str, season: int, game_row: Dict[str, Any]) -> Optional[str]:
    """Placeholder resolver that surfaces the raw event_id when already embedded.

    Args:
        league: League identifier ('nfl' or 'cfb').
        season: Season year (unused placeholder for future expansion).
        game_row: Week row record containing raw_sources metadata.

    Returns:
        The embedded event_id string when present; otherwise None.
    """
    raw_sources = game_row.get("raw_sources") or {}
    odds_row = raw_sources.get("odds_row") or {}
    raw_event = odds_row.get("raw_event") or {}
    event_id = raw_event.get("event_id")
    return event_id if isinstance(event_id, str) and event_id else None


__all__ = ["compute_ats", "load_pinned_event_index", "select_closing_spread", "resolve_event_id"]

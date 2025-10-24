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

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.odds.odds_api_client import get_current_spread, get_historical_spread


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
) -> Optional[Dict[str, Any]]:
    """Resolve a closing spread snapshot using historical odds first, then current odds.

    Args:
        league: League identifier ('nfl' or 'cfb').
        event_id: Odds API event identifier.
        kickoff_iso: Kickoff ISO8601 timestamp (UTC).
        home_name: Home team schedule label.
        away_name: Away team schedule label.
        season: Included for parity with higher-level helpers (unused placeholder).
        kickoff: Kickoff datetime (UTC expected); overrides kickoff_iso when provided.

    Returns:
        Dictionary containing spread payload augmented with a 'source' key, or None.
    """
    if not event_id:
        return None

    kickoff_value = _normalize_kickoff(kickoff, kickoff_iso)
    if not kickoff_value:
        return None

    home_label = home_name or ""
    away_label = away_name or ""

    historical = get_historical_spread(league, event_id, kickoff_value, home_label, away_label)
    if historical:
        historical["source"] = "history"
        return historical

    current = get_current_spread(league, event_id, kickoff_value, home_label, away_label)
    if current:
        current["source"] = "current"
        return current

    return None


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


__all__ = ["compute_ats", "select_closing_spread", "resolve_event_id"]

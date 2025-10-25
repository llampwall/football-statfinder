"""
ATS backfill primitives (helpers only; no I/O).

Purpose:
    Provide ATS computation, event resolution, and historical odds selection
    helpers that operate purely in-memory for the weekly refresh pipelines.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - All datetime values handled in UTC.
    - Event lookups favour pinned maps, then historical participants/events.
    - Historical odds queries use pre-kick snapshots with Pinnacle priority.
Side effects:
    - None; HTTP requests delegated to odds_api_client.
Do not:
    - Perform file I/O or orchestrator wiring (reserved for refresh modules).
Log contract:
    - No direct logging; callers capture diagnostics via ATS_DEBUG paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.common.io_utils import ensure_out_dir
from src.odds.historical_events import list_week_events
from src.odds.odds_api_client import get_current_spread, get_historical_spread
from src.odds.participants_cache import match_team_name, canonical_equals

_WEEK_WINDOW_CACHE: Dict[Tuple[str, int, int], Tuple[datetime, datetime]] = {}


def compute_ats(home_score: int, away_score: int, favored: str, spread: float) -> Optional[Dict[str, Any]]:
    """Compute ATS outcomes and margin deltas for a completed game."""
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


def resolve_event_id(
    league: str,
    season: int,
    week: int,
    game_row: Dict[str, Any],
    *,
    pinned_index: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], str]:
    """Resolve the Odds API event id for a game via pinned map or historical events."""
    game_key = str(game_row.get("game_key") or "")
    if pinned_index and game_key:
        pinned = pinned_index.get(game_key)
        if isinstance(pinned, str) and pinned:
            return pinned, "pinned"

    kickoff_dt = _extract_kickoff(game_row)
    if kickoff_dt is None:
        return None, "failed"

    home_name = game_row.get("home_team_norm") or game_row.get("home_team_raw")
    away_name = game_row.get("away_team_norm") or game_row.get("away_team_raw")
    canonical_home = match_team_name(league, str(home_name or ""))
    canonical_away = match_team_name(league, str(away_name or ""))
    print(
        f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
        f"h='{home_name}'→'{canonical_home or '-'}' a='{away_name}'→'{canonical_away or '-'}'",
        flush=True,
    )
    if not canonical_home or not canonical_away:
        return None, "failed"

    week_start, week_end = _week_window(league, season, week, kickoff_dt)
    events = list_week_events(league, week_start, week_end)

    best_event = None
    best_delta = None
    for event in events:
        event_home_raw = str(event.get("home_team") or "")
        event_away_raw = str(event.get("away_team") or "")
        if not canonical_equals(league, canonical_home, event_home_raw) or not canonical_equals(
            league, canonical_away, event_away_raw
        ):
            continue
        commence = _parse_ts(event.get("commence_time"))
        if not commence:
            continue
        delta_seconds = abs((commence - kickoff_dt).total_seconds())
        if delta_seconds > 90 * 60:
            continue
        if best_delta is None or delta_seconds < best_delta:
            best_event = event
            best_delta = delta_seconds

    if best_event is None:
        return None, "failed"

    event_id = best_event.get("id")
    return (str(event_id) if isinstance(event_id, (str, int)) else None, "events")


def select_closing_spread(
    league: str,
    event_id: str,
    kickoff_iso: str,
    home_name: str,
    away_name: str,
) -> Optional[Dict[str, Any]]:
    """Fetch historical odds snapshot at kickoff; fall back to current odds if needed."""
    kickoff_value = _normalize_kickoff(None, kickoff_iso)
    if not kickoff_value:
        return None

    historical = get_historical_spread(league, event_id, kickoff_value, home_name, away_name)
    if historical:
        return historical

    current = get_current_spread(league, event_id, kickoff_value, home_name, away_name)
    return current


def _extract_kickoff(game: Dict[str, Any]) -> Optional[datetime]:
    kickoff = (
        game.get("kickoff_ts")
        or game.get("kickoff_iso_utc")
        or game.get("kickoff_iso")
        or (game.get("raw_sources", {}).get("schedule_row", {}).get("commence_time"))
    )
    if not kickoff:
        return None
    return _parse_ts(str(kickoff))


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_kickoff(kickoff: Optional[datetime], kickoff_iso: Optional[str]) -> Optional[str]:
    if kickoff is not None:
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        else:
            kickoff = kickoff.astimezone(timezone.utc)
        return kickoff.isoformat()
    return kickoff_iso


def _week_window(
    league: str,
    season: int,
    week: int,
    reference_dt: datetime,
) -> Tuple[datetime, datetime]:
    key = (league.lower(), season, week)
    if key in _WEEK_WINDOW_CACHE:
        return _WEEK_WINDOW_CACHE[key]

    dt = reference_dt.astimezone(timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    start = datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
    end = start + timedelta(days=7) - timedelta(seconds=1)
    _WEEK_WINDOW_CACHE[key] = (start, end)
    return start, end


__all__ = [
    "compute_ats",
    "load_pinned_event_index",
    "resolve_event_id",
    "select_closing_spread",
]

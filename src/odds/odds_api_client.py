"""
Odds API v4 client for selecting closing spreads (NFL/CFB).

Purpose:
    Provide deterministic, pre-kick spread selection helpers for ATS backfill.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - All timestamps normalized to UTC and compared using timezone-aware datetimes.
    - Only pre-kick snapshots (timestamp <= kickoff) are considered.
    - Functions are pure HTTP helpers with no file I/O; idempotent per call.
Side effects:
    - None. No disk writes; only external HTTP calls to The Odds API.
Do not:
    - Mutate caller-provided objects.
    - Use deprecated /odds-history endpoint variants.
Log contract:
    - No direct logging; callers may inspect ODDS_API_USAGE for rate-limit reporting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.common.io_utils import getenv

_THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}

# One-run usage counters (callers can use this for a single summary line).
ODDS_API_USAGE: Dict[str, Optional[str]] = {"remaining": None, "used": None}


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Return a UTC datetime parsed from an ISO8601 string (or None on failure)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _is_finite(value: Any) -> bool:
    """True when value can be converted to a finite float."""
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _sport_key(league: str) -> Optional[str]:
    """Map league code to Odds API sport key."""
    return _SPORT_KEYS.get((league or "").lower())


def _normalize(league: str):
    """Return the appropriate team normalization function for the target league."""
    from src.common.team_names import team_merge_key
    from src.common.team_names_cfb import team_merge_key_cfb

    return team_merge_key_cfb if (league or "").lower() == "cfb" else team_merge_key


def _pick_book_pre_kick(
    bookmakers: List[Dict[str, Any]], kickoff: datetime
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Select the preferred bookmaker snapshot that satisfies the pre-kick constraint.

    Bookmaker priority: 'pinnacle' first (latest valid snapshot), else the latest snapshot
    across all bookmakers.
    """
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for book in bookmakers or []:
        for market in book.get("markets") or []:
            if market.get("key") != "spreads":
                continue

            snapshots = market.get("odds")
            if isinstance(snapshots, list) and snapshots:
                sorted_snapshots = sorted(
                    snapshots,
                    key=lambda snap: _parse_ts(snap.get("timestamp") or snap.get("last_update"))
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                valid_snapshot = next(
                    (
                        snap
                        for snap in sorted_snapshots
                        if (_parse_ts(snap.get("timestamp") or snap.get("last_update")) or datetime.min.replace(tzinfo=timezone.utc))
                        <= kickoff
                    ),
                    None,
                )
                if not valid_snapshot:
                    continue
                market_copy = dict(market)
                market_copy["outcomes"] = valid_snapshot.get("outcomes") or []
                market_copy["__ts__"] = _parse_ts(valid_snapshot.get("timestamp") or valid_snapshot.get("last_update"))
                candidates.append((book, market_copy))
            else:
                ts = _parse_ts(market.get("last_update"))
                if ts and ts <= kickoff:
                    market_copy = dict(market)
                    market_copy["outcomes"] = market.get("outcomes") or []
                    market_copy["__ts__"] = ts
                    candidates.append((book, market_copy))

    if not candidates:
        return None

    def _sort_key(item: Tuple[Dict[str, Any], Dict[str, Any]]) -> datetime:
        return item[1].get("__ts__") or datetime.min.replace(tzinfo=timezone.utc)

    pinnacle_candidates = [
        candidate for candidate in candidates if (candidate[0].get("key") or "").lower() == "pinnacle"
    ]
    if pinnacle_candidates:
        return sorted(pinnacle_candidates, key=_sort_key, reverse=True)[0]

    return sorted(candidates, key=_sort_key, reverse=True)[0]


def _extract_spread_from_market(
    league: str, market: Dict[str, Any], home_name: str, away_name: str
) -> Optional[Tuple[str, float]]:
    """
    Produce a normalized spread tuple of (favored_side, spread_abs).

    Returns:
        Tuple[str, float]: favored side ('HOME', 'AWAY', 'PICK') and absolute spread.
        None when the market lacks sufficient information.
    """
    outcomes = market.get("outcomes") or []
    if not outcomes:
        return None

    normalizer = _normalize(league)
    home_token = normalizer(home_name or "")
    away_token = normalizer(away_name or "")

    home_point: Optional[float] = None
    away_point: Optional[float] = None

    for outcome in outcomes:
        name = (outcome.get("name") or "").strip()
        point = outcome.get("point")
        if not _is_finite(point):
            continue

        token = normalizer(name)
        if token == home_token or name.lower() == "home":
            home_point = float(point)
        elif token == away_token or name.lower() == "away":
            away_point = float(point)

    if home_point is None and _is_finite(away_point):
        home_point = -float(away_point)  # enforce symmetry when only one side provided

    if not _is_finite(home_point):
        return None

    if home_point < 0:
        return ("HOME", abs(home_point))
    if home_point > 0:
        return ("AWAY", abs(home_point))
    return ("PICK", 0.0)


def _update_usage(resp: requests.Response) -> None:
    """Update the shared rate-limit usage counters from a response."""
    try:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            ODDS_API_USAGE["remaining"] = remaining
        if used is not None:
            ODDS_API_USAGE["used"] = used
    except Exception:
        # Header parsing issues should not break caller flows.
        return


def get_historical_spread(
    league: str, event_id: str, kickoff_iso: str, home_name: str, away_name: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the closing spread snapshot from the historical odds endpoint.

    Args:
        league: League identifier ('nfl' or 'cfb').
        event_id: Odds API event identifier.
        kickoff_iso: Kickoff ISO8601 string (UTC expected).
        home_name: Schedule home team name.
        away_name: Schedule away team name.

    Returns:
        Optional dictionary with keys {'favored_team','spread','book','fetched_ts'}.
        Returns None when no qualifying snapshot exists or errors occur.
    """
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport or not event_id:
        return None

    kickoff = _parse_ts(kickoff_iso) or datetime.min.replace(tzinfo=timezone.utc)

    try:
        response = requests.get(
            f"{_THE_ODDS_BASE}/historical/sports/{sport}/events/{event_id}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads",
                "oddsFormat": "american",
                "date": kickoff_iso,
            },
            timeout=20,
        )
        _update_usage(response)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        payload = response.json()
        bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else payload
        selection = _pick_book_pre_kick(bookmakers or [], kickoff)
        if not selection:
            return None

        book, market = selection
        normalized = _extract_spread_from_market(league, market, home_name, away_name)
        if not normalized:
            return None

        favored, spread = normalized
        timestamp = market.get("__ts__") or kickoff
        return {
            "favored_team": favored,
            "spread": float(spread),
            "book": (book.get("key") or book.get("title") or ""),
            "fetched_ts": timestamp.isoformat(),
        }
    except requests.RequestException:
        return None


def get_current_spread(
    league: str, event_id: str, kickoff_iso: str, home_name: str, away_name: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the closing spread snapshot from the current odds endpoint.

    Args:
        league: League identifier ('nfl' or 'cfb').
        event_id: Odds API event identifier.
        kickoff_iso: Kickoff ISO8601 string (UTC expected).
        home_name: Schedule home team name.
        away_name: Schedule away team name.

    Returns:
        Optional dictionary with keys {'favored_team','spread','book','fetched_ts'}.
        Returns None when no qualifying snapshot exists or errors occur.
    """
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport or not event_id:
        return None

    kickoff = _parse_ts(kickoff_iso) or datetime.min.replace(tzinfo=timezone.utc)

    try:
        response = requests.get(
            f"{_THE_ODDS_BASE}/sports/{sport}/events/{event_id}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads",
                "oddsFormat": "american",
            },
            timeout=20,
        )
        _update_usage(response)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        payload = response.json()
        bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else payload
        selection = _pick_book_pre_kick(bookmakers or [], kickoff)
        if not selection:
            return None

        book, market = selection
        normalized = _extract_spread_from_market(league, market, home_name, away_name)
        if not normalized:
            return None

        favored, spread = normalized
        timestamp = market.get("__ts__") or kickoff
        return {
            "favored_team": favored,
            "spread": float(spread),
            "book": (book.get("key") or book.get("title") or ""),
            "fetched_ts": timestamp.isoformat(),
        }
    except requests.RequestException:
        return None


def find_event_id(
    league: str,
    kickoff_utc: datetime,
    home_name: str,
    away_name: str,
) -> Optional[str]:
    """
    Resolve an Odds API event identifier by querying the events endpoint within a window.

    Args:
        league: League identifier ('nfl' or 'cfb').
        kickoff_utc: Kickoff timestamp in UTC.
        home_name: Home team name from the schedule.
        away_name: Away team name from the schedule.

    Returns:
        The matching event_id string when found; otherwise None.
    """
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport or kickoff_utc is None:
        return None

    kickoff = kickoff_utc.astimezone(timezone.utc)
    window_start = kickoff - timedelta(minutes=30)
    window_end = kickoff + timedelta(minutes=30)

    try:
        response = requests.get(
            f"{_THE_ODDS_BASE}/sports/{sport}/events",
            params={
                "apiKey": api_key,
                "regions": "us",
                "commenceTimeFrom": window_start.isoformat(),
                "commenceTimeTo": window_end.isoformat(),
            },
            timeout=20,
        )
        _update_usage(response)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    if not isinstance(payload, list):
        return None

    normalizer = _normalize(league)
    home_token = normalizer(home_name or "")
    away_token = normalizer(away_name or "")
    if not home_token or not away_token:
        return None

    matches: List[Tuple[float, Dict[str, Any]]] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        commence_time = event.get("commence_time")
        event_dt = _parse_ts(commence_time)
        if not event_dt:
            continue

        tokens = set()
        for key in ("home_team", "away_team"):
            value = event.get(key)
            if isinstance(value, str):
                tokens.add(normalizer(value))
        for participant in event.get("teams") or []:
            tokens.add(normalizer(str(participant or "")))

        if {home_token, away_token}.issubset(tokens):
            delta = abs((event_dt - kickoff).total_seconds())
            matches.append((delta, event))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    chosen = matches[0][1]
    event_id = chosen.get("id")
    return str(event_id) if isinstance(event_id, (str, int)) and str(event_id).strip() else None


__all__ = [
    "ODDS_API_USAGE",
    "get_historical_spread",
    "get_current_spread",
    "find_event_id",
]

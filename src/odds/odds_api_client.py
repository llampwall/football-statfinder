"""
Odds API HTTP helpers for ATS backfill workflows.

Purpose:
    Provide deterministic helpers for fetching participants, historical events,
    and historical odds snapshots from The Odds API.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - All timestamps handled in UTC.
    - Historical queries use snapshot timestamps (Odds API historical endpoints).
Side effects:
    - No disk writes; callers manage caching separately.
Do not:
    - Call live /events endpoints for past games (historical endpoints only).
Log contract:
    - HTTP errors surface via `_log_api_error` (red text in console) and update
      `ODDS_API_USAGE` headers for summary reporting.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.common.io_utils import getenv

_THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}

# One-run usage counters (callers can emit a single summary line).
ODDS_API_USAGE: Dict[str, Optional[str]] = {"remaining": None, "used": None}


def _log_api_error(message: str) -> None:
    """Emit API errors in red to satisfy diagnostics guardrails."""
    red = "\033[91m"
    reset = "\033[0m"
    print(f"{red}{message}{reset}", file=sys.stderr)


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _sport_key(league: str) -> Optional[str]:
    return _SPORT_KEYS.get((league or "").lower())


def _normalize(league: str):
    from src.common.team_names import team_merge_key
    from src.common.team_names_cfb import team_merge_key_cfb

    return team_merge_key_cfb if (league or "").lower() == "cfb" else team_merge_key


def _pick_book_pre_kick(
    bookmakers: List[Dict[str, Any]], kickoff: datetime
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for book in bookmakers or []:
        for market in book.get("markets") or []:
            if market.get("key") != "spreads":
                continue
            snapshots = market.get("odds")
            if isinstance(snapshots, list) and snapshots:
                sorted_snaps = sorted(
                    snapshots,
                    key=lambda snap: _parse_ts(snap.get("timestamp") or snap.get("last_update"))
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                snapshot = next(
                    (
                        snap
                        for snap in sorted_snaps
                        if (_parse_ts(snap.get("timestamp") or snap.get("last_update")) or datetime.min.replace(tzinfo=timezone.utc))
                        <= kickoff
                    ),
                    None,
                )
                if snapshot:
                    market_copy = dict(market)
                    market_copy["outcomes"] = snapshot.get("outcomes") or []
                    market_copy["__ts__"] = _parse_ts(snapshot.get("timestamp") or snapshot.get("last_update"))
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

    pinnacle = [
        candidate for candidate in candidates if (candidate[0].get("key") or "").lower() == "pinnacle"
    ]
    if pinnacle:
        return sorted(pinnacle, key=_sort_key, reverse=True)[0]
    return sorted(candidates, key=_sort_key, reverse=True)[0]


def _extract_spread_from_market(
    league: str, market: Dict[str, Any], home_name: str, away_name: str
) -> Optional[Tuple[str, float]]:
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
        home_point = -float(away_point)
    if not _is_finite(home_point):
        return None

    if home_point < 0:
        return ("HOME", abs(home_point))
    if home_point > 0:
        return ("AWAY", abs(home_point))
    return ("PICK", 0.0)


def _update_usage(resp: requests.Response) -> None:
    try:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            ODDS_API_USAGE["remaining"] = remaining
        if used is not None:
            ODDS_API_USAGE["used"] = used
    except Exception:
        pass


def get_historical_spread(
    league: str, event_id: str, kickoff_iso: str, home_name: str, away_name: str
) -> Optional[Dict[str, Any]]:
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
            "source": "history",
        }
    except requests.RequestException as exc:
        _log_api_error(f"ODDS_API_ERROR(get_historical_spread): league={league} error={exc}")
        return None


def get_current_spread(
    league: str, event_id: str, kickoff_iso: str, home_name: str, away_name: str
) -> Optional[Dict[str, Any]]:
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
            "source": "current",
        }
    except requests.RequestException as exc:
        _log_api_error(f"ODDS_API_ERROR(get_current_spread): league={league} error={exc}")
        return None


def get_participants(league: str) -> Optional[List[str]]:
    """
    Fetch participant names for a league from The Odds API participants endpoint.
    Returns a normalized List[str]. Tolerates payloads:
      - list[str]
      - list[dict{name:str}]
      - dict with 'participants' or 'data' arrays.
    """
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport:
        return None
    try:
        response = requests.get(
            f"{_THE_ODDS_BASE}/sports/{sport}/participants",
            params={"apiKey": api_key},
            timeout=20,
        )
        _update_usage(response)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            seq = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("participants"), list):
                seq = payload["participants"]
            elif isinstance(payload.get("data"), list):
                seq = payload["data"]
            else:
                seq = None
        else:
            seq = None

        names: List[str] = []
        if isinstance(seq, list):
            for entry in seq:
                if isinstance(entry, str):
                    token = entry.strip()
                    if token:
                        names.append(token)
                elif isinstance(entry, dict):
                    token = (entry.get("name") or "").strip()
                    if token:
                        names.append(token)
            return names
        _log_api_error(
            f"ODDS_API_PAYLOAD_ERROR(get_participants): unexpected shape {type(payload).__name__}"
        )
    except requests.RequestException as exc:
        _log_api_error(f"ODDS_API_ERROR(get_participants): league={league} error={exc}")
    return None


def get_historical_events(
    league: str,
    snapshot_iso: str,
    *,
    commence_from: Optional[str] = None,
    commence_to: Optional[str] = None,
) -> Optional[List[dict]]:
    """GET /v4/historical/sports/{sport}/events?apiKey=...&date={snapshot_iso}"""
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport:
        return None

    params: Dict[str, str] = {"apiKey": api_key, "date": snapshot_iso}
    if commence_from:
        params["commenceTimeFrom"] = commence_from
    if commence_to:
        params["commenceTimeTo"] = commence_to

    try:
        response = requests.get(
            f"{_THE_ODDS_BASE}/historical/sports/{sport}/events",
            params=params,
            timeout=20,
        )
        _update_usage(response)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return [event for event in payload if isinstance(event, dict)]
        _log_api_error(
            f"ODDS_API_PAYLOAD_ERROR(get_historical_events): expected list, got {type(payload).__name__}"
        )
    except requests.RequestException as exc:
        _log_api_error(f"ODDS_API_ERROR(get_historical_events): league={league} error={exc}")
    return None


__all__ = [
    "ODDS_API_USAGE",
    "get_historical_spread",
    "get_current_spread",
    "get_participants",
    "get_historical_events",
]

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
from typing import Any, Dict, List, Optional, Tuple

from src.common.io_utils import ensure_out_dir
from src.odds.historical_events import get_last_snapshot, list_week_events
from src.odds.odds_api_client import get_historical_spread
from src.odds.participants_cache import provider_name_for, provider_token

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
) -> Tuple[Optional[str], str, Optional[str]]:
    """Resolve the Odds API event id for a game via pinned map or historical events."""
    game_key = str(game_row.get("game_key") or "")

    home_name = game_row.get("home_team_norm") or game_row.get("home_team_raw")
    away_name = game_row.get("away_team_norm") or game_row.get("away_team_raw")
    provider_home, home_status = provider_name_for(league, str(home_name or ""))
    provider_away, away_status = provider_name_for(league, str(away_name or ""))
    mapping = f"h='{home_name}'->'{provider_home or '?'}' a='{away_name}'->'{provider_away or '?'}'"
    provider_issue = any(status != "mapped" for status in (home_status, away_status))
    provider_reason = "no_provider_map" if provider_issue else None

    if pinned_index and game_key:
        pinned = pinned_index.get(game_key)
        if isinstance(pinned, str) and pinned:
            print(
                f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
                f"{mapping} resolver=pinned event_id={pinned}",
                flush=True,
            )
            return pinned, "pinned", None

    kickoff_dt = _extract_kickoff(game_row)
    if kickoff_dt is None:
        reason = "no_kickoff"
        print(
            f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
            f"{mapping} resolver=failed reason={reason} event_id=-",
            flush=True,
        )
        return None, "failed", reason

    if provider_issue:
        reason = provider_reason
        print(
            f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
            f"{mapping} resolver=failed reason={reason} event_id=-",
            flush=True,
        )
        return None, "failed", reason

    target_home_token = provider_token(league, provider_home or "")
    target_away_token = provider_token(league, provider_away or "")

    week_start, week_end = _week_window(league, season, week, kickoff_dt)
    events = list_week_events(league, week_start, week_end)

    best_event = None
    best_delta: Optional[float] = None
    guard_violation = False
    for event in events:
        event_home_raw = str(event.get("home_team") or "")
        event_away_raw = str(event.get("away_team") or "")
        if (
            provider_token(league, event_home_raw) != target_home_token
            or provider_token(league, event_away_raw) != target_away_token
        ):
            continue
        commence = _parse_ts(event.get("commence_time"))
        if not commence:
            continue
        delta_seconds = abs((commence - kickoff_dt).total_seconds())
        if delta_seconds > 90 * 60:
            guard_violation = True
            continue
        if best_delta is None or delta_seconds < best_delta:
            best_event = event
            best_delta = delta_seconds

    if best_event is None:
        reason = "time_guard_miss" if guard_violation else "no_event_match"
        print(
            f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
            f"{mapping} resolver=failed reason={reason} event_id=-",
            flush=True,
        )
        return None, "failed", reason

    event_id = best_event.get("id")
    if isinstance(event_id, (str, int)):
        resolved_id = str(event_id)
        print(
            f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
            f"{mapping} resolver=events event_id={resolved_id}",
            flush=True,
        )
        return resolved_id, "events", None

    print(
        f"ATSDBG(RESOLVE): league={league} week={season}-{week} game={game_key} "
        f"{mapping} resolver=failed reason=invalid_event_id event_id=-",
        flush=True,
    )
    return None, "failed", "invalid_event_id"



def select_closing_spread(
    league: str,
    event_id: str,
    kickoff_iso: str,
    home_name: str,
    away_name: str,
) -> Optional[Dict[str, Any]]:
    """Fetch per-game historical odds snapshot using the weekly historical snapshot."""
    kickoff_dt = _parse_ts(kickoff_iso)
    if kickoff_dt is None:
        return {
            "status": "hist_odds_none",
            "raw_book_count": 0,
            "kept_book_count": 0,
            "kept_book_names": [],
            "source": "history",
            "probe_steps": 0,
            "reason": "no_kickoff",
        }

    snapshot_dt = get_last_snapshot(league)
    if snapshot_dt is not None:
        snapshot_iso = snapshot_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshot_reason = None
    else:
        snapshot_iso = _normalize_kickoff(kickoff_dt, kickoff_iso)
        snapshot_reason = "no_snapshot"

    provider_home_name, _ = provider_name_for(league, home_name)
    provider_away_name, _ = provider_name_for(league, away_name)

    result = get_historical_spread(
        league,
        event_id,
        snapshot_iso,
        provider_home_name or home_name,
        provider_away_name or away_name,
        kickoff_dt,
    )
    result.setdefault("snapshot_date", snapshot_iso)
    result.setdefault("snapshot_used", snapshot_iso)
    result.setdefault("probe_steps", 1)
    if snapshot_reason and "reason" not in result:
        result["reason"] = snapshot_reason
    return result


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
        return kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    if kickoff_iso:
        parsed = _parse_ts(str(kickoff_iso))
        if parsed:
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(kickoff_iso, str) and kickoff_iso.endswith("+00:00"):
            return kickoff_iso.replace("+00:00", "Z")
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

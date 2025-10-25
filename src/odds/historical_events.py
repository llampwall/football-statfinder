"""
Historical event snapshot utilities.

Purpose:
    Retrieve a single historical snapshot for a league week and filter events
    into a caller-provided window.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - Snapshot timestamp chosen at the end of the week (23:59:59Z).
    - Returned events are limited to the supplied week window.
Side effects:
    - None; results cached in-memory per run.
Do not:
    - Issue multiple snapshot calls for the same league/week unless necessary.
Log contract:
    - Underlying HTTP logging handled by odds_api_client.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from src.odds.odds_api_client import get_historical_events

_EVENT_CACHE: Dict[Tuple[str, str], List[dict]] = {}


def list_week_events(
    league: str,
    week_start_utc: datetime,
    week_end_utc: datetime,
) -> List[dict]:
    """Return historical events for the league within [week_start, week_end]."""
    week_start = week_start_utc.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_end = week_end_utc.astimezone(timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    snapshot = week_end
    cache_key = (league.lower(), snapshot.isoformat())

    if cache_key not in _EVENT_CACHE:
        events = get_historical_events(
            league,
            snapshot_dt=snapshot,
            commence_from=week_start,
            commence_to=week_end,
        ) or []
        print(
            f"ATSDBG(HIST-EVENTS): league={league} snapshot={snapshot.isoformat()} fetched={len(events)} window=[{week_start.isoformat()}->{week_end.isoformat()}]",
            flush=True,
        )
        filtered: List[dict] = []
        for event in events:
            commence_dt = _parse_ts(event.get("commence_time"))
            if not commence_dt:
                continue
            if week_start <= commence_dt <= week_end:
                filtered.append(event)
        print(
            f"ATSDBG(HIST-EVENTS): league={league} filtered={len(filtered)}",
            flush=True,
        )
        _EVENT_CACHE[cache_key] = filtered

    return list(_EVENT_CACHE.get(cache_key, []))


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


__all__ = ["list_week_events"]

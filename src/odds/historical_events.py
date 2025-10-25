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
from src.odds.participants_cache import canonical_equals

_EVENT_CACHE: Dict[Tuple[str, str], List[dict]] = {}
_TIME_GUARD_SECONDS = 90 * 60  # temporary widened guard


def list_week_events(
    league: str,
    week_start_utc: datetime,
    week_end_utc: datetime,
) -> List[dict]:
    """Return historical events for the league within [week_start, week_end]."""
    start = week_start_utc.astimezone(timezone.utc)
    end = week_end_utc.astimezone(timezone.utc)
    snapshot = end.replace(hour=23, minute=59, second=59, microsecond=0)
    snapshot_iso = snapshot.isoformat()
    cache_key = (league.lower(), snapshot_iso)

    if cache_key not in _EVENT_CACHE:
        events = get_historical_events(
            league,
            snapshot_iso,
            commence_from=start.isoformat(),
            commence_to=(end + timedelta(seconds=1)).isoformat(),
        ) or []
        print(
            f"ATSDBG(HIST-EVENTS): league={league} snapshot={snapshot_iso} fetched={len(events)} window=[{start.isoformat()}→{end.isoformat()}]",
            flush=True,
        )
        filtered: List[dict] = []
        for event in events:
            commence_dt = _parse_ts(event.get("commence_time"))
            if not commence_dt:
                continue
            if start <= commence_dt <= end:
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

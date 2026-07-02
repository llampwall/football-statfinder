"""Historical event snapshot cache (one snapshot call per league-week).

Port of the harvested ``src/odds/historical_events.py``. Logic preserved
(snapshot chosen inside the active week: Tuesday 12:00Z relative to the
Monday-anchored window; events filtered to the caller's window); adaptations:

* Module-level ``_EVENT_CACHE``/``_LAST_SNAPSHOT`` globals become instance
  state on :class:`HistoricalEventsCache`.
* The HTTP fetch goes through an injected :class:`OddsApiClient` (which honors
  ``settings.odds.cache_only``); ``print`` diagnostics become ``logging``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .odds_api import OddsApiClient, _parse_ts

logger = logging.getLogger(__name__)


class HistoricalEventsCache:
    """Per-run cache of historical events snapshots keyed by week start."""

    def __init__(self, client: OddsApiClient) -> None:
        self._client = client
        self._events: Dict[str, List[dict]] = {}
        self._last_snapshot: Optional[datetime] = None

    def list_week_events(
        self, week_start_utc: datetime, week_end_utc: datetime
    ) -> List[dict]:
        """Return historical events within ``[week_start, week_end]``."""
        week_start = week_start_utc.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        week_end = week_end_utc.astimezone(timezone.utc).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        snapshot = week_start + timedelta(days=1, hours=12)
        self._last_snapshot = snapshot
        cache_key = week_start.isoformat()

        if cache_key not in self._events:
            events = (
                self._client.get_historical_events(
                    snapshot, commence_from=week_start, commence_to=week_end
                )
                or []
            )
            filtered: List[dict] = []
            for event in events:
                commence_dt = _parse_ts(event.get("commence_time"))
                if not commence_dt:
                    continue
                if week_start <= commence_dt <= week_end:
                    filtered.append(event)
            logger.debug(
                "hist-events: league=%s snapshot=%s fetched=%d kept=%d window=[%s -> %s]",
                self._client.league.code,
                snapshot.isoformat(),
                len(events),
                len(filtered),
                week_start.isoformat(),
                week_end.isoformat(),
            )
            self._events[cache_key] = filtered

        return list(self._events.get(cache_key, []))

    def last_snapshot(self) -> Optional[datetime]:
        return self._last_snapshot


__all__ = ["HistoricalEventsCache"]

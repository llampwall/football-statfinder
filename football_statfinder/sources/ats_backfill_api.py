"""ATS-from-API resolution: event ids and historical closing spreads.

Port of the harvested ``src/odds/ats_backfill_api.py``. Logic preserved
(pinned-event-id map first, then historical-events matching with a 90-minute
kickoff guard; weekly snapshot reuse for the odds query); adaptations:

* Module-level ``_WEEK_WINDOW_CACHE`` becomes instance state on
  :class:`AtsBackfillApi`; ``print`` diagnostics become ``logging``.
* The pinned event-id index reads ``paths.odds_pinned_jsonl`` via the counted
  JSONL reader instead of a bare ``json.loads``/``continue`` loop (bug 10).
* The duplicate ``compute_ats`` was NOT ported: the pipeline uses the single
  validated implementation in ``football_statfinder.pipeline.ats``
  (``compute_game_ats``).
* ``resolve_closing_spread`` wraps resolve-then-select into the narrow
  interface the ATS pipeline's paid tier expects (returns ``None`` unless a
  usable spread came back).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .. import paths
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League
from .historical_events import HistoricalEventsCache
from .odds_api import OddsApiClient, _parse_ts, _to_iso_z
from .participants_cache import ParticipantsCache, provider_token

logger = logging.getLogger(__name__)

# Maximum |commence_time - kickoff| for an event match (legacy guard).
_EVENT_TIME_GUARD_SECONDS = 90 * 60


def load_pinned_event_index(
    league: League, season: int, *, out_root: Optional[Path] = None
) -> Dict[str, str]:
    """Return game_key -> Odds API event_id from the pinned spreads ledger."""
    path = paths.odds_pinned_jsonl(league.code, season, out_root=out_root)
    mapping: Dict[str, str] = {}
    for record in read_jsonl(path).rows:
        if record.get("market") != "spreads":
            continue
        game_key = record.get("game_key")
        event_id = (record.get("raw_event") or {}).get("event_id")
        if isinstance(game_key, str) and isinstance(event_id, str):
            mapping[game_key] = event_id
    return mapping


def _extract_kickoff(game: Mapping[str, Any]) -> Optional[datetime]:
    raw_sources = game.get("raw_sources")
    schedule_row = raw_sources.get("schedule_row", {}) if isinstance(raw_sources, dict) else {}
    kickoff = (
        game.get("kickoff_ts")
        or game.get("kickoff_iso_utc")
        or game.get("kickoff_iso")
        or (schedule_row.get("commence_time") if isinstance(schedule_row, dict) else None)
    )
    if not kickoff:
        return None
    return _parse_ts(str(kickoff))


class AtsBackfillApi:
    """Resolves closing spreads for finished games via the historical API.

    One instance per league per run: it shares one participants fetch, one
    historical-events snapshot per week, and one pinned event-id index per
    season across every game it resolves.
    """

    def __init__(
        self,
        league: League,
        settings: Settings,
        *,
        client: Optional[OddsApiClient] = None,
        participants: Optional[ParticipantsCache] = None,
        events: Optional[HistoricalEventsCache] = None,
        out_root: Optional[Path] = None,
    ) -> None:
        self._league = league
        self._settings = settings
        self._out_root = out_root
        self._client = client or OddsApiClient(league, settings, out_root=out_root)
        self._participants = participants or ParticipantsCache(
            league, self._client, out_root=out_root
        )
        self._events = events or HistoricalEventsCache(self._client)
        self._week_windows: Dict[Tuple[int, int], Tuple[datetime, datetime]] = {}
        self._pinned_event_index: Dict[int, Dict[str, str]] = {}

    # -- event resolution -------------------------------------------------------

    def _week_window(
        self, season: int, week: int, reference_dt: datetime
    ) -> Tuple[datetime, datetime]:
        key = (season, week)
        if key in self._week_windows:
            return self._week_windows[key]
        dt = reference_dt.astimezone(timezone.utc)
        monday = dt - timedelta(days=dt.weekday())
        start = datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
        end = start + timedelta(days=7) - timedelta(seconds=1)
        self._week_windows[key] = (start, end)
        return start, end

    def _pinned_index_for(self, season: int) -> Dict[str, str]:
        if season not in self._pinned_event_index:
            self._pinned_event_index[season] = load_pinned_event_index(
                self._league, season, out_root=self._out_root
            )
        return self._pinned_event_index[season]

    def resolve_event_id(
        self,
        season: int,
        week: int,
        game_row: Mapping[str, Any],
        *,
        pinned_index: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Resolve the Odds API event id via pinned map, then historical events.

        Returns ``(event_id, resolver, failure_reason)``.
        """
        game_key = str(game_row.get("game_key") or "")
        if pinned_index is None:
            pinned_index = self._pinned_index_for(season)

        home_name = game_row.get("home_team_norm") or game_row.get("home_team_raw")
        away_name = game_row.get("away_team_norm") or game_row.get("away_team_raw")
        self._participants.build_provider_map(
            [str(home_name or ""), str(away_name or "")]
        )
        provider_home, home_status = self._participants.provider_name_for(str(home_name or ""))
        provider_away, away_status = self._participants.provider_name_for(str(away_name or ""))
        provider_issue = any(status != "mapped" for status in (home_status, away_status))

        if pinned_index and game_key:
            pinned = pinned_index.get(game_key)
            if isinstance(pinned, str) and pinned:
                logger.debug(
                    "ats resolve: league=%s week=%s-%s game=%s resolver=pinned event_id=%s",
                    self._league.code, season, week, game_key, pinned,
                )
                return pinned, "pinned", None

        kickoff_dt = _extract_kickoff(game_row)
        if kickoff_dt is None:
            return self._resolve_failed(season, week, game_key, "no_kickoff")
        if provider_issue:
            return self._resolve_failed(season, week, game_key, "no_provider_map")

        target_home_token = provider_token(self._league, provider_home or "")
        target_away_token = provider_token(self._league, provider_away or "")

        week_start, week_end = self._week_window(season, week, kickoff_dt)
        events = self._events.list_week_events(week_start, week_end)

        best_event: Optional[dict] = None
        best_delta: Optional[float] = None
        guard_violation = False
        for event in events:
            if (
                provider_token(self._league, str(event.get("home_team") or "")) != target_home_token
                or provider_token(self._league, str(event.get("away_team") or "")) != target_away_token
            ):
                continue
            commence = _parse_ts(event.get("commence_time"))
            if not commence:
                continue
            delta_seconds = abs((commence - kickoff_dt).total_seconds())
            if delta_seconds > _EVENT_TIME_GUARD_SECONDS:
                guard_violation = True
                continue
            if best_delta is None or delta_seconds < best_delta:
                best_event = event
                best_delta = delta_seconds

        if best_event is None:
            reason = "time_guard_miss" if guard_violation else "no_event_match"
            return self._resolve_failed(season, week, game_key, reason)

        event_id = best_event.get("id")
        if isinstance(event_id, (str, int)):
            resolved_id = str(event_id)
            logger.debug(
                "ats resolve: league=%s week=%s-%s game=%s resolver=events event_id=%s",
                self._league.code, season, week, game_key, resolved_id,
            )
            return resolved_id, "events", None
        return self._resolve_failed(season, week, game_key, "invalid_event_id")

    def _resolve_failed(
        self, season: int, week: int, game_key: str, reason: str
    ) -> Tuple[None, str, str]:
        logger.debug(
            "ats resolve: league=%s week=%s-%s game=%s resolver=failed reason=%s",
            self._league.code, season, week, game_key, reason,
        )
        return None, "failed", reason

    # -- spread selection ---------------------------------------------------------

    def select_closing_spread(
        self,
        event_id: str,
        kickoff_iso: str,
        home_name: str,
        away_name: str,
    ) -> Dict[str, Any]:
        """Fetch the per-game historical spread using the weekly snapshot date."""
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

        snapshot_dt = self._events.last_snapshot()
        if snapshot_dt is not None:
            snapshot_iso = _to_iso_z(snapshot_dt)
            snapshot_reason = None
        else:
            snapshot_iso = _to_iso_z(kickoff_dt)
            snapshot_reason = "no_snapshot"

        provider_home, _ = self._participants.provider_name_for(home_name)
        provider_away, _ = self._participants.provider_name_for(away_name)

        result = self._client.get_historical_spread(
            event_id,
            snapshot_iso,
            provider_home or home_name,
            provider_away or away_name,
            kickoff_dt,
        )
        result.setdefault("snapshot_date", snapshot_iso)
        result.setdefault("snapshot_used", snapshot_iso)
        result.setdefault("probe_steps", 1)
        if snapshot_reason and "reason" not in result:
            result["reason"] = snapshot_reason
        return result

    # -- pipeline-facing tier ------------------------------------------------------

    def resolve_closing_spread(
        self, season: int, week: int, game_row: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Resolve a usable closing spread for one game, or ``None``.

        This is the paid tier the ATS pipeline calls after the free pinned
        ledger misses. It never raises on a per-game miss; hard config errors
        (missing API key) propagate.
        """
        event_id, _resolver, _reason = self.resolve_event_id(season, week, game_row)
        if not event_id:
            return None
        kickoff_dt = _extract_kickoff(game_row)
        if kickoff_dt is None:
            return None
        home_name = str(game_row.get("home_team_norm") or game_row.get("home_team_raw") or "")
        away_name = str(game_row.get("away_team_norm") or game_row.get("away_team_raw") or "")
        payload = self.select_closing_spread(
            event_id, _to_iso_z(kickoff_dt), home_name, away_name
        )
        if not payload or payload.get("status") != "ok":
            return None
        if payload.get("favored_team") is None or payload.get("spread") is None:
            return None
        return {
            "favored_team": payload["favored_team"],
            "spread": float(payload["spread"]),
            "source": "history",
            "book": payload.get("book"),
            "fetched_ts": payload.get("fetched_ts"),
        }


__all__ = ["AtsBackfillApi", "load_pinned_event_index"]

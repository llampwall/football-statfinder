"""The Odds API client (participants, historical events, historical odds).

Port of the harvested ``src/odds/odds_api_client.py`` from the
``feature/ats-api-backfill`` branch — the implementation that replaces the
dead closing-spread tiers of season 1 (REBUILD.md bug 4). Logic is preserved;
the adaptations are mechanical:

* No ``getenv()``: the client is constructed with a ``League`` and ``Settings``
  and reads the API key / cache-only / debug flags from them. Missing keys are
  surfaced via ``Settings.require`` at call time, never as silent empties.
* ``settings.odds.cache_only`` now blocks *every* paid endpoint (the legacy
  module only guarded the historical-odds call); cache misses return ``None``.
* Module-level ``ODDS_API_USAGE`` global replaced by a per-client ``usage``
  dict; ANSI-red ``print`` diagnostics replaced by ``logging``.
* Cache files move from ``out/debug/hist_odds/{league}/`` to
  ``paths.hist_odds_cache_dir`` (``out/staging/hist_odds/{league}/``).
* Fixed a latent legacy ``NameError``: ``get_current_spread`` called a
  nonexistent ``_pick_book_pre_kick``; it now uses ``_select_book_pre_kick``.
* HTTP is injectable (``http_get``) so tests never touch the network.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from .. import paths
from ..config import Settings
from ..leagues import League
from ..common.io_atomic import write_atomic_json

logger = logging.getLogger(__name__)

_THE_ODDS_BASE = "https://api.the-odds-api.com/v4"

# Preferred books for the closing spread, in order (identical for both leagues
# in the harvested branch).
BOOK_PREFERENCE: Tuple[str, ...] = (
    "pinnacle",
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "betrivers",
)

HttpGet = Callable[..., requests.Response]


def _to_iso_z(dt: datetime) -> str:
    """Return ISO8601 UTC with a trailing 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _snapshot_token(snapshot_iso: str) -> str:
    return snapshot_iso.replace(":", "").replace("/", "-")


def _build_url(path: str, params: Dict[str, Any]) -> str:
    query_items: List[Tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, datetime):
            value = _to_iso_z(value)
        query_items.append((key, str(value)))
    query = urlencode(query_items, doseq=True, safe=":+,")
    base = f"{_THE_ODDS_BASE}{path}"
    return f"{base}?{query}" if query else base


def _select_book_pre_kick(
    bookmakers: List[Dict[str, Any]],
    kickoff: datetime,
    preference: Tuple[str, ...] = BOOK_PREFERENCE,
) -> Tuple[Optional[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    """Return the preferred bookmaker spread snapshot at/before kickoff."""
    kickoff = kickoff.astimezone(timezone.utc)

    candidates: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    saw_spread_market = False
    saw_snapshot = False
    saw_pre = False

    for book in bookmakers or []:
        if not isinstance(book, dict):
            continue
        book_name = (book.get("key") or book.get("title") or "").strip()
        if not book_name:
            continue
        book_key = book_name.lower()
        markets = book.get("markets") or []

        best_candidate: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        best_ts: Optional[datetime] = None

        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "spreads":
                continue
            saw_spread_market = True
            snapshots = market.get("odds")
            snapshot_iter: List[Tuple[Optional[datetime], List[dict]]]
            if isinstance(snapshots, list) and snapshots:
                snapshot_iter = [
                    (
                        _parse_ts(snap.get("timestamp") or snap.get("last_update")),
                        snap.get("outcomes") or [],
                    )
                    for snap in snapshots
                    if isinstance(snap, dict)
                ]
            else:
                snapshot_iter = [(_parse_ts(market.get("last_update")), market.get("outcomes") or [])]

            for ts, outcomes in snapshot_iter:
                if not ts:
                    continue
                ts = ts.astimezone(timezone.utc)
                saw_snapshot = True
                if ts > kickoff:
                    continue
                saw_pre = True
                if best_ts is None or ts > best_ts:
                    best_ts = ts
                    market_copy = dict(market)
                    market_copy["outcomes"] = outcomes
                    market_copy["__ts__"] = ts
                    market_copy["__book_name__"] = book_name
                    best_candidate = (book, market_copy)

        if best_candidate:
            candidates[book_key] = best_candidate

    candidate_names = [
        candidate[1].get("__book_name__")
        or (candidate[0].get("key") or candidate[0].get("title") or "").strip()
        for candidate in candidates.values()
    ]
    diagnostics = {
        "saw_spread_market": saw_spread_market,
        "saw_snapshot": saw_snapshot,
        "saw_pre": saw_pre,
        "candidate_names": candidate_names,
    }

    if not candidates:
        return None, diagnostics

    for preferred in preference:
        candidate = candidates.get(preferred)
        if candidate:
            return candidate, diagnostics

    latest_candidate = max(
        candidates.values(),
        key=lambda item: item[1].get("__ts__") or datetime.min.replace(tzinfo=timezone.utc),
    )
    return latest_candidate, diagnostics


def _extract_spread_from_market(
    league: League, market: Dict[str, Any], home_name: str, away_name: str
) -> Optional[Tuple[str, float]]:
    outcomes = market.get("outcomes") or []
    if not outcomes:
        return None

    home_token = league.merge_key(home_name or "")
    away_token = league.merge_key(away_name or "")

    home_point: Optional[float] = None
    away_point: Optional[float] = None
    for outcome in outcomes:
        name = (outcome.get("name") or "").strip()
        point = outcome.get("point")
        if not _is_finite(point):
            continue
        token = league.merge_key(name)
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


class OddsApiClient:
    """Per-league Odds API client bound to explicit :class:`Settings`."""

    def __init__(
        self,
        league: League,
        settings: Settings,
        *,
        out_root: Optional[Path] = None,
        http_get: Optional[HttpGet] = None,
    ) -> None:
        self._league = league
        self._settings = settings
        self._out_root = out_root
        self._http_get: HttpGet = http_get if http_get is not None else requests.get
        # One-run usage counters (callers can emit a single summary line).
        self.usage: Dict[str, Optional[str]] = {"remaining": None, "used": None}

    # -- configuration -----------------------------------------------------

    @property
    def league(self) -> League:
        return self._league

    @property
    def cache_only(self) -> bool:
        return bool(self._settings.odds.cache_only)

    @property
    def _debug(self) -> bool:
        return bool(self._settings.backfill.ats_debug)

    def _api_key(self) -> str:
        self._settings.require("the_odds_api_key")
        return str(self._settings.the_odds_api_key)

    def _paid_calls_blocked(self, context: str) -> bool:
        if self.cache_only:
            logger.info(
                "odds api call blocked by cache_only: league=%s context=%s",
                self._league.code,
                context,
            )
            return True
        return False

    # -- historical odds cache ----------------------------------------------

    def _hist_odds_cache_path(self, event_id: str, snapshot_iso: str) -> Optional[Path]:
        if not event_id or not snapshot_iso:
            return None
        root = paths.hist_odds_cache_dir(self._league.code, out_root=self._out_root)
        return root / f"{event_id}__{_snapshot_token(snapshot_iso)}.json"

    def hist_odds_cache_exists(self, event_id: str, snapshot_iso: str) -> bool:
        path = self._hist_odds_cache_path(event_id, snapshot_iso)
        return bool(path and path.exists())

    def _load_hist_odds_cache(
        self, event_id: str, snapshot_iso: str
    ) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        path = self._hist_odds_cache_path(event_id, snapshot_iso)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else None
        if not isinstance(bookmakers, list):
            return None
        filtered = [book for book in bookmakers if isinstance(book, dict)]
        return filtered, payload

    # -- HTTP plumbing ------------------------------------------------------

    def _update_usage(self, resp: requests.Response) -> None:
        try:
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            if remaining is not None:
                self.usage["remaining"] = remaining
            if used is not None:
                self.usage["used"] = used
        except Exception:  # pragma: no cover - defensive, mirrors legacy
            pass

    def _log_http_problem(self, context: str, response: Optional[requests.Response], url: str) -> None:
        status = getattr(response, "status_code", "n/a")
        try:
            body = response.text if response is not None else ""
        except Exception:
            body = "<unavailable>"
        logger.error("odds api http error (%s): url=%s status=%s body=%s", context, url, status, body)

    # -- endpoints -----------------------------------------------------------

    def get_historical_event_odds(
        self,
        event_id: str,
        snapshot_iso: str,
        kickoff_iso: Optional[str],
    ) -> Optional[Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """Historical per-event odds at a snapshot; disk cache first, then paid."""
        cache_hit = self._load_hist_odds_cache(event_id, snapshot_iso)
        if cache_hit is not None:
            return cache_hit

        if self._paid_calls_blocked("get_historical_event_odds"):
            logger.info(
                "hist-odds cache miss (cache_only): league=%s event_id=%s snapshot=%s",
                self._league.code,
                event_id,
                snapshot_iso,
            )
            return None

        if not event_id:
            return [], None
        api_key = self._api_key()

        url = _build_url(
            f"/historical/sports/{self._league.odds_sport_key}/events/{event_id}/odds",
            {
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads",
                "oddsFormat": "american",
                "date": snapshot_iso,
            },
        )
        redacted_url = url.replace(api_key, "<REDACTED>")

        payload_data: Optional[Any] = None
        response: Optional[requests.Response] = None
        try:
            logger.debug(
                "hist-odds request: league=%s event_id=%s date=%s kickoff=%s",
                self._league.code,
                event_id,
                snapshot_iso,
                kickoff_iso,
            )
            response = self._http_get(url, timeout=20)
            self._update_usage(response)
            if response.status_code >= 400:
                self._log_http_problem("get_historical_event_odds", response, redacted_url)
                return [], None
            payload = response.json()
            payload_data = payload
            if isinstance(payload, dict):
                if isinstance(payload.get("bookmakers"), list):
                    bookmakers = payload.get("bookmakers")
                elif isinstance(payload.get("data"), dict):
                    bookmakers = payload.get("data", {}).get("bookmakers")
                else:
                    bookmakers = None
            else:
                bookmakers = payload
            if isinstance(bookmakers, list):
                filtered = [book for book in bookmakers if isinstance(book, dict)]
                return filtered, payload if self._debug else None
            return [], payload if self._debug and isinstance(payload, dict) else None
        except requests.RequestException as exc:
            logger.error("odds api error (get_historical_event_odds): url=%s error=%s", redacted_url, exc)
        except ValueError:
            self._log_http_problem("get_historical_event_odds - decode", response, redacted_url)
        return [], payload_data if (self._debug and isinstance(payload_data, dict)) else None

    def get_historical_spread(
        self,
        event_id: str,
        snapshot_iso: str,
        home_name: str,
        away_name: str,
        kickoff_dt: datetime,
    ) -> Dict[str, Any]:
        """Fetch a historical spread snapshot at/before kickoff.

        Always returns a status payload (``status == "ok"`` carries
        ``favored_team``/``spread``/``book``/``fetched_ts``).
        """
        if kickoff_dt.tzinfo is None:
            kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
        else:
            kickoff_dt = kickoff_dt.astimezone(timezone.utc)

        fetch_result = self.get_historical_event_odds(event_id, snapshot_iso, _to_iso_z(kickoff_dt))
        cache_miss = False
        if fetch_result is None:
            bookmakers: List[Dict[str, Any]] = []
            raw_payload: Optional[Dict[str, Any]] = None
            cache_miss = self.cache_only
        else:
            bookmakers, raw_payload = fetch_result
        raw_books = len(bookmakers)

        selection, diagnostics = _select_book_pre_kick(bookmakers, kickoff_dt)
        kept_names = diagnostics.get("candidate_names", [])
        kept_books = len(kept_names)

        favored: Optional[str] = None
        spread: Optional[float] = None
        chosen_book: Optional[str] = None
        fetched_ts: Optional[datetime] = None
        status = "hist_odds_none" if raw_books == 0 else "hist_odds_filtered"
        reason: Optional[str] = None

        if selection:
            book, market = selection
            normalized = _extract_spread_from_market(self._league, market, home_name, away_name)
            if normalized:
                favored, spread = normalized
                fetched_ts = market.get("__ts__") or kickoff_dt
                chosen_book = (
                    market.get("__book_name__") or book.get("key") or book.get("title") or ""
                ).strip()
                status = "ok"
            else:
                reason = "outcome_parse"
        else:
            if cache_miss:
                reason = "cache_miss"
            elif raw_books == 0:
                reason = "raw_zero"
            elif diagnostics.get("saw_spread_market") is False:
                reason = "no_spread_market"
            elif diagnostics.get("saw_snapshot") and not diagnostics.get("saw_pre"):
                reason = "time_guard_miss"
            else:
                reason = "no_pre_kick_snapshot"

        logger.debug(
            "hist-odds: league=%s event=%s raw_books=%s kept_books=%s snapshot=%s "
            "favored=%s spread=%s book=%s reason=%s",
            self._league.code,
            event_id,
            raw_books,
            kept_books,
            snapshot_iso,
            favored,
            spread,
            chosen_book,
            reason,
        )

        if self._debug and raw_payload is not None:
            try:
                debug_path = self._hist_odds_cache_path(event_id, snapshot_iso)
                if debug_path is not None:
                    bookmakers_debug: Any
                    if isinstance(raw_payload, dict):
                        bookmakers_debug = raw_payload.get("bookmakers")
                        if bookmakers_debug is None and isinstance(raw_payload.get("data"), dict):
                            bookmakers_debug = raw_payload["data"].get("bookmakers")
                    else:
                        bookmakers_debug = raw_payload
                    write_atomic_json(debug_path, {"bookmakers": bookmakers_debug})
            except Exception:  # pragma: no cover - debug writes are best-effort
                logger.debug("hist-odds debug write failed", exc_info=True)

        payload: Dict[str, Any] = {
            "status": status,
            "raw_book_count": raw_books,
            "kept_book_count": kept_books,
            "kept_book_names": kept_names,
            "source": "history",
            "snapshot_date": snapshot_iso,
            "probe_steps": 1,
        }
        if reason:
            payload["reason"] = reason
        if status != "ok":
            return payload

        payload.update(
            {
                "favored_team": favored,
                "spread": float(spread or 0.0),
                "book": chosen_book,
                "fetched_ts": (fetched_ts or kickoff_dt).isoformat(),
            }
        )
        return payload

    def get_current_spread(
        self, event_id: str, kickoff_iso: str, home_name: str, away_name: str
    ) -> Optional[Dict[str, Any]]:
        if not event_id or self._paid_calls_blocked("get_current_spread"):
            return None
        api_key = self._api_key()
        kickoff = _parse_ts(kickoff_iso) or datetime.min.replace(tzinfo=timezone.utc)

        url = _build_url(
            f"/sports/{self._league.odds_sport_key}/events/{event_id}/odds",
            {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"},
        )
        response: Optional[requests.Response] = None
        try:
            response = self._http_get(url, timeout=20)
            self._update_usage(response)
            if response.status_code >= 400:
                self._log_http_problem("get_current_spread", response, url)
                return None
            payload = response.json()
            bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else payload
            selection, _diagnostics = _select_book_pre_kick(bookmakers or [], kickoff)
            if not selection:
                return None
            book, market = selection
            normalized = _extract_spread_from_market(self._league, market, home_name, away_name)
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
            logger.error("odds api error (get_current_spread): url=%s error=%s", url, exc)
            return None
        except ValueError:
            self._log_http_problem("get_current_spread - decode", response, url)
            return None

    def get_participants(self) -> Optional[List[Dict[str, str]]]:
        """Fetch participant names; tolerates every payload shape the API uses."""
        if self._paid_calls_blocked("get_participants"):
            return None
        api_key = self._api_key()
        url = _build_url(f"/sports/{self._league.odds_sport_key}/participants", {"apiKey": api_key})
        response: Optional[requests.Response] = None
        try:
            response = self._http_get(url, timeout=20)
            self._update_usage(response)
            if response.status_code >= 400:
                self._log_http_problem("get_participants", response, url)
                return None
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

            if isinstance(seq, list):
                entries: List[Dict[str, str]] = []
                for entry in seq:
                    if isinstance(entry, str):
                        token = entry.strip()
                        if token:
                            entries.append({"name": token})
                    elif isinstance(entry, dict):
                        token = (
                            entry.get("name")
                            or entry.get("full_name")
                            or entry.get("fullName")
                            or entry.get("team")
                            or ""
                        ).strip()
                        if token:
                            record: Dict[str, str] = {"name": token}
                            participant_id = (
                                entry.get("id") or entry.get("participant_id") or entry.get("par_id")
                            )
                            if isinstance(participant_id, str) and participant_id.strip():
                                record["id"] = participant_id.strip()
                            entries.append(record)
                return entries
            logger.error(
                "odds api payload error (get_participants): url=%s unexpected shape %s",
                url,
                type(payload).__name__,
            )
        except requests.RequestException as exc:
            logger.error("odds api error (get_participants): url=%s error=%s", url, exc)
        except ValueError:
            self._log_http_problem("get_participants - decode", response, url)
        return None

    def get_historical_events(
        self,
        snapshot_dt: datetime,
        *,
        commence_from: Optional[datetime] = None,
        commence_to: Optional[datetime] = None,
        event_ids: Optional[List[str]] = None,
    ) -> Optional[List[dict]]:
        """Fetch a historical events snapshot for the league."""
        if self._paid_calls_blocked("get_historical_events"):
            return None
        api_key = self._api_key()

        params: Dict[str, Any] = {
            "apiKey": api_key,
            "date": _to_iso_z(snapshot_dt),
            "dateFormat": "iso",
            "eventIds": ",".join(event_ids[:1000]) if event_ids else None,
        }
        if commence_from is not None:
            params["commenceTimeFrom"] = _to_iso_z(commence_from)
        if commence_to is not None:
            params["commenceTimeTo"] = _to_iso_z(commence_to)
        url = _build_url(f"/historical/sports/{self._league.odds_sport_key}/events", params)

        response: Optional[requests.Response] = None
        try:
            response = self._http_get(url, timeout=20)
            self._update_usage(response)
            if response.status_code >= 400:
                self._log_http_problem("get_historical_events", response, url)
                return None
            payload = response.json()
            if isinstance(payload, dict):
                if isinstance(payload.get("events"), list):
                    payload = payload["events"]
                elif isinstance(payload.get("data"), list):
                    payload = payload["data"]
            if isinstance(payload, list):
                if not payload:
                    logger.warning(
                        "odds api empty (get_historical_events): league=%s snapshot=%s",
                        self._league.code,
                        _to_iso_z(snapshot_dt),
                    )
                return [event for event in payload if isinstance(event, dict)]
            logger.error(
                "odds api payload error (get_historical_events): url=%s payload=%r", url, payload
            )
        except requests.RequestException as exc:
            logger.error("odds api error (get_historical_events): url=%s error=%s", url, exc)
        except ValueError:
            self._log_http_problem("get_historical_events - decode", response, url)
        return None


__all__ = [
    "BOOK_PREFERENCE",
    "OddsApiClient",
]

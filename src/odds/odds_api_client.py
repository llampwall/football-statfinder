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

import json
import math
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from src.common.io_utils import ensure_out_dir, getenv, write_atomic_json

_THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}

# One-run usage counters (callers can emit a single summary line).
ODDS_API_USAGE: Dict[str, Optional[str]] = {"remaining": None, "used": None}
_ATS_DEBUG = getenv("ATS_DEBUG", "0") == "1"

_BOOK_PREFERENCE: Dict[str, List[str]] = {
    "nfl": ["pinnacle", "fanduel", "draftkings", "betmgm", "caesars", "betrivers"],
    "cfb": ["pinnacle", "fanduel", "draftkings", "betmgm", "caesars", "betrivers"],
}



def _to_iso_z(dt: datetime) -> str:
    """Return ISO8601 UTC with a trailing 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _select_book_pre_kick(
    league: str, bookmakers: List[Dict[str, Any]], kickoff: datetime
) -> Tuple[Optional[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, bool]]:
    """
    Return the preferred bookmaker snapshot at/before kickoff plus diagnostics.
    """

    kickoff = kickoff.astimezone(timezone.utc)
    preference = [token.lower() for token in _BOOK_PREFERENCE.get(league.lower(), [])]

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
        local_saw_snapshot = False
        local_saw_pre = False

        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("key") != "spreads":
                continue
            saw_spread_market = True
            snapshots = market.get("odds")
            if isinstance(snapshots, list) and snapshots:
                for snap in snapshots:
                    if not isinstance(snap, dict):
                        continue
                    ts = _parse_ts(snap.get("timestamp") or snap.get("last_update"))
                    if not ts:
                        continue
                    ts = ts.astimezone(timezone.utc)
                    local_saw_snapshot = True
                    saw_snapshot = True
                    market_copy = dict(market)
                    market_copy["outcomes"] = snap.get("outcomes") or []
                    market_copy["__ts__"] = ts
                    market_copy["__book_name__"] = book_name
                    if ts <= kickoff:
                        local_saw_pre = True
                        saw_pre = True
                        if best_ts is None or ts > best_ts:
                            best_ts = ts
                            best_candidate = (book, market_copy)
            else:
                ts = _parse_ts(market.get("last_update"))
                if ts:
                    ts = ts.astimezone(timezone.utc)
                    local_saw_snapshot = True
                    saw_snapshot = True
                    market_copy = dict(market)
                    market_copy["outcomes"] = market.get("outcomes") or []
                    market_copy["__ts__"] = ts
                    market_copy["__book_name__"] = book_name
                    if ts <= kickoff:
                        local_saw_pre = True
                        saw_pre = True
                        if best_ts is None or ts > best_ts:
                            best_ts = ts
                            best_candidate = (book, market_copy)

        if best_candidate:
            candidates[book_key] = best_candidate
        elif local_saw_snapshot and not local_saw_pre:
            # ensure time_guard diagnostics reflect books with post-kick only snapshots
            pass

    candidate_names: List[str] = [
        candidate[1].get("__book_name__") or (candidate[0].get("key") or candidate[0].get("title") or "").strip()
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

    # Pick the latest timestamp overall if no preferred book present.
    latest_candidate = max(
        candidates.values(),
        key=lambda item: item[1].get("__ts__") or datetime.min.replace(tzinfo=timezone.utc),
    )
    return latest_candidate, diagnostics


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


def _build_url(path: str, params: Dict[str, Any]) -> str:
    query_items: List[Tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, datetime):
            # Normalise datetimes defensively in case callers pass them directly.
            value = _to_iso_z(value)
        query_items.append((key, str(value)))
    query = urlencode(query_items, doseq=True, safe=":+,")
    base = f"{_THE_ODDS_BASE}{path}"
    return f"{base}?{query}" if query else base


def _log_http_problem(context: str, response: Optional[requests.Response], url: str) -> None:
    status = getattr(response, "status_code", "n/a")
    try:
        body = response.text if response is not None else ""
    except Exception:
        body = "<unavailable>"
    request_url = getattr(getattr(response, "request", None), "url", None)
    full_url = request_url or url
    _log_api_error(
        f"ODDS_API_HTTP_ERROR({context}): url={full_url} status={status} body={body}"
    )


def _log_request_exception(context: str, url: str, error: Exception) -> None:
    _log_api_error(f"ODDS_API_ERROR({context}): url={url} error={error}")


def get_historical_event_odds(
    league: str,
    event_id: str,
    snapshot_iso: str,
    kickoff_iso: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport or not event_id:
        return [], None

    url = _build_url(
        f"/historical/sports/{sport}/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "american",
            "date": snapshot_iso,
        },
    )

    redacted_url = url.replace(api_key, "<REDACTED>")
    print(
        f"HIST-ODDS-URL: league={league} url={redacted_url}",
        flush=True,
    )
    request_log = {
        "league": league,
        "event_id": event_id,
        "date": snapshot_iso,
        "kickoff_ts": kickoff_iso,
    }
    print(f"HIST-ODDS-REQUEST: {json.dumps(request_log, ensure_ascii=False)}", flush=True)

    payload_data: Optional[Any] = None
    try:
        response = requests.get(url, timeout=20)
        _update_usage(response)
        if response.status_code >= 400:
            detail = ""
            try:
                detail = response.text.strip()
            except Exception:
                detail = "<unavailable>"
            _log_api_error(
                "ODDS_API_ERROR(get_historical_event_odds): status={status} url={url} detail={detail}".format(
                    status=response.status_code,
                    url=redacted_url,
                    detail=detail,
                )
            )
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
            return filtered, payload if _ATS_DEBUG else None
        return [], payload if _ATS_DEBUG and isinstance(payload, dict) else None
    except requests.RequestException as exc:
        _log_request_exception("get_historical_event_odds", url, exc)
    except ValueError:
        _log_http_problem("get_historical_event_odds - decode", locals().get("response"), url)
    return [], payload_data if (_ATS_DEBUG and isinstance(payload_data, dict)) else None


def get_historical_spread(
    league: str,
    event_id: str,
    snapshot_iso: str,
    home_name: str,
    away_name: str,
    kickoff_dt: datetime,
) -> Optional[Dict[str, Any]]:
    """Fetch a historical spread snapshot at/before kickoff using weekly snapshot date."""

    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    else:
        kickoff_dt = kickoff_dt.astimezone(timezone.utc)

    kickoff_iso = _to_iso_z(kickoff_dt)

    bookmakers, raw_payload = get_historical_event_odds(league, event_id, snapshot_iso, kickoff_iso)
    raw_books = len(bookmakers)

    selection, diagnostics = _select_book_pre_kick(league, bookmakers, kickoff_dt)
    kept_names = diagnostics.get("candidate_names", [])
    kept_books = len(kept_names)

    favored_for_log: Optional[str] = None
    spread_for_log: Optional[float] = None
    chosen_book_key: Optional[str] = None
    fetched_ts: Optional[datetime] = None
    status = "hist_odds_none" if raw_books == 0 else "hist_odds_filtered"
    reason = None

    if selection:
        book, market = selection
        normalized = _extract_spread_from_market(league, market, home_name, away_name)
        if normalized:
            favored_for_log, spread_for_log = normalized
            fetched_ts = market.get("__ts__") or kickoff_dt
            chosen_book_key = (market.get("__book_name__") or book.get("key") or book.get("title") or "").strip()
            status = "ok"
        else:
            reason = "outcome_parse"
    else:
        if raw_books == 0:
            reason = "raw_zero"
        elif diagnostics.get("saw_spread_market") is False:
            reason = "no_spread_market"
        elif diagnostics.get("saw_snapshot") and not diagnostics.get("saw_pre"):
            reason = "time_guard_miss"
        else:
            reason = "no_pre_kick_snapshot"

    log_line = (
        "ATSDBG(HIST-ODDS): league={league} event={event} raw_books={raw} kept_books={kept} "
        "names={names} snapshot={snapshot} favored={favored} spread={spread} book={book}"
    ).format(
        league=league,
        event=event_id,
        raw=raw_books,
        kept=kept_books,
        names=kept_names,
        snapshot=snapshot_iso,
        favored=favored_for_log,
        spread=spread_for_log,
        book=chosen_book_key,
    )
    if reason:
        log_line += f" reason={reason}"
    print(log_line, flush=True)

    if _ATS_DEBUG and raw_payload is not None:
        try:
            out_root = ensure_out_dir() / "debug" / "hist_odds" / league.lower()
            out_root.mkdir(parents=True, exist_ok=True)
            snapshot_token = snapshot_iso.replace(":", "").replace("/", "-")
            debug_path = out_root / f"{event_id}__{snapshot_token}.json"
            bookmakers_debug: Any
            if isinstance(raw_payload, dict):
                bookmakers_debug = raw_payload.get("bookmakers")
                if bookmakers_debug is None and isinstance(raw_payload.get("data"), dict):
                    bookmakers_debug = raw_payload["data"].get("bookmakers")
            else:
                bookmakers_debug = raw_payload
            write_atomic_json(
                debug_path,
                {"bookmakers": bookmakers_debug},
            )
        except Exception:
            # Debug writes are best-effort; ignore failures.
            pass

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
            "favored_team": favored_for_log,
            "spread": float(spread_for_log or 0.0),
            "book": chosen_book_key,
            "fetched_ts": (fetched_ts or kickoff_dt).isoformat(),
        }
    )
    return payload


def get_current_spread(
    league: str, event_id: str, kickoff_iso: str, home_name: str, away_name: str
) -> Optional[Dict[str, Any]]:
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport or not event_id:
        return None

    kickoff = _parse_ts(kickoff_iso) or datetime.min.replace(tzinfo=timezone.utc)

    url = _build_url(
        f"/sports/{sport}/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "american",
        },
    )

    try:
        response = requests.get(url, timeout=20)
        _update_usage(response)
        if response.status_code >= 400:
            _log_http_problem("get_current_spread", response, url)
            return None
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
        _log_request_exception("get_current_spread", url, exc)
        return None
    except ValueError:
        _log_http_problem("get_current_spread - decode", response, url)
        return None


def get_participants(league: str) -> Optional[List[Dict[str, str]]]:
    """
    Fetch participant names for a league from The Odds API participants endpoint.
    Returns a normalized List[{"name": str, "id": Optional[str]}]. Tolerates payloads:
      - list[str]
      - list[dict{name|full_name|fullName|team:str, id?:str}]
      - dict with 'participants' or 'data' arrays containing either of the above.
    """
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport:
        return None

    url = _build_url(
        f"/sports/{sport}/participants",
        {"apiKey": api_key},
    )
    try:
        response = requests.get(url, timeout=20)
        _update_usage(response)
        if response.status_code >= 400:
            _log_http_problem("get_participants", response, url)
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

        entries: List[Dict[str, str]] = []
        if isinstance(seq, list):
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
                        participant_id = entry.get("id") or entry.get("participant_id") or entry.get("par_id")
                        if isinstance(participant_id, str) and participant_id.strip():
                            record["id"] = participant_id.strip()
                        entries.append(record)
            return entries
        _log_api_error(
            f"ODDS_API_PAYLOAD_ERROR(get_participants): url={url} unexpected shape {type(payload).__name__}"
        )
    except requests.RequestException as exc:
        _log_request_exception("get_participants", url, exc)
    except ValueError:
        _log_http_problem("get_participants - decode", response, url)
    return None


def get_historical_events(
    league: str,
    snapshot_dt: datetime,
    *,
    commence_from: Optional[datetime] = None,
    commence_to: Optional[datetime] = None,
    event_ids: Optional[List[str]] = None,
) -> Optional[List[dict]]:
    """Fetch historical events snapshot for the league."""
    api_key = getenv("THE_ODDS_API_KEY")
    sport = _sport_key(league)
    if not api_key or not sport:
        return None

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
    url = _build_url(f"/historical/sports/{sport}/events", params)

    try:
        response = requests.get(url, timeout=20)
        _update_usage(response)
        if response.status_code >= 400:
            _log_api_error(
                f"ODDS_API_ERROR(get_historical_events): league={league} status={response.status_code}"
            )
            _log_http_problem("get_historical_events", response, url)
            return None
        payload = response.json()

        if isinstance(payload, dict):
            if isinstance(payload.get("events"), list):
                payload = payload["events"]
            elif isinstance(payload.get("data"), list):
                payload = payload["data"]

        if isinstance(payload, list):
            if not payload:
                _log_api_error(
                    f"ODDS_API_EMPTY(get_historical_events): url={response.request.url} "
                    f"status={response.status_code} body={response.text}"
                )
            return [event for event in payload if isinstance(event, dict)]

        preview = payload
        try:
            preview = json.dumps(payload, ensure_ascii=False)
        except Exception:
            preview = str(payload)
        _log_api_error(
            f"ODDS_API_PAYLOAD_ERROR(get_historical_events): url={response.request.url} "
            f"status={response.status_code} payload={preview}"
        )
    except requests.RequestException as exc:
        _log_request_exception("get_historical_events", url, exc)
    except ValueError:
        _log_http_problem("get_historical_events - decode", response, url)
    return None


__all__ = [
    "ODDS_API_USAGE",
    "get_historical_event_odds",
    "get_historical_spread",
    "get_current_spread",
    "get_participants",
    "get_historical_events",
]

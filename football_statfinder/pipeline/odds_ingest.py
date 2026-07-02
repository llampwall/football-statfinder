"""Raw odds ingestion: The Odds API -> append-only staging files.

One league-parameterized port of the season-1 twins ``src/odds/nfl_ingest.py``
and ``src/odds/cfb_ingest.py`` (~95% identical). Every provider record is
staged unfiltered — no week/bookmaker pre-filtering — so pinning and promotion
stay deterministic and debuggable.

Changes from the legacy behavior, all deliberate:

* A missing THE_ODDS_API_KEY raises ``ConfigError`` via ``settings.require``
  instead of silently returning an empty result, and fetch exceptions
  propagate to the caller instead of being swallowed (REBUILD.md bug 8; the
  legacy twins kept the refresh green on auth/network failure).
* Configuration (staging enable flag) comes from an explicit ``Settings``
  argument, never from ``getenv``.
* League constants (sport key, display label, name normalizers) come from the
  ``League`` object; the CFB twin's imports from the 791-line legacy CLI
  module ``fetch_week_odds_cfb`` are not carried over.
* Outcome names are tokenized as ``merge_key(normalize_odds_name(name))`` for
  both leagues (the legacy NFL twin tokenized raw outcome names without
  normalizing first; CFB's normalize-then-token order was the correct one).
* Paths come from ``football_statfinder.paths`` and nothing is created at
  import time (the legacy NFL twin ran ``mkdir`` on import).
* The staging file is written through ``io_atomic``.

The HTTP fetch is injectable (``fetch_events`` parameter) so tests never
touch the network.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

from .. import paths
from ..common.io_atomic import write_atomic_jsonl
from ..config import Settings
from ..leagues import League

logger = logging.getLogger(__name__)

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MARKET_KEYS = ("spreads", "totals", "h2h")

# Signature of an injectable fetcher: (api_key, sport_key) -> list of events.
FetchEvents = Callable[[str, str], List[dict]]


def fetch_events_http(api_key: str, sport_key: str) -> List[dict]:
    """Fetch current odds events from The Odds API (network; not for tests).

    Raises on HTTP errors — the legacy twins swallowed every exception into
    an empty result (bug 8).
    """
    url = f"{THE_ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        logger.warning("odds API returned non-list payload for %s; treating as empty", sport_key)
        return []
    return data


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value in (None, "", "null"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_outcomes(league: League, outcomes: Iterable[dict]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for outcome in outcomes or []:
        name = outcome.get("name")
        token = league.merge_key(league.normalize_odds_name(name or ""))
        sanitized.append(
            {
                "name": name,
                "token": token,
                "price": _to_int(outcome.get("price")),
                "point": _to_float(outcome.get("point")),
            }
        )
    return sanitized


def _build_market_payload(league: League, market: Optional[dict]) -> Dict[str, Any]:
    if not market:
        return {"key": None, "last_update": None, "outcomes": []}
    last_dt = _parse_utc(market.get("last_update"))
    return {
        "key": market.get("key"),
        "last_update": _isoformat(last_dt) if last_dt else None,
        "outcomes": _sanitize_outcomes(league, market.get("outcomes") or []),
    }


def ingest_raw(
    league: League,
    settings: Settings,
    *,
    fetch_events: Optional[FetchEvents] = None,
    now: Optional[datetime] = None,
    out_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fetch, normalize, and stage raw odds provider records for one league.

    Writes ``out/staging/odds_raw/{league}/<YYYYMMDDTHHMMSSZ>.jsonl``
    atomically (one line per bookmaker+market snapshot) and returns::

        {records, fetch_ts, path, counts: {books, markets}, skipped_reason}

    ``skipped_reason`` is non-None only when staging is disabled in settings.
    Missing THE_ODDS_API_KEY raises ``ConfigError``; fetch errors propagate.
    """
    if not settings.odds.staging_enable:
        logger.info("%s odds staging disabled by settings; skipping ingest", league.display)
        return {
            "records": [],
            "fetch_ts": None,
            "path": None,
            "counts": {},
            "skipped_reason": "staging_disabled",
        }

    settings.require("the_odds_api_key")
    api_key = settings.the_odds_api_key
    assert api_key is not None  # settings.require guarantees this

    fetcher = fetch_events or fetch_events_http
    events = fetcher(api_key, league.odds_sport_key)

    fetch_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fetch_iso = _isoformat(fetch_dt)
    fetch_token = fetch_dt.strftime("%Y%m%dT%H%M%SZ")

    records: List[Dict[str, Any]] = []
    books_counter: Counter[str] = Counter()
    markets_counter: Counter[str] = Counter()

    for event in events:
        commence_dt = _parse_utc(event.get("commence_time"))
        commence_iso = _isoformat(commence_dt) if commence_dt else None
        home_raw = event.get("home_team") or ""
        away_raw = event.get("away_team") or ""
        home_norm = league.normalize_odds_name(home_raw)
        away_norm = league.normalize_odds_name(away_raw)
        home_token = league.merge_key(home_norm)
        away_token = league.merge_key(away_norm)

        for bookmaker in event.get("bookmakers") or []:
            book_key = bookmaker.get("key") or "unknown"
            book_title = bookmaker.get("title")
            books_counter[book_key] += 1
            markets = bookmaker.get("markets") or []
            for market_key in MARKET_KEYS:
                market = next((m for m in markets if m.get("key") == market_key), None)
                if not market:
                    continue
                markets_counter[market_key] += 1
                records.append(
                    {
                        "fetch_ts": fetch_iso,
                        "event_id": event.get("id"),
                        "event_start": commence_iso,
                        "book": book_key,
                        "book_title": book_title,
                        "market": market_key,
                        "market_payload": _build_market_payload(league, market),
                        "home_raw": home_raw,
                        "away_raw": away_raw,
                        "home_norm": home_norm,
                        "away_norm": away_norm,
                        "home_token": home_token,
                        "away_token": away_token,
                        "league": league.display,
                        "source": "the-odds-api",
                    }
                )

    raw_path = None
    if records:
        raw_path = paths.odds_raw_dir(league.code, out_root=out_root) / f"{fetch_token}.jsonl"
        write_atomic_jsonl(raw_path, records)
        logger.info(
            "%s odds ingest staged %d record(s) to %s", league.display, len(records), raw_path
        )
    else:
        logger.info("%s odds ingest produced no records (events=%d)", league.display, len(events))

    return {
        "records": records,
        "fetch_ts": fetch_iso,
        "path": raw_path,
        "counts": {"books": dict(books_counter), "markets": dict(markets_counter)},
        "skipped_reason": None,
    }


__all__ = ["FetchEvents", "MARKET_KEYS", "THE_ODDS_API_BASE", "fetch_events_http", "ingest_raw"]

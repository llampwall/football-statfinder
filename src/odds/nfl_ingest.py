"""NFL odds raw ingestion staging utilities (append-only).

Purpose & scope:
    Retrieve every available NFL bookmaker snapshot from The Odds API
    without pre-filtering by week, normalize team identities, and append
    the records to a staging area so downstream pinning/promotions can
    operate deterministically.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B1, F, H, I)

Invariants:
    * All timestamps are UTC ISO8601 with `Z` suffix.
    * Staging writes are append-only; existing files are never mutated.
    * Team normalization relies on `src.common.team_names` utilities.

Side effects:
    * Appends to `staging/odds_raw/nfl/<YYYYMMDDTHHMMSSZ>.jsonl`.

Do not:
    * Filter by season/week or bookmaker; keep every provider record.
    * Emit staging rows when staging is disabled or API credentials absent.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from src.common.io_utils import ensure_out_dir, read_env, getenv
from src.common.team_names import normalize_team_display, team_merge_key

OUT_ROOT = ensure_out_dir()
RAW_DIR = OUT_ROOT / "staging" / "odds_raw" / "nfl"
RAW_DIR.mkdir(parents=True, exist_ok=True)

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
MARKET_KEYS = ("spreads", "totals", "h2h")


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_int(value: Any) -> Optional[int]:
    try:
        if value in (None, "", "null"):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_events(api_key: str) -> List[dict]:
    url = f"{THE_ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
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
        return []
    return data


def _sanitize_outcomes(outcomes: Iterable[dict]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for outcome in outcomes or []:
        name = outcome.get("name")
        token = team_merge_key(name or "")
        sanitized.append(
            {
                "name": name,
                "token": token,
                "price": _to_int(outcome.get("price")),
                "point": _to_float(outcome.get("point")),
            }
        )
    return sanitized


def _build_market_payload(market: Optional[dict]) -> Dict[str, Any]:
    if not market:
        return {"key": None, "last_update": None, "outcomes": []}
    last_update_raw = market.get("last_update")
    try:
        last_dt = datetime.fromisoformat(str(last_update_raw).replace("Z", "+00:00"))
        last_update = _isoformat(last_dt)
    except Exception:
        last_update = None
    return {
        "key": market.get("key"),
        "last_update": last_update,
        "outcomes": _sanitize_outcomes(market.get("outcomes") or []),
    }


def _is_enabled(flag: Optional[str]) -> bool:
    if flag is None:
        return True
    return flag.strip().lower() not in {"0", "false", "off", "disabled"}


def ingest_nfl_odds_raw() -> Dict[str, Any]:
    """Fetch, normalize, and stage raw NFL odds provider records."""
    if not _is_enabled(os.getenv("ODDS_STAGING_ENABLE", "1")):
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    env = read_env(["THE_ODDS_API_KEY"])
    api_key = env.get("THE_ODDS_API_KEY")
    if not api_key:
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    try:
        events = _fetch_events(api_key)
    except Exception:
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    fetch_dt = datetime.now(timezone.utc)
    fetch_iso = _isoformat(fetch_dt)
    fetch_token = _now_token()

    records: List[Dict[str, Any]] = []
    books_counter: Counter[str] = Counter()
    markets_counter: Counter[str] = Counter()

    for event in events:
        commence_iso = event.get("commence_time")
        try:
            commence_dt = datetime.fromisoformat(str(commence_iso).replace("Z", "+00:00"))
            commence_str = _isoformat(commence_dt)
        except Exception:
            commence_str = None
        home_raw = event.get("home_team") or ""
        away_raw = event.get("away_team") or ""
        home_norm = normalize_team_display(home_raw)
        away_norm = normalize_team_display(away_raw)
        home_token = team_merge_key(home_norm)
        away_token = team_merge_key(away_norm)

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
                payload = {
                    "fetch_ts": fetch_iso,
                    "event_id": event.get("id"),
                    "event_start": commence_str,
                    "book": book_key,
                    "book_title": book_title,
                    "market": market_key,
                    "market_payload": _build_market_payload(market),
                    "home_raw": home_raw,
                    "away_raw": away_raw,
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "home_token": home_token,
                    "away_token": away_token,
                    "league": "NFL",
                    "source": "the-odds-api",
                }
                records.append(payload)

    if not records:
        return {"records": [], "fetch_ts": fetch_iso, "path": None, "counts": {}}

    raw_path = RAW_DIR / f"{fetch_token}.jsonl"
    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
    os.replace(tmp_path, raw_path)

    counts = {
        "books": dict(books_counter),
        "markets": dict(markets_counter),
    }
    return {"records": records, "fetch_ts": fetch_iso, "path": raw_path, "counts": counts}


__all__ = ["ingest_nfl_odds_raw"]

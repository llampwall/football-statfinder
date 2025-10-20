"""CFB odds raw ingestion staging utilities (read-only promotion phase).

Spec reference: /context/global_week_and_provider_decoupling.md

WHY THIS EXISTS:
    Task 2 introduces an append-only staging layer that captures every odds
    provider event before pinning or promotion. Keeping the raw snapshot
    allows us to debug upstream issues without touching the Week/Game views.

Key invariants:
    * All timestamps emitted in the raw artifacts are UTC ISO8601 (Z suffix).
    * No filtering by kickoff window -- every provider record is preserved.
    * Team normalization relies on src/common/team_names_cfb.py utilities.
    * Raw files are written atomically (tmp write -> rename).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.common.io_utils import ensure_out_dir, read_env
from src.common.team_names_cfb import normalize_team_name_cfb_odds, team_merge_key_cfb
from src.fetch_week_odds_cfb import (
    extract_market,
    fetch_theoddsapi_events,
    float_or_none,
    int_or_none,
    parse_utc,
)

OUT_ROOT = ensure_out_dir()
RAW_DIR = OUT_ROOT / "staging" / "odds_raw" / "cfb"

MARKET_KEYS = ("spreads", "totals", "h2h")


def _is_enabled(flag: Optional[str]) -> bool:
    """Return True when the provided flag represents an enabled state."""
    if flag is None:
        return True
    return flag.strip().lower() not in {"0", "false", "off", "disable", "disabled"}


def _sanitize_outcomes(
    outcomes: Iterable[dict],
) -> List[Dict[str, Any]]:
    """Normalize Odds API outcomes down to token/price/point tuples."""
    sanitized: List[Dict[str, Any]] = []
    for outcome in outcomes or []:
        token = team_merge_key_cfb(normalize_team_name_cfb_odds(outcome.get("name")))
        sanitized.append(
            {
                "name": outcome.get("name"),
                "token": token,
                "price": int_or_none(outcome.get("price")),
                "point": float_or_none(outcome.get("point")),
            }
        )
    return sanitized


def _build_market_payload(market: dict) -> Dict[str, Any]:
    """Serialize market data with normalized outcomes."""
    if not market:
        return {"key": None, "last_update": None, "outcomes": []}
    last_update_dt = parse_utc(market.get("last_update"))
    last_update_iso = (
        last_update_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(last_update_dt, datetime)
        else None
    )
    return {
        "key": market.get("key"),
        "last_update": last_update_iso,
        "outcomes": _sanitize_outcomes(market.get("outcomes") or []),
    }


def ingest_cfb_odds_raw() -> Dict[str, Any]:
    """Fetch, normalize, and stage raw CFB odds provider records.

    Returns:
        Dict containing:
            records: list of raw staging dicts (per bookmaker+market)
            fetch_ts: UTC ISO string for the fetch run (or None when skipped)
            path: Path to the raw staging file (or None when skipped)
            counts: Counter with `books` / `markets`

    Notes:
        - Respects the `ODDS_STAGING_ENABLE` flag (default enabled).
        - Requires `THE_ODDS_API_KEY`; when absent, the function is a no-op.
        - Raw staging files are named `<YYYYMMDDTHHMMSSZ>.jsonl` and written
          atomically under `staging/odds_raw/cfb/`.
    """
    if not _is_enabled(os.getenv("ODDS_STAGING_ENABLE", "1")):
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    env = read_env(["THE_ODDS_API_KEY"])
    api_key = env.get("THE_ODDS_API_KEY")
    if not api_key:
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    try:
        events = fetch_theoddsapi_events(api_key) or []
    except Exception:
        return {"records": [], "fetch_ts": None, "path": None, "counts": {}}

    fetch_dt = datetime.now(timezone.utc)
    fetch_iso = fetch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_token = fetch_dt.strftime("%Y%m%dT%H%M%SZ")

    records: List[Dict[str, Any]] = []
    book_counter: Counter[str] = Counter()
    market_counter: Counter[str] = Counter()

    for event in events:
        commence_dt = parse_utc(event.get("commence_time"))
        commence_iso = (
            commence_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(commence_dt, datetime)
            else None
        )
        home_raw = event.get("home_team") or ""
        away_raw = event.get("away_team") or ""
        home_norm = normalize_team_name_cfb_odds(home_raw)
        away_norm = normalize_team_name_cfb_odds(away_raw)
        home_token = team_merge_key_cfb(home_norm)
        away_token = team_merge_key_cfb(away_norm)

        for bookmaker in event.get("bookmakers") or []:
            book_key = bookmaker.get("key") or "unknown"
            book_counter[book_key] += 1
            markets = bookmaker.get("markets") or []
            for market_key in MARKET_KEYS:
                market = extract_market(markets, market_key)
                if not market:
                    continue
                market_counter[market_key] += 1
                payload = {
                    "fetch_ts": fetch_iso,
                    "event_id": event.get("id"),
                    "event_start": commence_iso,
                    "book": book_key,
                    "book_title": bookmaker.get("title"),
                    "market": market_key,
                    "market_payload": _build_market_payload(market),
                    "home_raw": home_raw,
                    "away_raw": away_raw,
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "home_token": home_token,
                    "away_token": away_token,
                    "league": "CFB",
                    "source": "the-odds-api",
                }
                records.append(payload)

    if not records:
        return {"records": [], "fetch_ts": fetch_iso, "path": None, "counts": {}}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{fetch_token}.jsonl"
    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
    os.replace(tmp_path, raw_path)

    counts = {
        "books": dict(book_counter),
        "markets": dict(market_counter),
    }
    return {"records": records, "fetch_ts": fetch_iso, "path": raw_path, "counts": counts}


__all__ = ["ingest_cfb_odds_raw"]

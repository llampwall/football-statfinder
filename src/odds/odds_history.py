"""Odds API history helpers for closing spreads."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from src.common.io_utils import ensure_out_dir, getenv
from src.common.team_names import team_merge_key
from src.common.team_names_cfb import team_merge_key_cfb

_THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
_OUT_ROOT = ensure_out_dir()
_PINNED_ROOT = _OUT_ROOT / "staging" / "odds_pinned"

_PINNED_EVENT_CACHE: Dict[Tuple[str, int], Dict[str, str]] = {}


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc)


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_team(league: str, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    func = team_merge_key_cfb if league.lower() == "cfb" else team_merge_key
    return func(name)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def _pinned_event_id(league: str, season: int, game_key: str) -> Optional[str]:
    cache_key = (league, season)
    if cache_key not in _PINNED_EVENT_CACHE:
        path = _PINNED_ROOT / league.lower() / f"{season}.jsonl"
        mapping: Dict[str, str] = {}
        for record in _load_jsonl(path):
            if record.get("market") != "spreads":
                continue
            key = record.get("game_key")
            raw_event = record.get("raw_event") or {}
            event_id = raw_event.get("event_id")
            if isinstance(key, str) and isinstance(event_id, str):
                mapping[key] = event_id
        _PINNED_EVENT_CACHE[cache_key] = mapping
    return _PINNED_EVENT_CACHE[cache_key].get(game_key)


def _extract_event_id(league: str, season: int, game_row: Dict[str, Any]) -> Optional[str]:
    raw_sources = game_row.get("raw_sources")
    if isinstance(raw_sources, dict):
        odds_row = raw_sources.get("odds_row")
        if isinstance(odds_row, dict):
            raw_event = odds_row.get("raw_event") or {}
            event_id = raw_event.get("event_id")
            if isinstance(event_id, str):
                return event_id
    game_key = game_row.get("game_key")
    if isinstance(game_key, str):
        return _pinned_event_id(league, season, game_key)
    return None


def _outcomes_to_payload(league: str, outcomes: list[dict], home_name: Optional[str], away_name: Optional[str]) -> Optional[Dict[str, Any]]:
    home_token = _normalize_team(league, home_name)
    away_token = _normalize_team(league, away_name)
    home_point = None
    away_point = None
    for outcome in outcomes:
        name = outcome.get("name")
        point = outcome.get("point")
        token = _normalize_team(league, name)
        if token == home_token and _is_finite(point):
            home_point = float(point)
        elif token == away_token and _is_finite(point):
            away_point = float(point)
    if home_point is None and away_point is not None:
        home_point = -away_point
    if home_point is None or home_point == 0:
        return None
    favored_team = "HOME" if home_point < 0 else "AWAY"
    spread = abs(home_point)
    return {"favored_team": favored_team, "spread": spread}


def get_closing_spread(league: str, game_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    api_key = getenv("THE_ODDS_API_KEY")
    if not api_key:
        return None
    sport_key = _SPORT_KEYS.get(league.lower())
    if not sport_key:
        return None
    season = game_row.get("season")
    if not isinstance(season, int):
        try:
            season = int(season)
        except Exception:
            season = None
    event_id = _extract_event_id(league, season or 0, game_row)
    if not event_id:
        return None

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american",
    }
    url = f"{_THE_ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds-history"
    try:
        resp = requests.get(url, params=params, timeout=25)
    except requests.RequestException:
        return None
    if resp.status_code == 404:
        return None
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        return None

    payload = resp.json()
    bookmakers = payload.get("bookmakers") if isinstance(payload, dict) else payload
    if not isinstance(bookmakers, list):
        return None

    best: Optional[Dict[str, Any]] = None
    best_ts: Optional[datetime] = None
    home_name = game_row.get("home_team_norm") or game_row.get("home_team_raw")
    away_name = game_row.get("away_team_norm") or game_row.get("away_team_raw")

    for bookmaker in bookmakers:
        book_title = bookmaker.get("title") or bookmaker.get("key")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "spreads":
                continue
            odds_history = market.get("odds") or []
            if isinstance(odds_history, list) and odds_history:
                odds_history = sorted(
                    odds_history,
                    key=lambda entry: _parse_ts(entry.get("timestamp") or entry.get("last_update"))
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
                outcomes = odds_history[0].get("outcomes") or []
                ts_value = odds_history[0].get("timestamp") or odds_history[0].get("last_update")
            else:
                outcomes = market.get("outcomes") or []
                ts_value = market.get("last_update")
            line_payload = _outcomes_to_payload(league, outcomes, home_name, away_name)
            if not line_payload:
                continue
            ts_dt = _parse_ts(ts_value) or datetime.min.replace(tzinfo=timezone.utc)
            if best is None or ts_dt > (best_ts or datetime.min.replace(tzinfo=timezone.utc)):
                best = {
                    "favored_team": line_payload["favored_team"],
                    "spread": line_payload["spread"],
                    "book": book_title,
                    "fetched_ts": ts_value,
                }
                best_ts = ts_dt
    return best

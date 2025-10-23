"""Closing spread resolution and ATS helpers.

Provides utilities to resolve the best-available closing spread for a game
from pinned odds, local snapshots, or The Odds API history fallback, and
compute against-the-spread (ATS) outcomes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io_utils import ensure_out_dir
from src.common.team_names import team_merge_key
from src.common.team_names_cfb import team_merge_key_cfb
from .odds_history import get_closing_spread

_OUT_ROOT = ensure_out_dir()
_PINNED_ROOT = _OUT_ROOT / "staging" / "odds_pinned"
_SNAPSHOT_PATTERNS = {
    "nfl": lambda season, week: _OUT_ROOT / f"{season}_week{week}" / f"odds_{season}_wk{week}.jsonl",
    "cfb": lambda season, week: _OUT_ROOT / "cfb" / f"{season}_week{week}" / f"odds_{season}_wk{week}.jsonl",
}
_PINNED_CACHE: Dict[Tuple[str, int], Dict[str, dict]] = {}
_SNAPSHOT_CACHE: Dict[Tuple[str, int, int], List[dict]] = {}


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


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
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


def _load_pinned_index(league: str, season: int) -> Dict[str, dict]:
    cache_key = (league, season)
    if cache_key in _PINNED_CACHE:
        return _PINNED_CACHE[cache_key]
    path = _PINNED_ROOT / league.lower() / f"{season}.jsonl"
    index: Dict[str, dict] = {}
    for record in _load_jsonl(path):
        if record.get("market") != "spreads":
            continue
        if record.get("season") != season:
            continue
        game_key = record.get("game_key")
        if not isinstance(game_key, str):
            continue
        existing = index.get(game_key)
        candidate_ts = _parse_ts(record.get("fetch_ts")) or datetime.min.replace(tzinfo=timezone.utc)
        existing_ts = _parse_ts(existing.get("fetch_ts")) if existing else None
        if existing is None or (existing_ts is not None and candidate_ts > existing_ts):
            index[game_key] = record
    _PINNED_CACHE[cache_key] = index
    return index


def _load_snapshots(league: str, season: int, week: int) -> List[dict]:
    cache_key = (league, season, week)
    if cache_key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[cache_key]
    factory = _SNAPSHOT_PATTERNS.get(league.lower())
    if not factory:
        _SNAPSHOT_CACHE[cache_key] = []
        return []
    path = factory(season, week)
    rows = _load_jsonl(path)
    _SNAPSHOT_CACHE[cache_key] = rows
    return rows


def _record_to_payload(
    record: dict,
    league: str,
    home_norm: Optional[str],
    away_norm: Optional[str],
) -> Optional[Dict[str, Any]]:
    line = record.get("line") or record
    spread_home_relative = line.get("spread_home_relative")
    if not _is_finite(spread_home_relative):
        raw = line.get("raw_outcomes") or []
        home_key = _normalize_team(league, home_norm)
        away_key = _normalize_team(league, away_norm)
        home_point = None
        away_point = None
        for outcome in raw:
            name = outcome.get("name")
            point = outcome.get("point")
            token = _normalize_team(league, name)
            if token == home_key and _is_finite(point):
                home_point = float(point)
            elif token == away_key and _is_finite(point):
                away_point = float(point)
        if home_point is not None:
            spread_home_relative = home_point
        elif away_point is not None:
            spread_home_relative = -away_point
    favored_side = (line.get("favored_side") or "").upper()
    if not favored_side and _is_finite(spread_home_relative):
        value = float(spread_home_relative)
        if value < 0:
            favored_side = "HOME"
        elif value > 0:
            favored_side = "AWAY"
    if not favored_side:
        return None
    if not _is_finite(spread_home_relative):
        return None
    spread_value = abs(float(spread_home_relative))
    favored_team = "HOME" if favored_side == "HOME" else "AWAY"
    return {
        "favored_team": favored_team,
        "spread": spread_value,
        "book": record.get("book") or record.get("book_label"),
        "fetched_ts": record.get("fetch_ts") or record.get("snapshot_at"),
    }


def resolve_closing_spread(
    league: str,
    season: int,
    game_row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve closing spread for a game using pinned, snapshot, then history."""
    game_key = game_row.get("game_key")
    if not isinstance(game_key, str):
        return None
    week = game_row.get("week")
    try:
        week_int = int(week)
    except Exception:
        week_int = None
    kickoff_iso = game_row.get("kickoff_iso_utc") or game_row.get("kickoff_iso")
    kickoff_dt = _parse_ts(kickoff_iso) if kickoff_iso else None
    home_norm = game_row.get("home_team_norm") or game_row.get("home_team_raw")
    away_norm = game_row.get("away_team_norm") or game_row.get("away_team_raw")

    pinned_index = _load_pinned_index(league, season)
    pinned_record = pinned_index.get(game_key)
    if pinned_record:
        payload = _record_to_payload(pinned_record, league, home_norm, away_norm)
        if payload:
            payload["source"] = "pinned"
            return payload

    if week_int is not None:
        snapshots = _load_snapshots(league, season, week_int)
        candidates: List[Tuple[datetime, dict]] = []
        row_home_token = _normalize_team(league, home_norm)
        row_away_token = _normalize_team(league, away_norm)
        for snap in snapshots:
            snap_home = _normalize_team(league, snap.get("home_team_norm") or snap.get("home_team_raw"))
            snap_away = _normalize_team(league, snap.get("away_team_norm") or snap.get("away_team_raw"))
            if row_home_token and snap_home and snap_home != row_home_token:
                continue
            if row_away_token and snap_away and snap_away != row_away_token:
                continue
            ts = _parse_ts(snap.get("snapshot_at")) or datetime.min.replace(tzinfo=timezone.utc)
            if kickoff_dt and ts > kickoff_dt:
                continue
            if snap.get("spread_home_relative") is None:
                continue
            candidates.append((ts, snap))
        if not candidates:
            for snap in snapshots:
                if snap.get("spread_home_relative") is None:
                    continue
                ts = _parse_ts(snap.get("snapshot_at")) or datetime.min.replace(tzinfo=timezone.utc)
                candidates.append((ts, snap))
                break
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            payload = _record_to_payload(candidates[0][1], league, home_norm, away_norm)
            if payload:
                payload["source"] = "snapshot"
                return payload

    history_payload = get_closing_spread(league, game_row)
    if history_payload:
        history_payload["source"] = "history"
        return history_payload
    return None


def compute_ats(
    home_score: int,
    away_score: int,
    favored_team: str,
    spread: float,
) -> Optional[Dict[str, Any]]:
    try:
        home_score = int(home_score)
        away_score = int(away_score)
        spread = float(abs(spread))
    except (TypeError, ValueError):
        return None
    favored = (favored_team or "").upper()
    if favored not in {"HOME", "AWAY"}:
        return None
    home_line = -spread if favored == "HOME" else spread
    margin = home_score - away_score
    home_vs_line = margin + home_line
    away_vs_line = -home_vs_line

    def _label(value: float) -> str:
        if value > 0:
            return "W"
        if value < 0:
            return "L"
        return "P"

    return {
        "home_ats": _label(home_vs_line),
        "away_ats": _label(away_vs_line),
        "to_margin_home": round(home_vs_line, 2),
        "to_margin_away": round(away_vs_line, 2),
    }


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False

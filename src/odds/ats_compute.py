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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.common.io_utils import ensure_out_dir
from src.common.team_names import team_merge_key
from src.common.team_names_cfb import team_merge_key_cfb
from src.odds.odds_history import get_closing_spread   # safe to import; it can no-op when offline

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
    """
    Build and cache an index of pinned spread records for a league/season:
      { game_key: record }
    """
    cache_key = (league.lower(), int(season))
    if cache_key in _PINNED_CACHE:
        return _PINNED_CACHE[cache_key]

    idx: Dict[str, dict] = {}
    path = _PINNED_ROOT / league.lower() / f"{season}.jsonl"
    if path.exists():
        for rec in _load_jsonl(path):
            # only spreads, and only rows that have a game_key
            if rec.get("market") == "spreads" and isinstance(rec.get("game_key"), str):
                # use the existing helper to handle “newer fetch_ts wins”
                load_pinned_index(idx, league, season, rec["game_key"], rec)

    _PINNED_CACHE[cache_key] = idx
    return idx


# def load_pinned_index(idx: dict,
#                       league: str,
#                       season: int,
#                       game_key: str,
#                       rec: dict) -> dict:
#     """
#     Insert/replace the pinned odds record for a game_key.

#     Replace if:
#       - there is no existing record, OR
#       - existing record has no fetch_ts, OR
#       - new rec.fetch_ts is newer than existing fetch_ts
#     """
#     g = idx.setdefault((league, season), {})
#     cur = g.get(game_key)
#     def _to_dt(x):
#         if not x: return None
#         if isinstance(x, (int, float)):
#             return datetime.fromtimestamp(float(x), tz=timezone.utc)
#         return datetime.fromisoformat(str(x).replace("Z", "+00:00"))

#     new_when = _to_dt(rec.get("fetch_ts"))
#     cur_when = _to_dt(cur.get("fetch_ts") if isinstance(cur, dict) else None)

#     if (cur is None) or (cur_when is None) or (new_when and cur_when and new_when > cur_when):
#         g[game_key] = rec
#     return idx


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



def pick_latest_before(records: Iterable[Dict[str, Any]],
                       cutoff: datetime) -> Optional[Dict[str, Any]]:
    """
    Return the record with the greatest fetch_ts <= cutoff.
    Records missing/invalid fetch_ts are ignored.
    """
    best: Tuple[datetime, Dict[str, Any]] | None = None
    for r in records or []:
        ts = r.get("fetch_ts")
        if not ts:
            continue
        try:
            # allow both iso string and epoch float
            if isinstance(ts, (int, float)):
                when = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            else:
                when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if when <= cutoff and (best is None or when > best[0]):
            best = (when, r)
    return best[1] if best else None


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


def resolve_closing_spread(league: str, season: int, game_row: dict) -> Optional[dict]:
    """Resolve closing spread using odds history, pinned data, then local snapshots."""
    history = get_closing_spread(league, game_row)
    if history and _is_finite(history.get("spread")) and history.get("favored_team"):
        payload = {
            "favored_team": history.get("favored_team"),
            "spread": float(abs(history["spread"])),
            "book": history.get("book"),
            "fetched_ts": history.get("fetched_ts"),
            "source": "history",
        }
        return payload
    
    # 2) pinned (from out/staging/odds_pinned/<league>/<season>.jsonl)
    pinned_index = _load_pinned_index(league, season)
    pinned_record = pinned_index.get(game_row.get("game_key"))
    if pinned_record:
        home_norm = game_row.get("home_team_norm") or game_row.get("home_team_raw")
        away_norm = game_row.get("away_team_norm") or game_row.get("away_team_raw")
        payload = _record_to_payload(pinned_record, league, home_norm, away_norm)
        if payload:
            payload["source"] = "pinned"
            return payload

    # 3) latest pre-kick local snapshot (odds_{season}_wk{week}.jsonl)
    try:
        week = int(game_row.get("week"))
    except Exception:
        week = None
    kickoff_ts = _parse_ts(game_row.get("kickoff_iso_utc") or game_row.get("kickoff_iso"))
    cutoff = kickoff_ts or datetime.max.replace(tzinfo=timezone.utc)
    if week is not None:
        home_norm = game_row.get("home_team_norm") or game_row.get("home_team_raw")
        away_norm = game_row.get("away_team_norm") or game_row.get("away_team_raw")
        snapshots = _load_snapshots(league, season, week)
        snap_record = pick_latest_before(snapshots, cutoff)
        if snap_record:
            payload = _record_to_payload(snap_record, league, home_norm, away_norm)
            if payload:
                payload["source"] = "snapshot"
                return payload

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

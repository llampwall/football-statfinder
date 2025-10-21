"""NFL odds pinning utilities (season-wide schedule index).

Purpose & scope:
    Match normalized NFL odds provider snapshots to schedule master rows,
    capturing the resolved `game_key`, week, and kickoff details while
    preserving unmatched entries for diagnostics.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B2, F, H, I)

Invariants:
    * Schedule master (or existing game sidecars) is the source of truth.
    * Time comparisons operate in UTC and honor configurable tolerances.
    * Pinned staging writes are append-only; unmatched rows are quarantined
      to timestamped JSONL files for debugging.

Side effects:
    * Appends to `staging/odds_pinned/nfl/<season>.jsonl`.
    * Writes unmatched rows (when present) to
      `staging/odds_unmatched/nfl/<YYYYMMDDTHHMMSSZ>.jsonl` atomically.

Do not:
    * Modify existing pinned records or delete unmatched diagnostics.
    * Extend tolerances without specification updates.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from src.common.io_utils import ensure_out_dir
from src.common.team_names import normalize_team_display, team_merge_key

OUT_ROOT = ensure_out_dir()
PINNED_DIR = OUT_ROOT / "staging" / "odds_pinned" / "nfl"
UNMATCHED_DIR = OUT_ROOT / "staging" / "odds_unmatched" / "nfl"
MASTER_PATH = OUT_ROOT / "master" / "nfl_schedule_master.csv"

PINNED_DIR.mkdir(parents=True, exist_ok=True)
UNMATCHED_DIR.mkdir(parents=True, exist_ok=True)


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def _slug(name: str) -> str:
    cleaned = name.lower()
    tokens = []
    for ch in cleaned:
        if ch.isalnum():
            tokens.append(ch)
        elif tokens and tokens[-1] != "_":
            tokens.append("_")
    return "".join(tokens).strip("_")


@dataclass(frozen=True)
class ScheduleGame:
    season: int
    week: int
    game_key: str
    kickoff: datetime
    home_norm: str
    away_norm: str
    home_token: str
    away_token: str
    neutral_site: Optional[bool]


@dataclass(frozen=True)
class SeasonScheduleIndex:
    season: int
    games_by_pair: Dict[Tuple[str, str], List[ScheduleGame]]

    def lookup(self, home_token: str, away_token: str) -> List[ScheduleGame]:
        return self.games_by_pair.get((home_token, away_token), [])


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1", "neutral", "neutral_site"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "home"}:
            return False
    return None


def _load_schedule_from_master() -> Tuple[Optional[int], List[ScheduleGame]]:
    if not MASTER_PATH.exists():
        return None, []
    try:
        df = pd.read_csv(MASTER_PATH)
    except Exception:
        return None, []
    if df.empty or "season" not in df.columns:
        return None, []

    df = df.dropna(subset=["season", "kickoff_iso_utc", "home_team_norm", "away_team_norm"])
    if df.empty:
        return None, []

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    if df.empty:
        return None, []

    current_season = int(df["season"].astype(int).max())
    df = df[df["season"] == current_season]
    games: List[ScheduleGame] = []
    for row in df.itertuples(index=False):
        kickoff_dt = _parse_utc(getattr(row, "kickoff_iso_utc", None))
        if not isinstance(kickoff_dt, datetime):
            continue
        home_norm = normalize_team_display(getattr(row, "home_team_norm", None))
        away_norm = normalize_team_display(getattr(row, "away_team_norm", None))
        if not home_norm or not away_norm:
            continue
        season_val = int(getattr(row, "season", current_season) or current_season)
        week_val = int(getattr(row, "week", 0) or 0)
        home_token = team_merge_key(home_norm)
        away_token = team_merge_key(away_norm)
        key_ts = kickoff_dt.strftime("%Y%m%d_%H%M")
        game_key = f"{key_ts}_{_slug(away_norm)}_{_slug(home_norm)}"
        neutral = _coerce_bool(getattr(row, "neutral_site", None))
        games.append(
            ScheduleGame(
                season=season_val,
                week=week_val,
                game_key=game_key,
                kickoff=kickoff_dt,
                home_norm=home_norm,
                away_norm=away_norm,
                home_token=home_token,
                away_token=away_token,
                neutral_site=neutral,
            )
        )
    return current_season, games


def _load_schedule_from_weeks() -> Tuple[Optional[int], List[ScheduleGame]]:
    nfl_dir = OUT_ROOT / "nfl"
    if not nfl_dir.exists():
        return None, []
    week_pattern = re.compile(r"^(?P<season>\d{4})_week(?P<week>\d+)$")
    entries: List[Tuple[int, int, Path]] = []
    for child in nfl_dir.iterdir():
        if not child.is_dir():
            continue
        match = week_pattern.match(child.name)
        if not match:
            continue
        season_val = int(match.group("season"))
        week_val = int(match.group("week"))
        jsonl = child / f"games_week_{season_val}_{week_val}.jsonl"
        if jsonl.exists():
            entries.append((season_val, week_val, jsonl))
    if not entries:
        return None, []

    target_season = max(e[0] for e in entries)
    games: Dict[str, ScheduleGame] = {}
    for season_val, week_val, json_path in sorted(entries):
        if season_val != target_season:
            continue
        try:
            lines = json_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            game_key = payload.get("game_key")
            kickoff_iso = payload.get("kickoff_iso_utc") or payload.get("kickoff_iso")
            kickoff_dt = _parse_utc(kickoff_iso)
            home_norm = normalize_team_display(payload.get("home_team_norm"))
            away_norm = normalize_team_display(payload.get("away_team_norm"))
            if not isinstance(game_key, str) or not kickoff_dt or not home_norm or not away_norm:
                continue
            neutral = None
            raw_sources = payload.get("raw_sources")
            if isinstance(raw_sources, dict):
                schedule_row = raw_sources.get("schedule_row")
                if isinstance(schedule_row, dict):
                    neutral = _coerce_bool(schedule_row.get("neutral_site"))
            games[game_key] = ScheduleGame(
                season=season_val,
                week=week_val,
                game_key=game_key,
                kickoff=kickoff_dt,
                home_norm=home_norm,
                away_norm=away_norm,
                home_token=team_merge_key(home_norm),
                away_token=team_merge_key(away_norm),
                neutral_site=neutral,
            )
    return target_season, list(games.values())


def _build_schedule_index() -> Optional[SeasonScheduleIndex]:
    season, games = _load_schedule_from_master()
    if not games:
        season, games = _load_schedule_from_weeks()
    if not games or season is None:
        return None
    games_by_pair: Dict[Tuple[str, str], List[ScheduleGame]] = {}
    for game in games:
        games_by_pair.setdefault((game.home_token, game.away_token), []).append(game)
    for pair_games in games_by_pair.values():
        pair_games.sort(key=lambda item: item.kickoff)
    return SeasonScheduleIndex(season=season, games_by_pair=games_by_pair)


def _within_day_window(event_dt: datetime, kickoff: datetime, day_window: int) -> bool:
    delta_days = abs((kickoff.date() - event_dt.date()).days)
    return delta_days <= day_window


def _collect_candidates(
    games: Iterable[ScheduleGame],
    event_dt: datetime,
    day_window: int,
    max_delta_seconds: float,
    swapped_flag: bool,
) -> List[Tuple[ScheduleGame, bool, float]]:
    collected: List[Tuple[ScheduleGame, bool, float]] = []
    for game in games:
        delta_seconds = abs((game.kickoff - event_dt).total_seconds())
        if max_delta_seconds >= 0 and delta_seconds > max_delta_seconds:
            continue
        if not _within_day_window(event_dt, game.kickoff, day_window):
            continue
        collected.append((game, swapped_flag, delta_seconds))
    return collected


def _sort_key(candidate: Tuple[ScheduleGame, bool, float]) -> Tuple[float, int, str]:
    game, _, delta_seconds = candidate
    neutral_rank = 0 if game.neutral_site else 1
    return (delta_seconds, neutral_rank, game.game_key)


def _extract_spread_line(
    outcomes: List[Dict[str, Any]],
    home_token: str,
    away_token: str,
    swapped: bool,
) -> Dict[str, Any]:
    expected_home = away_token if swapped else home_token
    expected_away = home_token if swapped else away_token
    home_outcome = next((o for o in outcomes if o.get("token") == expected_home), None)
    away_outcome = next((o for o in outcomes if o.get("token") == expected_away), None)
    home_point = home_outcome.get("point") if home_outcome else None
    away_point = away_outcome.get("point") if away_outcome else None
    if home_point is None and away_point is not None:
        home_point = -away_point

    favored_side = None
    favored_spread = None
    if isinstance(home_point, (int, float)):
        if home_point < 0:
            favored_side = "HOME"
            favored_spread = home_point
        elif home_point > 0:
            favored_side = "AWAY"
            favored_spread = -abs(home_point)
        else:
            favored_side = "PICK"
            favored_spread = 0.0
    return {
        "spread_home_relative": home_point,
        "favored_side": favored_side,
        "spread_favored_team": favored_spread,
        "home_price": home_outcome.get("price") if home_outcome else None,
        "away_price": away_outcome.get("price") if away_outcome else None,
        "raw_outcomes": outcomes,
    }


def _extract_totals_line(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_points = None
    over_price = None
    under_price = None
    for outcome in outcomes:
        name = (outcome.get("name") or "").lower()
        if name.startswith("over"):
            total_points = outcome.get("point")
            over_price = outcome.get("price")
        elif name.startswith("under"):
            under_price = outcome.get("price")
            if total_points is None:
                total_points = outcome.get("point")
    if total_points is None and outcomes:
        total_points = outcomes[0].get("point")
    return {
        "total_points": total_points,
        "over_price": over_price,
        "under_price": under_price,
        "raw_outcomes": outcomes,
    }


def _extract_moneyline(outcomes: List[Dict[str, Any]], home_token: str, away_token: str, swapped: bool) -> Dict[str, Any]:
    expected_home = away_token if swapped else home_token
    expected_away = home_token if swapped else away_token
    home_price = None
    away_price = None
    for outcome in outcomes:
        token = outcome.get("token")
        if token == expected_home:
            home_price = outcome.get("price")
        elif token == expected_away:
            away_price = outcome.get("price")
    return {
        "moneyline_home": home_price,
        "moneyline_away": away_price,
        "raw_outcomes": outcomes,
    }


def _build_line_payload(record: Mapping[str, Any], schedule_game: ScheduleGame, swapped: bool) -> Dict[str, Any]:
    market = record.get("market")
    payload = record.get("market_payload") or {}
    outcomes = payload.get("outcomes") or []
    if market == "spreads":
        return _extract_spread_line(outcomes, schedule_game.home_token, schedule_game.away_token, swapped)
    if market == "totals":
        return _extract_totals_line(outcomes)
    if market == "h2h":
        return _extract_moneyline(outcomes, schedule_game.home_token, schedule_game.away_token, swapped)
    return {"raw_outcomes": outcomes}


def _ensure_jsonl_append(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_unmatched(rows: Iterable[dict]) -> Optional[Path]:
    rows = list(rows)
    if not rows:
        return None
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = UNMATCHED_DIR / f"{token}.jsonl"
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    os.replace(tmp_path, target)
    return target


def pin_nfl_odds(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    day_window: int,
    max_delta_hours: float,
    role_swap_tolerance: bool,
) -> Dict[str, Any]:
    """Pin raw NFL odds records onto schedule games."""
    counts = {
        "raw": len(raw_records),
        "pinned": 0,
        "unmatched": 0,
        "books": {},
        "markets": {},
        "candidate_sets_zero": 0,
        "candidate_sets_multi": 0,
    }
    if not raw_records:
        return {
            "pinned_records": [],
            "unmatched_records": [],
            "pinned_path": None,
            "unmatched_path": None,
            "counts": counts,
            "examples_unmatched": [],
        }

    schedule_index = _build_schedule_index()
    if schedule_index is None:
        counts["unmatched"] = len(raw_records)
        return {
            "pinned_records": [],
            "unmatched_records": list(raw_records),
            "pinned_path": None,
            "unmatched_path": None,
            "counts": counts,
            "examples_unmatched": [],
        }

    max_delta_seconds = -1.0 if max_delta_hours < 0 else max_delta_hours * 3600.0

    pinned: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    books_counter: Counter[str] = Counter()
    markets_counter: Counter[str] = Counter()

    for record in raw_records:
        event_dt = _parse_utc(record.get("event_start"))
        if not isinstance(event_dt, datetime):
            entry = dict(record)
            entry["why"] = "invalid_event_time"
            unmatched.append(entry)
            continue
        home_token = record.get("home_token") or ""
        away_token = record.get("away_token") or ""

        direct_candidates = _collect_candidates(
            schedule_index.lookup(home_token, away_token),
            event_dt,
            day_window,
            max_delta_seconds,
            False,
        )
        candidate_pool = direct_candidates
        if not candidate_pool and role_swap_tolerance:
            candidate_pool = _collect_candidates(
                schedule_index.lookup(away_token, home_token),
                event_dt,
                day_window,
                max_delta_seconds,
                True,
            )
        candidate_count = len(candidate_pool)
        if candidate_count == 0:
            counts["candidate_sets_zero"] += 1
            entry = dict(record)
            entry["why"] = "no_candidate"
            unmatched.append(entry)
            continue

        candidate_pool.sort(key=_sort_key)
        best_candidate, swapped, _ = candidate_pool[0]
        ambiguous = candidate_count > 1 and _sort_key(candidate_pool[0]) == _sort_key(candidate_pool[1])
        if candidate_count > 1:
            counts["candidate_sets_multi"] += 1
        if ambiguous:
            entry = dict(record)
            entry["why"] = "ambiguous"
            unmatched.append(entry)
            continue

        line_payload = _build_line_payload(record, best_candidate, swapped)
        pinned.append(
            {
                "fetch_ts": record.get("fetch_ts"),
                "source": record.get("source"),
                "season": best_candidate.season,
                "week": best_candidate.week,
                "game_key": best_candidate.game_key,
                "market": record.get("market"),
                "book": record.get("book"),
                "line": line_payload,
                "home_norm": best_candidate.home_norm,
                "away_norm": best_candidate.away_norm,
                "kickoff_utc": best_candidate.kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "role_swapped": swapped,
                "raw_event": {
                    "event_id": record.get("event_id"),
                    "event_start": record.get("event_start"),
                },
            }
        )
        books_counter[record.get("book") or "unknown"] += 1
        markets_counter[record.get("market") or "unknown"] += 1

    pinned_path: Optional[Path] = None
    if pinned:
        pinned_path = PINNED_DIR / f"{schedule_index.season}.jsonl"
        _ensure_jsonl_append(pinned_path, pinned)

    unmatched_path = _write_unmatched(unmatched)

    examples = []
    for sample in unmatched[:3]:
        home = sample.get("home_norm") or sample.get("home_raw")
        away = sample.get("away_norm") or sample.get("away_raw")
        examples.append(f"{home} vs {away} ({sample.get('why')})")

    counts.update(
        {
            "pinned": len(pinned),
            "unmatched": len(unmatched),
            "books": dict(books_counter),
            "markets": dict(markets_counter),
        }
    )
    return {
        "pinned_records": pinned,
        "unmatched_records": unmatched,
        "pinned_path": pinned_path,
        "unmatched_path": unmatched_path,
        "counts": counts,
        "examples_unmatched": examples,
    }


__all__ = ["pin_nfl_odds"]

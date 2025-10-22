"""CFB odds pinning utilities (schedule-aligned staging records).

Purpose & scope:
    Pin normalized College Football odds snapshots onto schedule master rows so
    that downstream builders can promote lines week-by-week without losing any
    future inventory.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B1–B3, E, F, H, I)

Invariants:
    * Schedule master (or week sidecars) is the single source of truth.
    * All time comparisons run in UTC and honor configured day/hour tolerances.
    * Append-only staging writes; unmatched snapshots are quarantined with why.

Side effects:
    * Appends to staging/odds_pinned/cfb/<season>.jsonl (UTF-8, ensure_ascii=False).
    * Writes unmatched payloads to staging/odds_unmatched/cfb/<timestamp>.jsonl
      via atomic tmp-write -> rename.

Do not:
    * Do not mutate historical pinned rows or drop unmatched diagnostics.
    * Do not widen search tolerances without spec approval.

Log contract:
    * No direct logging; callers consume return payload counts for summaries.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.common.io_utils import ensure_out_dir
from src.common.team_names_cfb import normalize_team_name_cfb_odds, team_merge_key_cfb
from src.fetch_week_odds_cfb import parse_utc

OUT_ROOT = ensure_out_dir()
PINNED_DIR = OUT_ROOT / "staging" / "odds_pinned" / "cfb"
UNMATCHED_DIR = OUT_ROOT / "staging" / "odds_unmatched" / "cfb"
MASTER_PATH = OUT_ROOT / "master" / "cfb_schedule_master.csv"


def _slug(norm_name: str) -> str:
    """Convert display names to snake-case slugs (lowercase, underscores)."""
    cleaned = (norm_name or "").lower().replace("&", "and")
    slug = ""
    for char in cleaned:
        if char.isalnum():
            slug += char
        elif slug and slug[-1] != "_":
            slug += "_"
    return slug.strip("_")


@dataclass(frozen=True)
class ScheduleGame:
    """Normalized schedule row used for pinning."""

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
    """Season-wide lookup for schedule games keyed by normalized tokens."""

    season: int
    games_by_pair: Dict[Tuple[str, str], List[ScheduleGame]]

    def lookup(self, home_token: str, away_token: str) -> List[ScheduleGame]:
        """Return schedule games for the provided home/away token pair."""
        return self.games_by_pair.get((home_token, away_token), [])


def _coerce_neutral_flag(raw_value: Any) -> Optional[bool]:
    """Best-effort conversion for neutral-site indicators.

    Args:
        raw_value: Value sourced from schedule master or sidecar payloads.

    Returns:
        Optional[bool]: True when explicitly neutral, False when explicitly home,
        otherwise None when indeterminate.
    """
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        if isinstance(raw_value, float):
            if math.isnan(raw_value):
                return None
        return bool(int(raw_value))
    if isinstance(raw_value, str):
        cleaned = raw_value.strip().lower()
        if cleaned in {"neutral", "neutral_site", "true", "t", "1", "y", "yes"}:
            return True
        if cleaned in {"false", "f", "home", "0", "n", "no"}:
            return False
    return None


def _load_schedule_games_from_master() -> Tuple[Optional[int], List[ScheduleGame]]:
    """Load current-season schedule games from the master CSV.

    Returns:
        Tuple containing the detected season (or None) and schedule games.
    """
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
    if df.empty:
        return None, []

    games: List[ScheduleGame] = []
    for row in df.itertuples(index=False):
        kickoff_iso = getattr(row, "kickoff_iso_utc", None)
        kickoff_dt = parse_utc(kickoff_iso)
        if not isinstance(kickoff_dt, datetime):
            continue
        kickoff_dt = kickoff_dt.astimezone(timezone.utc)
        home_norm = normalize_team_name_cfb_odds(getattr(row, "home_team_norm", None))
        away_norm = normalize_team_name_cfb_odds(getattr(row, "away_team_norm", None))
        if not home_norm or not away_norm:
            continue
        season_val = int(getattr(row, "season", current_season) or current_season)
        week_val = int(getattr(row, "week", 0) or 0)
        away_slug = _slug(away_norm)
        home_slug = _slug(home_norm)
        key_ts = kickoff_dt.strftime("%Y%m%d_%H%M")
        game_key = f"{key_ts}_{away_slug}_{home_slug}"
        neutral = _coerce_neutral_flag(getattr(row, "neutral_site", None))
        games.append(
            ScheduleGame(
                season=season_val,
                week=week_val,
                game_key=game_key,
                kickoff=kickoff_dt,
                home_norm=home_norm,
                away_norm=away_norm,
                home_token=team_merge_key_cfb(home_norm),
                away_token=team_merge_key_cfb(away_norm),
                neutral_site=neutral,
            )
        )
    return current_season, games


def _load_schedule_games_from_week_files() -> Tuple[Optional[int], List[ScheduleGame]]:
    """Fallback loader using built week files (latest season only).

    Returns:
        Tuple containing the detected season (or None) and schedule games.
    """
    cfb_dir = OUT_ROOT / "cfb"
    if not cfb_dir.exists():
        return None, []

    week_entries: List[Tuple[int, int, Path]] = []
    week_pattern = re.compile(r"^(?P<season>\d{4})_week(?P<week>\d+)$")
    for child in cfb_dir.iterdir():
        if not child.is_dir():
            continue
        match = week_pattern.match(child.name)
        if not match:
            continue
        season_val = int(match.group("season"))
        week_val = int(match.group("week"))
        week_file = child / f"games_week_{season_val}_{week_val}.jsonl"
        if week_file.exists():
            week_entries.append((season_val, week_val, week_file))

    if not week_entries:
        return None, []

    target_season = max(entry[0] for entry in week_entries)
    games_by_key: Dict[str, ScheduleGame] = {}
    for _, week_val, path in sorted(entry for entry in week_entries if entry[0] == target_season):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kickoff_iso = payload.get("kickoff_iso_utc")
            kickoff_dt = parse_utc(kickoff_iso)
            if not isinstance(kickoff_dt, datetime):
                continue
            kickoff_dt = kickoff_dt.astimezone(timezone.utc)
            home_norm = normalize_team_name_cfb_odds(payload.get("home_team_norm"))
            away_norm = normalize_team_name_cfb_odds(payload.get("away_team_norm"))
            game_key = payload.get("game_key")
            if not home_norm or not away_norm or not isinstance(game_key, str):
                continue
            neutral = None
            raw_sources = payload.get("raw_sources")
            if isinstance(raw_sources, dict):
                schedule_row = raw_sources.get("schedule_row")
                if isinstance(schedule_row, dict):
                    neutral = _coerce_neutral_flag(schedule_row.get("neutral_site"))
            schedule_game = ScheduleGame(
                season=int(payload.get("season", target_season) or target_season),
                week=int(payload.get("week", week_val) or week_val),
                game_key=game_key,
                kickoff=kickoff_dt,
                home_norm=home_norm,
                away_norm=away_norm,
                home_token=team_merge_key_cfb(home_norm),
                away_token=team_merge_key_cfb(away_norm),
                neutral_site=neutral,
            )
            games_by_key[game_key] = schedule_game
    return target_season, list(games_by_key.values())


def _build_schedule_index() -> Optional[SeasonScheduleIndex]:
    """Build and return a season-wide schedule index for pinning.

    Returns:
        SeasonScheduleIndex for the active season, or None when unavailable.
    """
    season, games = _load_schedule_games_from_master()
    if not games:
        season, games = _load_schedule_games_from_week_files()
    if not games or season is None:
        return None

    games_by_pair: Dict[Tuple[str, str], List[ScheduleGame]] = defaultdict(list)
    for game in games:
        games_by_pair[(game.home_token, game.away_token)].append(game)
    for pair in games_by_pair:
        games_by_pair[pair].sort(key=lambda item: item.kickoff)
    return SeasonScheduleIndex(season=season, games_by_pair=dict(games_by_pair))


def _within_day_window(event_dt: datetime, kickoff: datetime, day_window: int) -> bool:
    """Return True when kickoff falls within the +/-day_window tolerance."""
    delta_days = abs((kickoff.date() - event_dt.date()).days)
    return delta_days <= day_window


def _match_candidates(
    event_dt: datetime,
    home_token: str,
    away_token: str,
    schedule_index: SeasonScheduleIndex,
    day_window: int,
    max_delta_hours: float,
    allow_swap: bool,
) -> Tuple[Optional[ScheduleGame], bool, List[ScheduleGame], int, bool]:
    """Return the best candidate schedule game and matching metadata.

    Args:
        event_dt: Provider event kickoff in UTC.
        home_token: Expected schedule home token (normalized).
        away_token: Expected schedule away token (normalized).
        schedule_index: Pre-built season index for lookup.
        day_window: +/- day tolerance for candidate filtering.
        max_delta_hours: Maximum kickoff delta tolerance (hours).
        allow_swap: Whether to consider role-swapped candidates.

    Returns:
        Tuple of:
            selected ScheduleGame (or None),
            swapped flag,
            ordered list of candidate schedule games,
            candidate count,
            ambiguous flag (True when multiple equally good matches exist).
    """

    def collect_candidates(
        search_games: Iterable[ScheduleGame],
        swapped_flag: bool,
        *,
        limit_seconds: float,
    ) -> List[Tuple[ScheduleGame, bool, float]]:
        collected: List[Tuple[ScheduleGame, bool, float]] = []
        for schedule_game in search_games:
            delta_seconds = abs((schedule_game.kickoff - event_dt).total_seconds())
            if limit_seconds >= 0 and delta_seconds > limit_seconds:
                continue
            if not _within_day_window(event_dt, schedule_game.kickoff, day_window):
                continue
            collected.append((schedule_game, swapped_flag, delta_seconds))
        return collected

    limit_seconds = float("inf") if max_delta_hours < 0 else max_delta_hours * 3600.0
    direct_candidates = collect_candidates(
        schedule_index.lookup(home_token, away_token),
        False,
        limit_seconds=limit_seconds,
    )
    candidate_pool = direct_candidates
    if not candidate_pool and allow_swap:
        swap_candidates = collect_candidates(
            schedule_index.lookup(away_token, home_token),
            True,
            limit_seconds=limit_seconds,
        )
        candidate_pool = swap_candidates

    candidate_count = len(candidate_pool)
    if candidate_count == 0:
        return None, False, [], 0, False

    def sort_key(item: Tuple[ScheduleGame, bool, float]) -> Tuple[float, int, str]:
        schedule_game, _, delta_seconds = item
        neutral_rank = 0 if schedule_game.neutral_site else 1
        return (delta_seconds, neutral_rank, schedule_game.game_key)

    candidate_pool.sort(key=sort_key)
    best_game, swapped, _ = candidate_pool[0]
    ambiguous = False
    if candidate_count > 1 and sort_key(candidate_pool[0]) == sort_key(candidate_pool[1]):
        ambiguous = True
        return None, False, [candidate for candidate, _, _ in candidate_pool], candidate_count, True

    return best_game, swapped, [candidate for candidate, _, _ in candidate_pool], candidate_count, ambiguous


def _extract_spread_line(
    outcomes: List[Dict[str, Any]],
    home_token: str,
    away_token: str,
    role_swapped: bool,
) -> Dict[str, Any]:
    """Build spread line fields relative to the schedule home team."""
    expected_home = away_token if role_swapped else home_token
    expected_away = home_token if role_swapped else away_token

    home_outcome = next((o for o in outcomes if o["token"] == expected_home), None)
    away_outcome = next((o for o in outcomes if o["token"] == expected_away), None)

    home_point = home_outcome.get("point") if home_outcome else None
    away_point = away_outcome.get("point") if away_outcome else None
    if home_point is None and away_point is not None:
        home_point = -away_point

    favored_side = None
    spread_favored_team = None
    if isinstance(home_point, (int, float)):
        if home_point < 0:
            favored_side = "HOME"
            spread_favored_team = home_point
        elif home_point > 0:
            favored_side = "AWAY"
            spread_favored_team = -abs(home_point)
        else:
            favored_side = "PICK"
            spread_favored_team = 0.0

    return {
        "spread_home_relative": home_point,
        "favored_side": favored_side,
        "spread_favored_team": spread_favored_team,
        "home_price": home_outcome.get("price") if home_outcome else None,
        "away_price": away_outcome.get("price") if away_outcome else None,
        "raw_outcomes": outcomes,
    }


def _extract_totals_line(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return total points information."""
    total = None
    over_price = None
    under_price = None
    for outcome in outcomes:
        name = (outcome.get("name") or "").lower()
        if name.startswith("over"):
            total = outcome.get("point")
            over_price = outcome.get("price")
        elif name.startswith("under"):
            under_price = outcome.get("price")
            if total is None:
                total = outcome.get("point")
    if total is None and outcomes:
        total = outcomes[0].get("point")
    return {
        "total_points": total,
        "over_price": over_price,
        "under_price": under_price,
        "raw_outcomes": outcomes,
    }


def _extract_moneyline(outcomes: List[Dict[str, Any]], home_token: str, away_token: str, swapped: bool) -> Dict[str, Any]:
    """Return moneyline prices aligned to the schedule home/away."""
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


def _build_line_payload(
    record: Dict[str, Any],
    game: ScheduleGame,
    role_swapped: bool,
) -> Dict[str, Any]:
    """Compute normalized line values for the pinned staging record."""
    payload = record.get("market_payload") or {}
    outcomes = payload.get("outcomes") or []
    market = record.get("market")

    if market == "spreads":
        return _extract_spread_line(outcomes, game.home_token, game.away_token, role_swapped)
    if market == "totals":
        return _extract_totals_line(outcomes)
    if market == "h2h":
        return _extract_moneyline(outcomes, game.home_token, game.away_token, role_swapped)
    return {"raw_outcomes": outcomes}


def pin_cfb_odds(
    raw_records: List[Dict[str, Any]],
    *,
    day_window: int,
    max_delta_hours: float,
    role_swap_tolerance: bool,
) -> Dict[str, Any]:
    """Pin raw odds records onto schedule games and write staging artifacts.

    Args:
        raw_records: Output from ``ingest_cfb_odds_raw`` (per market snapshot).
        day_window: +/-day search window for candidate schedule games.
        max_delta_hours: Maximum kickoff delta tolerance (hours).
        role_swap_tolerance: Whether to attempt away/home swaps during matching.

    Returns:
        Dict with fields:
            pinned_records, unmatched_records, pinned_path, unmatched_path,
            counts (raw, pinned, unmatched, books, markets), examples_unmatched.

    Notes:
        - Append-only writes for ``odds_pinned`` (season file).
        - Unmatched entries are written to a new timestamped file each run.
        - When staging is disabled (env flag), this function behaves as a no-op.
    """
    if not raw_records:
        return {
            "pinned_records": [],
            "unmatched_records": [],
            "pinned_path": None,
            "unmatched_path": None,
            "counts": {
                "raw": 0,
                "pinned": 0,
                "unmatched": 0,
                "books": {},
                "markets": {},
            },
            "examples_unmatched": [],
        }

    schedule_index = _build_schedule_index()
    if schedule_index is None:
        return {
            "pinned_records": [],
            "unmatched_records": raw_records,
            "pinned_path": None,
            "unmatched_path": None,
            "counts": {
                "raw": len(raw_records),
                "pinned": 0,
                "unmatched": len(raw_records),
                "books": {},
                "markets": {},
                "candidate_sets_zero": len(raw_records),
                "candidate_sets_multi": 0,
            },
            "examples_unmatched": [],
        }

    pinned: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    books_counter: Counter[str] = Counter()
    markets_counter: Counter[str] = Counter()
    candidate_sets_zero = 0
    candidate_sets_multi = 0

    now_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unmatched_path: Optional[Path] = None

    for record in raw_records:
        event_start_iso = record.get("event_start")
        event_dt = parse_utc(event_start_iso)
        if not isinstance(event_dt, datetime):
            record = dict(record)
            record["why"] = "invalid_event_time"
            unmatched.append(record)
            continue
        event_dt = event_dt.astimezone(timezone.utc)
        home_token = record.get("home_token") or ""
        away_token = record.get("away_token") or ""

        candidate, swapped, _candidate_games, candidate_count, ambiguous = _match_candidates(
            event_dt=event_dt,
            home_token=home_token,
            away_token=away_token,
            schedule_index=schedule_index,
            day_window=day_window,
            max_delta_hours=max_delta_hours,
            allow_swap=role_swap_tolerance,
        )
        if candidate_count == 0:
            candidate_sets_zero += 1
        elif candidate_count > 1:
            candidate_sets_multi += 1
        if candidate is None:
            record = dict(record)
            record["why"] = "ambiguous" if ambiguous else "no_candidate"
            unmatched.append(record)
            continue

        line_payload = _build_line_payload(record, candidate, swapped)
        pinned.append(
            {
                "fetch_ts": record.get("fetch_ts"),
                "season": candidate.season,
                "game_key": candidate.game_key,
                "market": record.get("market"),
                "book": record.get("book"),
                "line": line_payload,
                "home_norm": candidate.home_norm,
                "away_norm": candidate.away_norm,
                "week": candidate.week,
                "kickoff_utc": candidate.kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "role_swapped": swapped,
                "source": record.get("source"),
                "raw_event": {
                    "event_id": record.get("event_id"),
                    "event_start": record.get("event_start"),
                },
            }
        )
        books_counter[record.get("book") or "unknown"] += 1
        markets_counter[record.get("market") or "unknown"] += 1

    # Append pinned rows to season file.
    pinned_path: Optional[Path] = None
    if pinned:
        season = schedule_index.season
        PINNED_DIR.mkdir(parents=True, exist_ok=True)
        pinned_path = PINNED_DIR / f"{season}.jsonl"
        with pinned_path.open("a", encoding="utf-8") as handle:
            for row in pinned:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    # Write unmatched rows (if any) to timestamped file.
    if unmatched:
        UNMATCHED_DIR.mkdir(parents=True, exist_ok=True)
        unmatched_path = UNMATCHED_DIR / f"{now_token}.jsonl"
        tmp_path = unmatched_path.with_suffix(unmatched_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in unmatched:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, unmatched_path)

    examples = []
    for sample in unmatched[:3]:
        home = sample.get("home_norm") or sample.get("home_raw")
        away = sample.get("away_norm") or sample.get("away_raw")
        examples.append(f"{home} vs {away} ({sample.get('why')})")

    counts = {
        "raw": len(raw_records),
        "pinned": len(pinned),
        "unmatched": len(unmatched),
        "books": dict(books_counter),
        "markets": dict(markets_counter),
        "candidate_sets_zero": candidate_sets_zero,
        "candidate_sets_multi": candidate_sets_multi,
    }
    return {
        "pinned_records": pinned,
        "unmatched_records": unmatched,
        "pinned_path": pinned_path,
        "unmatched_path": unmatched_path,
        "counts": counts,
        "examples_unmatched": examples,
    }


__all__ = ["pin_cfb_odds"]


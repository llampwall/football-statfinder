"""Pin raw odds records to schedule games (staging middle layer).

One league-parameterized port of the near line-for-line twins
``src/odds/nfl_pin_to_schedule.py`` and ``src/odds/cfb_pin_to_schedule.py``.
The matching model is kept intact — it is the part REBUILD.md section 5 calls
the soundest design in the repo: team-token pair lookup plus a kickoff window
(day window and max kickoff delta), a role-swap fallback, deterministic
tie-breaking, and an unmatched quarantine that records WHY each record missed
(``invalid_event_time`` / ``no_candidate`` / ``ambiguous``).

Changes from the legacy behavior, all deliberate:

* Pinned records carry the canonical ``build_game_key(...)`` for the league.
  The legacy NFL pin built ``{ts}_{away}_{home}`` keys that could never match
  the frontend's ``{ts}_{home}_{away}`` game rows, which is part of why
  ATS-from-pinned was dead (REBUILD.md bug 4).
* Slugging goes through ``common.game_key.slug`` (one slug rule for both
  leagues; the legacy NFL slug did not map ``&`` to ``and``).
* The pinned season ledger is deduped on append. The legacy twins appended
  every pinned record on every run, so the ledger grew unboundedly with
  duplicates (REBUILD.md section 3). Identity is
  ``(fetch_ts, game_key, market, book)``.
* The schedule arrives as a parameter (``ScheduleGame`` objects or mapping
  rows) instead of the module reading schedule masters itself;
  ``load_schedule_master`` is provided for the orchestrator's convenience.
  The legacy NFL fallback that scanned the wrong ``out/nfl/`` week tree is
  gone with it.
* Tolerance knobs come from ``settings.odds`` (``pin_day_window``,
  ``pin_max_kickoff_delta_hours``, ``role_swap_tolerance``), never from env.
* Ledger reads/writes go through the counted-skip JSONL reader and the
  unmatched quarantine is written atomically.
* Role-swap line inversion fixed: the legacy extractors selected the home
  outcome as ``away_token if swapped else home_token``, but outcome tokens
  identify TEAMS (not provider roles), so a role-swapped match pinned the
  schedule-away team's spread/moneyline as the home line — flipped sign,
  wrong favored side. Extraction now always keys on the schedule's own
  tokens; ``role_swapped`` survives on the record as a diagnostic only.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .. import paths
from ..common.game_key import build_game_key
from ..common.io_atomic import write_atomic_jsonl
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduleGame:
    """Normalized schedule row used for pinning (kickoff is aware UTC)."""

    season: int
    week: int
    game_key: str
    kickoff: datetime
    home_norm: str
    away_norm: str
    home_token: str
    away_token: str
    neutral_site: Optional[bool]


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _coerce_neutral(value: Any) -> Optional[bool]:
    """Best-effort neutral-site coercion (True/False/indeterminate None)."""
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


def make_schedule_game(
    league: League,
    *,
    season: int,
    week: int,
    kickoff: datetime,
    home: str,
    away: str,
    neutral_site: Any = None,
) -> ScheduleGame:
    """Build a ``ScheduleGame`` with normalized names, tokens, canonical key."""
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    kickoff = kickoff.astimezone(timezone.utc)
    home_norm = league.normalize_display(home)
    away_norm = league.normalize_display(away)
    return ScheduleGame(
        season=int(season),
        week=int(week),
        game_key=build_game_key(league, kickoff, home_norm, away_norm),
        kickoff=kickoff,
        home_norm=home_norm,
        away_norm=away_norm,
        home_token=league.merge_key(home_norm),
        away_token=league.merge_key(away_norm),
        neutral_site=_coerce_neutral(neutral_site),
    )


def schedule_game_from_row(league: League, row: Mapping[str, Any]) -> Optional[ScheduleGame]:
    """Coerce a schedule-master-shaped mapping into a ``ScheduleGame``.

    Expected columns: season, week, kickoff_iso_utc, home_team_norm,
    away_team_norm, neutral_site. Returns None (logged) for unusable rows.
    """
    kickoff = _parse_utc(str(row.get("kickoff_iso_utc") or ""))
    home = str(row.get("home_team_norm") or "").strip()
    away = str(row.get("away_team_norm") or "").strip()
    if kickoff is None or not home or not away:
        return None
    try:
        season = int(float(row.get("season")))  # type: ignore[arg-type]
        week = int(float(row.get("week") or 0))
    except (TypeError, ValueError):
        return None
    return make_schedule_game(
        league,
        season=season,
        week=week,
        kickoff=kickoff,
        home=home,
        away=away,
        neutral_site=row.get("neutral_site"),
    )


def load_schedule_master(
    league: League,
    csv_path: Optional[Path] = None,
    *,
    out_root: Optional[Path] = None,
) -> List[ScheduleGame]:
    """Load schedule games from the master CSV (orchestrator convenience).

    Reads ``paths.schedule_master_csv(league.code)`` unless a path is given.
    Unusable rows are skipped and counted in a single log line (legacy loaders
    dropped them silently).
    """
    import pandas as pd

    target = (
        csv_path
        if csv_path is not None
        else paths.schedule_master_csv(league.code, out_root=out_root)
    )
    if not target.exists():
        logger.warning("%s schedule master missing at %s", league.display, target)
        return []
    df = pd.read_csv(target)
    games: List[ScheduleGame] = []
    skipped = 0
    for row in df.to_dict(orient="records"):
        game = schedule_game_from_row(league, row)
        if game is None:
            skipped += 1
            continue
        games.append(game)
    if skipped:
        logger.warning("%s schedule master: skipped %d unusable row(s)", league.display, skipped)
    return games


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_Candidate = Tuple[ScheduleGame, bool, float]  # (game, role_swapped, delta_seconds)


def _within_day_window(event_dt: datetime, kickoff: datetime, day_window: int) -> bool:
    return abs((kickoff.date() - event_dt.date()).days) <= day_window


def _collect_candidates(
    games: Iterable[ScheduleGame],
    event_dt: datetime,
    day_window: int,
    max_delta_seconds: float,
    swapped_flag: bool,
) -> List[_Candidate]:
    collected: List[_Candidate] = []
    for game in games:
        delta_seconds = abs((game.kickoff - event_dt).total_seconds())
        if max_delta_seconds >= 0 and delta_seconds > max_delta_seconds:
            continue
        if not _within_day_window(event_dt, game.kickoff, day_window):
            continue
        collected.append((game, swapped_flag, delta_seconds))
    return collected


def _sort_key(candidate: _Candidate) -> Tuple[float, int, str]:
    game, _, delta_seconds = candidate
    neutral_rank = 0 if game.neutral_site else 1
    return (delta_seconds, neutral_rank, game.game_key)


# ---------------------------------------------------------------------------
# Line extraction (aligned to the schedule's home/away roles)
# ---------------------------------------------------------------------------


def _extract_spread_line(
    outcomes: List[Dict[str, Any]],
    home_token: str,
    away_token: str,
) -> Dict[str, Any]:
    # Outcome tokens identify teams, so the schedule tokens are used directly.
    # (Legacy swapped these under role-swap matches, inverting the spread.)
    home_outcome = next((o for o in outcomes if o.get("token") == home_token), None)
    away_outcome = next((o for o in outcomes if o.get("token") == away_token), None)
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


def _extract_moneyline(
    outcomes: List[Dict[str, Any]], home_token: str, away_token: str
) -> Dict[str, Any]:
    # Same identity-token rule as spreads (legacy swapped prices on role swap).
    home_price = None
    away_price = None
    for outcome in outcomes:
        token = outcome.get("token")
        if token == home_token:
            home_price = outcome.get("price")
        elif token == away_token:
            away_price = outcome.get("price")
    return {
        "moneyline_home": home_price,
        "moneyline_away": away_price,
        "raw_outcomes": outcomes,
    }


def _build_line_payload(record: Mapping[str, Any], game: ScheduleGame) -> Dict[str, Any]:
    market = record.get("market")
    payload = record.get("market_payload") or {}
    outcomes = payload.get("outcomes") or []
    if market == "spreads":
        return _extract_spread_line(outcomes, game.home_token, game.away_token)
    if market == "totals":
        return _extract_totals_line(outcomes)
    if market == "h2h":
        return _extract_moneyline(outcomes, game.home_token, game.away_token)
    return {"raw_outcomes": outcomes}


# ---------------------------------------------------------------------------
# Ledger append with dedupe
# ---------------------------------------------------------------------------

_DedupeKey = Tuple[Any, Any, Any, Any]


def _dedupe_key(record: Mapping[str, Any]) -> _DedupeKey:
    return (
        record.get("fetch_ts"),
        record.get("game_key"),
        record.get("market"),
        record.get("book"),
    )


def _append_deduped(path: Path, rows: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    """Append rows not already present in the ledger; returns (appended, skipped)."""
    existing = {_dedupe_key(row) for row in read_jsonl(path).rows}
    appended = 0
    skipped = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = _dedupe_key(row)
            if key in existing:
                skipped += 1
                continue
            existing.add(key)
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            appended += 1
    return appended, skipped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

ScheduleInput = Union[ScheduleGame, Mapping[str, Any]]


def pin_to_schedule(
    league: League,
    raw_records: Sequence[Mapping[str, Any]],
    schedule_games: Sequence[ScheduleInput],
    settings: Settings,
    *,
    out_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pin raw odds records onto schedule games and stage the results.

    Args:
        league: League constants and normalizers.
        raw_records: ``ingest_raw`` output records (one per bookmaker+market).
        schedule_games: ``ScheduleGame`` objects or schedule-master-shaped
            mappings (see ``schedule_game_from_row``).
        settings: Tolerances read from ``settings.odds``.
        out_root: Optional output-root override (tests); defaults to the repo
            ``out/`` tree.

    Returns a dict with ``pinned_records``, ``unmatched_records``,
    ``pinned_paths`` (per-season ledger paths), ``unmatched_path``, ``counts``
    (raw/pinned/appended/duplicates_skipped/unmatched/books/markets/
    candidate_sets_zero/candidate_sets_multi/unmatched_reasons), and
    ``examples_unmatched``.

    Side effects: dedupe-appends to ``odds_pinned/{league}/{season}.jsonl``
    and atomically writes unmatched rows (with a ``why`` field each) to
    ``odds_unmatched/{league}/<timestamp>.jsonl``.
    """
    day_window = settings.odds.pin_day_window
    max_delta_hours = settings.odds.pin_max_kickoff_delta_hours
    role_swap_tolerance = settings.odds.role_swap_tolerance
    max_delta_seconds = -1.0 if max_delta_hours < 0 else max_delta_hours * 3600.0

    counts: Dict[str, Any] = {
        "raw": len(raw_records),
        "pinned": 0,
        "appended": 0,
        "duplicates_skipped": 0,
        "unmatched": 0,
        "books": {},
        "markets": {},
        "candidate_sets_zero": 0,
        "candidate_sets_multi": 0,
        "unmatched_reasons": {},
    }
    result: Dict[str, Any] = {
        "pinned_records": [],
        "unmatched_records": [],
        "pinned_paths": [],
        "unmatched_path": None,
        "counts": counts,
        "examples_unmatched": [],
    }
    if not raw_records:
        return result

    games: List[ScheduleGame] = []
    for item in schedule_games:
        if isinstance(item, ScheduleGame):
            games.append(item)
        else:
            game = schedule_game_from_row(league, item)
            if game is not None:
                games.append(game)

    unmatched: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    def _quarantine(record: Mapping[str, Any], why: str) -> None:
        entry = dict(record)
        entry["why"] = why
        unmatched.append(entry)
        reasons[why] += 1

    if not games:
        for record in raw_records:
            _quarantine(record, "no_schedule")
        counts["unmatched"] = len(unmatched)
        counts["unmatched_reasons"] = dict(reasons)
        result["unmatched_records"] = unmatched
        result["unmatched_path"] = _write_unmatched(league, unmatched, out_root=out_root)
        return result

    games_by_pair: Dict[Tuple[str, str], List[ScheduleGame]] = defaultdict(list)
    for game in games:
        games_by_pair[(game.home_token, game.away_token)].append(game)
    for pair_games in games_by_pair.values():
        pair_games.sort(key=lambda item: item.kickoff)

    pinned: List[Dict[str, Any]] = []
    books_counter: Counter[str] = Counter()
    markets_counter: Counter[str] = Counter()

    for record in raw_records:
        event_dt = _parse_utc(record.get("event_start"))
        if not isinstance(event_dt, datetime):
            _quarantine(record, "invalid_event_time")
            continue
        home_token = record.get("home_token") or league.merge_key(
            str(record.get("home_norm") or "")
        )
        away_token = record.get("away_token") or league.merge_key(
            str(record.get("away_norm") or "")
        )

        candidate_pool = _collect_candidates(
            games_by_pair.get((home_token, away_token), []),
            event_dt,
            day_window,
            max_delta_seconds,
            False,
        )
        if not candidate_pool and role_swap_tolerance:
            candidate_pool = _collect_candidates(
                games_by_pair.get((away_token, home_token), []),
                event_dt,
                day_window,
                max_delta_seconds,
                True,
            )

        candidate_count = len(candidate_pool)
        if candidate_count == 0:
            counts["candidate_sets_zero"] += 1
            _quarantine(record, "no_candidate")
            continue
        if candidate_count > 1:
            counts["candidate_sets_multi"] += 1

        candidate_pool.sort(key=_sort_key)
        best, swapped, _ = candidate_pool[0]
        if candidate_count > 1 and _sort_key(candidate_pool[0]) == _sort_key(candidate_pool[1]):
            _quarantine(record, "ambiguous")
            continue

        line_payload = _build_line_payload(record, best)
        pinned.append(
            {
                "fetch_ts": record.get("fetch_ts"),
                "source": record.get("source"),
                "season": best.season,
                "week": best.week,
                # Canonical key (bug 4 fix): the legacy NFL pin wrote
                # {ts}_{away}_{home} which never matched NFL game rows.
                "game_key": best.game_key,
                "market": record.get("market"),
                "book": record.get("book"),
                "line": line_payload,
                "home_norm": best.home_norm,
                "away_norm": best.away_norm,
                "kickoff_utc": best.kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "role_swapped": swapped,
                "raw_event": {
                    "event_id": record.get("event_id"),
                    "event_start": record.get("event_start"),
                },
            }
        )
        books_counter[record.get("book") or "unknown"] += 1
        markets_counter[record.get("market") or "unknown"] += 1

    pinned_paths: List[Path] = []
    appended_total = 0
    duplicates_total = 0
    if pinned:
        by_season: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in pinned:
            by_season[int(row["season"])].append(row)
        for season in sorted(by_season):
            ledger = paths.odds_pinned_jsonl(league.code, season, out_root=out_root)
            appended, skipped = _append_deduped(ledger, by_season[season])
            appended_total += appended
            duplicates_total += skipped
            pinned_paths.append(ledger)
        if duplicates_total:
            logger.info(
                "%s odds pin: skipped %d duplicate ledger row(s)", league.display, duplicates_total
            )

    unmatched_path = _write_unmatched(league, unmatched, out_root=out_root)

    examples = []
    for sample in unmatched[:3]:
        home = sample.get("home_norm") or sample.get("home_raw")
        away = sample.get("away_norm") or sample.get("away_raw")
        examples.append(f"{home} vs {away} ({sample.get('why')})")

    counts.update(
        {
            "pinned": len(pinned),
            "appended": appended_total,
            "duplicates_skipped": duplicates_total,
            "unmatched": len(unmatched),
            "books": dict(books_counter),
            "markets": dict(markets_counter),
            "unmatched_reasons": dict(reasons),
        }
    )
    result.update(
        {
            "pinned_records": pinned,
            "unmatched_records": unmatched,
            "pinned_paths": pinned_paths,
            "unmatched_path": unmatched_path,
            "examples_unmatched": examples,
        }
    )
    logger.info(
        "%s odds pin: raw=%d pinned=%d appended=%d dupes=%d unmatched=%d",
        league.display,
        len(raw_records),
        len(pinned),
        appended_total,
        duplicates_total,
        len(unmatched),
    )
    return result


def _write_unmatched(
    league: League,
    rows: Sequence[Mapping[str, Any]],
    *,
    out_root: Optional[Path] = None,
) -> Optional[Path]:
    if not rows:
        return None
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = paths.odds_unmatched_dir(league.code, out_root=out_root) / f"{token}.jsonl"
    write_atomic_jsonl(target, rows)
    return target


__all__ = [
    "ScheduleGame",
    "load_schedule_master",
    "make_schedule_game",
    "pin_to_schedule",
    "schedule_game_from_row",
]

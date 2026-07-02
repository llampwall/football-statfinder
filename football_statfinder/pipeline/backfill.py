"""One league-parameterized scores/ATS backfill for prior weeks.

Replaces the season-1 twins ``src/scores/nfl_backfill.py`` +
``src/scores/cfb_backfill.py`` (~130 duplicated ATS lines). Legacy behavior
deliberately changed while porting:

* Bug 5: W-L(-T) season records are tallied and read back under ONE key case
  (the league merge key). The NFL twin tallied under upper-cased abbreviations
  but read back ``.lower()`` keys out of a defaultdict, minting fresh zeroed
  entries — every backfilled record was "0-0".
* Bug 6: a score-only change persists. The CFB twin's canonical no-op
  comparison excluded ``home_score``/``away_score``, so a run whose only
  effect was filling final scores compared equal and was skipped. Here score
  changes set the change flag directly; there is no canonical comparison.
* Bug 11: no subprocess relaunch of the game-view builders
  (``subprocess.run(check=False)`` swallowed failures). The result names the
  weeks that changed (``changed_weeks``) so the orchestrator rebuilds them,
  and an optionally injected ``rebuild`` callback is invoked per changed week
  with exceptions propagating.
* Bug 3 stays fixed: the sidecar updater is the CFB signature
  (``entries, season, week, *, ats, margin``) — the correct one.
* Unified score semantics: incoming scores win whenever they differ (the NFL
  twin's behavior, and the merge-preserve invariant); the CFB twin only
  filled missing scores.
* Score fetch is injectable (``score_source``); the default reads the
  league's schedule master CSV — the source both twins actually used (the
  masters are refreshed from nflverse/CFBD earlier in a run). Game keys come
  from the one shared builder, so master rows and week rows can no longer
  disagree on key shape (the season-1 NFL abbreviation-key mismatch).
* Merge-preserve (``common.backfill_merge``): incoming scores win, existing
  promoted odds/rating fields survive. Treated as law.
* Deterministic output ordering (season, week, kickoff, game_key) kept from
  the CFB twin.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd

from .. import paths
from ..common.backfill_merge import merge_games_week, summarize_preservation
from ..common.game_key import build_game_key
from ..common.io_atomic import write_atomic_csv, write_atomic_json, write_atomic_jsonl
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League
from .ats import (
    ClosingSpreadApi,
    compute_game_ats,
    is_blank,
    load_pinned_spread_index,
    resolve_closing_spread,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredGame:
    """One schedule-master game as the backfill consumes it."""

    game_key: str
    kickoff_iso_utc: Optional[str]
    home_team: str  # normalized display name
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]


# (league, season) -> finished/scheduled games with any known finals.
ScoreSource = Callable[[League, int], Sequence[ScoredGame]]


def _parse_kickoff(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return int(round(number))


def master_score_source(
    league: League, season: int, *, out_root: Optional[Path] = None
) -> List[ScoredGame]:
    """Default score source: the league's schedule master CSV.

    Game keys are rebuilt with the shared ``build_game_key`` from the master's
    normalized team names, matching what the week builders emit.
    """
    path = paths.schedule_master_csv(league.code, out_root=out_root)
    if not path.exists():
        logger.warning("schedule master missing: %s (no scores to backfill)", path)
        return []
    df = pd.read_csv(path)
    if df.empty or "season" not in df.columns:
        return []
    df = df[pd.to_numeric(df["season"], errors="coerce") == season]

    games: List[ScoredGame] = []
    for row in df.to_dict(orient="records"):
        kickoff_raw = row.get("kickoff_iso_utc") or row.get("kickoff_iso")
        kickoff = _parse_kickoff(kickoff_raw)
        if kickoff is None:
            continue
        home = str(row.get("home_team_norm") or row.get("home_team") or "").strip()
        away = str(row.get("away_team_norm") or row.get("away_team") or "").strip()
        if not home or not away:
            continue
        home_norm = league.normalize_display(home)
        away_norm = league.normalize_display(away)
        games.append(
            ScoredGame(
                game_key=build_game_key(league, kickoff, home_norm, away_norm),
                kickoff_iso_utc=str(kickoff_raw),
                home_team=home_norm,
                away_team=away_norm,
                home_score=_coerce_score(row.get("home_score")),
                away_score=_coerce_score(row.get("away_score")),
            )
        )
    return games


def _score_and_record_maps(
    league: League, games: Sequence[ScoredGame]
) -> Tuple[Dict[str, Tuple[int, int]], Dict[Tuple[str, str], str]]:
    """Final-score lookup plus per-(game, side) season records after the game.

    Bug 5 fix: one tally key — ``league.merge_key(team)`` — for both writing
    and reading, and no defaultdict minting zeroed entries on the read path.
    """
    score_lookup: Dict[str, Tuple[int, int]] = {}
    for game in games:
        if game.home_score is None or game.away_score is None:
            continue
        score_lookup[game.game_key] = (game.home_score, game.away_score)

    def _sort_key(game: ScoredGame) -> Tuple[str, str]:
        return (game.kickoff_iso_utc or "", game.game_key)

    records_after_game: Dict[Tuple[str, str], str] = {}
    tallies: Dict[str, Dict[str, int]] = {}

    for game in sorted(games, key=_sort_key):
        scores = score_lookup.get(game.game_key)
        if not scores:
            continue
        home_score, away_score = scores
        home_key = league.merge_key(game.home_team)
        away_key = league.merge_key(game.away_team)
        if not home_key or not away_key:
            continue
        home_rec = tallies.setdefault(home_key, {"w": 0, "l": 0, "t": 0})
        away_rec = tallies.setdefault(away_key, {"w": 0, "l": 0, "t": 0})
        if home_score > away_score:
            home_rec["w"] += 1
            away_rec["l"] += 1
        elif home_score < away_score:
            home_rec["l"] += 1
            away_rec["w"] += 1
        else:
            home_rec["t"] += 1
            away_rec["t"] += 1
        for side, rec in (("home", home_rec), ("away", away_rec)):
            record_str = f"{rec['w']}-{rec['l']}"
            if rec["t"] > 0:
                record_str += f"-{rec['t']}"
            records_after_game[(game.game_key, side)] = record_str

    return score_lookup, records_after_game


# ---------------------------------------------------------------------------
# Sidecar / ATS helpers (ported from the CFB twin — the correct signatures)
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_sidecar(
    sidecar_dir: Path, game_key: str, cache: Dict[str, Optional[dict]]
) -> Optional[dict]:
    if game_key in cache:
        return cache[game_key]
    path = sidecar_dir / f"{game_key}.json"
    if not path.exists():
        cache[game_key] = None
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("unreadable sidecar skipped: %s", path)
        cache[game_key] = None
        return None
    cache[game_key] = {"path": path, "data": payload, "dirty": False}
    return cache[game_key]


def _update_sidecar_entry(
    entries: Optional[Iterable[dict]],
    season: int,
    week: int,
    *,
    ats: Optional[str],
    margin: Optional[float],
) -> bool:
    """Fill blank per-game ATS/to-margin on the matching YTD entry.

    This is the CFB twin's definition — the correct one; the NFL twin's paste
    had a mismatched signature that raised ``TypeError`` (bug 3).
    """
    updated = False
    for entry in entries or []:
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except (TypeError, ValueError):
            continue
        if entry_season != season or entry_week != week:
            continue
        if ats and is_blank(entry.get("ats")):
            entry["ats"] = ats
            updated = True
        if margin is not None and not _is_finite_number(entry.get("to_margin")):
            entry["to_margin"] = round(float(margin), 2)
            updated = True
        break
    return updated


def _sidecar_needs_ats(entries: Optional[Iterable[dict]], season: int, week: int) -> bool:
    for entry in entries or []:
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except (TypeError, ValueError):
            continue
        if entry_season != season or entry_week != week:
            continue
        if is_blank(entry.get("ats")):
            return True
        if not _is_finite_number(entry.get("to_margin")):
            return True
        return False
    return True


def _compute_team_ats(
    entries: Optional[Iterable[dict]], season: int, thru_week: int
) -> Tuple[str, Optional[float]]:
    wins = losses = pushes = 0
    margins: List[float] = []
    for entry in entries or []:
        try:
            entry_season = int(entry.get("season"))
            entry_week = int(entry.get("week"))
        except (TypeError, ValueError):
            continue
        if entry_season != season or entry_week > thru_week:
            continue
        result = (entry.get("ats") or "").strip().upper()
        if result == "W":
            wins += 1
        elif result == "L":
            losses += 1
        elif result == "P":
            pushes += 1
        margin = entry.get("to_margin")
        if _is_finite_number(margin):
            margins.append(float(margin))
    record = f"{wins}-{losses}-{pushes}"
    avg_margin = sum(margins) / len(margins) if margins else None
    return record, avg_margin


def _game_needs_ats(row: Mapping[str, Any]) -> bool:
    if is_blank(row.get("home_ats")) or is_blank(row.get("away_ats")):
        return True
    if not _is_finite_number(row.get("home_to_margin_pg")) or not _is_finite_number(
        row.get("away_to_margin_pg")
    ):
        return True
    return False


_SINGLE_GAME_RECORD = {"W": "1-0-0", "L": "0-1-0", "P": "0-0-1"}


def _apply_ats_backfill(
    league: League,
    season: int,
    week: int,
    rows: List[dict],
    sidecar_dir: Path,
    *,
    settings: Settings,
    pinned_index: Dict[str, List[dict]],
    spread_api: Optional[ClosingSpreadApi],
) -> Tuple[int, Counter, bool]:
    """Fill blank ATS fields on finished games using resolved closing spreads."""
    sidecar_cache: Dict[str, Optional[dict]] = {}
    source_counts: Counter = Counter()
    games_fixed = 0
    rows_changed = False

    for row in rows:
        game_key = row.get("game_key")
        if not isinstance(game_key, str):
            continue
        row_needs = _game_needs_ats(row)
        sidecar_entry = _load_sidecar(sidecar_dir, game_key, sidecar_cache)
        sidecar_needs = False
        if sidecar_entry:
            data = sidecar_entry["data"]
            sidecar_needs = _sidecar_needs_ats(
                data.get("home_ytd"), season, week
            ) or _sidecar_needs_ats(data.get("away_ytd"), season, week)
        if not row_needs and not sidecar_needs:
            continue
        try:
            home_score_int = int(row.get("home_score"))
            away_score_int = int(row.get("away_score"))
        except (TypeError, ValueError):
            continue
        closing = resolve_closing_spread(
            league,
            season,
            row,
            settings=settings,
            pinned_index=pinned_index,
            api=spread_api,
        )
        if not closing:
            continue
        ats_payload = compute_game_ats(
            home_score_int, away_score_int, closing.get("favored_team"), closing.get("spread")
        )
        if not ats_payload:
            continue

        game_updated = False

        if sidecar_entry:
            data = sidecar_entry["data"]
            home_changed = _update_sidecar_entry(
                data.get("home_ytd"),
                season,
                week,
                ats=ats_payload["home_ats"],
                margin=ats_payload["to_margin_home"],
            )
            away_changed = _update_sidecar_entry(
                data.get("away_ytd"),
                season,
                week,
                ats=ats_payload["away_ats"],
                margin=ats_payload["to_margin_away"],
            )
            if home_changed or away_changed:
                sidecar_entry["dirty"] = True
                game_updated = True

            home_record, home_avg = _compute_team_ats(data.get("home_ytd"), season, week)
            away_record, away_avg = _compute_team_ats(data.get("away_ytd"), season, week)

            if home_record and row.get("home_ats") != home_record:
                row["home_ats"] = home_record
                game_updated = True
            if away_record and row.get("away_ats") != away_record:
                row["away_ats"] = away_record
                game_updated = True
            for side, avg in (("home", home_avg), ("away", away_avg)):
                if avg is None:
                    continue
                avg_val = round(float(avg), 2)
                current = row.get(f"{side}_to_margin_pg")
                if not _is_finite_number(current) or abs(float(current) - avg_val) > 1e-6:
                    row[f"{side}_to_margin_pg"] = avg_val
                    game_updated = True
        else:
            for side in ("home", "away"):
                if is_blank(row.get(f"{side}_ats")):
                    row[f"{side}_ats"] = _SINGLE_GAME_RECORD[ats_payload[f"{side}_ats"]]
                    game_updated = True
            if not _is_finite_number(row.get("home_to_margin_pg")):
                row["home_to_margin_pg"] = round(ats_payload["to_margin_home"], 2)
                game_updated = True
            if not _is_finite_number(row.get("away_to_margin_pg")):
                row["away_to_margin_pg"] = round(ats_payload["to_margin_away"], 2)
                game_updated = True

        if game_updated:
            row.setdefault("raw_sources", {})["closing_spread"] = {
                "source": closing.get("source"),
                "book": closing.get("book"),
                "spread": closing.get("spread"),
                "favored_team": closing.get("favored_team"),
                "fetched_ts": closing.get("fetched_ts"),
            }
            source_counts[closing.get("source", "unknown")] += 1
            games_fixed += 1
            rows_changed = True

    for entry in sidecar_cache.values():
        if entry and entry.get("dirty"):
            write_atomic_json(entry["path"], entry["data"])

    return games_fixed, source_counts, rows_changed


# ---------------------------------------------------------------------------
# The backfill
# ---------------------------------------------------------------------------


@dataclass
class BackfillResult:
    """Structured outcome; the orchestrator rebuilds ``changed_weeks`` itself."""

    weeks_scanned: List[int] = field(default_factory=list)
    updated: int = 0
    skipped: int = 0
    files_rewritten: int = 0
    preserved_odds: int = 0
    preserved_rvo: int = 0
    ats_fixed: int = 0
    ats_sources: Dict[str, int] = field(default_factory=dict)
    changed_weeks: List[int] = field(default_factory=list)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_sort_key(row: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    return (
        _safe_int(row.get("season")),
        _safe_int(row.get("week")),
        str(row.get("kickoff_iso_utc") or row.get("kickoff_iso") or ""),
        str(row.get("game_key") or ""),
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _align_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = [col for col in columns if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + remainder)


def backfill_scores(
    league: League,
    season: int,
    week: int,
    settings: Settings,
    *,
    score_source: Optional[ScoreSource] = None,
    spread_api: Optional[ClosingSpreadApi] = None,
    promote_week: Optional[Callable[[List[dict], int, int], Mapping[str, Any]]] = None,
    rebuild: Optional[Callable[[int, int], None]] = None,
    out_root: Optional[Path] = None,
) -> BackfillResult:
    """Backfill final scores, records, and ATS for recent prior weeks.

    Args:
        league / season / week: current position; only weeks
            ``[week - settings.backfill.weeks, week)`` (>= 1) are touched.
        settings: honors ``backfill.scores_enable``, ``backfill.weeks``,
            ``backfill.promote_prev``, ``backfill.ats_enable``,
            ``backfill.ats_source``, and (via the API client)
            ``odds.cache_only``.
        score_source: injectable ``(league, season) -> Sequence[ScoredGame]``;
            defaults to the schedule master CSV reader.
        spread_api: paid closing-spread tier (``AtsBackfillApi`` in
            production, a stub in tests); optional.
        promote_week: optional in-place odds promoter for prior weeks
            (runs only when ``backfill.promote_prev`` is set).
        rebuild: optional ``(season, week)`` callback invoked for each week
            whose files changed. Exceptions propagate — a failed rebuild is a
            failed backfill, never a swallowed exit code (bug 11).
        out_root: test override for the ``out/`` root.

    Returns:
        :class:`BackfillResult`; ``changed_weeks`` names every week the
        orchestrator must rebuild downstream artifacts for.
    """
    result = BackfillResult()
    if not settings.backfill.scores_enable:
        logger.info("scores backfill disabled; league=%s", league.code)
        return result

    include_weeks = settings.backfill.weeks
    if include_weeks <= 0 or week <= 1:
        return result

    weeks = sorted(w for w in range(week - include_weeks, week) if w >= 1)
    if not weeks:
        return result
    result.weeks_scanned = weeks

    source = score_source if score_source is not None else (
        lambda lg, sn: master_score_source(lg, sn, out_root=out_root)
    )
    score_lookup, record_lookup = _score_and_record_maps(league, list(source(league, season)))

    ats_enabled = settings.backfill.ats_enable
    pinned_index = (
        load_pinned_spread_index(league, season, out_root=out_root) if ats_enabled else {}
    )
    ats_source_counts: Counter = Counter()

    for target_week in weeks:
        json_path = paths.games_week_jsonl(league.code, season, target_week, out_root=out_root)
        csv_path = paths.games_week_csv(league.code, season, target_week, out_root=out_root)
        existing_rows = read_jsonl(json_path).rows
        if not existing_rows:
            continue

        incoming_rows = deepcopy(existing_rows)
        csv_df = pd.read_csv(csv_path) if csv_path.exists() else None
        file_changed = False
        row_updates = 0

        for row in incoming_rows:
            game_key = row.get("game_key")
            if not isinstance(game_key, str):
                continue
            scores = score_lookup.get(game_key)
            if not scores:
                if _is_missing(row.get("home_score")) or _is_missing(row.get("away_score")):
                    result.skipped += 1
                continue
            home_score, away_score = scores

            row_changed = False
            # Incoming scores win (merge-preserve invariant); a score-only
            # change is a change (bug 6).
            if row.get("home_score") != home_score or row.get("away_score") != away_score:
                row["home_score"] = home_score
                row["away_score"] = away_score
                row_changed = True

            home_record = record_lookup.get((game_key, "home"))
            away_record = record_lookup.get((game_key, "away"))
            if home_record and row.get("home_su") != home_record:
                row["home_su"] = home_record
                row_changed = True
            if away_record and row.get("away_su") != away_record:
                row["away_su"] = away_record
                row_changed = True

            raw_sources = row.get("raw_sources")
            schedule_row = raw_sources.get("schedule_row") if isinstance(raw_sources, dict) else None
            if isinstance(schedule_row, dict):
                if schedule_row.get("home_score") != home_score:
                    schedule_row["home_score"] = home_score
                    row_changed = True
                if schedule_row.get("away_score") != away_score:
                    schedule_row["away_score"] = away_score
                    row_changed = True

            if row_changed:
                file_changed = True
                result.updated += 1
                row_updates += 1

        merged_rows = merge_games_week(existing_rows, incoming_rows)
        preservation = summarize_preservation(existing_rows, merged_rows)
        final_rows: List[dict] = merged_rows
        promoted_games = 0

        if settings.backfill.promote_prev and target_week < week:
            if promote_week is None:
                logger.info(
                    "promote_prev set but no promoter injected; skipping "
                    "re-promotion for %s week %s-%s",
                    league.code,
                    season,
                    target_week,
                )
            else:
                promoted_rows = deepcopy(merged_rows)
                promote_stats = promote_week(promoted_rows, season, target_week)
                promoted_games = int(promote_stats.get("promoted_games", 0) or 0)
                final_rows = promoted_rows
                if promoted_games > 0:
                    file_changed = True
                logger.info(
                    "odds re-promote: league=%s week=%s-%s promoted=%d",
                    league.code,
                    season,
                    target_week,
                    promoted_games,
                )

        if ats_enabled:
            sidecar_dir = paths.sidecar_dir(league.code, season, target_week, out_root=out_root)
            ats_fixed, ats_counts, ats_rows_changed = _apply_ats_backfill(
                league,
                season,
                target_week,
                final_rows,
                sidecar_dir,
                settings=settings,
                pinned_index=pinned_index,
                spread_api=spread_api,
            )
            result.ats_fixed += ats_fixed
            ats_source_counts.update(ats_counts)
            if ats_rows_changed:
                file_changed = True
            logger.info(
                "ats backfill: league=%s week=%s-%s games_fixed=%d sources=%s",
                league.code,
                season,
                target_week,
                ats_fixed,
                dict(ats_counts),
            )

        if not file_changed:
            continue

        final_rows = sorted(final_rows, key=_row_sort_key)
        result.preserved_odds += preservation["preserved_odds"]
        result.preserved_rvo += preservation["preserved_rvo"]

        write_atomic_jsonl(json_path, final_rows)
        final_df = pd.DataFrame(final_rows)
        if csv_df is not None:
            final_df = _align_columns(final_df, list(csv_df.columns))
        write_atomic_csv(csv_path, final_df)
        result.files_rewritten += 1
        result.changed_weeks.append(target_week)

        logger.info(
            "backfill merge: league=%s week=%s-%s updated_scores=%d "
            "preserved_odds=%d preserved_rvo=%d",
            league.code,
            season,
            target_week,
            row_updates,
            preservation["preserved_odds"],
            preservation["preserved_rvo"],
        )

    result.ats_sources = dict(ats_source_counts)

    if rebuild is not None:
        for changed_week in result.changed_weeks:
            # Bug 11 fix: rebuild failures propagate to the orchestrator.
            rebuild(season, changed_week)

    return result


__all__ = [
    "BackfillResult",
    "ScoreSource",
    "ScoredGame",
    "backfill_scores",
    "master_score_source",
]

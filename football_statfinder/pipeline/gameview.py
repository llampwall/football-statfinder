"""The one gameview builder for both leagues.

Ports and merges ``src/gameview_build.py`` (NFL, 822 lines) and
``src/gameview_build_cfb.py`` (699 lines). The output schema and derived-field
names are the frozen frontend contract (``web/week_view.js`` /
``web/game_view.js`` read these records); :data:`FROZEN_RECORD_FIELDS`
enumerates them and must not change.

Deliberate changes from legacy behavior, all documented:

* BUG 7 FIX — exactly ONE rating-vs-odds formula. The legacy CFB builder
  computed ``rating_vs_odds`` twice with conflicting formulas
  (``_compute_rating_vectors``: no HFA in the rating delta, and a
  favored-perspective value that SUBTRACTED the favored spread;
  ``_compute_rating_vs_odds``: HFA applied to the home side, favored spread
  ADDED) and emitted a row mixing both. This builder uses only the
  ``common.metrics`` trio (``rating_diff`` / ``team_centric_spread`` /
  ``rating_vs_odds``), which is exactly the legacy NFL builder's math:
  ``rating_diff = home_pr + hfa - away_pr`` (HFA included), home-centric
  ``rating_vs_odds = rating_diff + spread_home_relative``, and the favored
  fields derived from the same numbers (see :func:`derive_rating_fields`).
* One stat-sourcing model for both leagues: the builder joins a prebuilt
  :class:`~football_statfinder.sources.stats.TeamStats` mapping (the CFB
  model). The legacy NFL builder recomputed per-team stats in-process and
  re-ranked the league six times per game (O(n^2), one of the three duplicate
  dense-rank implementations), then overwrote almost all of it from the
  league_metrics CSV anyway; that path is gone.
* ``favored_side`` / ``spread_favored_team`` are always derived from
  ``spread_home_relative`` (negative = home favored; pick-em counts as HOME,
  the legacy NFL tie-break), never trusted from an odds payload.
* ``game_key`` comes from ``common.game_key.build_game_key`` — the only
  constructor — fed with the league-normalized display names.
* One pass per week: given the same inputs the build is deterministic
  (records sorted by kickoff then game_key), so the legacy CFB
  run-the-builder-twice workaround (REBUILD.md bug 12) is unnecessary.
* Missing stats emit ``None`` rather than 0.0 (REBUILD.md error policy:
  distinguish null from zero; legacy NFL emitted 0.0 for teams with no
  games).
* Writes go through ``common.io_atomic`` to the unified
  ``out/{league}/{season}_week{week}/`` layout from ``paths``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .. import paths
from ..common import metrics
from ..common.game_key import build_game_key
from ..common.io_atomic import write_atomic_csv, write_atomic_json, write_atomic_jsonl
from ..leagues import League
from ..sources.stats import TeamStats

logger = logging.getLogger(__name__)

# The frozen games_week record schema (field names AND CSV column order).
# Enumerated from the legacy builders: src/gameview_build_cfb.py
# OUTPUT_COLUMNS, cross-checked against the record dict assembled in
# src/gameview_build.py build_gameview() — the two sets are identical.
FROZEN_RECORD_FIELDS: Tuple[str, ...] = (
    "season",
    "week",
    "kickoff_iso_utc",
    "game_key",
    "source_uid",
    "home_team_raw",
    "home_team_norm",
    "away_team_raw",
    "away_team_norm",
    "spread_home_relative",
    "total",
    "moneyline_home",
    "moneyline_away",
    "odds_source",
    "is_closing",
    "snapshot_at",
    "home_pr",
    "home_pr_rank",
    "away_pr",
    "away_pr_rank",
    "home_sos",
    "away_sos",
    "home_sos_rank",
    "away_sos_rank",
    "hfa",
    "rating_diff",
    "rating_vs_odds",
    "favored_side",
    "spread_favored_team",
    "rating_diff_favored_team",
    "rating_vs_odds_favored_team",
    "home_pf_pg",
    "home_pa_pg",
    "home_ry_pg",
    "home_py_pg",
    "home_ty_pg",
    "home_ry_allowed_pg",
    "home_py_allowed_pg",
    "home_ty_allowed_pg",
    "home_to_margin_pg",
    "home_su",
    "home_ats",
    "home_rush_rank",
    "home_pass_rank",
    "home_tot_off_rank",
    "home_rush_def_rank",
    "home_pass_def_rank",
    "home_tot_def_rank",
    "away_pf_pg",
    "away_pa_pg",
    "away_ry_pg",
    "away_py_pg",
    "away_ty_pg",
    "away_ry_allowed_pg",
    "away_py_allowed_pg",
    "away_ty_allowed_pg",
    "away_to_margin_pg",
    "away_su",
    "away_ats",
    "away_rush_rank",
    "away_pass_rank",
    "away_tot_off_rank",
    "away_rush_def_rank",
    "away_pass_def_rank",
    "away_tot_def_rank",
    "raw_sources",
)

_STAT_VALUE_FIELDS: Tuple[str, ...] = (
    "pf_pg",
    "pa_pg",
    "ry_pg",
    "py_pg",
    "ty_pg",
    "ry_allowed_pg",
    "py_allowed_pg",
    "ty_allowed_pg",
    "to_margin_pg",
)
_STAT_RANK_FIELDS: Tuple[str, ...] = (
    "rush_rank",
    "pass_rank",
    "tot_off_rank",
    "rush_def_rank",
    "pass_def_rank",
    "tot_def_rank",
)

# Sagarin team spellings that disagree with schedule/metrics spellings after
# merge-keying (ported verbatim from src/gameview_build_cfb.py).
SAGARIN_TOKEN_OVERRIDES: Dict[str, str] = {
    "appalachianstate": "appstate",
    "armywestpoint": "army",
    "centralfloridaucf": "ucf",
    "connecticut": "uconn",
    "flainternational": "floridainternational",
    "louisianalafayette": "louisiana",
    "louisianamonroeulm": "ulmonroe",
    "miamiflorida": "miami",
    "miamiohio": "miamioh",
    "mississippi": "olemiss",
    "samhoustonstate": "samhouston",
    "sanjosestate": "sanjosstate",
    "southerncalifornia": "usctrojans",
}

RECEIPT_NAME = "gameview_build_receipt.json"


@dataclass(frozen=True)
class ScheduleGame:
    """One normalized schedule row (the builder's per-game input)."""

    season: int
    week: int
    kickoff_utc: datetime
    home_team_raw: str
    away_team_raw: str
    home_team_norm: str
    away_team_norm: str
    source_uid: Optional[Any] = None
    # Passed through verbatim as raw_sources["schedule_row"]; the frontend
    # optionally reads game_no / rotation / gsis from it.
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SagarinEntry:
    """One team's Sagarin snapshot row for the join."""

    team: str
    pr: Optional[float] = None
    pr_rank: Optional[int] = None
    sos: Optional[float] = None
    sos_rank: Optional[int] = None
    hfa: Optional[float] = None


@dataclass
class GameviewBuild:
    records: List[Dict[str, Any]]
    receipt: Dict[str, Any]


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _i(value: Any) -> Optional[int]:
    result = _f(value)
    return None if result is None else int(round(result))


def _round2(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 2)


def sagarin_token(league: League, name: str) -> str:
    """League merge key for a Sagarin team name, with the CFB override table."""
    base = league.merge_key(name)
    if league.code == "cfb":
        return SAGARIN_TOKEN_OVERRIDES.get(base, base)
    return base


def sagarin_map_from_rows(league: League, rows: Iterable[Mapping[str, Any]]) -> Dict[str, SagarinEntry]:
    """Build the Sagarin join mapping from snapshot rows (team_norm keyed)."""
    mapping: Dict[str, SagarinEntry] = {}
    for row in rows:
        name = row.get("team_norm") or row.get("team")
        if not name:
            continue
        token = sagarin_token(league, str(name))
        if not token:
            continue
        mapping[token] = SagarinEntry(
            team=str(name),
            pr=_round2(_f(row.get("pr"))),
            pr_rank=_i(row.get("pr_rank") if row.get("pr_rank") is not None else row.get("rank")),
            sos=_round2(_f(row.get("sos"))),
            sos_rank=_i(row.get("sos_rank")),
            hfa=_round2(_f(row.get("hfa"))),
        )
    return mapping


def derive_favored_fields(
    spread_home_relative: Optional[float],
) -> Tuple[Optional[str], Optional[float]]:
    """Favored side and favored-team spread from the home-relative line.

    Convention (frozen): negative home-relative spread means the home team is
    favored; a pick-em (0) counts as HOME favored (legacy NFL tie-break).
    """
    if spread_home_relative is None:
        return None, None
    s_home = metrics.team_centric_spread(spread_home_relative, "HOME")
    favored = "AWAY" if s_home > 0 else "HOME"
    return favored, metrics.team_centric_spread(spread_home_relative, favored)


def derive_rating_fields(
    home_pr: Optional[float],
    away_pr: Optional[float],
    hfa: Optional[float],
    spread_home_relative: Optional[float],
) -> Dict[str, Any]:
    """The single rating-vs-odds computation for both leagues (bug 7 fix).

    Formula (the ``common.metrics`` trio, matching the legacy NFL builder):

    * ``rating_diff = (home_pr + hfa) - away_pr``           (home-centric)
    * ``rating_vs_odds = rating_diff + spread_home_relative`` (home-centric)
    * ``rating_diff_favored_team = ±rating_diff`` flipped to the favorite
    * ``rating_vs_odds_favored_team = rating_diff_favored_team +
      spread_favored_team``

    Invariant: ``rating_vs_odds_favored_team == rating_vs_odds`` when the home
    team is favored and ``== -rating_vs_odds`` when the away team is favored.
    """
    favored_side, spread_favored = derive_favored_fields(spread_home_relative)
    out: Dict[str, Any] = {
        "favored_side": favored_side,
        "spread_favored_team": _round2(spread_favored),
        "rating_diff": None,
        "rating_vs_odds": None,
        "rating_diff_favored_team": None,
        "rating_vs_odds_favored_team": None,
    }
    if home_pr is None or away_pr is None:
        return out
    rdiff = metrics.rating_diff(home_pr, away_pr, hfa if hfa is not None else 0.0)
    out["rating_diff"] = _round2(rdiff)
    if spread_home_relative is None or favored_side is None:
        return out
    s_home = metrics.team_centric_spread(spread_home_relative, "HOME")
    out["rating_vs_odds"] = _round2(metrics.rating_vs_odds(rdiff, s_home))
    rdiff_favored = rdiff if favored_side == "HOME" else -rdiff
    out["rating_diff_favored_team"] = _round2(rdiff_favored)
    out["rating_vs_odds_favored_team"] = _round2(
        metrics.rating_vs_odds(rdiff_favored, spread_favored)
    )
    return out


def _kickoff_iso(kickoff_utc: datetime) -> str:
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
    return kickoff_utc.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _apply_team_stats(record: Dict[str, Any], prefix: str, stats: Optional[TeamStats]) -> None:
    if stats is None:
        return
    for name in _STAT_VALUE_FIELDS:
        value = getattr(stats, name)
        record[f"{prefix}_{name}"] = _round2(value)
    for name in _STAT_RANK_FIELDS:
        record[f"{prefix}_{name}"] = getattr(stats, name)
    record[f"{prefix}_su"] = stats.su
    record[f"{prefix}_ats"] = stats.ats


def _sagarin_payload(entry: SagarinEntry) -> Dict[str, Any]:
    # Frontend contract: raw_sources.sagarin_row_{home,away} must expose
    # .team and .hfa (web/js/game_metrics.js, week_view.js).
    return {
        "team": entry.team,
        "pr": entry.pr,
        "pr_rank": entry.pr_rank,
        "sos": entry.sos,
        "sos_rank": entry.sos_rank,
        "hfa": entry.hfa,
    }


def gameview_sort_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    """The one games_week row ordering: kickoff first, game_key as tiebreak.

    Exposed (not inlined) so :mod:`football_statfinder.storage.export` can
    reproduce the exact same ordering for DB-sourced rows instead of
    duplicating the sort — WP-D byte-parity requirement.
    """
    return (record.get("kickoff_iso_utc") or "", record.get("game_key") or "")


def order_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Sort arbitrary games_week records into the frozen output order."""
    return sorted((dict(rec) for rec in records), key=gameview_sort_key)


def build_gameview(
    league: League,
    season: int,
    week: int,
    *,
    schedule: Sequence[ScheduleGame],
    team_stats: Mapping[str, TeamStats],
    sagarin: Mapping[str, SagarinEntry],
    odds: Mapping[str, Mapping[str, Any]],
    fbs_only: Optional[bool] = None,
) -> GameviewBuild:
    """Assemble the week's games_week records (no I/O).

    ``team_stats`` and ``sagarin`` are keyed by the league merge key (use
    :func:`sagarin_map_from_rows` for the latter); ``odds`` is keyed by
    game_key with the promoted/pinned odds line shape
    (``spread_home_relative``, ``total``, ``moneyline_home``/``_away``,
    ``odds_source``, ``is_closing``, ``snapshot_at``).

    ``fbs_only`` defaults to True for CFB: games where either team is missing
    from ``team_stats`` are dropped (legacy CFB's implicit FBS filter).
    """
    if fbs_only is None:
        fbs_only = league.code == "cfb"

    records: List[Dict[str, Any]] = []
    skipped_non_fbs = 0
    joined_stats = joined_odds = joined_sagarin = 0
    missing_odds: List[str] = []
    missing_sagarin: List[str] = []
    missing_stats: List[str] = []

    for game in schedule:
        home_token = league.merge_key(game.home_team_norm)
        away_token = league.merge_key(game.away_team_norm)
        if fbs_only and (home_token not in team_stats or away_token not in team_stats):
            skipped_non_fbs += 1
            continue

        game_key = build_game_key(league, game.kickoff_utc, game.home_team_norm, game.away_team_norm)
        record: Dict[str, Any] = {name: None for name in FROZEN_RECORD_FIELDS}
        record.update(
            {
                "season": int(game.season),
                "week": int(game.week),
                "kickoff_iso_utc": _kickoff_iso(game.kickoff_utc),
                "game_key": game_key,
                "source_uid": game.source_uid,
                "home_team_raw": game.home_team_raw,
                "home_team_norm": game.home_team_norm,
                "away_team_raw": game.away_team_raw,
                "away_team_norm": game.away_team_norm,
                "is_closing": False,
            }
        )

        home_stats = team_stats.get(home_token)
        away_stats = team_stats.get(away_token)
        if home_stats and away_stats:
            joined_stats += 1
        else:
            missing_stats.append(game_key)
        _apply_team_stats(record, "home", home_stats)
        _apply_team_stats(record, "away", away_stats)

        odds_payload = odds.get(game_key)
        spread_home_relative: Optional[float] = None
        if odds_payload:
            joined_odds += 1
            spread_home_relative = _round2(_f(odds_payload.get("spread_home_relative")))
            record.update(
                {
                    "spread_home_relative": spread_home_relative,
                    "total": _round2(_f(odds_payload.get("total"))),
                    "moneyline_home": _i(odds_payload.get("moneyline_home")),
                    "moneyline_away": _i(odds_payload.get("moneyline_away")),
                    "odds_source": odds_payload.get("odds_source"),
                    "is_closing": bool(odds_payload.get("is_closing")),
                    "snapshot_at": odds_payload.get("snapshot_at"),
                }
            )
        else:
            missing_odds.append(game_key)

        home_sag = sagarin.get(home_token)
        away_sag = sagarin.get(away_token)
        hfa: Optional[float] = None
        if home_sag and away_sag:
            joined_sagarin += 1
            if home_sag.hfa is not None:
                hfa = home_sag.hfa
            elif away_sag.hfa is not None:
                hfa = away_sag.hfa
            else:
                hfa = 0.0
            record.update(
                {
                    "home_pr": home_sag.pr,
                    "home_pr_rank": home_sag.pr_rank,
                    "home_sos": home_sag.sos,
                    "home_sos_rank": home_sag.sos_rank,
                    "away_pr": away_sag.pr,
                    "away_pr_rank": away_sag.pr_rank,
                    "away_sos": away_sag.sos,
                    "away_sos_rank": away_sag.sos_rank,
                    "hfa": hfa,
                }
            )
        else:
            missing_sagarin.append(game_key)

        record.update(
            derive_rating_fields(
                record["home_pr"], record["away_pr"], hfa, spread_home_relative
            )
        )

        raw_sources: Dict[str, Any] = {"schedule_row": dict(game.extra)}
        raw_sources["odds_row"] = dict(odds_payload) if odds_payload else None
        raw_sources["sagarin_row_home"] = _sagarin_payload(home_sag) if home_sag else None
        raw_sources["sagarin_row_away"] = _sagarin_payload(away_sag) if away_sag else None
        if home_stats is not None:
            raw_sources["league_metrics_home"] = dict(home_stats.raw)
        if away_stats is not None:
            raw_sources["league_metrics_away"] = dict(away_stats.raw)
        record["raw_sources"] = raw_sources

        records.append(record)

    # Deterministic output ordering: kickoff first, game_key as tiebreak.
    records.sort(key=gameview_sort_key)

    total = len(records)
    receipt: Dict[str, Any] = {
        "league": league.code,
        "season": int(season),
        "week": int(week),
        "schedule_games": len(schedule),
        "output_rows": total,
        "skipped_non_fbs": skipped_non_fbs,
        "joined_stats_rows": joined_stats,
        "joined_odds_rows": joined_odds,
        "joined_sagarin_rows": joined_sagarin,
        "coverage_stats": (joined_stats / total) if total else 0.0,
        "coverage_odds": (joined_odds / total) if total else 0.0,
        "coverage_sagarin": (joined_sagarin / total) if total else 0.0,
        "samples": {
            "missing_odds": missing_odds[:10],
            "missing_sagarin": missing_sagarin[:10],
            "missing_stats": missing_stats[:10],
        },
    }
    logger.info(
        "gameview %s %s wk%s: rows=%d stats=%d odds=%d sagarin=%d skipped_non_fbs=%d",
        league.display,
        season,
        week,
        total,
        joined_stats,
        joined_odds,
        joined_sagarin,
        skipped_non_fbs,
    )
    return GameviewBuild(records=records, receipt=receipt)


def games_week_paths(
    league: League, season: int, week: int, *, out_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """(jsonl_path, csv_path) for a week, honoring ``out_dir`` test overrides."""
    if out_dir is None:
        return (
            paths.games_week_jsonl(league.code, season, week),
            paths.games_week_csv(league.code, season, week),
        )
    return (
        out_dir / f"games_week_{int(season)}_{int(week)}.jsonl",
        out_dir / f"games_week_{int(season)}_{int(week)}.csv",
    )


def _csv_cell_ready(record: Mapping[str, Any]) -> Dict[str, Any]:
    """A CSV-safe copy: nested values (``raw_sources``) become canonical JSON text.

    A CSV cell can't hold a nested dict/list natively; pandas falls back to
    ``str(value)`` — Python's ``repr()`` — whose key order mirrors whatever
    order the dict object happens to carry. A freshly built record's
    ``raw_sources`` has one (insertion) order; the same record read back from
    storage (``storage/store.py`` payloads are canonical JSON with
    ``sort_keys=True``) has another (alphabetical) — same data, different
    ``repr()`` bytes, which broke the WP-D export byte-parity gate for no
    functional reason. Encoding nested values with the package's one
    canonical JSON dump (matching every other JSON writer here) makes the
    cell text depend only on the data, not on which code path produced the
    dict.
    """
    out = dict(record)
    for key, value in list(out.items()):
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return out


def write_games_week_files(
    league: League,
    season: int,
    week: int,
    records: Sequence[Mapping[str, Any]],
    *,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Atomically write games_week JSONL + CSV for already-built records.

    The one games_week serializer: :func:`write_gameview` calls this for a
    fresh build, and :mod:`football_statfinder.storage.export` calls it again
    for DB-sourced payloads (already put in :func:`gameview_sort_key` order)
    so there is exactly one JSONL/CSV writer to keep byte-identical, never two
    copies to drift apart.
    """
    jsonl_path, csv_path = games_week_paths(league, season, week, out_dir=out_dir)
    write_atomic_jsonl(jsonl_path, records)
    frame = pd.DataFrame([_csv_cell_ready(rec) for rec in records], columns=list(FROZEN_RECORD_FIELDS))
    write_atomic_csv(csv_path, frame)
    return jsonl_path, csv_path


def write_gameview(
    league: League,
    season: int,
    week: int,
    build: GameviewBuild,
    *,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Atomically write games_week JSONL + CSV (and the build receipt).

    Defaults to the unified ``paths`` layout; ``out_dir`` exists for tests.
    """
    if out_dir is None:
        receipt_path = paths.week_dir(league.code, season, week) / RECEIPT_NAME
    else:
        receipt_path = out_dir / RECEIPT_NAME
    jsonl_path, csv_path = write_games_week_files(league, season, week, build.records, out_dir=out_dir)
    write_atomic_json(receipt_path, build.receipt)
    logger.info("wrote gameview: %s (%d rows)", jsonl_path, len(build.records))
    return jsonl_path, csv_path


__all__ = [
    "FROZEN_RECORD_FIELDS",
    "GameviewBuild",
    "RECEIPT_NAME",
    "SAGARIN_TOKEN_OVERRIDES",
    "SagarinEntry",
    "ScheduleGame",
    "build_gameview",
    "derive_favored_fields",
    "derive_rating_fields",
    "gameview_sort_key",
    "games_week_paths",
    "order_records",
    "sagarin_map_from_rows",
    "sagarin_token",
    "write_games_week_files",
    "write_gameview",
]

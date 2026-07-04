"""Per-game ``game_schedules/{game_key}.json`` sidecar writer.

Ports and merges the season-1 twins ``src/build_team_timelines.py`` (NFL) and
``src/build_team_timelines_cfb.py`` (CFB). The sidecar schema is a frozen
frontend contract (``web/game_view.js`` reads it): top-level
``game_key, home_ytd, away_ytd, home_prev, away_prev``; each timeline entry
carries exactly :data:`SIDECAR_ENTRY_FIELDS`.

Deliberate changes from legacy behavior:

* League-parameterized: one implementation; the schedule master and Sagarin
  master are injected as DataFrames (no fetching, no ``ensure_weeks_present``
  re-ingest inside the sidecar stage — the legacy CFB builder refetched the
  whole CFBD schedule here).
* Sagarin enrichment uses the CFB nearest-week fallback for BOTH leagues
  (the NFL twin only joined exact (season, week) rows and then patched ranks;
  nearest-week is strictly more complete and was CFB production behavior).
* The rank fallback (when a master row has values but no rank) now uses
  ``common.metrics.dense_rank`` — the only ranking implementation — instead
  of the legacy order-dependent sequential rank that split ties arbitrarily.
* The legacy NFL "rank maps from the league_metrics CSV" tier is dropped:
  that CSV has never contained PR/SoS columns, so the tier could not fire in
  production (verified dead code).
* Sidecar files are written atomically via ``common.io_atomic`` (legacy used
  bare ``write_text``).
* ``ats`` and ``to_margin`` entries start ``None`` exactly as in season 1
  (the score/ATS backfill stage fills them later).
* Sidecar JSON is serialized with ``sort_keys=True`` (Phase 2 WP-D decision):
  every JSON writer in this package now shares one canonical encoding
  (``json.dumps(payload, ensure_ascii=False, sort_keys=True)``, matching
  ``storage/store.py``'s DB payload encoding and ``common/io_atomic.py``'s
  JSONL writer) so a DB-sourced export can byte-match the flat file. Key
  order is not a frontend contract (``web/`` parses JSON, never reads raw
  bytes).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .. import paths
from ..common import metrics
from ..common.io_atomic import write_atomic_json, write_atomic_text
from ..leagues import League

logger = logging.getLogger(__name__)

SIDECAR_TOP_LEVEL_FIELDS: Tuple[str, ...] = (
    "game_key",
    "home_ytd",
    "away_ytd",
    "home_prev",
    "away_prev",
)

# Frozen entry schema (order preserved from the legacy builders).
SIDECAR_ENTRY_FIELDS: Tuple[str, ...] = (
    "season",
    "week",
    "date",
    "opp",
    "site",
    "pf",
    "pa",
    "result",
    "ats",
    "to_margin",
    "pr",
    "pr_rank",
    "sos",
    "sos_rank",
    "opp_pr",
    "opp_pr_rank",
    "opp_sos",
    "opp_sos_rank",
)

RECEIPT_NAME = "sidecars_receipt.json"


class SidecarError(RuntimeError):
    """Raised when a strict sidecar build cannot cover every game."""


def write_sidecar_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one ``game_schedules/{game_key}.json`` sidecar.

    The one sidecar-payload writer: ``build_sidecars`` calls this per game and
    :mod:`football_statfinder.storage.export` calls it again for DB-sourced
    payloads, so the encoding (``sort_keys=True``, matching every other JSON
    writer in the package — see module docstring) never has two copies to
    drift apart.
    """
    write_atomic_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _f(value: Any) -> Optional[float]:
    if not _is_finite(value):
        return None
    return float(value)


def _i(value: Any) -> Optional[int]:
    result = _f(value)
    return None if result is None else int(round(result))


def _date_of(iso: Optional[str]) -> Optional[str]:
    return iso.split("T")[0] if isinstance(iso, str) and "T" in iso else None


def _result(pf: Optional[int], pa: Optional[int]) -> Optional[str]:
    if pf is None or pa is None:
        return None
    if pf > pa:
        return "W"
    if pf < pa:
        return "L"
    return "T"


def _schedule_rows(league: League, schedule_master: pd.DataFrame, seasons: Sequence[int]) -> List[dict]:
    """Normalize the schedule master into keyed plain dicts for slicing."""
    frame = schedule_master.copy()
    if "league" in frame.columns:
        frame = frame[frame["league"].astype(str).str.upper() == league.display]
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].astype(str).str.upper() == "REG"]
    frame["season"] = pd.to_numeric(frame.get("season"), errors="coerce")
    frame["week"] = pd.to_numeric(frame.get("week"), errors="coerce")
    frame = frame[frame["season"].isin(list(seasons))]
    rows: List[dict] = []
    for row in frame.to_dict(orient="records"):
        home_norm = row.get("home_team_norm")
        away_norm = row.get("away_team_norm")
        if not home_norm or not away_norm:
            continue
        rows.append(
            {
                "season": int(row["season"]) if pd.notna(row["season"]) else None,
                "week": int(row["week"]) if pd.notna(row["week"]) else None,
                "kickoff_iso_utc": str(row.get("kickoff_iso_utc") or ""),
                "home_team_norm": str(home_norm),
                "away_team_norm": str(away_norm),
                "home_key": league.merge_key(str(home_norm)),
                "away_key": league.merge_key(str(away_norm)),
                "home_score": _i(row.get("home_score")),
                "away_score": _i(row.get("away_score")),
            }
        )
    return rows


def _sagarin_lookup(
    league: League, sagarin_master: pd.DataFrame
) -> Tuple[Dict[Tuple[int, int, str], Dict[str, Any]], Dict[Tuple[int, str], List[int]]]:
    """(season, week, team_key) -> ratings, plus per-(season, team) week index."""
    frame = sagarin_master.copy()
    if "league" in frame.columns:
        frame = frame[frame["league"].astype(str).str.upper() == league.display]
    if frame.empty:
        return {}, {}
    frame["season"] = pd.to_numeric(frame.get("season"), errors="coerce")
    frame["week"] = pd.to_numeric(frame.get("week"), errors="coerce")
    rank_col = "pr_rank" if "pr_rank" in frame.columns else ("rank" if "rank" in frame.columns else None)

    lookup: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    week_index: Dict[Tuple[int, str], List[int]] = {}
    for row in frame.to_dict(orient="records"):
        if pd.isna(row.get("season")) or pd.isna(row.get("week")):
            continue
        team_key = league.merge_key(str(row.get("team_norm") or ""))
        if not team_key:
            continue
        season = int(row["season"])
        week = int(row["week"])
        lookup[(season, week, team_key)] = {
            "pr": _f(row.get("pr")),
            "pr_rank": _i(row.get(rank_col)) if rank_col else None,
            "sos": _f(row.get("sos")),
            "sos_rank": _i(row.get("sos_rank")),
        }
        week_index.setdefault((season, team_key), []).append(week)
    for pair, weeks in week_index.items():
        week_index[pair] = sorted(set(weeks))
    return lookup, week_index


def _nearest_week_entry(
    season: int,
    week: int,
    team_key: str,
    lookup: Mapping[Tuple[int, int, str], Dict[str, Any]],
    week_index: Mapping[Tuple[int, str], List[int]],
) -> Optional[Dict[str, Any]]:
    """Exact (season, week) hit, else nearest prior week, else nearest later."""
    weeks = week_index.get((season, team_key))
    if not weeks:
        return None
    direct = lookup.get((season, week, team_key))
    if direct:
        return direct
    earlier = [w for w in weeks if w <= week]
    if earlier:
        return lookup.get((season, max(earlier), team_key))
    later = [w for w in weeks if w >= week]
    if not later:
        return None
    return lookup.get((season, min(later), team_key))


def _week_rank_maps(
    season: int,
    week: int,
    lookup: Mapping[Tuple[int, int, str], Dict[str, Any]],
    cache: Dict[Tuple[int, int], Tuple[Dict[str, int], Dict[str, int]]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-week (pr_ranks, sos_ranks): master ranks, else dense_rank on values."""
    key = (season, week)
    if key in cache:
        return cache[key]
    pr_values: Dict[str, float] = {}
    pr_ranks: Dict[str, int] = {}
    sos_values: Dict[str, float] = {}
    sos_ranks: Dict[str, int] = {}
    for (s, w, team_key), payload in lookup.items():
        if s != season or w != week:
            continue
        if payload.get("pr") is not None:
            pr_values[team_key] = payload["pr"]
        if payload.get("pr_rank") is not None:
            pr_ranks[team_key] = payload["pr_rank"]
        if payload.get("sos") is not None:
            sos_values[team_key] = payload["sos"]
        if payload.get("sos_rank") is not None:
            sos_ranks[team_key] = payload["sos_rank"]
    if not pr_ranks and pr_values:
        ranked = metrics.dense_rank(pd.Series(pr_values), higher_is_better=True)
        pr_ranks = {k: int(v) for k, v in ranked.to_dict().items()}
    if not sos_ranks and sos_values:
        ranked = metrics.dense_rank(pd.Series(sos_values), higher_is_better=True)
        sos_ranks = {k: int(v) for k, v in ranked.to_dict().items()}
    cache[key] = (pr_ranks, sos_ranks)
    return cache[key]


def _team_timeline(
    schedule_rows: Sequence[dict],
    team_key: str,
    season: int,
    *,
    cutoff_iso: Optional[str] = None,
    cutoff_week: Optional[int] = None,
    lookup: Mapping[Tuple[int, int, str], Dict[str, Any]],
    week_index: Mapping[Tuple[int, str], List[int]],
    rank_cache: Dict[Tuple[int, int], Tuple[Dict[str, int], Dict[str, int]]],
    coverage: Dict[str, int],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for row in schedule_rows:
        if row["season"] != season:
            continue
        if row["home_key"] != team_key and row["away_key"] != team_key:
            continue
        if cutoff_iso is not None and not (row["kickoff_iso_utc"] < cutoff_iso):
            continue
        if cutoff_week is not None and row["week"] is not None and not (row["week"] < cutoff_week):
            continue
        is_home = row["home_key"] == team_key
        opp_norm = row["away_team_norm"] if is_home else row["home_team_norm"]
        opp_key = row["away_key"] if is_home else row["home_key"]
        pf = row["home_score"] if is_home else row["away_score"]
        pa = row["away_score"] if is_home else row["home_score"]
        entry: Dict[str, Any] = {name: None for name in SIDECAR_ENTRY_FIELDS}
        entry.update(
            {
                "season": row["season"],
                "week": row["week"],
                "date": _date_of(row["kickoff_iso_utc"]),
                "opp": opp_norm,
                "site": "H" if is_home else "A",
                "pf": pf,
                "pa": pa,
                "result": _result(pf, pa),
            }
        )
        week = row["week"]
        if week is not None:
            coverage["rows_considered"] += 1
            team_entry = _nearest_week_entry(season, week, team_key, lookup, week_index)
            if team_entry:
                entry["pr"] = team_entry["pr"]
                entry["pr_rank"] = team_entry["pr_rank"]
                entry["sos"] = team_entry["sos"]
                entry["sos_rank"] = team_entry["sos_rank"]
                coverage["rows_enriched"] += 1
            opp_entry = _nearest_week_entry(season, week, opp_key, lookup, week_index) if opp_key else None
            if opp_entry:
                entry["opp_pr"] = opp_entry["pr"]
                entry["opp_pr_rank"] = opp_entry["pr_rank"]
                entry["opp_sos"] = opp_entry["sos"]
                entry["opp_sos_rank"] = opp_entry["sos_rank"]
                coverage["opp_rows_enriched"] += 1
            pr_ranks, sos_ranks = _week_rank_maps(season, week, lookup, rank_cache)
            if entry["pr_rank"] is None and team_key in pr_ranks:
                entry["pr_rank"] = pr_ranks[team_key]
            if entry["sos_rank"] is None and team_key in sos_ranks:
                entry["sos_rank"] = sos_ranks[team_key]
            if entry["opp_pr_rank"] is None and opp_key and opp_key in pr_ranks:
                entry["opp_pr_rank"] = pr_ranks[opp_key]
            if entry["opp_sos_rank"] is None and opp_key and opp_key in sos_ranks:
                entry["opp_sos_rank"] = sos_ranks[opp_key]
        entries.append(entry)
    entries.sort(
        key=lambda e: (
            e["season"] if e["season"] is not None else -1,
            e["week"] if e["week"] is not None else -1,
            e["date"] or "",
        )
    )
    return entries


def build_sidecars(
    league: League,
    season: int,
    week: int,
    *,
    games: Sequence[Mapping[str, Any]],
    schedule_master: pd.DataFrame,
    sagarin_master: pd.DataFrame,
    out_dir: Optional[Path] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Write one sidecar JSON per games_week record; return the receipt.

    ``games`` is the week's games_week records (needs ``game_key``,
    ``kickoff_iso_utc``, ``week``, ``home_team_norm``, ``away_team_norm``).
    ``schedule_master`` and ``sagarin_master`` are the master tables
    (DataFrames, already loaded — this stage does no fetching).

    With ``strict`` (default), missing schedule joins raise
    :class:`SidecarError` after the receipt is written, mirroring the legacy
    hard-fail policy of both orchestrators.
    """
    week_dir = out_dir if out_dir is not None else paths.week_dir(league.code, season, week)
    side_dir = week_dir / "game_schedules"

    seasons = (int(season), int(season) - 1)
    schedule_rows = _schedule_rows(league, schedule_master, seasons)
    if not schedule_rows:
        raise SidecarError(f"schedule master has no {league.display} REG rows for seasons {seasons}")
    lookup, week_index = _sagarin_lookup(league, sagarin_master)
    rank_cache: Dict[Tuple[int, int], Tuple[Dict[str, int], Dict[str, int]]] = {}
    coverage = {"rows_considered": 0, "rows_enriched": 0, "opp_rows_enriched": 0}

    schedule_pairs = {
        (row["season"], row["week"], row["home_key"], row["away_key"]) for row in schedule_rows
    }

    written = 0
    join_issues: List[Dict[str, Any]] = []
    payloads: Dict[str, Dict[str, Any]] = {}
    for game in games:
        game_key = game.get("game_key")
        kickoff_iso = game.get("kickoff_iso_utc")
        game_week = int(game["week"]) if game.get("week") is not None else None
        home_norm = game.get("home_team_norm") or game.get("home_team_raw") or ""
        away_norm = game.get("away_team_norm") or game.get("away_team_raw") or ""
        home_key = league.merge_key(str(home_norm))
        away_key = league.merge_key(str(away_norm))

        if (int(season), game_week, home_key, away_key) not in schedule_pairs:
            join_issues.append({"game_key": game_key, "reason": "missing_schedule"})
            continue

        def timeline(team_key: str, timeline_season: int, *, ytd: bool) -> List[Dict[str, Any]]:
            return _team_timeline(
                schedule_rows,
                team_key,
                timeline_season,
                cutoff_iso=str(kickoff_iso) if ytd and kickoff_iso else None,
                cutoff_week=game_week if ytd else None,
                lookup=lookup,
                week_index=week_index,
                rank_cache=rank_cache,
                coverage=coverage,
            )

        payload = {
            "game_key": game_key,
            "home_ytd": timeline(home_key, int(season), ytd=True),
            "away_ytd": timeline(away_key, int(season), ytd=True),
            "home_prev": timeline(home_key, int(season) - 1, ytd=False),
            "away_prev": timeline(away_key, int(season) - 1, ytd=False),
        }
        side_path = side_dir / f"{game_key}.json"
        write_sidecar_json(side_path, payload)
        payloads[game_key] = payload
        written += 1

    rows_considered = coverage["rows_considered"]
    receipt: Dict[str, Any] = {
        "league": league.code,
        "season": int(season),
        "week": int(week),
        "games_total": len(games),
        "sidecars_written": written,
        "join_issues": join_issues,
        "sagarin_rows_considered": rows_considered,
        "sagarin_rows_enriched": coverage["rows_enriched"],
        "sagarin_opp_rows_enriched": coverage["opp_rows_enriched"],
        "sagarin_coverage_fraction": (
            coverage["rows_enriched"] / rows_considered if rows_considered else 1.0
        ),
    }
    write_atomic_json(week_dir / RECEIPT_NAME, receipt)
    logger.info(
        "sidecars %s %s wk%s: written=%d/%d sagarin_coverage=%.2f",
        league.display,
        season,
        week,
        written,
        len(games),
        receipt["sagarin_coverage_fraction"],
    )

    # In-memory payloads keyed by game_key, for callers that dual-write to
    # storage without re-reading the just-written sidecar files (Phase 2:
    # football_statfinder/storage/). Not persisted to the receipt file itself.
    receipt["payloads"] = payloads

    if strict and written != len(games):
        missing = [issue["game_key"] for issue in join_issues]
        raise SidecarError(
            f"sidecar build incomplete: wrote {written}/{len(games)} (missing schedule joins: {missing})"
        )
    return receipt


__all__ = [
    "RECEIPT_NAME",
    "SIDECAR_ENTRY_FIELDS",
    "SIDECAR_TOP_LEVEL_FIELDS",
    "SidecarError",
    "build_sidecars",
    "write_sidecar_json",
]

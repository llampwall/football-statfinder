"""One league-refresh orchestrator for both leagues.

Replaces the three season-1 orchestrators (`refresh_week_data.py`,
`refresh_week_data_nfl.py`, `refresh_week_data_cfb.py`) with a single
in-process stage runner. Design (REBUILD.md section 6 error policy):

* Each stage either contributes its artifact and a StageResult or raises;
  the failure is recorded in the run summary and aborts the remaining stages
  (downstream stages depend on upstream artifacts).
* Per-league isolation lives in :func:`refresh_all`: one league failing never
  blocks the other.
* No subprocesses, no stdout grepping: the machine-readable outcome is the
  ``RunSummary`` JSON status file, plus the legacy one-line NOTIFY contract
  emitted by the CLI.
* Stage order: schedule -> sagarin -> stats -> odds staging (ingest+pin) ->
  gameview -> odds promotion (then derived-field recompute) -> sidecars ->
  scores backfill -> ATS. Promotion rewrites the week rows with staged odds,
  so the orchestrator recomputes the rating-vs-odds fields afterwards with the
  same single formula the builder uses (the season-1 system left them stale).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import paths
from .common.current_week import get_current_week
from .common.jsonl import read_jsonl
from .config import Settings, get_settings
from .leagues import League
from .pipeline import ats as ats_mod
from .pipeline import backfill as backfill_mod
from .pipeline import gameview as gameview_mod
from .pipeline import odds_ingest, odds_pin, odds_promote
from .pipeline import sidecars as sidecars_mod
from .run_summary import RunSummary, StageResult
from .sources import schedule as schedule_mod
from .sources import schedule_master as master_mod
from .sources import sagarin as sagarin_mod
from .sources import stats as stats_mod
from .storage import db as db_mod
from .storage import store as store_mod

logger = logging.getLogger(__name__)

# A stage body returns (value, counts, notes).
StageBody = Callable[[], Tuple[Any, Dict[str, int], List[str]]]


def _int_counts(obj: Any) -> Dict[str, int]:
    """Numeric fields of a dataclass/mapping, for StageResult counts."""
    data = asdict(obj) if not isinstance(obj, Mapping) else dict(obj)
    return {k: int(v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _run_stage(summary: RunSummary, name: str, body: StageBody) -> Any:
    start = time.monotonic()
    logger.info("stage %s: start", name)
    try:
        value, counts, notes = body()
    except Exception as exc:
        duration = time.monotonic() - start
        summary.add(StageResult(name=name, ok=False, duration_s=duration, error=f"{type(exc).__name__}: {exc}"))
        logger.error("stage %s: FAILED after %.1fs: %s", name, duration, exc)
        raise
    duration = time.monotonic() - start
    summary.add(StageResult(name=name, ok=True, duration_s=duration, counts=counts, notes=notes))
    logger.info("stage %s: ok in %.1fs %s", name, duration, counts)
    return value


def _schedule_games_for_week(league: League, week_df: pd.DataFrame) -> List[gameview_mod.ScheduleGame]:
    games: List[gameview_mod.ScheduleGame] = []
    for row in week_df.to_dict(orient="records"):
        kickoff = pd.to_datetime(row.get("kickoff_iso_utc"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            logger.warning("schedule row without parseable kickoff skipped: %s @ %s",
                           row.get("away_team_norm"), row.get("home_team_norm"))
            continue
        games.append(
            gameview_mod.ScheduleGame(
                season=int(row["season"]),
                week=int(row["week"]),
                kickoff_utc=kickoff.to_pydatetime(),
                home_team_raw=str(row.get("home_team_raw") or row.get("home_team_norm") or ""),
                away_team_raw=str(row.get("away_team_raw") or row.get("away_team_norm") or ""),
                home_team_norm=str(row.get("home_team_norm") or ""),
                away_team_norm=str(row.get("away_team_norm") or ""),
                source_uid=row.get("source_uid"),
                extra=row,
            )
        )
    return games


def _recompute_rating_fields(league: League, season: int, week: int) -> int:
    """Recompute derived rating fields after promotion rewrote the spreads.

    Uses the same single formula as the builder (bug 7 stays fixed); returns
    the number of rows whose derived fields changed.
    """
    json_path = paths.games_week_jsonl(league.code, season, week)
    result = read_jsonl(json_path)
    if not result.rows:
        return 0
    changed = 0
    for row in result.rows:
        derived = gameview_mod.derive_rating_fields(
            row.get("home_pr"), row.get("away_pr"), row.get("hfa"), row.get("spread_home_relative")
        )
        if any(row.get(k) != v for k, v in derived.items()):
            row.update(derived)
            changed += 1
    if changed:
        odds_promote.write_week_outputs(league, result.rows, season, week)
    return changed


def _rebuild_sidecars(league: League, season: int, week: int) -> Dict[str, Any]:
    rows = read_jsonl(paths.games_week_jsonl(league.code, season, week)).rows
    schedule_master = master_mod.load_master(league)
    sagarin_csv = paths.sagarin_master_csv(league.code)
    sagarin_master = pd.read_csv(sagarin_csv) if sagarin_csv.exists() else pd.DataFrame()
    return sidecars_mod.build_sidecars(
        league, season, week,
        games=rows,
        schedule_master=schedule_master,
        sagarin_master=sagarin_master,
    )


def refresh_league(
    league: League,
    settings: Optional[Settings] = None,
    *,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> RunSummary:
    """Run the full weekly refresh for one league; returns the run summary.

    Raises on the first failed stage after recording it (callers that need
    isolation use :func:`refresh_all`). The summary JSON is written either way.
    """
    cfg = settings if settings is not None else get_settings()
    summary = RunSummary(league=league.display)
    build: Optional[gameview_mod.GameviewBuild] = None
    storage_conn: Optional[sqlite3.Connection] = None
    logger.info("%s refresh starting; %s", league.display, cfg.banner())

    try:
        if season is None or week is None:
            season, week, _ = get_current_week(league, settings=cfg)
        summary.season, summary.week = int(season), int(week)
        logger.info("%s current week: season=%s week=%s", league.display, season, week)

        if cfg.storage.enable:
            storage_conn = db_mod.connect(cfg.storage.db_path)

        def _record_games_now() -> None:
            """Re-record the current week's games rows from the on-disk JSONL.

            Used after stages (promotion, recompute, ATS) that rewrite the
            games_week file in place without returning the updated rows —
            reading the file they just wrote is the only faithful way to get
            the post-stage state back into memory.
            """
            if storage_conn is None:
                return
            rows = read_jsonl(paths.games_week_jsonl(league.code, season, week)).rows
            store_mod.record_games(storage_conn, league, int(season), int(week), rows)

        def schedule_stage() -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
            refresh_master = league.code == "nfl" or cfg.cfbd_refresh
            inserted = updated = 0
            if refresh_master:
                df = schedule_mod.fetch_schedule(league, int(season), cfg)
                inserted, updated = master_mod.upsert_schedule_rows(league, df)
            master = master_mod.load_master(league)
            week_df = master[
                (pd.to_numeric(master["season"], errors="coerce") == int(season))
                & (pd.to_numeric(master["week"], errors="coerce") == int(week))
            ].copy()
            if week_df.empty:
                raise RuntimeError(f"schedule produced 0 rows for {league.display} {season} week {week}")
            schedule_mod.write_schedule_artifact(league, int(season), int(week), week_df)
            if storage_conn is not None:
                store_mod.record_schedule(storage_conn, league, week_df)
            return week_df, {"rows": len(week_df), "inserted": inserted, "updated": updated}, []

        week_df = _run_stage(summary, "schedule", schedule_stage)

        def sagarin_stage() -> Tuple[Dict[str, gameview_mod.SagarinEntry], Dict[str, int], List[str]]:
            result = sagarin_mod.run_sagarin_staging(league, int(season), int(week), cfg)
            rows = read_jsonl(result.weekly_jsonl).rows if result.weekly_jsonl else []
            mapping = gameview_mod.sagarin_map_from_rows(league, rows)
            if storage_conn is not None:
                store_mod.record_sagarin(storage_conn, league, int(season), int(week), rows)
            counts = {"teams_parsed": result.teams_parsed, "teams_selected": result.teams_selected,
                      "mapped": len(mapping)}
            notes = [f"page_stamp={result.page_stamp}"] if result.page_stamp else []
            return mapping, counts, notes

        sagarin_map = _run_stage(summary, "sagarin", sagarin_stage)

        def stats_stage() -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
            provider = stats_mod.get_stats_provider(league, cfg)
            rows = provider.league_metrics_rows(int(season), int(week))
            stats_mod.write_league_metrics_csv(league, int(season), int(week), rows)
            if storage_conn is not None:
                store_mod.record_metrics(storage_conn, league, int(season), int(week), rows)
            team_stats = stats_mod.team_stats_from_metrics_rows(league, rows)
            return team_stats, {"teams": len(team_stats)}, []

        team_stats = _run_stage(summary, "stats", stats_stage)

        if cfg.odds.staging_enable:
            def odds_staging_stage() -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
                ingest = odds_ingest.ingest_raw(league, cfg)
                schedule_games = odds_pin.load_schedule_master(league)
                pin = odds_pin.pin_to_schedule(league, ingest.get("records") or [], schedule_games, cfg)
                if storage_conn is not None:
                    store_mod.record_pinned_odds(storage_conn, league, pin.get("pinned_records") or [])
                counts = {"raw": len(ingest.get("records") or []), **_int_counts(pin.get("counts") or {})}
                notes = [str(x) for x in (pin.get("examples_unmatched") or [])[:3]]
                return pin, counts, notes

            _run_stage(summary, "odds_staging", odds_staging_stage)
        else:
            logger.info("odds staging disabled by config")

        def gameview_stage() -> Tuple[gameview_mod.GameviewBuild, Dict[str, int], List[str]]:
            games = _schedule_games_for_week(league, week_df)
            build = gameview_mod.build_gameview(
                league, int(season), int(week),
                schedule=games,
                team_stats=team_stats,
                sagarin=sagarin_map,
                odds={},
            )
            gameview_mod.write_gameview(league, int(season), int(week), build)
            if storage_conn is not None:
                store_mod.record_games(storage_conn, league, int(season), int(week), build.records)
            return build, _int_counts(build.receipt.get("counts") or build.receipt), []

        build = _run_stage(summary, "gameview", gameview_stage)

        if cfg.odds.promotion_enable:
            def promotion_stage() -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
                promo = odds_promote.promote_week(league, int(season), int(week), cfg)
                recomputed = _recompute_rating_fields(league, int(season), int(week))
                _record_games_now()
                counts = {**_int_counts(promo), "rating_fields_recomputed": recomputed}
                notes = []
                if promo.get("skipped_reason"):
                    notes.append(str(promo["skipped_reason"]))
                if promo.get("coverage_ok") is False:
                    notes.append("coverage gate FAILED (see receipt)")
                return promo, counts, notes

            _run_stage(summary, "odds_promotion", promotion_stage)
        else:
            logger.info("odds promotion disabled by config")

        def sidecars_stage() -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
            receipt = _rebuild_sidecars(league, int(season), int(week))
            if storage_conn is not None:
                payloads = list((receipt.get("payloads") or {}).values())
                store_mod.record_sidecars(storage_conn, league, int(season), int(week), payloads)
            return receipt, _int_counts(receipt.get("counts") or receipt), []

        _run_stage(summary, "sidecars", sidecars_stage)

        if cfg.backfill.scores_enable and int(week) > 1:
            def backfill_stage() -> Tuple[backfill_mod.BackfillResult, Dict[str, int], List[str]]:
                result = backfill_mod.backfill_scores(
                    league, int(season), int(week), cfg,
                    promote_week=(lambda _rows, s, w: odds_promote.promote_week(league, s, w, cfg))
                    if cfg.backfill.promote_prev else None,
                )
                for changed_week in result.changed_weeks:
                    changed_receipt = _rebuild_sidecars(league, int(season), int(changed_week))
                    if storage_conn is not None:
                        changed_rows = read_jsonl(
                            paths.games_week_jsonl(league.code, season, changed_week)
                        ).rows
                        store_mod.record_games(
                            storage_conn, league, int(season), int(changed_week), changed_rows
                        )
                        changed_payloads = list((changed_receipt.get("payloads") or {}).values())
                        store_mod.record_sidecars(
                            storage_conn, league, int(season), int(changed_week), changed_payloads
                        )
                counts = _int_counts(result)
                counts["changed_weeks"] = len(result.changed_weeks)
                return result, counts, []

            _run_stage(summary, "scores_backfill", backfill_stage)
        else:
            logger.info("scores backfill skipped (disabled or week 1)")

        if cfg.backfill.ats_enable:
            def ats_stage() -> Tuple[Any, Dict[str, int], List[str]]:
                result = ats_mod.run_ats(league, int(season), int(week), cfg)
                _record_games_now()
                return result, _int_counts(result), []

            _run_stage(summary, "ats", ats_stage)
        else:
            logger.info("ATS disabled by config")

        return summary
    finally:
        if storage_conn is not None:
            storage_conn.close()
        summary.write()
        rows = len(build.records) if build is not None else 0
        status = "ok" if summary.ok else "FAILED"
        logger.info("%s refresh %s: %d stages, %d rows", league.display, status, len(summary.stages), rows)


def refresh_all(
    leagues: Sequence[League],
    settings: Optional[Settings] = None,
    *,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> Dict[str, RunSummary]:
    """Refresh each league in isolation; one failure never blocks the next."""
    cfg = settings if settings is not None else get_settings()
    summaries: Dict[str, RunSummary] = {}
    for league in leagues:
        try:
            summaries[league.display] = refresh_league(league, cfg, season=season, week=week)
        except Exception:
            logger.exception("%s refresh failed; continuing with remaining leagues", league.display)
            summary_path = paths.OUT_ROOT / "state" / f"run_summary_{league.code}.json"
            logger.info("%s failure detail in %s", league.display, summary_path)
            failed = RunSummary(league=league.display)
            failed.add(StageResult(name="run", ok=False, duration_s=0.0, error="see run summary JSON"))
            summaries[league.display] = failed
    return summaries


__all__ = ["refresh_all", "refresh_league"]

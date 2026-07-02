"""One upsert writer for the per-league schedule master CSV.

Replaces the season-1 twins ``src/schedule_master.py`` and
``src/schedule_master_cfb.py`` (same upsert-keyed idea, diverged constants).
The master is what :mod:`football_statfinder.common.current_week` reads, so
the ``season``, ``week`` and ``kickoff_iso_utc`` columns are load-bearing.

Legacy behavior deliberately changed:

* Paths always come from ``paths.schedule_master_csv(league.code)`` — several
  legacy versions used CWD-relative ``out/master`` (masters silently split
  when run from another directory) and created the directory at import time.
* Writes go through the atomic CSV writer instead of a bare ``to_csv``.
* No env reads: the CFB refresh toggle is ``settings.cfbd_refresh``.

Semantics preserved: KEEP/KEY column sets, score-present and source-priority
tie-breaking on duplicate keys, and the post-upsert duplicate-key hard fail.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import pandas as pd

from .. import paths
from ..common.io_atomic import write_atomic_csv
from ..config import Settings
from ..leagues import League
from .schedule import FetchRaw, fetch_schedule

logger = logging.getLogger(__name__)

KEEP = [
    "league",
    "season",
    "week",
    "game_type",
    "kickoff_iso_utc",
    "home_team_norm",
    "away_team_norm",
    "home_team_key",
    "away_team_key",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "source",
]

KEY = [
    "league",
    "season",
    "week",
    "game_type",
    "home_team_key",
    "away_team_key",
    "kickoff_iso_utc",
]

# Provider rows beat seed rows on key collisions (legacy tie-break, unified).
_SOURCE_PRIORITY = {"seed": 0, "nflverse": 1, "cfbd": 1}


def _coerce_types(league: League, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce master columns to their canonical dtypes (legacy semantics)."""
    df = df.copy()
    for col in KEEP:
        if col not in df.columns:
            df[col] = None
    df["league"] = df["league"].fillna(league.display).astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    kickoff = pd.to_datetime(df["kickoff_iso_utc"], errors="coerce", utc=True)
    df["kickoff_iso_utc"] = kickoff.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace("+0000", "+00:00")
    for col in ("home_team_norm", "away_team_norm"):
        values = df[col].where(pd.notna(df[col]), None)
        df[col] = values.apply(lambda v: v.strip() if isinstance(v, str) else v)
    if df["home_team_key"].isna().all():
        df["home_team_key"] = df["home_team_norm"].map(league.merge_key)
    if df["away_team_key"].isna().all():
        df["away_team_key"] = df["away_team_norm"].map(league.merge_key)
    df["home_team_key"] = df["home_team_key"].astype(str)
    df["away_team_key"] = df["away_team_key"].astype(str)
    for col in ("home_score", "away_score", "spread_line", "total_line"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = df["source"].fillna("seed").astype(str)
    df["game_type"] = df["game_type"].fillna("REG").astype(str)
    return df[KEEP + [c for c in df.columns if c not in KEEP]]


def load_master(league: League) -> pd.DataFrame:
    """Load the league's schedule master (empty frame when absent)."""
    master_csv = paths.schedule_master_csv(league.code)
    if master_csv.exists():
        return _coerce_types(league, pd.read_csv(master_csv))
    return pd.DataFrame(columns=KEEP)


def upsert_schedule_rows(league: League, df: pd.DataFrame) -> Tuple[int, int]:
    """Upsert normalized schedule rows into the master; returns (before, after).

    Keyed on :data:`KEY`; on collisions, rows with scores beat rows without,
    then provider sources beat seed rows, keeping the legacy tie-break.
    """
    df = _coerce_types(league, df)[KEEP]
    master_df = load_master(league)
    if not master_df.empty:
        master_df = master_df[KEEP]
    if df.empty:
        count = len(master_df)
        return count, count
    combined = pd.concat([master_df, df], ignore_index=True) if not master_df.empty else df.copy()
    combined["score_present"] = (
        combined["home_score"].notna() & combined["away_score"].notna()
    ).astype(int)
    combined["source_priority"] = combined["source"].map(_SOURCE_PRIORITY).fillna(0).astype(int)
    combined = (
        combined.sort_values(KEY + ["score_present", "source_priority"], kind="mergesort")
        .drop_duplicates(KEY, keep="last")
        .reset_index(drop=True)
    )
    combined = combined.drop(columns=["score_present", "source_priority"])
    before = len(master_df)
    after = len(combined)
    dups = combined[combined.duplicated(KEY, keep=False)]
    if not dups.empty:
        logger.error("duplicate keys after upsert: %s", dups.head().to_dict(orient="records"))
        raise RuntimeError(f"{league.display} schedule master still has duplicate keys")
    write_atomic_csv(paths.schedule_master_csv(league.code), combined)
    logger.info(
        "%s schedule master upsert: before=%d after=%d delta=%d",
        league.display,
        before,
        after,
        after - before,
    )
    return before, after


def ensure_seasons_present(
    league: League,
    seasons: Iterable[int],
    settings: Settings,
    *,
    fetch_raw: Optional[FetchRaw] = None,
) -> Tuple[int, int]:
    """Fetch each season's schedule and upsert it into the master.

    Port of the legacy ``ensure_weeks_present`` twins. The CFB refresh toggle
    now comes from ``settings.cfbd_refresh`` instead of the ``CFBD_REFRESH``
    env flag; fetch failures propagate (bug 8 fix).
    """
    seasons = list(seasons)
    if league.code == "cfb" and not settings.cfbd_refresh:
        logger.info("CFBD refresh disabled via settings; skipping schedule fetch")
        count = len(load_master(league))
        return count, count
    frames = []
    for season in seasons:
        season_df = fetch_schedule(league, season, settings, fetch_raw=fetch_raw)
        if season_df.empty:
            logger.warning("%s schedule season=%s returned 0 rows", league.display, season)
            continue
        frames.append(season_df)
    if not frames:
        logger.warning("%s: no schedule rows fetched for seasons=%s", league.display, seasons)
        count = len(load_master(league))
        return count, count
    combined = pd.concat(frames, ignore_index=True)
    before, after = upsert_schedule_rows(league, combined)
    logger.info(
        "%s ensure_seasons_present seasons=%s -> delta %d", league.display, seasons, after - before
    )
    return before, after


__all__ = ["KEEP", "KEY", "ensure_seasons_present", "load_master", "upsert_schedule_rows"]

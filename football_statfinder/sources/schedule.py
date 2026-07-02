"""League-parameterized schedule ingest.

One module replacing the season-1 twins ``src/fetch_games.py`` (NFL: nflverse
games.csv download) and ``src/fetch_games_cfb.py`` (CFB: CFBD API). Both
leagues normalize into one schedule schema (:data:`SCHEDULE_COLUMNS`).

Legacy behavior deliberately changed:

* CFB requires ``settings.require("cfbd_api_key")`` before its default fetch
  and lets fetch exceptions propagate — a missing key or a network failure is
  the stage's error, never a silent empty DataFrame (REBUILD.md bug 8).
* The week's schedule persists to the week dir via
  :func:`write_schedule_artifact` so one fetch per run is reused instead of
  every downstream stage refetching live (the churn behind bugs 12/13; the
  legacy CFB "schedule ingest" stage was a hardcoded no-op).
* ``game_type`` keeps the provider's real value instead of the legacy NFL
  master's hardcoded ``"REG"``.
* Rows without a parseable kickoff get ``game_key=None`` (logged) instead of
  the legacy CFB ``00000000_0000`` placeholder stamp.

No env reads, no import-time side effects; fetchers are injectable so tests
never touch the network.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests

from .. import paths
from ..common.game_key import build_game_key
from ..common.io_atomic import write_atomic_csv
from ..config import Settings
from ..leagues import League

logger = logging.getLogger(__name__)

try:  # pragma: no cover - fallback for Python <3.9 mirrors legacy guard
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

NFLVERSE_GAMES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
)
CFBD_BASE_URL = "https://api.collegefootballdata.com"

TZ_NY = ZoneInfo("America/New_York") if ZoneInfo else None

# Unified schedule schema. Superset of the master KEEP columns so the frame
# feeds schedule_master.upsert_schedule_rows unchanged.
SCHEDULE_COLUMNS = [
    "league",
    "season",
    "week",
    "game_type",
    "kickoff_iso_utc",
    "home_team_raw",
    "away_team_raw",
    "home_team_norm",
    "away_team_norm",
    "home_team_key",
    "away_team_key",
    "neutral_site",
    "venue",
    "conference_game",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "game_key",
    "source",
]

FetchRaw = Callable[[int], Any]


# --- default (network) fetchers ------------------------------------------------


def _download_nflverse_games(season: int) -> pd.DataFrame:
    """Download the nflverse games table (all seasons; filtered later)."""
    logger.info("downloading nflverse games.csv for season=%s", season)
    resp = requests.get(NFLVERSE_GAMES_URL, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def _download_cfbd_games(season: int, api_key: str) -> list:
    """Fetch the CFBD /games payload for a season (regular season)."""
    logger.info("fetching CFBD /games for season=%s", season)
    resp = requests.get(
        f"{CFBD_BASE_URL}/games",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"year": season, "seasonType": "regular"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"CFBD /games returned non-list payload for season {season}")
    return data


# --- normalization helpers ------------------------------------------------------


def _column(df: pd.DataFrame, options: list) -> pd.Series:
    for name in options:
        if name in df.columns:
            return df[name]
    return pd.Series([None] * len(df), index=df.index)


def _parse_nfl_kickoff(row: pd.Series) -> Optional[datetime]:
    """Kickoff to UTC from nflverse columns (verbatim legacy semantics)."""
    if "start_time_utc" in row and pd.notna(row["start_time_utc"]):
        try:
            return pd.to_datetime(row["start_time_utc"], utc=True).to_pydatetime()
        except Exception:
            pass
    gameday = next(
        (str(row[c]) for c in ("gameday", "gamedate", "game_date") if c in row and pd.notna(row[c])),
        None,
    )
    gametime = next(
        (str(row[c]) for c in ("gametime", "game_time_eastern", "start_time") if c in row and pd.notna(row[c])),
        None,
    )
    if not gameday or not gametime:
        return None
    try:
        dt_naive = pd.to_datetime(f"{gameday} {gametime}").to_pydatetime()
    except Exception:
        return None
    if TZ_NY is None:  # pragma: no cover - zoneinfo always present on 3.10+
        return dt_naive.replace(tzinfo=timezone.utc)
    return dt_naive.replace(tzinfo=TZ_NY).astimezone(timezone.utc)


def _parse_cfb_kickoff(row: pd.Series) -> Optional[datetime]:
    """Kickoff to UTC from the CFBD start-date ISO string."""
    start = None
    for key in ("start_date", "startDate", "kickoff", "start_time"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            start = value
            break
    if not start:
        return None
    dt = pd.to_datetime(start, utc=True, errors="coerce")
    if dt is None or pd.isna(dt):
        return None
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return None


def _kickoff_iso(dt: Optional[datetime]) -> Optional[str]:
    if not isinstance(dt, datetime):
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _finalize(work: pd.DataFrame, league: League) -> pd.DataFrame:
    """Shared tail: norm names, merge keys, game_key, column order, sort."""
    work["home_team_norm"] = work["home_team_raw"].astype(str).map(league.normalize_display)
    work["away_team_norm"] = work["away_team_raw"].astype(str).map(league.normalize_display)
    work["home_team_key"] = work["home_team_norm"].map(league.merge_key)
    work["away_team_key"] = work["away_team_norm"].map(league.merge_key)
    work["kickoff_iso_utc"] = work["kickoff_dt_utc"].map(_kickoff_iso)

    def _row_game_key(row: pd.Series) -> Optional[str]:
        dt = row["kickoff_dt_utc"]
        if not isinstance(dt, datetime):
            return None
        return build_game_key(league, dt, row["home_team_norm"], row["away_team_norm"])

    work["game_key"] = work.apply(_row_game_key, axis=1) if not work.empty else None
    missing_kickoff = int(work["kickoff_iso_utc"].isna().sum()) if not work.empty else 0
    if missing_kickoff:
        logger.warning(
            "%s schedule: %d row(s) without parseable kickoff (game_key=None)",
            league.display,
            missing_kickoff,
        )
    work["league"] = league.display
    for col in SCHEDULE_COLUMNS:
        if col not in work.columns:
            work[col] = None
    out = work[SCHEDULE_COLUMNS].copy()
    out = out.sort_values(
        ["season", "week", "kickoff_iso_utc", "home_team_key"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _normalize_nfl(raw: pd.DataFrame, season: int, league: League) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    work = raw.copy()
    work["season"] = pd.to_numeric(work.get("season"), errors="coerce")
    work = work[work["season"] == season].copy()
    if work.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    work["week"] = pd.to_numeric(_column(work, ["week"]), errors="coerce")
    work["game_type"] = (
        _column(work, ["game_type"]).fillna("REG").astype(str) if "game_type" in work.columns else "REG"
    )
    work["home_team_raw"] = _column(work, ["home_team"]).astype(str)
    work["away_team_raw"] = _column(work, ["away_team"]).astype(str)
    work["kickoff_dt_utc"] = work.apply(_parse_nfl_kickoff, axis=1)
    location = _column(work, ["location"]).astype(str).str.strip().str.lower()
    work["neutral_site"] = location.eq("neutral")
    work["venue"] = _column(work, ["stadium", "venue"])
    work["conference_game"] = None
    for col in ("home_score", "away_score", "spread_line", "total_line"):
        work[col] = pd.to_numeric(_column(work, [col]), errors="coerce")
    work["source"] = "nflverse"
    return _finalize(work, league)


def _normalize_cfb(raw: pd.DataFrame, season: int, league: League) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    work = raw.copy()
    work["season"] = pd.to_numeric(_column(work, ["season", "year"]), errors="coerce").fillna(season)
    work = work[work["season"] == season].copy()
    if work.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    work["week"] = pd.to_numeric(_column(work, ["week"]), errors="coerce")
    season_type = _column(work, ["seasonType", "season_type"]).astype(str).str.lower()
    work["game_type"] = season_type.map(lambda s: "POST" if s == "postseason" else "REG")
    work["home_team_raw"] = _column(work, ["home_team", "homeTeam"]).astype(str)
    work["away_team_raw"] = _column(work, ["away_team", "awayTeam"]).astype(str)
    work["kickoff_dt_utc"] = work.apply(_parse_cfb_kickoff, axis=1)
    work["neutral_site"] = _column(work, ["neutral_site", "neutralSite"]).fillna(False).astype(bool)
    work["venue"] = _column(work, ["venue", "venue_name"])
    work["conference_game"] = _column(work, ["conference_game", "conferenceGame"]).fillna(False).astype(bool)
    work["home_score"] = pd.to_numeric(_column(work, ["home_points", "homePoints", "home_score"]), errors="coerce")
    work["away_score"] = pd.to_numeric(_column(work, ["away_points", "awayPoints", "away_score"]), errors="coerce")
    work["spread_line"] = pd.to_numeric(_column(work, ["spread", "spread_line"]), errors="coerce")
    work["total_line"] = pd.to_numeric(_column(work, ["overUnder", "over_under", "total_line"]), errors="coerce")
    work["source"] = "cfbd"
    return _finalize(work, league)


# --- public API -------------------------------------------------------------------


def fetch_schedule(
    league: League,
    season: int,
    settings: Settings,
    *,
    fetch_raw: Optional[FetchRaw] = None,
) -> pd.DataFrame:
    """Fetch and normalize the season schedule for a league.

    ``fetch_raw(season)`` overrides the network fetcher (tests, caches). Its
    return value is league-shaped: an nflverse-style DataFrame/records for
    NFL, a CFBD ``/games`` list of dicts for CFB. When it is omitted, CFB
    fails loud on a missing ``CFBD_API_KEY`` before any network call, and all
    fetch exceptions propagate (bug 8 fix).
    """
    if league.code == "nfl":
        raw = fetch_raw(season) if fetch_raw is not None else _download_nflverse_games(season)
        frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        df = _normalize_nfl(frame, season, league)
    elif league.code == "cfb":
        if fetch_raw is not None:
            records = fetch_raw(season)
        else:
            settings.require("cfbd_api_key")
            records = _download_cfbd_games(season, settings.cfbd_api_key)  # type: ignore[arg-type]
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records or [])
        df = _normalize_cfb(frame, season, league)
    else:
        raise ValueError(f"no schedule source for league {league.code!r}")
    logger.info("%s schedule season=%s: %d row(s)", league.display, season, len(df))
    return df


def schedule_artifact_path(league: League, season: int, week: int) -> Path:
    """Path of the persisted per-week schedule artifact."""
    return paths.week_dir(league.code, season, week) / f"schedule_{int(season)}_wk{int(week)}.csv"


def write_schedule_artifact(league: League, season: int, week: int, df: pd.DataFrame) -> Path:
    """Persist the week's slice of a season schedule to the week dir.

    Fetch once per run, write here, and have downstream stages read the
    artifact back via :func:`read_schedule_artifact` instead of refetching
    live (fixes the churn behind bugs 12/13).
    """
    if df.empty:
        weekly = df.copy()
    else:
        weekly = df[
            (pd.to_numeric(df["season"], errors="coerce") == int(season))
            & (pd.to_numeric(df["week"], errors="coerce") == int(week))
        ].copy()
    target = schedule_artifact_path(league, season, week)
    write_atomic_csv(target, weekly)
    logger.info(
        "%s schedule artifact season=%s week=%s: wrote %d row(s) -> %s",
        league.display,
        season,
        week,
        len(weekly),
        target,
    )
    return target


def read_schedule_artifact(league: League, season: int, week: int) -> Optional[pd.DataFrame]:
    """Read back the persisted week schedule; None when not yet written."""
    target = schedule_artifact_path(league, season, week)
    if not target.exists():
        return None
    return pd.read_csv(target)


__all__ = [
    "CFBD_BASE_URL",
    "NFLVERSE_GAMES_URL",
    "SCHEDULE_COLUMNS",
    "fetch_schedule",
    "read_schedule_artifact",
    "schedule_artifact_path",
    "write_schedule_artifact",
]

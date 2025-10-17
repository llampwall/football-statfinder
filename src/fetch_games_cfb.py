"""CFB schedule helpers mirroring the NFL fetcher API."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.cfb_source import fetch_cfbd_games
from src.common.io_utils import ensure_out_dir, read_env
from src.common.team_names_cfb import normalize_team_name_cfb, team_merge_key_cfb

RAW_COLUMNS = [
    "season",
    "week",
    "id",
    "home_team",
    "away_team",
    "start_date",
    "venue",
    "conference_game",
]

SCHEDULE_COLUMNS = RAW_COLUMNS + [
    "kickoff_dt_utc",
    "kickoff_iso_utc",
    "home_team_norm",
    "away_team_norm",
    "home_team_key",
    "away_team_key",
    "game_key",
]


def _empty_raw_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def _ensure_week_dirs(season: int, week: int) -> Path:
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "game_schedules").mkdir(parents=True, exist_ok=True)
    return out_dir


def _column(series: pd.DataFrame, options: list[str]) -> pd.Series:
    for name in options:
        if name in series.columns:
            return series[name]
    return pd.Series([None] * len(series), index=series.index)


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_games(season: int) -> pd.DataFrame:
    """Fetch raw CFB games for the given season (regular season only)."""
    env = read_env(["CFBD_API_KEY"])
    api_key = env.get("CFBD_API_KEY")
    if not api_key:
        return _empty_raw_df()
    try:
        records = fetch_cfbd_games(season, None, api_key) or []
    except Exception:
        return _empty_raw_df()
    if not records:
        return _empty_raw_df()
    df = pd.DataFrame(records)
    df["season"] = season
    if "week" not in df.columns:
        df["week"] = pd.to_numeric(df.get("week"), errors="coerce")
    for column, aliases in {
        "start_date": ["start_date", "startDate", "kickoff", "start_time"],
        "home_team": ["home_team", "homeTeam"],
        "away_team": ["away_team", "awayTeam"],
        "venue": ["venue", "venue_name"],
        "conference_game": ["conference_game", "conferenceGame"],
    }.items():
        if column not in df.columns:
            df[column] = _column(df, aliases)
    return df


def filter_week_reg(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Return only regular-season rows for the requested week."""
    if games.empty:
        return games.copy()
    df = games.copy()
    df = df[(pd.to_numeric(df.get("season"), errors="coerce") == season) & (pd.to_numeric(df.get("week"), errors="coerce") == week)]
    return df


def parse_kickoff_utc(row: pd.Series) -> Optional[datetime]:
    """Parse kickoff ISO string into UTC datetime."""
    start = row.get("start_date") or row.get("startDate") or row.get("kickoff") or row.get("start_time")
    if not start:
        return None
    try:
        dt = pd.to_datetime(start, utc=True, errors="coerce")
    except Exception:
        dt = None
    if dt is None or pd.isna(dt):
        return None
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return None


def home_relative_spread(_row: pd.Series) -> Optional[float]:
    """CFB stub currently has no odds lines."""
    return None


def total_from_schedule(_row: pd.Series) -> Optional[float]:
    """CFB stub currently has no totals."""
    return None


def get_schedule_df(season: int) -> pd.DataFrame:
    """Return normalized schedule with kickoff and team display columns."""
    raw = load_games(season)
    if raw.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    df = raw.copy()
    if "kickoff_dt_utc" not in df.columns:
        df["kickoff_dt_utc"] = df.apply(parse_kickoff_utc, axis=1)
    df["kickoff_iso_utc"] = df["kickoff_dt_utc"].apply(
        lambda dt: dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(dt, datetime)
        else None
    )
    df["home_team"] = _column(df, ["home_team", "homeTeam"]).astype(str)
    df["away_team"] = _column(df, ["away_team", "awayTeam"]).astype(str)
    df["home_team_norm"] = df["home_team"].map(normalize_team_name_cfb)
    df["away_team_norm"] = df["away_team"].map(normalize_team_name_cfb)
    df["home_team_key"] = df["home_team_norm"].map(team_merge_key_cfb)
    df["away_team_key"] = df["away_team_norm"].map(team_merge_key_cfb)
    venue_series = _column(df, ["venue", "venue_name"]).astype(str)
    df["venue"] = venue_series
    conf_series = _column(df, ["conference_game", "conferenceGame"]).fillna(False)
    df["conference_game"] = conf_series.astype(bool)

    def build_key(row: pd.Series) -> str:
        kickoff = row.get("kickoff_iso_utc") or ""
        home = row.get("home_team_norm") or row.get("home_team") or ""
        away = row.get("away_team_norm") or row.get("away_team") or ""
        seq = row.get("id") or ""
        if kickoff:
            key = f"{kickoff.replace(':', '').replace('-', '')}_{_sanitize(away)}_{_sanitize(home)}"
        elif seq:
            key = f"{seq}_{_sanitize(away)}_{_sanitize(home)}"
        else:
            key = f"{_sanitize(away)}_{_sanitize(home)}"
        return key

    df["game_key"] = df.apply(build_key, axis=1)
    return df[SCHEDULE_COLUMNS].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFB schedule (parity helper).")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"CFB schedule fetch: season={args.season} week={args.week}")
    out_dir = _ensure_week_dirs(args.season, args.week)

    games = load_games(args.season)
    if games.empty:
        print("CFB schedule unavailable (empty result).")
        return 0

    weekly = filter_week_reg(games, args.season, args.week)
    schedule_df = get_schedule_df(args.season)
    weekly_df = schedule_df[schedule_df["week"] == args.week]
    rows = len(weekly_df)
    print(f"PASS: normalized schedule rows={rows}")

    if os.environ.get("CFB_WRITE_DEBUG_SCHEDULE") == "1":
        debug_path = out_dir / "_schedule_norm.csv"
        weekly_df.to_csv(debug_path, index=False)
        print(f"DEBUG: wrote {debug_path}")
    return 0


__all__ = [
    "load_games",
    "filter_week_reg",
    "parse_kickoff_utc",
    "home_relative_spread",
    "total_from_schedule",
    "get_schedule_df",
]


if __name__ == "__main__":
    raise SystemExit(main())

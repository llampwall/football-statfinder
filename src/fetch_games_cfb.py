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


def _team_token(name: Optional[str]) -> str:
    token = (name or "").lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = token.strip("_")
    return token or "unknown"


def load_games(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """Fetch CFB games and return a normalized schedule DataFrame."""
    env = read_env(["CFBD_API_KEY"])
    api_key = env.get("CFBD_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    try:
        records = fetch_cfbd_games(season, week, api_key) or []
    except Exception:
        records = []
    if not records:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    raw = pd.DataFrame(records)
    raw["season"] = season
    for column, aliases in {
        "start_date": ["start_date", "startDate", "kickoff", "start_time"],
        "home_team": ["home_team", "homeTeam"],
        "away_team": ["away_team", "awayTeam"],
        "venue": ["venue", "venue_name"],
        "conference_game": ["conference_game", "conferenceGame"],
    }.items():
        if column not in raw.columns:
            raw[column] = _column(raw, aliases)
    normalized = _normalize_schedule_df(raw, season)
    if week is not None:
        normalized = normalized[pd.to_numeric(normalized.get("week"), errors="coerce") == week]
    return normalized.reset_index(drop=True)


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


def _build_game_key(row: pd.Series) -> str:
    dt = row.get("kickoff_dt_utc")
    if isinstance(dt, datetime):
        dt = dt.astimezone(timezone.utc)
        yyyymmdd = dt.strftime("%Y%m%d")
        hhmm = dt.strftime("%H%M")
    else:
        yyyymmdd = "00000000"
        hhmm = "0000"
    away_token = _team_token(row.get("away_team_norm") or row.get("away_team"))
    home_token = _team_token(row.get("home_team_norm") or row.get("home_team"))
    return f"{yyyymmdd}_{hhmm}_{away_token}_{home_token}"


def _normalize_schedule_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    work = df.copy()
    work["season"] = pd.to_numeric(work.get("season"), errors="coerce").fillna(season).astype(int)
    work["week"] = pd.to_numeric(work.get("week"), errors="coerce")
    if "kickoff_dt_utc" not in work.columns:
        work["kickoff_dt_utc"] = work.apply(parse_kickoff_utc, axis=1)
    work["kickoff_iso_utc"] = work["kickoff_dt_utc"].apply(
        lambda dt: dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(dt, datetime) else None
    )
    work["home_team"] = _column(work, ["home_team", "homeTeam"]).astype(str)
    work["away_team"] = _column(work, ["away_team", "awayTeam"]).astype(str)
    work["home_team_norm"] = work["home_team"].map(normalize_team_name_cfb)
    work["away_team_norm"] = work["away_team"].map(normalize_team_name_cfb)
    work["home_team_key"] = work["home_team_norm"].map(team_merge_key_cfb)
    work["away_team_key"] = work["away_team_norm"].map(team_merge_key_cfb)
    work["venue"] = _column(work, ["venue", "venue_name"]).astype(str)
    work["conference_game"] = _column(work, ["conference_game", "conferenceGame"]).fillna(False).astype(bool)
    work["game_key"] = work.apply(_build_game_key, axis=1)
    return work[SCHEDULE_COLUMNS].copy()


def home_relative_spread(_row: pd.Series) -> Optional[float]:
    """CFB stub currently has no odds lines."""
    return None


def total_from_schedule(_row: pd.Series) -> Optional[float]:
    """CFB stub currently has no totals."""
    return None


def get_schedule_df(season: int) -> pd.DataFrame:
    """Return normalized schedule with kickoff and team display columns."""
    return load_games(season)


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

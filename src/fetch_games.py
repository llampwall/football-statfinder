"""NFL schedule helpers sourced from nflverse releases.

Purpose:
    Download the nflverse games table and expose helpers for downstream modules.
Inputs:
    Target season and week numbers.
Outputs:
    Filtered DataFrames plus per-row helpers for kickoff time and betting lines.
Source(s) of truth:
    https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
Example:
    >>> games = load_games(2025)
    >>> week6 = filter_week_reg(games, 2025, 6)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.common.io_utils import download_csv
from src.common.team_names import normalize_team_display

try:
    from zoneinfo import ZoneInfo  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for Python <3.9
    ZoneInfo = None  # type: ignore


NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TZ_NY = ZoneInfo("America/New_York") if ZoneInfo else None


def load_games(season: int) -> pd.DataFrame:
    """Download nflverse games.csv and return it as a DataFrame."""
    df = download_csv(NFLVERSE_GAMES_URL)
    df = df[df["season"] == season].copy()
    return df


def filter_week_reg(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Return only regular season rows for the requested week."""
    if "game_type" in games.columns:
        mask_type = games["game_type"] == "REG"
    else:
        mask_type = True
    subset = games[
        (games["season"] == season)
        & (games["week"] == week)
        & mask_type
    ].copy()
    subset["kickoff_dt_utc"] = subset.apply(parse_kickoff_utc, axis=1)
    return subset


def parse_kickoff_utc(row: pd.Series) -> Optional[datetime]:
    """Parse kickoff to UTC datetime using nflverse columns."""
    if "start_time_utc" in row and pd.notna(row["start_time_utc"]):
        try:
            dt = pd.to_datetime(row["start_time_utc"], utc=True)
            return dt.to_pydatetime()
        except Exception:
            pass

    gameday = next((str(row[c]) for c in ("gameday", "gamedate", "game_date") if c in row and pd.notna(row[c])), None)
    gametime = next((str(row[c]) for c in ("gametime", "game_time_eastern", "start_time") if c in row and pd.notna(row[c])), None)
    if not gameday or not gametime:
        return None

    ts = f"{gameday} {gametime}"
    try:
        dt_naive = pd.to_datetime(ts).to_pydatetime()
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %I:%M %p"):
            try:
                dt_naive = datetime.strptime(ts, fmt)
                break
            except Exception:
                dt_naive = None
        if dt_naive is None:
            return None

    if TZ_NY is None:
        return dt_naive.replace(tzinfo=timezone.utc)
    return dt_naive.replace(tzinfo=TZ_NY).astimezone(timezone.utc)


def home_relative_spread(row: pd.Series) -> Optional[float]:
    """Best-effort home-relative spread from schedule columns."""
    for column in ("spread_line", "spread", "spread_favorite"):
        if column in row and pd.notna(row[column]):
            try:
                return float(row[column])
            except Exception:
                continue

    fav_key = next((c for c in ("spread_favorite", "favorite", "fav") if c in row), None)
    line_key = next((c for c in ("spread_line", "spread", "line") if c in row), None)
    if fav_key and line_key and pd.notna(row[fav_key]) and pd.notna(row[line_key]):
        try:
            line_val = float(row[line_key])
        except Exception:
            return None
        fav = str(row[fav_key]).strip().lower()
        home = str(row.get("home_team", "")).strip().lower()
        away = str(row.get("away_team", "")).strip().lower()
        if fav in (home, "home"):
            return -abs(line_val)
        if fav in (away, "away"):
            return abs(line_val)
    return None


def total_from_schedule(row: pd.Series) -> Optional[float]:
    """Best-effort total (over/under) from schedule columns."""
    for column in ("total_line", "total", "ou_total"):
        if column in row and pd.notna(row[column]):
            try:
                return float(row[column])
            except Exception:
                continue
    return None


__all__ = [
    "load_games",
    "filter_week_reg",
    "parse_kickoff_utc",
    "home_relative_spread",
    "total_from_schedule",
    "get_schedule_df",
]


def get_schedule_df(season: int) -> pd.DataFrame:
    """Return full schedule for a season with normalized team names and kickoff ISO timestamps."""
    df = load_games(season)
    if "kickoff_dt_utc" not in df.columns:
        df["kickoff_dt_utc"] = df.apply(parse_kickoff_utc, axis=1)
    df["kickoff_iso_utc"] = df["kickoff_dt_utc"].apply(
        lambda dt: dt.replace(microsecond=0).isoformat() if isinstance(dt, datetime) else None
    )
    df["home_team_norm"] = df["home_team"].astype(str).map(normalize_team_display)
    df["away_team_norm"] = df["away_team"].astype(str).map(normalize_team_display)
    return df

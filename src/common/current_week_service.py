"""Current week service for league scheduling (read-only consumers)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, time, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

OUT_ROOT = Path(__file__).resolve().parents[2] / "out"
STATE_PATH = OUT_ROOT / "state" / "current_week.json"

MASTER_PATHS: Dict[str, Path] = {
    "CFB": OUT_ROOT / "master" / "cfb_schedule_master.csv",
    # "NFL": OUT_ROOT / "master" / "nfl_schedule_master.csv",  # placeholder
}

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_force_value(raw: str) -> Optional[Tuple[int, int]]:
    for sep in ("-", ":", ",", "/"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            break
    else:
        parts = raw.split()
        if len(parts) == 2:
            left, right = parts
        else:
            return None
    try:
        season = int(left.strip())
        week = int(right.strip())
        return season, week
    except ValueError:
        return None


def _window_start(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC)
    # Tuesday is weekday=1 (Mon=0).
    days_back = (dt.weekday() - 1) % 7
    start_date = (dt - timedelta(days=days_back)).date()
    start_dt = datetime.combine(start_date, time(0, 0, tzinfo=UTC))
    if dt < start_dt:
        start_dt -= timedelta(days=7)
    return start_dt


def _load_schedule(league: str) -> pd.DataFrame:
    path = MASTER_PATHS.get(league)
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df


def _prepare_windows(df: pd.DataFrame) -> Dict[int, Tuple[datetime, datetime]]:
    if df.empty or "kickoff_iso_utc" not in df.columns:
        return {}
    frame = df.copy()
    frame = frame.dropna(subset=["kickoff_iso_utc", "week", "season"])
    if frame.empty:
        return {}
    frame["kickoff_dt"] = pd.to_datetime(frame["kickoff_iso_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["kickoff_dt"])
    if frame.empty:
        return {}
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["week"])
    windows: Dict[int, Tuple[datetime, datetime]] = {}
    for week_value, group in frame.groupby("week"):
        kickoffs = group["kickoff_dt"].dropna()
        if kickoffs.empty:
            continue
        min_dt = kickoffs.min().to_pydatetime().astimezone(UTC)
        window_start = _window_start(min_dt)
        window_end = window_start + timedelta(days=7)
        windows[int(week_value)] = (window_start, window_end)
    return dict(sorted(windows.items(), key=lambda item: item[0]))


def _select_week(now: datetime, windows: Dict[int, Tuple[datetime, datetime]]) -> Optional[int]:
    if not windows:
        return None
    for week, (start, end) in windows.items():
        if start <= now < end:
            return week
    # If now precedes first window, use earliest week; if after, use latest.
    weeks = sorted(windows.items(), key=lambda item: item[1][0])
    if now < weeks[0][1][0]:
        return weeks[0][0]
    return weeks[-1][0]


def _write_state(league: str, season: int, week: int, computed_at: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict[str, object]] = {}
    if STATE_PATH.exists():
        try:
            existing = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
    entry = {"season": season, "week": week, "computed_at": computed_at}
    existing[league] = entry
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)


def get_current_week(league: str, *, persist: bool = True) -> Tuple[int, int, str]:
    """
    Return (season, week, computed_at_iso_utc) using schedule master only.

    Honors WEEK_FORCE_LEAGUE and WEEK_FORCE overrides. Always UTC.
    """
    league_upper = (league or "").strip().upper()
    if not league_upper:
        raise ValueError("League is required")
    # Environment overrides.
    force_league = os.getenv("WEEK_FORCE_LEAGUE")
    force_value = os.getenv("WEEK_FORCE")
    computed_at = _now_utc()
    if force_league and force_value:
        if force_league.strip().upper() in {league_upper, "ALL", "*"}:
            forced = _parse_force_value(force_value.strip())
            if forced:
                season_val, week_val = forced
                iso_ts = computed_at.isoformat().replace("+00:00", "Z")
                if persist:
                    _write_state(league_upper, season_val, week_val, iso_ts)
                return season_val, week_val, iso_ts

    df = _load_schedule(league_upper)
    if df.empty:
        raise RuntimeError(f"Schedule master missing or empty for league={league_upper}")

    df = df.dropna(subset=["season"])
    if df.empty:
        raise RuntimeError(f"Schedule master lacks season data for league={league_upper}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    if df.empty:
        raise RuntimeError(f"Schedule master invalid season rows for league={league_upper}")

    df["season"] = df["season"].astype(int)
    current_season = int(df["season"].max())
    season_df = df[df["season"] == current_season]
    windows = _prepare_windows(season_df)
    now_utc = computed_at
    selected_week = _select_week(now_utc, windows)
    if selected_week is None:
        # Fallback to earliest week available.
        week_values = sorted(set(int(v) for v in season_df["week"].dropna()))
        if not week_values:
            raise RuntimeError(f"Unable to determine week for league={league_upper}")
        selected_week = week_values[0]

    iso_ts = computed_at.isoformat().replace("+00:00", "Z")
    if persist:
        _write_state(league_upper, current_season, selected_week, iso_ts)
    return current_season, selected_week, iso_ts


__all__ = ["get_current_week"]

"""Current-week resolution from schedule kickoff windows.

Port of season-1 ``src/common/current_week_service.py`` (a keeper per
REBUILD.md section 5), parameterized by League and driven by typed Settings
instead of raw env reads. Semantics preserved:

* One persisted ``{season, week}`` per league, computed only from schedule
  master kickoff times bucketed into Tuesday-00:00-UTC anchored windows;
  every provider's own week label is ignored.
* ``WEEK_FORCE`` / ``WEEK_FORCE_LEAGUE`` override (now via Settings).
* State persists to ``out/state/current_week.json`` keyed by league display
  code, same file and shape the season-1 tree used.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, time, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from ..config import Settings, get_settings
from ..leagues import League
from ..paths import STATE_PATH, schedule_master_csv

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _window_start(dt: datetime) -> datetime:
    """Start of the Tuesday-anchored 7-day window containing ``dt``."""
    dt = dt.astimezone(UTC)
    days_back = (dt.weekday() - 1) % 7  # Tuesday is weekday=1 (Mon=0)
    start_date = (dt - timedelta(days=days_back)).date()
    start_dt = datetime.combine(start_date, time(0, 0, tzinfo=UTC))
    if dt < start_dt:
        start_dt -= timedelta(days=7)
    return start_dt


def _prepare_windows(df: pd.DataFrame) -> Dict[int, Tuple[datetime, datetime]]:
    if df.empty or "kickoff_iso_utc" not in df.columns:
        return {}
    frame = df.dropna(subset=["kickoff_iso_utc", "week", "season"]).copy()
    if frame.empty:
        return {}
    frame["kickoff_dt"] = pd.to_datetime(frame["kickoff_iso_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["kickoff_dt"])
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["week"])
    windows: Dict[int, Tuple[datetime, datetime]] = {}
    for week_value, group in frame.groupby("week"):
        kickoffs = group["kickoff_dt"].dropna()
        if kickoffs.empty:
            continue
        min_dt = kickoffs.min().to_pydatetime().astimezone(UTC)
        start = _window_start(min_dt)
        windows[int(week_value)] = (start, start + timedelta(days=7))
    return dict(sorted(windows.items()))


def _select_week(now: datetime, windows: Dict[int, Tuple[datetime, datetime]]) -> Optional[int]:
    if not windows:
        return None
    for week, (start, end) in windows.items():
        if start <= now < end:
            return week
    ordered = sorted(windows.items(), key=lambda item: item[1][0])
    if now < ordered[0][1][0]:
        return ordered[0][0]
    return ordered[-1][0]


def _write_state(league_display: str, season: int, week: int, computed_at: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict[str, object]] = {}
    if STATE_PATH.exists():
        try:
            loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing[league_display] = {"season": season, "week": week, "computed_at": computed_at}
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)


def get_current_week(
    league: League,
    *,
    settings: Optional[Settings] = None,
    persist: bool = True,
) -> Tuple[int, int, str]:
    """Return ``(season, week, computed_at_iso_utc)`` for a league."""
    cfg = settings if settings is not None else get_settings()
    computed_at = _now_utc()
    iso_ts = computed_at.isoformat().replace("+00:00", "Z")

    force = cfg.week_force
    if force.applies_to(league.display):
        assert force.season is not None and force.week is not None
        if persist:
            _write_state(league.display, force.season, force.week, iso_ts)
        return force.season, force.week, iso_ts

    master = schedule_master_csv(league.code)
    if not master.exists():
        raise RuntimeError(f"schedule master missing for league={league.display}: {master}")
    try:
        df = pd.read_csv(master)
    except Exception as exc:
        raise RuntimeError(f"schedule master unreadable for league={league.display}: {exc}") from exc

    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    df = df.dropna(subset=["season"])
    if df.empty:
        raise RuntimeError(f"schedule master has no valid season rows for league={league.display}")

    current_season = int(df["season"].astype(int).max())
    season_df = df[df["season"] == current_season]
    selected_week = _select_week(computed_at, _prepare_windows(season_df))
    if selected_week is None:
        week_values = sorted({int(v) for v in season_df["week"].dropna()})
        if not week_values:
            raise RuntimeError(f"unable to determine week for league={league.display}")
        selected_week = week_values[0]

    if persist:
        _write_state(league.display, current_season, selected_week, iso_ts)
    return current_season, selected_week, iso_ts


__all__ = ["get_current_week"]

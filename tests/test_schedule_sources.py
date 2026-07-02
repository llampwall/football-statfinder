"""Tests for football_statfinder.sources.schedule and .schedule_master.

No network: every fetch is injected via ``fetch_raw``. All filesystem output
is sandboxed by monkeypatching the path roots in football_statfinder.paths
(the path helpers read those module globals at call time).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from football_statfinder import paths
from football_statfinder.common.current_week import get_current_week
from football_statfinder.config import ConfigError, Settings
from football_statfinder.leagues import CFB, NFL
from football_statfinder.sources import schedule, schedule_master


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    out_root = tmp_path / "out"
    monkeypatch.setattr(paths, "OUT_ROOT", out_root)
    monkeypatch.setattr(paths, "MASTER_ROOT", out_root / "master")
    monkeypatch.setattr(paths, "STAGING_ROOT", out_root / "staging")
    monkeypatch.setattr(paths, "SAGARIN_RAW_ROOT", tmp_path / "data" / "sagarin" / "raw")
    monkeypatch.setattr(paths, "STATE_PATH", out_root / "state" / "current_week.json")
    return tmp_path


def _nflverse_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "KC",
                "away_team": "BUF",
                "location": "Home",
                "home_score": None,
                "away_score": None,
                "spread_line": -3.5,
                "total_line": 47.5,
                "stadium": "Arrowhead",
            },
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-09-13",
                "gametime": "16:25",
                "home_team": "Jaguars",
                "away_team": "MIA",
                "location": "Neutral",
                "home_score": None,
                "away_score": None,
                "spread_line": 2.5,
                "total_line": 41.0,
                "stadium": "Wembley",
            },
            {
                # different season: must be filtered out
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "13:00",
                "home_team": "KC",
                "away_team": "BUF",
                "location": "Home",
                "home_score": 21,
                "away_score": 17,
                "spread_line": -1.0,
                "total_line": 44.0,
                "stadium": "Arrowhead",
            },
        ]
    )


def _cfbd_payload() -> list:
    return [
        {
            "id": 1001,
            "season": 2026,
            "week": 3,
            "seasonType": "regular",
            "startDate": "2026-09-19T19:30:00.000Z",
            "neutralSite": False,
            "homeTeam": "Ohio State",
            "awayTeam": "Texas",
            "homePoints": None,
            "awayPoints": None,
            "venue": "Ohio Stadium",
            "conferenceGame": False,
        },
        {
            "id": 1002,
            "season": 2026,
            "week": 4,
            "seasonType": "regular",
            "startDate": "2026-09-26T23:00:00.000Z",
            "neutralSite": True,
            "homeTeam": "Georgia",
            "awayTeam": "Alabama",
            "homePoints": 24,
            "awayPoints": 20,
            "venue": "Mercedes-Benz Stadium",
            "conferenceGame": True,
        },
    ]


# --- fetch_schedule: NFL -----------------------------------------------------


def test_nfl_schedule_normalizes():
    df = schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=lambda season: _nflverse_frame())
    assert list(df.columns) == schedule.SCHEDULE_COLUMNS
    assert len(df) == 2  # 2025 row filtered out
    assert set(df["season"]) == {2026}
    row = df[df["home_team_norm"] == "Kansas City Chiefs"].iloc[0]
    assert row["away_team_norm"] == "Buffalo Bills"
    # 13:00 America/New_York in September is 17:00 UTC (EDT)
    assert row["kickoff_iso_utc"] == "2026-09-13T17:00:00+00:00"
    assert bool(row["neutral_site"]) is False
    # NFL game_key is home-first (frozen frontend contract)
    assert row["game_key"] == "20260913_1700_kansas_city_chiefs_buffalo_bills"
    assert row["source"] == "nflverse"
    neutral = df[df["home_team_norm"] == "Jacksonville Jaguars"].iloc[0]
    assert bool(neutral["neutral_site"]) is True


def test_nfl_fetch_error_propagates():
    def boom(season):
        raise RuntimeError("nflverse down")

    with pytest.raises(RuntimeError, match="nflverse down"):
        schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=boom)


# --- fetch_schedule: CFB -----------------------------------------------------


def test_cfb_requires_api_key_before_fetching():
    # Bug 8 fix: missing key is a loud startup error, not an empty DataFrame.
    with pytest.raises(ConfigError, match="CFBD_API_KEY"):
        schedule.fetch_schedule(CFB, 2026, Settings())


def test_cfb_schedule_normalizes():
    df = schedule.fetch_schedule(
        CFB, 2026, Settings(cfbd_api_key="k"), fetch_raw=lambda season: _cfbd_payload()
    )
    assert list(df.columns) == schedule.SCHEDULE_COLUMNS
    assert len(df) == 2
    row = df[df["home_team_norm"] == "Ohio State Buckeyes"].iloc[0]
    assert row["away_team_norm"] == "Texas Longhorns"
    assert row["kickoff_iso_utc"] == "2026-09-19T19:30:00+00:00"
    # CFB game_key is away-first (frozen frontend contract)
    assert row["game_key"] == "20260919_1930_texas_longhorns_ohio_state_buckeyes"
    assert bool(row["neutral_site"]) is False
    assert row["source"] == "cfbd"
    scored = df[df["week"] == 4].iloc[0]
    assert bool(scored["neutral_site"]) is True
    assert scored["home_score"] == 24 and scored["away_score"] == 20


def test_cfb_fetch_error_propagates():
    def boom(season):
        raise RuntimeError("cfbd 401")

    with pytest.raises(RuntimeError, match="cfbd 401"):
        schedule.fetch_schedule(CFB, 2026, Settings(cfbd_api_key="k"), fetch_raw=boom)


# --- per-week schedule artifact ----------------------------------------------


def test_write_and_read_schedule_artifact(sandbox):
    df = schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=lambda season: _nflverse_frame())
    target = schedule.write_schedule_artifact(NFL, 2026, 1, df)
    assert target == paths.week_dir("nfl", 2026, 1) / "schedule_2026_wk1.csv"
    assert target.exists()
    back = schedule.read_schedule_artifact(NFL, 2026, 1)
    assert back is not None
    assert len(back) == 2
    assert set(back["week"]) == {1}
    assert schedule.read_schedule_artifact(NFL, 2026, 9) is None


# --- schedule master upsert ----------------------------------------------------


def test_master_upsert_idempotent(sandbox):
    df = schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=lambda season: _nflverse_frame())
    before, after = schedule_master.upsert_schedule_rows(NFL, df)
    assert (before, after) == (0, 2)
    before2, after2 = schedule_master.upsert_schedule_rows(NFL, df)
    assert (before2, after2) == (2, 2)
    written = pd.read_csv(paths.schedule_master_csv("nfl"))
    # columns the current-week service depends on must survive
    for col in ("season", "week", "kickoff_iso_utc"):
        assert col in written.columns
    assert len(written) == 2


def test_master_upsert_scores_win(sandbox):
    df = schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=lambda season: _nflverse_frame())
    schedule_master.upsert_schedule_rows(NFL, df)
    scored = df.copy()
    scored["home_score"] = 27.0
    scored["away_score"] = 13.0
    before, after = schedule_master.upsert_schedule_rows(NFL, scored)
    assert (before, after) == (2, 2)
    written = pd.read_csv(paths.schedule_master_csv("nfl"))
    assert written["home_score"].notna().all()
    assert set(written["home_score"]) == {27.0}


def test_master_upsert_cfb_uses_league_path(sandbox):
    df = schedule.fetch_schedule(
        CFB, 2026, Settings(cfbd_api_key="k"), fetch_raw=lambda season: _cfbd_payload()
    )
    schedule_master.upsert_schedule_rows(CFB, df)
    assert paths.schedule_master_csv("cfb").exists()
    assert not paths.schedule_master_csv("nfl").exists()
    written = pd.read_csv(paths.schedule_master_csv("cfb"))
    assert set(written["league"]) == {"CFB"}


def test_ensure_seasons_present_respects_cfbd_refresh_toggle(sandbox):
    calls = []

    def spy(season):
        calls.append(season)
        return _cfbd_payload()

    settings_off = Settings(cfbd_api_key="k", cfbd_refresh=False)
    schedule_master.ensure_seasons_present(CFB, [2026], settings_off, fetch_raw=spy)
    assert calls == []
    settings_on = Settings(cfbd_api_key="k", cfbd_refresh=True)
    schedule_master.ensure_seasons_present(CFB, [2026], settings_on, fetch_raw=spy)
    assert calls == [2026]
    assert paths.schedule_master_csv("cfb").exists()


# --- integration: master feeds the current-week service ------------------------


def test_master_feeds_current_week(sandbox):
    now = datetime.now(timezone.utc)
    kick1 = (now + timedelta(hours=1)).replace(microsecond=0)
    kick2 = (kick1 + timedelta(days=7)).replace(microsecond=0)
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": kick1.strftime("%Y-%m-%d"),
                "gametime": None,
                "start_time_utc": kick1.isoformat(),
                "home_team": "KC",
                "away_team": "BUF",
                "location": "Home",
            },
            {
                "season": 2026,
                "week": 2,
                "game_type": "REG",
                "gameday": kick2.strftime("%Y-%m-%d"),
                "gametime": None,
                "start_time_utc": kick2.isoformat(),
                "home_team": "MIA",
                "away_team": "NYJ",
                "location": "Home",
            },
        ]
    )
    df = schedule.fetch_schedule(NFL, 2026, Settings(), fetch_raw=lambda season: frame)
    schedule_master.upsert_schedule_rows(NFL, df)
    season, week, _ = get_current_week(NFL, settings=Settings(), persist=False)
    assert season == 2026
    assert week == 1

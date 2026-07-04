"""Integration test: refresh_league end-to-end with all fetches faked.

Exercises the orchestrator glue (stage sequencing, schedule->gameview mapping,
Sagarin map join, stats join, sidecars, run summary) against a tmp output
root. Odds staging/promotion/backfill/ATS are covered by their own module
tests; here they are disabled via Settings so no stage touches the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from football_statfinder import paths as paths_mod
from football_statfinder import refresh as refresh_mod
from football_statfinder import run_summary as run_summary_mod
from football_statfinder.common.game_key import build_game_key
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.config import BackfillSettings, OddsSettings, Settings, StorageSettings
from football_statfinder.leagues import NFL
from football_statfinder.sources import sagarin as sagarin_mod
from football_statfinder.sources import schedule as schedule_mod
from football_statfinder.sources import stats as stats_mod
from football_statfinder.common.io_atomic import write_atomic_jsonl

SEASON, WEEK = 2026, 2

GAMES = [
    ("Green Bay Packers", "Chicago Bears", "2026-09-13T17:00:00Z"),
    ("Kansas City Chiefs", "Buffalo Bills", "2026-09-13T20:25:00Z"),
]
TEAMS = sorted({name for pair in GAMES for name in pair[:2]})


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setattr(paths_mod, "OUT_ROOT", root)
    monkeypatch.setattr(run_summary_mod, "OUT_ROOT", root)
    return root


def _settings() -> Settings:
    return Settings(
        odds=OddsSettings(staging_enable=False, promotion_enable=False),
        backfill=BackfillSettings(scores_enable=False, ats_enable=False),
        # Storage defaults to enabled (StorageSettings.enable=True) and would
        # otherwise touch the real repo-anchored data/statfinder.sqlite3 as a
        # side effect of this test; storage's own dual-write behavior is
        # covered separately in tests/test_storage.py.
        storage=StorageSettings(enable=False),
    )


def _fake_schedule_df() -> pd.DataFrame:
    rows = []
    for home, away, kick in GAMES:
        kickoff = datetime.fromisoformat(kick.replace("Z", "+00:00"))
        rows.append(
            {
                "league": "nfl",
                "season": SEASON,
                "week": WEEK,
                "game_type": "REG",
                "kickoff_iso_utc": kick,
                "home_team_raw": home,
                "away_team_raw": away,
                "home_team_norm": home,
                "away_team_norm": away,
                "home_team_key": NFL.merge_key(home),
                "away_team_key": NFL.merge_key(away),
                "neutral_site": False,
                "venue": "",
                "conference_game": "",
                "home_score": None,
                "away_score": None,
                "spread_line": None,
                "total_line": None,
                "game_key": build_game_key(NFL, kickoff, home, away),
                "source": "nflverse",
            }
        )
    return pd.DataFrame(rows, columns=schedule_mod.SCHEDULE_COLUMNS)


def _fake_sagarin_result(out_root) -> sagarin_mod.SagarinStagingResult:
    weekly_jsonl = out_root / "staging" / "sagarin_latest" / "nfl" / f"weekly_{SEASON}_wk{WEEK}.jsonl"
    rows = [
        {
            "team_norm": team,
            "pr": 25.0 - i,
            "pr_rank": i + 1,
            "sos": 20.0 - i,
            "sos_rank": i + 1,
            "hfa": 2.3,
        }
        for i, team in enumerate(TEAMS)
    ]
    write_atomic_jsonl(weekly_jsonl, rows)
    return sagarin_mod.SagarinStagingResult(
        league="NFL",
        season=SEASON,
        week=WEEK,
        page_season=SEASON,
        page_week=WEEK,
        teams_parsed=len(rows),
        teams_selected=len(rows),
        master_before=0,
        master_after=len(rows),
        latest_fetch_ts="2026-09-10T00:00:00Z",
        hfa=2.3,
        page_stamp="test fixture",
        source_url="fixture://sagarin",
        raw_html_path=None,
        staging_path=weekly_jsonl.parent,
        weekly_csv=weekly_jsonl.with_suffix(".csv"),
        weekly_jsonl=weekly_jsonl,
    )


def _fake_metrics_rows() -> list[dict]:
    rows = []
    for i, team in enumerate(TEAMS):
        rows.append(
            {
                "Team": team,
                "RY(O)": 120.0 + i, "R(O)_RY": i + 1,
                "PY(O)": 220.0 + i, "R(O)_PY": i + 1,
                "TY(O)": 340.0 + i, "R(O)_TY": i + 1,
                "RY(D)": 100.0 + i, "R(D)_RY": i + 1,
                "PY(D)": 200.0 + i, "R(D)_PY": i + 1,
                "TY(D)": 300.0 + i, "R(D)_TY": i + 1,
                "TO": 0.5, "PF": 24.0, "PA": 20.0,
                "SU": "1-0", "ATS": "1-0-0",
            }
        )
    return rows


class _FakeProvider:
    def league_metrics_rows(self, season, week, *, as_of_week=None):
        assert (season, week) == (SEASON, WEEK)
        return _fake_metrics_rows()

    def team_stats(self, season, week, *, as_of_week=None):
        return stats_mod.team_stats_from_metrics_rows(NFL, self.league_metrics_rows(season, week))


def test_refresh_league_end_to_end(out_root, monkeypatch):
    monkeypatch.setattr(schedule_mod, "fetch_schedule", lambda league, season, settings, **kw: _fake_schedule_df())
    monkeypatch.setattr(
        refresh_mod.sagarin_mod, "run_sagarin_staging",
        lambda league, season, week, settings, **kw: _fake_sagarin_result(out_root),
    )
    monkeypatch.setattr(stats_mod, "get_stats_provider", lambda league, settings: _FakeProvider())

    summary = refresh_mod.refresh_league(NFL, _settings(), season=SEASON, week=WEEK)

    assert summary.ok, [s.error for s in summary.stages if not s.ok]
    assert [s.name for s in summary.stages] == ["schedule", "sagarin", "stats", "gameview", "sidecars"]
    assert (summary.season, summary.week) == (SEASON, WEEK)

    # games_week artifacts under the unified out/{league}/{S}_week{W}/ layout
    json_path = paths_mod.games_week_jsonl("nfl", SEASON, WEEK)
    assert json_path.exists()
    assert paths_mod.games_week_csv("nfl", SEASON, WEEK).exists()
    rows = read_jsonl(json_path).rows
    assert len(rows) == len(GAMES)
    by_key = {row["game_key"]: row for row in rows}
    expected_key = build_game_key(
        NFL, datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc), "Green Bay Packers", "Chicago Bears"
    )
    assert expected_key in by_key
    packers_row = by_key[expected_key]
    # Sagarin joined, ratings derived; no odds -> rating_vs_odds stays None
    assert packers_row["home_pr"] is not None
    assert packers_row["rating_diff"] is not None
    assert packers_row["rating_vs_odds"] is None
    # stats joined
    assert packers_row["home_ty_pg"] is not None

    # schedule artifact + master exist
    assert schedule_mod.read_schedule_artifact(NFL, SEASON, WEEK) is not None
    assert paths_mod.schedule_master_csv("nfl").exists()

    # sidecars written per game
    for key in by_key:
        assert paths_mod.sidecar_path("nfl", SEASON, WEEK, key).exists()

    # machine-readable run summary
    summary_path = out_root / "state" / "run_summary_nfl.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["league"] == "NFL"

    # NOTIFY contract line
    assert summary.notify_line(len(rows)) == f"NOTIFY: NFL refresh complete week={SEASON}-{WEEK} rows=2"


def test_refresh_failure_is_recorded_and_raises(out_root, monkeypatch):
    def boom(league, season, settings, **kw):
        raise RuntimeError("nflverse download failed")

    monkeypatch.setattr(schedule_mod, "fetch_schedule", boom)
    with pytest.raises(RuntimeError, match="nflverse download failed"):
        refresh_mod.refresh_league(NFL, _settings(), season=SEASON, week=WEEK)

    payload = json.loads((out_root / "state" / "run_summary_nfl.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["stages"][0]["name"] == "schedule"
    assert "nflverse download failed" in payload["stages"][0]["error"]


def test_recompute_rating_fields_after_promotion(out_root):
    json_path = paths_mod.games_week_jsonl("nfl", SEASON, WEEK)
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    key = build_game_key(NFL, kickoff, "Green Bay Packers", "Chicago Bears")
    row = {
        "season": SEASON,
        "week": WEEK,
        "game_key": key,
        "kickoff_iso_utc": "2026-09-13T17:00:00Z",
        "home_team_norm": "Green Bay Packers",
        "away_team_norm": "Chicago Bears",
        "home_pr": 25.0,
        "away_pr": 20.0,
        "hfa": 2.0,
        # promotion just landed this spread; derived fields still stale None
        "spread_home_relative": -3.5,
        "rating_diff": None,
        "rating_vs_odds": None,
        "favored_side": None,
        "spread_favored_team": None,
        "rating_diff_favored_team": None,
        "rating_vs_odds_favored_team": None,
    }
    write_atomic_jsonl(json_path, [row])

    changed = refresh_mod._recompute_rating_fields(NFL, SEASON, WEEK)
    assert changed == 1
    updated = read_jsonl(json_path).rows[0]
    assert updated["rating_diff"] == 7.0  # (25 + 2) - 20
    assert updated["rating_vs_odds"] == 3.5  # 7 + (-3.5)
    assert updated["favored_side"] == "HOME"
    assert updated["spread_favored_team"] == -3.5
    assert updated["rating_vs_odds_favored_team"] == 3.5

"""Tests for football_statfinder.storage.export: the WP-D byte-parity gate.

Runs a fake refresh (the ``tests/test_storage.py`` / ``test_refresh_integration.py``
injection pattern) with storage enabled into output root A, then calls
``export_week`` into a separate output root B, and asserts every exported file
pair is BYTE IDENTICAL to what the pipeline itself wrote. The whole point of
this gate (docs/PHASE2_SPEC.md, WP-D) is that any drift must be found and
fixed at its source, never laundered into a parsed-JSON-equality assertion.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from football_statfinder import paths as paths_mod
from football_statfinder import refresh as refresh_mod
from football_statfinder import run_summary as run_summary_mod
from football_statfinder.common.game_key import build_game_key
from football_statfinder.common.io_atomic import write_atomic_jsonl
from football_statfinder.config import BackfillSettings, OddsSettings, Settings, StorageSettings
from football_statfinder.leagues import NFL
from football_statfinder.sources import sagarin as sagarin_mod
from football_statfinder.sources import schedule as schedule_mod
from football_statfinder.sources import stats as stats_mod
from football_statfinder.storage import db as db_mod
from football_statfinder.storage.export import ExportError, export_week

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


def _settings(db_path) -> Settings:
    return Settings(
        odds=OddsSettings(staging_enable=False, promotion_enable=False),
        backfill=BackfillSettings(scores_enable=False, ats_enable=False),
        storage=StorageSettings(enable=True, db_path=db_path),
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
            "fetch_ts": "2026-09-10T00:00:00Z",
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


def _run_fake_refresh(tmp_path, out_root, monkeypatch):
    db_path = tmp_path / "storage_test.sqlite3"
    monkeypatch.setattr(schedule_mod, "fetch_schedule", lambda league, season, settings, **kw: _fake_schedule_df())
    monkeypatch.setattr(
        refresh_mod.sagarin_mod, "run_sagarin_staging",
        lambda league, season, week, settings, **kw: _fake_sagarin_result(out_root),
    )
    monkeypatch.setattr(stats_mod, "get_stats_provider", lambda league, settings: _FakeProvider())

    summary = refresh_mod.refresh_league(NFL, _settings(db_path), season=SEASON, week=WEEK)
    assert summary.ok, [s.error for s in summary.stages if not s.ok]
    return db_path


def test_export_week_byte_identical_to_pipeline_output(tmp_path, out_root, monkeypatch):
    db_path = _run_fake_refresh(tmp_path, out_root, monkeypatch)

    root_b = tmp_path / "out_b"
    conn = db_mod.connect(db_path)
    try:
        result = export_week(conn, NFL, SEASON, WEEK, out_root=root_b)
    finally:
        conn.close()

    # games_week jsonl + csv
    orig_jsonl = paths_mod.games_week_jsonl("nfl", SEASON, WEEK)
    orig_csv = paths_mod.games_week_csv("nfl", SEASON, WEEK)
    assert orig_jsonl.exists() and orig_csv.exists()
    assert result["games_jsonl"].read_bytes() == orig_jsonl.read_bytes()
    assert result["games_csv"].read_bytes() == orig_csv.read_bytes()

    # league_metrics csv
    orig_metrics = stats_mod.league_metrics_csv_path(NFL, SEASON, WEEK)
    assert orig_metrics.exists()
    assert result["league_metrics_csv"].read_bytes() == orig_metrics.read_bytes()

    # sidecars: every sidecar json byte-identical, same file set
    orig_side_dir = paths_mod.sidecar_dir("nfl", SEASON, WEEK)
    exported_sidecars = {p.name: p for p in result["sidecars"]}
    orig_sidecars = {p.name: p for p in orig_side_dir.glob("*.json")}
    assert exported_sidecars, "expected at least one sidecar to be exported"
    assert set(exported_sidecars) == set(orig_sidecars)
    for name, exported_path in exported_sidecars.items():
        assert exported_path.read_bytes() == orig_sidecars[name].read_bytes(), name


def test_export_absent_week_fails_loudly(tmp_path, out_root, monkeypatch):
    db_path = _run_fake_refresh(tmp_path, out_root, monkeypatch)
    conn = db_mod.connect(db_path)
    try:
        with pytest.raises(ExportError, match=r"NFL.*season=2026.*week=99"):
            export_week(conn, NFL, SEASON, 99, out_root=tmp_path / "out_c")
    finally:
        conn.close()

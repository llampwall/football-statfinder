"""Tests for football_statfinder.storage: schema, upserts, orchestrator wiring.

Covers the WP-C deliverables: schema creation (db.connect), upsert idempotence
for each record_* writer, the schema_version mismatch guard, and an end-to-end
orchestrator run (refresh_league with storage enabled) asserting per-table row
counts and payload round-trip equality against the flat files it also writes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from football_statfinder import paths as paths_mod
from football_statfinder import refresh as refresh_mod
from football_statfinder import run_summary as run_summary_mod
from football_statfinder.common.game_key import build_game_key
from football_statfinder.common.io_atomic import write_atomic_jsonl
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.config import BackfillSettings, OddsSettings, Settings, StorageSettings
from football_statfinder.leagues import NFL
from football_statfinder.sources import sagarin as sagarin_mod
from football_statfinder.sources import schedule as schedule_mod
from football_statfinder.sources import stats as stats_mod
from football_statfinder.storage import db as db_mod
from football_statfinder.storage import store as store_mod

# ---------------------------------------------------------------------------
# db.py: schema creation, transaction helper, schema_version guard
# ---------------------------------------------------------------------------


def test_connect_creates_schema(tmp_path):
    target = tmp_path / "statfinder.sqlite3"
    conn = db_mod.connect(target)
    try:
        assert target.exists()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "meta",
            "schedule_games",
            "sagarin_ratings",
            "team_metrics",
            "odds_pinned",
            "games",
            "sidecars",
        } <= tables
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == str(db_mod.SCHEMA_VERSION)
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.lower() == "wal"
    finally:
        conn.close()


def test_connect_is_idempotent_across_reopen(tmp_path):
    target = tmp_path / "statfinder.sqlite3"
    conn1 = db_mod.connect(target)
    conn1.close()
    conn2 = db_mod.connect(target)  # must not re-raise or duplicate the meta row
    try:
        rows = conn2.execute("SELECT COUNT(*) FROM meta WHERE key='schema_version'").fetchone()[0]
        assert rows == 1
    finally:
        conn2.close()


def test_schema_version_mismatch_raises(tmp_path):
    target = tmp_path / "statfinder.sqlite3"
    conn = db_mod.connect(target)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(db_mod.SchemaVersionError):
        db_mod.connect(target)


def test_transaction_rolls_back_on_error(tmp_path):
    conn = db_mod.connect(tmp_path / "statfinder.sqlite3")
    try:
        with pytest.raises(RuntimeError):
            with db_mod.transaction(conn):
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('probe', 'x')"
                )
                raise RuntimeError("boom")
        row = conn.execute("SELECT value FROM meta WHERE key='probe'").fetchone()
        assert row is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# store.py: upsert idempotence per table, payload/dumps-kwargs fidelity
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    connection = db_mod.connect(tmp_path / "statfinder.sqlite3")
    yield connection
    connection.close()


def test_dumps_matches_io_atomic_jsonl_kwargs(conn):
    """The payload encoding must match write_atomic_jsonl's json.dumps kwargs
    (ensure_ascii=False, sort_keys=True, default separators) — load-bearing
    for the later export byte-parity gate (WP-D)."""
    row = {"game_key": "k1", "season": 2026, "week": 2, "note": "café"}
    store_mod.record_games(conn, NFL, 2026, 2, [row])
    payload = conn.execute(
        "SELECT payload FROM games WHERE game_key='k1'"
    ).fetchone()[0]
    assert payload == json.dumps(row, ensure_ascii=False, sort_keys=True)
    assert json.loads(payload) == row


def test_record_games_upsert_idempotent(conn):
    row_v1 = {"game_key": "k1", "season": 2026, "week": 2, "home_pr": 1.0}
    row_v2 = {"game_key": "k1", "season": 2026, "week": 2, "home_pr": 2.0}
    store_mod.record_games(conn, NFL, 2026, 2, [row_v1])
    store_mod.record_games(conn, NFL, 2026, 2, [row_v2])
    rows = conn.execute("SELECT payload, updated_at FROM games WHERE game_key='k1'").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["home_pr"] == 2.0
    # timezone-aware UTC ISO string
    datetime.fromisoformat(rows[0][1].replace("Z", "+00:00"))


def test_record_schedule_from_dataframe(conn):
    df = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_key": "sk1",
                "kickoff_iso_utc": "2026-09-13T17:00:00Z",
                "home_team_key": "packers",
                "away_team_key": "bears",
            }
        ]
    )
    count = store_mod.record_schedule(conn, NFL, df)
    assert count == 1
    row = conn.execute(
        "SELECT season, week, kickoff_iso_utc, home_team_key, away_team_key, payload "
        "FROM schedule_games WHERE game_key='sk1'"
    ).fetchone()
    assert row[0] == 2026 and row[1] == 2
    assert row[2] == "2026-09-13T17:00:00Z"
    assert row[3] == "packers" and row[4] == "bears"
    assert json.loads(row[5])["game_key"] == "sk1"


def test_record_sagarin_upsert_and_fetch_ts_key(conn):
    rows = [
        {"team_norm": "Green Bay Packers", "pr": 25.0, "fetch_ts": "2026-09-10T00:00:00Z"},
        {"team_norm": "Green Bay Packers", "pr": 26.0, "fetch_ts": "2026-09-11T00:00:00Z"},
    ]
    store_mod.record_sagarin(conn, NFL, 2026, 2, rows)
    count = conn.execute("SELECT COUNT(*) FROM sagarin_ratings").fetchone()[0]
    assert count == 2  # distinct fetch_ts -> distinct rows, not an overwrite

    store_mod.record_sagarin(conn, NFL, 2026, 2, [rows[0]])  # same key again
    count_after = conn.execute("SELECT COUNT(*) FROM sagarin_ratings").fetchone()[0]
    assert count_after == 2  # idempotent


def test_record_metrics_upsert_idempotent(conn):
    row = {"Team": "Green Bay Packers", "PF": "24.0"}
    store_mod.record_metrics(conn, NFL, 2026, 2, [row])
    store_mod.record_metrics(conn, NFL, 2026, 2, [row])
    count = conn.execute("SELECT COUNT(*) FROM team_metrics").fetchone()[0]
    assert count == 1


def test_record_pinned_odds_key_matches_ledger_dedupe(conn):
    record = {
        "fetch_ts": "2026-09-10T00:00:00Z",
        "game_key": "gk1",
        "market": "spreads",
        "book": "draftkings",
        "line": {"spread_home_relative": -3.5},
    }
    store_mod.record_pinned_odds(conn, NFL, [record])
    store_mod.record_pinned_odds(conn, NFL, [record])  # duplicate -> idempotent
    count = conn.execute("SELECT COUNT(*) FROM odds_pinned").fetchone()[0]
    assert count == 1
    cols = [d[0] for d in conn.execute("SELECT * FROM odds_pinned LIMIT 0").description]
    assert "season" not in cols and "week" not in cols  # schema v1 has no season/week columns here


def test_record_sidecars_upsert_idempotent(conn):
    payload = {"game_key": "gk1", "home_ytd": [], "away_ytd": [], "home_prev": [], "away_prev": []}
    store_mod.record_sidecars(conn, NFL, 2026, 2, [payload])
    store_mod.record_sidecars(conn, NFL, 2026, 2, [payload])
    count = conn.execute("SELECT COUNT(*) FROM sidecars").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Orchestrator integration: refresh_league with storage enabled
# ---------------------------------------------------------------------------

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


def test_refresh_league_dual_writes_to_storage(tmp_path, out_root, monkeypatch):
    db_path = tmp_path / "storage_test.sqlite3"
    monkeypatch.setattr(schedule_mod, "fetch_schedule", lambda league, season, settings, **kw: _fake_schedule_df())
    monkeypatch.setattr(
        refresh_mod.sagarin_mod, "run_sagarin_staging",
        lambda league, season, week, settings, **kw: _fake_sagarin_result(out_root),
    )
    monkeypatch.setattr(stats_mod, "get_stats_provider", lambda league, settings: _FakeProvider())

    summary = refresh_mod.refresh_league(NFL, _settings(db_path), season=SEASON, week=WEEK)
    assert summary.ok, [s.error for s in summary.stages if not s.ok]

    conn = sqlite3.connect(str(db_path))
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "schedule_games",
                "sagarin_ratings",
                "team_metrics",
                "odds_pinned",
                "games",
                "sidecars",
            )
        }
        assert counts["schedule_games"] == len(GAMES)
        assert counts["sagarin_ratings"] == len(TEAMS)
        assert counts["team_metrics"] == len(TEAMS)
        assert counts["odds_pinned"] == 0  # odds staging disabled: zero DB touches for that table
        assert counts["games"] == len(GAMES)
        assert counts["sidecars"] == len(GAMES)

        # Payload round-trip: the DB's games payload must equal the flat file's row.
        json_path = paths_mod.games_week_jsonl("nfl", SEASON, WEEK)
        file_rows = {row["game_key"]: row for row in read_jsonl(json_path).rows}
        db_rows = conn.execute("SELECT game_key, payload FROM games").fetchall()
        assert len(db_rows) == len(file_rows)
        for game_key, payload in db_rows:
            assert json.loads(payload) == file_rows[game_key]

        # updated_at is a timezone-aware UTC ISO string.
        updated_at = conn.execute("SELECT updated_at FROM games LIMIT 1").fetchone()[0]
        parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
    finally:
        conn.close()


def test_storage_disabled_means_zero_db_touches(tmp_path, out_root, monkeypatch):
    db_path = tmp_path / "should_not_exist.sqlite3"
    monkeypatch.setattr(schedule_mod, "fetch_schedule", lambda league, season, settings, **kw: _fake_schedule_df())
    monkeypatch.setattr(
        refresh_mod.sagarin_mod, "run_sagarin_staging",
        lambda league, season, week, settings, **kw: _fake_sagarin_result(out_root),
    )
    monkeypatch.setattr(stats_mod, "get_stats_provider", lambda league, settings: _FakeProvider())

    settings = Settings(
        odds=OddsSettings(staging_enable=False, promotion_enable=False),
        backfill=BackfillSettings(scores_enable=False, ats_enable=False),
        storage=StorageSettings(enable=False, db_path=db_path),
    )
    summary = refresh_mod.refresh_league(NFL, settings, season=SEASON, week=WEEK)
    assert summary.ok, [s.error for s in summary.stages if not s.ok]
    assert not db_path.exists()

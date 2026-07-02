"""Tests for football_statfinder.pipeline.backfill.

Synthetic week dirs live under ``tmp_path`` via the paths.py ``out_root``
override; the score source is injected (never network); the paid ATS tier is
absent or stubbed. Regression coverage for REBUILD.md bugs 5, 6, 11 and the
merge-preserve invariant.
"""

from __future__ import annotations

import json

import pytest

from football_statfinder import paths
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.config import BackfillSettings, OddsSettings, Settings
from football_statfinder.leagues import CFB, NFL
from football_statfinder.pipeline.backfill import (
    ScoredGame,
    backfill_scores,
    master_score_source,
)

SEASON = 2025
KICKOFF_W1 = "2025-09-07T17:00:00Z"
KICKOFF_W2 = "2025-09-14T17:00:00Z"


def make_settings(**backfill_kwargs) -> Settings:
    return Settings(
        odds=OddsSettings(cache_only=True),
        backfill=BackfillSettings(**backfill_kwargs),
    )


def game_row(game_key, home, away, *, week, kickoff, home_score=None, away_score=None, **extra):
    row = {
        "game_key": game_key,
        "season": SEASON,
        "week": week,
        "kickoff_iso_utc": kickoff,
        "home_team_norm": home,
        "away_team_norm": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_su": None,
        "away_su": None,
        "home_ats": None,
        "away_ats": None,
    }
    row.update(extra)
    return row


def write_week(out_root, league, week, rows):
    path = paths.games_week_jsonl(league.code, SEASON, week, out_root=out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def read_week(out_root, league, week):
    rows = read_jsonl(paths.games_week_jsonl(league.code, SEASON, week, out_root=out_root)).rows
    return {row["game_key"]: row for row in rows}


def source_of(games):
    return lambda league, season: games


# ---------------------------------------------------------------------------
# Merge-preserve: scores land, promoted odds survive
# ---------------------------------------------------------------------------


def test_scores_land_and_promoted_odds_survive(tmp_path):
    write_week(tmp_path, NFL, 1, [
        game_row(
            "g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1,
            favored_side="HOME", spread_favored_team=-3.5, total=44.5,
            rating_vs_odds=2.2, odds_source="staging/odds_pinned",
        ),
    ])
    games = [ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 30, 20)]

    result = backfill_scores(
        NFL, SEASON, 2, make_settings(), score_source=source_of(games), out_root=tmp_path
    )

    assert result.updated == 1
    assert result.files_rewritten == 1
    assert result.changed_weeks == [1]
    row = read_week(tmp_path, NFL, 1)["g1"]
    assert (row["home_score"], row["away_score"]) == (30, 20)
    # The merge-preserve invariant: promoted odds/rating fields survive.
    assert row["spread_favored_team"] == -3.5
    assert row["favored_side"] == "HOME"
    assert row["total"] == 44.5
    assert row["rating_vs_odds"] == 2.2
    assert row["odds_source"] == "staging/odds_pinned"
    # CSV artifact rewritten alongside.
    assert paths.games_week_csv(NFL.code, SEASON, 1, out_root=tmp_path).exists()


# ---------------------------------------------------------------------------
# Bug 5: W-L record correctness (was always "0-0")
# ---------------------------------------------------------------------------


def test_su_records_are_correct_not_zero_zero(tmp_path):
    write_week(tmp_path, NFL, 1, [
        game_row("g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1),
    ])
    write_week(tmp_path, NFL, 2, [
        game_row("g2", "Gamma", "Alpha", week=2, kickoff=KICKOFF_W2),
        game_row("g3", "Delta", "Echo", week=2, kickoff=KICKOFF_W2),
    ])
    games = [
        ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 30, 20),   # Alpha 1-0, Beta 0-1
        ScoredGame("g2", KICKOFF_W2, "Gamma", "Alpha", 28, 21),  # Gamma 1-0, Alpha 1-1
        ScoredGame("g3", KICKOFF_W2, "Delta", "Echo", 20, 20),   # tie: 0-0-1 both
    ]

    result = backfill_scores(
        NFL, SEASON, 3, make_settings(), score_source=source_of(games), out_root=tmp_path
    )

    assert result.weeks_scanned == [1, 2]
    week1 = read_week(tmp_path, NFL, 1)
    assert week1["g1"]["home_su"] == "1-0"
    assert week1["g1"]["away_su"] == "0-1"
    week2 = read_week(tmp_path, NFL, 2)
    assert week2["g2"]["home_su"] == "1-0"   # Gamma
    assert week2["g2"]["away_su"] == "1-1"   # Alpha after two games
    assert week2["g3"]["home_su"] == "0-0-1"
    assert week2["g3"]["away_su"] == "0-0-1"


# ---------------------------------------------------------------------------
# Bug 6: a score-only change persists (and the backfill is idempotent)
# ---------------------------------------------------------------------------


def test_score_only_change_persists_then_second_run_is_noop(tmp_path):
    # CFB league: the twin that carried the over-suppressing canonical check.
    write_week(tmp_path, CFB, 1, [
        game_row(
            "g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1,
            favored_side="HOME", spread_favored_team=-6.0, rating_vs_odds=1.0,
        ),
    ])
    games = [ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 21, 14)]
    settings = make_settings()

    first = backfill_scores(
        CFB, SEASON, 2, settings, score_source=source_of(games), out_root=tmp_path
    )
    assert first.updated == 1
    assert first.files_rewritten == 1
    assert first.changed_weeks == [1]
    row = read_week(tmp_path, CFB, 1)["g1"]
    assert (row["home_score"], row["away_score"]) == (21, 14)

    second = backfill_scores(
        CFB, SEASON, 2, settings, score_source=source_of(games), out_root=tmp_path
    )
    assert second.updated == 0
    assert second.files_rewritten == 0
    assert second.changed_weeks == []


# ---------------------------------------------------------------------------
# Bug 11: rebuilds are the orchestrator's job; failures propagate
# ---------------------------------------------------------------------------


def test_rebuild_callback_invoked_per_changed_week(tmp_path):
    write_week(tmp_path, NFL, 1, [game_row("g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1)])
    write_week(tmp_path, NFL, 2, [game_row("g2", "Gamma", "Delta", week=2, kickoff=KICKOFF_W2)])
    games = [
        ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 30, 20),
        ScoredGame("g2", KICKOFF_W2, "Gamma", "Delta", 14, 10),
    ]
    rebuilt = []

    result = backfill_scores(
        NFL, SEASON, 3, make_settings(),
        score_source=source_of(games),
        rebuild=lambda season, week: rebuilt.append((season, week)),
        out_root=tmp_path,
    )

    assert result.changed_weeks == [1, 2]
    assert rebuilt == [(SEASON, 1), (SEASON, 2)]


def test_rebuild_failure_propagates(tmp_path):
    write_week(tmp_path, NFL, 1, [game_row("g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1)])
    games = [ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 30, 20)]

    def exploding_rebuild(season, week):
        raise RuntimeError("gameview rebuild failed")

    with pytest.raises(RuntimeError, match="gameview rebuild failed"):
        backfill_scores(
            NFL, SEASON, 2, make_settings(),
            score_source=source_of(games),
            rebuild=exploding_rebuild,
            out_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------


def test_scores_disabled_is_noop(tmp_path):
    path = write_week(tmp_path, NFL, 1, [game_row("g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1)])
    before = path.read_text(encoding="utf-8")
    games = [ScoredGame("g1", KICKOFF_W1, "Alpha", "Beta", 30, 20)]

    result = backfill_scores(
        NFL, SEASON, 2, make_settings(scores_enable=False),
        score_source=source_of(games), out_root=tmp_path,
    )

    assert result.updated == 0
    assert result.changed_weeks == []
    assert path.read_text(encoding="utf-8") == before


def test_week_window_respects_backfill_weeks(tmp_path):
    for week in (1, 2, 3):
        write_week(tmp_path, NFL, week, [
            game_row(f"g{week}", "Alpha", "Beta", week=week, kickoff=KICKOFF_W1),
        ])
    games = [ScoredGame(f"g{week}", KICKOFF_W1, "Alpha", "Beta", 30, 20) for week in (1, 2, 3)]

    result = backfill_scores(
        NFL, SEASON, 4, make_settings(weeks=1),
        score_source=source_of(games), out_root=tmp_path,
    )

    assert result.weeks_scanned == [3]
    assert result.changed_weeks == [3]
    assert read_week(tmp_path, NFL, 1)["g1"]["home_score"] is None  # untouched


# ---------------------------------------------------------------------------
# ATS backfill: sidecar + row fields from the pinned tier (bugs 3/4/19 path)
# ---------------------------------------------------------------------------


def test_ats_backfill_fills_sidecar_and_row_from_pinned(tmp_path):
    # Finished game already carrying scores; ATS fields blank — the home side
    # with the legacy em dash (bug 19), the away side with None.
    write_week(tmp_path, NFL, 1, [
        game_row(
            "g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1,
            home_score=30, away_score=20, home_ats="—", away_ats=None,
        ),
    ])
    sidecar = paths.sidecar_path(NFL.code, SEASON, 1, "g1", out_root=tmp_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "home_ytd": [{"season": SEASON, "week": 1, "ats": "—", "to_margin": None}],
        "away_ytd": [{"season": SEASON, "week": 1, "ats": None, "to_margin": None}],
    }), encoding="utf-8")

    pinned = paths.odds_pinned_jsonl(NFL.code, SEASON, out_root=tmp_path)
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text(json.dumps({
        "market": "spreads",
        "game_key": "g1",
        "fetch_ts": "2025-09-07T16:00:00Z",
        "book": "pinnacle",
        "line": {"spread_home_relative": -3.5, "favored_side": "HOME"},
        "raw_event": {"event_id": "ev1"},
    }) + "\n", encoding="utf-8")

    result = backfill_scores(
        NFL, SEASON, 2, make_settings(),
        score_source=source_of([]),  # no score changes; ATS only
        out_root=tmp_path,
    )

    assert result.ats_fixed == 1
    assert result.ats_sources == {"pinned": 1}
    assert result.changed_weeks == [1]

    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_data["home_ytd"][0]["ats"] == "W"      # 30-20 covers -3.5
    assert sidecar_data["home_ytd"][0]["to_margin"] == 6.5
    assert sidecar_data["away_ytd"][0]["ats"] == "L"
    assert sidecar_data["away_ytd"][0]["to_margin"] == -6.5

    row = read_week(tmp_path, NFL, 1)["g1"]
    assert row["home_ats"] == "1-0-0"
    assert row["away_ats"] == "0-1-0"
    assert row["home_to_margin_pg"] == 6.5
    assert row["away_to_margin_pg"] == -6.5
    assert row["raw_sources"]["closing_spread"]["source"] == "pinned"
    assert row["raw_sources"]["closing_spread"]["favored_team"] == "HOME"


def test_ats_backfill_without_sidecar_writes_single_game_record(tmp_path):
    write_week(tmp_path, NFL, 1, [
        game_row(
            "g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1,
            home_score=17, away_score=20,
        ),
    ])
    pinned = paths.odds_pinned_jsonl(NFL.code, SEASON, out_root=tmp_path)
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text(json.dumps({
        "market": "spreads",
        "game_key": "g1",
        "fetch_ts": "2025-09-07T16:00:00Z",
        "book": "draftkings",
        "line": {"spread_home_relative": 3.0, "favored_side": "AWAY"},
        "raw_event": {"event_id": "ev1"},
    }) + "\n", encoding="utf-8")

    result = backfill_scores(
        NFL, SEASON, 2, make_settings(),
        score_source=source_of([]),
        out_root=tmp_path,
    )

    assert result.ats_fixed == 1
    row = read_week(tmp_path, NFL, 1)["g1"]
    # Away favored by 3, wins by 3: exact push on both sides.
    assert row["home_ats"] == "0-0-1"
    assert row["away_ats"] == "0-0-1"


def test_ats_disabled_leaves_blank_fields(tmp_path):
    write_week(tmp_path, NFL, 1, [
        game_row("g1", "Alpha", "Beta", week=1, kickoff=KICKOFF_W1,
                 home_score=30, away_score=20),
    ])
    result = backfill_scores(
        NFL, SEASON, 2, make_settings(ats_enable=False),
        score_source=source_of([]),
        out_root=tmp_path,
    )
    assert result.ats_fixed == 0
    assert read_week(tmp_path, NFL, 1)["g1"]["home_ats"] is None


# ---------------------------------------------------------------------------
# Default score source (schedule master CSV; no network)
# ---------------------------------------------------------------------------


def test_master_score_source_builds_shared_game_keys(tmp_path):
    master = paths.schedule_master_csv(NFL.code, out_root=tmp_path)
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_text(
        "season,kickoff_iso_utc,home_team_norm,away_team_norm,home_score,away_score\n"
        f"{SEASON},{KICKOFF_W1},Dallas Cowboys,Philadelphia Eagles,24,20\n"
        f"{SEASON},{KICKOFF_W1},Green Bay Packers,Chicago Bears,,\n"
        f"{SEASON - 1},{KICKOFF_W1},Dallas Cowboys,Philadelphia Eagles,3,7\n",
        encoding="utf-8",
    )

    games = master_score_source(NFL, SEASON, out_root=tmp_path)

    assert len(games) == 2  # prior season filtered out
    scored = {g.game_key: g for g in games}
    # NFL game_key order is home-first (frozen contract).
    key = "20250907_1700_dallas_cowboys_philadelphia_eagles"
    assert key in scored
    assert (scored[key].home_score, scored[key].away_score) == (24, 20)
    unscored = scored["20250907_1700_green_bay_packers_chicago_bears"]
    assert unscored.home_score is None and unscored.away_score is None


def test_master_score_source_missing_file_is_empty(tmp_path):
    assert master_score_source(NFL, SEASON, out_root=tmp_path) == []

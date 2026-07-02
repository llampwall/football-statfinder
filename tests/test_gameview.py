"""Tests for the unified gameview builder and sidecar writer.

Synthetic schedule/ratings/odds inputs only — no network, no repo out/ writes
(everything lands in tmp_path).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from football_statfinder.common.game_key import build_game_key
from football_statfinder.leagues import CFB, NFL
from football_statfinder.pipeline.gameview import (
    FROZEN_RECORD_FIELDS,
    SagarinEntry,
    ScheduleGame,
    build_gameview,
    derive_favored_fields,
    derive_rating_fields,
    sagarin_map_from_rows,
    sagarin_token,
    write_gameview,
)
from football_statfinder.pipeline.sidecars import (
    SIDECAR_ENTRY_FIELDS,
    SIDECAR_TOP_LEVEL_FIELDS,
    SidecarError,
    build_sidecars,
)
from football_statfinder.sources.stats import team_stats_from_metrics_rows

# ---------------------------------------------------------------------------
# The single rating-vs-odds formula (bug 7 fix)
# ---------------------------------------------------------------------------


class TestDeriveRatingFields:
    def test_home_favorite_hand_computed(self):
        # rating_diff = (25 + 3) - 18 = 10; home line -7
        out = derive_rating_fields(25.0, 18.0, 3.0, -7.0)
        assert out["rating_diff"] == 10.0
        assert out["favored_side"] == "HOME"
        assert out["spread_favored_team"] == -7.0
        # home-centric: 10 + (-7) = 3 (model likes home 3 more than market)
        assert out["rating_vs_odds"] == 3.0
        assert out["rating_diff_favored_team"] == 10.0
        assert out["rating_vs_odds_favored_team"] == 3.0

    def test_away_favorite_hand_computed(self):
        # rating_diff = (18 + 2) - 25 = -5; home line +3 (away favored by 3)
        out = derive_rating_fields(18.0, 25.0, 2.0, 3.0)
        assert out["rating_diff"] == -5.0
        assert out["favored_side"] == "AWAY"
        assert out["spread_favored_team"] == -3.0
        assert out["rating_vs_odds"] == -2.0
        assert out["rating_diff_favored_team"] == 5.0
        assert out["rating_vs_odds_favored_team"] == 2.0

    def test_neutral_site_pick_em(self):
        # hfa 0 (neutral), spread 0 (pick-em -> HOME by convention)
        out = derive_rating_fields(20.0, 18.0, 0.0, 0.0)
        assert out["rating_diff"] == 2.0
        assert out["favored_side"] == "HOME"
        assert out["spread_favored_team"] == 0.0
        assert out["rating_vs_odds"] == 2.0
        assert out["rating_vs_odds_favored_team"] == 2.0

    def test_missing_ratings_keep_market_fields(self):
        out = derive_rating_fields(None, 25.0, 2.0, -7.0)
        assert out["favored_side"] == "HOME"
        assert out["spread_favored_team"] == -7.0
        assert out["rating_diff"] is None
        assert out["rating_vs_odds"] is None
        assert out["rating_vs_odds_favored_team"] is None

    def test_missing_spread_keeps_rating_diff_only(self):
        out = derive_rating_fields(25.0, 18.0, 3.0, None)
        assert out["rating_diff"] == 10.0
        assert out["favored_side"] is None
        assert out["rating_vs_odds"] is None
        assert out["rating_diff_favored_team"] is None

    @pytest.mark.parametrize(
        "home_pr,away_pr,hfa,spread",
        [(25.0, 18.0, 3.0, -7.0), (18.0, 25.0, 2.0, 3.0), (21.0, 21.0, 2.0, 1.5), (20.0, 18.0, 0.0, 0.0)],
    )
    def test_favored_fields_consistent_with_home_centric_sign(self, home_pr, away_pr, hfa, spread):
        out = derive_rating_fields(home_pr, away_pr, hfa, spread)
        if out["favored_side"] == "HOME":
            assert out["rating_vs_odds_favored_team"] == pytest.approx(out["rating_vs_odds"])
            assert out["rating_diff_favored_team"] == pytest.approx(out["rating_diff"])
        else:
            assert out["rating_vs_odds_favored_team"] == pytest.approx(-out["rating_vs_odds"])
            assert out["rating_diff_favored_team"] == pytest.approx(-out["rating_diff"])


def test_derive_favored_fields_sign_convention():
    assert derive_favored_fields(-7.0) == ("HOME", -7.0)
    assert derive_favored_fields(3.0) == ("AWAY", -3.0)
    assert derive_favored_fields(0.0) == ("HOME", 0.0)
    assert derive_favored_fields(None) == (None, None)


# ---------------------------------------------------------------------------
# Full build fixtures
# ---------------------------------------------------------------------------

KICK_EARLY = datetime(2026, 9, 20, 17, 0, tzinfo=timezone.utc)
KICK_LATE = datetime(2026, 9, 20, 20, 15, tzinfo=timezone.utc)


def _metrics_row(team: str, **overrides) -> dict:
    row = {
        "Team": team,
        "RY(O)": "150.0", "R(O)_RY": "1", "PY(O)": "250.0", "R(O)_PY": "2",
        "TY(O)": "400.0", "R(O)_TY": "1", "RY(D)": "90.0", "R(D)_RY": "1",
        "PY(D)": "210.0", "R(D)_PY": "2", "TY(D)": "300.0", "R(D)_TY": "1",
        "TO": "1.0", "PF": 57, "PA": 40, "SU": "2-0", "ATS": "2-0-0",
    }
    row.update(overrides)
    return row


def _nfl_inputs():
    teams = ["Buffalo Bills", "New York Jets", "Miami Dolphins", "New England Patriots"]
    team_stats = team_stats_from_metrics_rows(NFL, [_metrics_row(t) for t in teams])
    sagarin = sagarin_map_from_rows(
        NFL,
        [
            {"team_norm": "Buffalo Bills", "pr": 25.0, "rank": 1, "sos": 5.0, "sos_rank": 10, "hfa": 2.0},
            {"team_norm": "New York Jets", "pr": 18.0, "rank": 20, "sos": 4.0, "sos_rank": 15, "hfa": 2.0},
            {"team_norm": "Miami Dolphins", "pr": 21.0, "rank": 9, "sos": 3.0, "sos_rank": 20, "hfa": 2.0},
            {"team_norm": "New England Patriots", "pr": 19.0, "rank": 15, "sos": 2.0, "sos_rank": 25, "hfa": 2.0},
        ],
    )
    schedule = [
        ScheduleGame(
            season=2026, week=3, kickoff_utc=KICK_LATE,
            home_team_raw="MIA", away_team_raw="NE",
            home_team_norm="Miami Dolphins", away_team_norm="New England Patriots",
            source_uid="2026_03_NE_MIA", extra={"game_id": "2026_03_NE_MIA", "gsis": 123},
        ),
        ScheduleGame(
            season=2026, week=3, kickoff_utc=KICK_EARLY,
            home_team_raw="BUF", away_team_raw="NYJ",
            home_team_norm="Buffalo Bills", away_team_norm="New York Jets",
            source_uid="2026_03_NYJ_BUF", extra={"game_id": "2026_03_NYJ_BUF"},
        ),
    ]
    buf_key = build_game_key(NFL, KICK_EARLY, "Buffalo Bills", "New York Jets")
    mia_key = build_game_key(NFL, KICK_LATE, "Miami Dolphins", "New England Patriots")
    odds = {
        buf_key: {
            "spread_home_relative": -4.0, "total": 44.5,
            "moneyline_home": -190, "moneyline_away": 165,
            "odds_source": "theoddsapi", "is_closing": False,
            "snapshot_at": "2026-09-19T10:00:00+00:00",
        },
    }
    return schedule, team_stats, sagarin, odds, buf_key, mia_key


# ---------------------------------------------------------------------------
# The frozen frontend contract
# ---------------------------------------------------------------------------

# Enumerated independently from the legacy builders (src/gameview_build.py
# record dict == src/gameview_build_cfb.py OUTPUT_COLUMNS). If this set ever
# drifts from the builder's, the frontend contract broke.
LEGACY_RECORD_FIELDS = {
    "season", "week", "kickoff_iso_utc", "game_key", "source_uid",
    "home_team_raw", "home_team_norm", "away_team_raw", "away_team_norm",
    "spread_home_relative", "total", "moneyline_home", "moneyline_away",
    "odds_source", "is_closing", "snapshot_at",
    "home_pr", "home_pr_rank", "away_pr", "away_pr_rank",
    "home_sos", "away_sos", "home_sos_rank", "away_sos_rank",
    "hfa", "rating_diff", "rating_vs_odds", "favored_side",
    "spread_favored_team", "rating_diff_favored_team", "rating_vs_odds_favored_team",
    "home_pf_pg", "home_pa_pg", "home_ry_pg", "home_py_pg", "home_ty_pg",
    "home_ry_allowed_pg", "home_py_allowed_pg", "home_ty_allowed_pg",
    "home_to_margin_pg", "home_su", "home_ats",
    "home_rush_rank", "home_pass_rank", "home_tot_off_rank",
    "home_rush_def_rank", "home_pass_def_rank", "home_tot_def_rank",
    "away_pf_pg", "away_pa_pg", "away_ry_pg", "away_py_pg", "away_ty_pg",
    "away_ry_allowed_pg", "away_py_allowed_pg", "away_ty_allowed_pg",
    "away_to_margin_pg", "away_su", "away_ats",
    "away_rush_rank", "away_pass_rank", "away_tot_off_rank",
    "away_rush_def_rank", "away_pass_def_rank", "away_tot_def_rank",
    "raw_sources",
}


class TestGameviewBuild:
    def test_frozen_field_set(self):
        assert set(FROZEN_RECORD_FIELDS) == LEGACY_RECORD_FIELDS
        schedule, team_stats, sagarin, odds, _, _ = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        for record in build.records:
            assert set(record.keys()) == LEGACY_RECORD_FIELDS

    def test_game_key_and_kickoff(self):
        schedule, team_stats, sagarin, odds, buf_key, mia_key = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        keys = [rec["game_key"] for rec in build.records]
        # NFL keys are home-first; sorted by kickoff.
        assert keys == [buf_key, mia_key]
        assert buf_key == "20260920_1700_buffalo_bills_new_york_jets"
        assert build.records[0]["kickoff_iso_utc"] == "2026-09-20T17:00:00+00:00"

    def test_ordering_is_deterministic_and_input_order_independent(self):
        schedule, team_stats, sagarin, odds, _, _ = _nfl_inputs()
        build_a = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        build_b = build_gameview(
            NFL, 2026, 3, schedule=list(reversed(schedule)), team_stats=team_stats, sagarin=sagarin, odds=odds
        )
        assert build_a.records == build_b.records

    def test_kickoff_tie_breaks_on_game_key(self):
        schedule, team_stats, sagarin, odds, _, _ = _nfl_inputs()
        same_kick = [
            ScheduleGame(
                season=2026, week=3, kickoff_utc=KICK_EARLY,
                home_team_raw=g.home_team_raw, away_team_raw=g.away_team_raw,
                home_team_norm=g.home_team_norm, away_team_norm=g.away_team_norm,
            )
            for g in schedule
        ]
        build = build_gameview(NFL, 2026, 3, schedule=same_kick, team_stats=team_stats, sagarin=sagarin, odds={})
        keys = [rec["game_key"] for rec in build.records]
        assert keys == sorted(keys)

    def test_single_formula_applied_to_record(self):
        schedule, team_stats, sagarin, odds, buf_key, mia_key = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        rec = next(r for r in build.records if r["game_key"] == buf_key)
        # rating_diff = (25 + 2) - 18 = 9; rvo = 9 + (-4) = 5
        assert rec["hfa"] == 2.0
        assert rec["rating_diff"] == 9.0
        assert rec["rating_vs_odds"] == 5.0
        assert rec["favored_side"] == "HOME"
        assert rec["spread_favored_team"] == -4.0
        assert rec["rating_diff_favored_team"] == 9.0
        assert rec["rating_vs_odds_favored_team"] == 5.0
        # no-odds game: ratings only
        rec2 = next(r for r in build.records if r["game_key"] == mia_key)
        assert rec2["rating_diff"] == 4.0  # (21 + 2) - 19
        assert rec2["favored_side"] is None
        assert rec2["rating_vs_odds"] is None

    def test_stats_and_sagarin_join(self):
        schedule, team_stats, sagarin, odds, buf_key, _ = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        rec = next(r for r in build.records if r["game_key"] == buf_key)
        assert rec["home_ry_pg"] == 150.0
        assert rec["home_rush_rank"] == 1
        assert rec["home_su"] == "2-0"
        assert rec["home_ats"] == "2-0-0"
        assert rec["home_pr"] == 25.0
        assert rec["away_pr_rank"] == 20
        assert rec["moneyline_home"] == -190
        assert rec["total"] == 44.5
        assert rec["odds_source"] == "theoddsapi"

    def test_raw_sources_contract(self):
        schedule, team_stats, sagarin, odds, buf_key, mia_key = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        rec = next(r for r in build.records if r["game_key"] == buf_key)
        raw = rec["raw_sources"]
        # Frontend reads these exact keys (web/js/game_metrics.js etc.).
        assert raw["schedule_row"] == {"game_id": "2026_03_NYJ_BUF"}
        assert raw["sagarin_row_home"]["team"] == "Buffalo Bills"
        assert raw["sagarin_row_home"]["hfa"] == 2.0
        assert raw["sagarin_row_away"]["team"] == "New York Jets"
        assert raw["odds_row"]["spread_home_relative"] == -4.0
        assert raw["league_metrics_home"]["Team"] == "Buffalo Bills"
        rec2 = next(r for r in build.records if r["game_key"] == mia_key)
        assert rec2["raw_sources"]["odds_row"] is None

    def test_cfb_fbs_filter_drops_games_missing_stats(self):
        team_stats = team_stats_from_metrics_rows(CFB, [_metrics_row("Alpha"), _metrics_row("Beta")])
        schedule = [
            ScheduleGame(
                season=2026, week=3, kickoff_utc=KICK_EARLY,
                home_team_raw="Alpha", away_team_raw="Beta",
                home_team_norm="Alpha", away_team_norm="Beta",
            ),
            ScheduleGame(
                season=2026, week=3, kickoff_utc=KICK_LATE,
                home_team_raw="Alpha", away_team_raw="FCS Team",
                home_team_norm="Alpha", away_team_norm="FCS Team",
            ),
        ]
        build = build_gameview(CFB, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin={}, odds={})
        assert len(build.records) == 1
        assert build.receipt["skipped_non_fbs"] == 1
        # CFB game_key is away-first (frozen contract).
        assert build.records[0]["game_key"] == "20260920_1700_beta_alpha"

    def test_missing_stats_emit_none_not_zero(self):
        schedule, _, sagarin, odds, buf_key, _ = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats={}, sagarin=sagarin, odds=odds)
        rec = next(r for r in build.records if r["game_key"] == buf_key)
        assert rec["home_ry_pg"] is None
        assert rec["home_su"] is None

    def test_write_outputs(self, tmp_path):
        schedule, team_stats, sagarin, odds, _, _ = _nfl_inputs()
        build = build_gameview(NFL, 2026, 3, schedule=schedule, team_stats=team_stats, sagarin=sagarin, odds=odds)
        jsonl_path, csv_path = write_gameview(NFL, 2026, 3, build, out_dir=tmp_path)
        assert jsonl_path.name == "games_week_2026_3.jsonl"
        lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert set(lines[0].keys()) == LEGACY_RECORD_FIELDS
        frame = pd.read_csv(csv_path)
        assert list(frame.columns) == list(FROZEN_RECORD_FIELDS)
        assert (tmp_path / "gameview_build_receipt.json").exists()


def test_sagarin_token_overrides_apply_to_cfb_only():
    assert sagarin_token(CFB, "Mississippi") == "olemiss"
    assert sagarin_token(NFL, "Buffalo Bills") == NFL.merge_key("Buffalo Bills")


def test_sagarin_map_accepts_rank_or_pr_rank():
    mapping = sagarin_map_from_rows(NFL, [{"team_norm": "Buffalo Bills", "pr": 25.0, "rank": 3}])
    entry = mapping[NFL.merge_key("Buffalo Bills")]
    assert isinstance(entry, SagarinEntry)
    assert entry.pr_rank == 3


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def _schedule_master() -> pd.DataFrame:
    rows = [
        # 2026 YTD
        dict(league="NFL", season=2026, week=1, game_type="REG",
             kickoff_iso_utc="2026-09-06T17:00:00+00:00",
             home_team_norm="Buffalo Bills", away_team_norm="Miami Dolphins",
             home_score=30, away_score=20),
        dict(league="NFL", season=2026, week=2, game_type="REG",
             kickoff_iso_utc="2026-09-13T20:00:00+00:00",
             home_team_norm="New England Patriots", away_team_norm="Buffalo Bills",
             home_score=20, away_score=27),
        dict(league="NFL", season=2026, week=1, game_type="REG",
             kickoff_iso_utc="2026-09-06T20:00:00+00:00",
             home_team_norm="New York Jets", away_team_norm="New England Patriots",
             home_score=14, away_score=21),
        # the current game (week 3, unplayed)
        dict(league="NFL", season=2026, week=3, game_type="REG",
             kickoff_iso_utc="2026-09-20T17:00:00+00:00",
             home_team_norm="Buffalo Bills", away_team_norm="New York Jets",
             home_score=None, away_score=None),
        # previous season
        dict(league="NFL", season=2025, week=17, game_type="REG",
             kickoff_iso_utc="2025-12-28T18:00:00+00:00",
             home_team_norm="Buffalo Bills", away_team_norm="New York Jets",
             home_score=31, away_score=10),
        # non-REG row must be ignored
        dict(league="NFL", season=2025, week=19, game_type="POST",
             kickoff_iso_utc="2026-01-10T18:00:00+00:00",
             home_team_norm="Buffalo Bills", away_team_norm="Miami Dolphins",
             home_score=24, away_score=17),
    ]
    return pd.DataFrame(rows)


def _sagarin_master() -> pd.DataFrame:
    rows = [
        # week 1 only for 2026 -> nearest-week fallback must cover week 2/3.
        dict(league="NFL", season=2026, week=1, team_norm="Buffalo Bills",
             pr=25.0, rank=1, sos=5.0, sos_rank=None),
        dict(league="NFL", season=2026, week=1, team_norm="Miami Dolphins",
             pr=21.0, rank=9, sos=3.0, sos_rank=None),
        dict(league="NFL", season=2026, week=1, team_norm="New England Patriots",
             pr=19.0, rank=15, sos=2.0, sos_rank=None),
        dict(league="NFL", season=2026, week=1, team_norm="New York Jets",
             pr=18.0, rank=20, sos=4.0, sos_rank=None),
        dict(league="NFL", season=2025, week=17, team_norm="Buffalo Bills",
             pr=24.0, rank=2, sos=4.5, sos_rank=8),
        dict(league="NFL", season=2025, week=17, team_norm="New York Jets",
             pr=15.0, rank=28, sos=3.5, sos_rank=12),
    ]
    return pd.DataFrame(rows)


def _game_record() -> dict:
    return {
        "game_key": "20260920_1700_buffalo_bills_new_york_jets",
        "kickoff_iso_utc": "2026-09-20T17:00:00+00:00",
        "season": 2026,
        "week": 3,
        "home_team_norm": "Buffalo Bills",
        "away_team_norm": "New York Jets",
    }


class TestSidecars:
    def _build(self, tmp_path, **kwargs):
        return build_sidecars(
            NFL, 2026, 3,
            games=[_game_record()],
            schedule_master=_schedule_master(),
            sagarin_master=_sagarin_master(),
            out_dir=tmp_path,
            **kwargs,
        )

    def test_writes_sidecar_with_frozen_schema(self, tmp_path):
        receipt = self._build(tmp_path)
        assert receipt["sidecars_written"] == 1
        path = tmp_path / "game_schedules" / "20260920_1700_buffalo_bills_new_york_jets.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert tuple(payload.keys()) == SIDECAR_TOP_LEVEL_FIELDS
        for section in ("home_ytd", "away_ytd", "home_prev", "away_prev"):
            for entry in payload[section]:
                assert tuple(entry.keys()) == SIDECAR_ENTRY_FIELDS

    def test_ytd_cutoff_excludes_current_game(self, tmp_path):
        self._build(tmp_path)
        payload = json.loads(
            (tmp_path / "game_schedules" / "20260920_1700_buffalo_bills_new_york_jets.json").read_text(
                encoding="utf-8"
            )
        )
        home_weeks = [entry["week"] for entry in payload["home_ytd"]]
        assert home_weeks == [1, 2]  # sorted, week-3 game excluded
        entry = payload["home_ytd"][0]
        assert entry["opp"] == "Miami Dolphins"
        assert entry["site"] == "H"
        assert entry["pf"] == 30 and entry["pa"] == 20
        assert entry["result"] == "W"
        assert entry["ats"] is None and entry["to_margin"] is None  # filled by backfill later

    def test_nearest_week_sagarin_fallback(self, tmp_path):
        self._build(tmp_path)
        payload = json.loads(
            (tmp_path / "game_schedules" / "20260920_1700_buffalo_bills_new_york_jets.json").read_text(
                encoding="utf-8"
            )
        )
        week2 = next(entry for entry in payload["home_ytd"] if entry["week"] == 2)
        # no week-2 master rows: values fall back to the week-1 snapshot.
        assert week2["pr"] == 25.0
        assert week2["pr_rank"] == 1
        assert week2["opp_pr"] == 19.0  # NE, week-1 fallback

    def test_missing_sos_rank_filled_by_dense_rank_fallback(self, tmp_path):
        self._build(tmp_path)
        payload = json.loads(
            (tmp_path / "game_schedules" / "20260920_1700_buffalo_bills_new_york_jets.json").read_text(
                encoding="utf-8"
            )
        )
        week1 = next(entry for entry in payload["home_ytd"] if entry["week"] == 1)
        # sos values (BUF 5, NYJ 4, MIA 3, NE 2) -> BUF sos_rank 1, MIA opp 3
        assert week1["sos_rank"] == 1
        assert week1["opp_sos_rank"] == 3

    def test_prev_season_full_and_post_rows_excluded(self, tmp_path):
        self._build(tmp_path)
        payload = json.loads(
            (tmp_path / "game_schedules" / "20260920_1700_buffalo_bills_new_york_jets.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(payload["home_prev"]) == 1  # the POST row is excluded
        prev = payload["home_prev"][0]
        assert prev["season"] == 2025 and prev["week"] == 17
        assert prev["pr"] == 24.0 and prev["opp_pr"] == 15.0

    def test_receipt_written(self, tmp_path):
        receipt = self._build(tmp_path)
        assert (tmp_path / "sidecars_receipt.json").exists()
        assert receipt["sagarin_coverage_fraction"] == 1.0
        assert receipt["join_issues"] == []

    def test_strict_raises_on_missing_schedule_join(self, tmp_path):
        bad_game = dict(_game_record(), home_team_norm="Chicago Bears", game_key="20260920_1700_chicago_bears_new_york_jets")
        with pytest.raises(SidecarError):
            build_sidecars(
                NFL, 2026, 3,
                games=[bad_game],
                schedule_master=_schedule_master(),
                sagarin_master=_sagarin_master(),
                out_dir=tmp_path,
            )

    def test_non_strict_records_join_issue(self, tmp_path):
        bad_game = dict(_game_record(), home_team_norm="Chicago Bears", game_key="x")
        receipt = build_sidecars(
            NFL, 2026, 3,
            games=[bad_game],
            schedule_master=_schedule_master(),
            sagarin_master=_sagarin_master(),
            out_dir=tmp_path,
            strict=False,
        )
        assert receipt["sidecars_written"] == 0
        assert receipt["join_issues"][0]["reason"] == "missing_schedule"

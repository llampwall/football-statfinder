"""Tests for the league stats providers (synthetic frames, no network)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from football_statfinder.config import ConfigError, Settings
from football_statfinder.leagues import CFB, NFL
from football_statfinder.sources import stats as stats_mod
from football_statfinder.sources.stats import (
    CfbStatsProvider,
    LEAGUE_METRICS_COLUMNS,
    NflStatsProvider,
    build_cfb_league_metrics_rows,
    build_nfl_league_metrics_rows,
    fetch_teamrankings_turnover_margin,
    load_league_metrics_csv,
    team_stats_from_metrics_rows,
    write_league_metrics_csv,
)


# ---------------------------------------------------------------------------
# NFL fixtures
# ---------------------------------------------------------------------------


def _nfl_schedule() -> pd.DataFrame:
    rows = [
        # week 1 (played)
        dict(season=2026, week=1, game_type="REG", start_time_utc="2026-09-06T17:00:00Z",
             home_team="BUF", away_team="MIA", home_score=30, away_score=20, spread_line=7.0),
        dict(season=2026, week=1, game_type="REG", start_time_utc="2026-09-06T20:00:00Z",
             home_team="NYJ", away_team="NE", home_score=14, away_score=21, spread_line=-3.0),
        # week 2 (played)
        dict(season=2026, week=2, game_type="REG", start_time_utc="2026-09-13T17:00:00Z",
             home_team="MIA", away_team="NYJ", home_score=27, away_score=17, spread_line=1.0),
        dict(season=2026, week=2, game_type="REG", start_time_utc="2026-09-13T20:00:00Z",
             home_team="NE", away_team="BUF", home_score=20, away_score=27, spread_line=-2.5),
        # week 3 (upcoming)
        dict(season=2026, week=3, game_type="REG", start_time_utc="2026-09-20T17:00:00Z",
             home_team="BUF", away_team="NYJ", home_score=None, away_score=None, spread_line=-4.0),
        dict(season=2026, week=3, game_type="REG", start_time_utc="2026-09-20T20:00:00Z",
             home_team="MIA", away_team="NE", home_score=None, away_score=None, spread_line=2.0),
    ]
    return pd.DataFrame(rows)


def _nfl_team_week_stats() -> pd.DataFrame:
    per_team = {
        # team: (ry, py, opp_ry, opp_py, turnovers, takeaways)
        "BUF": (150, 250, 90, 210, 1, 2),
        "MIA": (120, 200, 130, 190, 1, 1),
        "NYJ": (150, 180, 110, 230, 1, 1),
        "NE": (100, 220, 120, 250, 1, 1),
    }
    rows = []
    for week in (1, 2):
        for team, (ry, py, dry, dpy, tov, take) in per_team.items():
            rows.append(
                dict(
                    season=2026,
                    team=team,
                    week=week,
                    opponent="X",
                    rushing_yards=ry,
                    net_passing_yards=py,
                    opponent_rushing_yards=dry,
                    opponent_net_passing_yards=dpy,
                    turnovers=tov,
                    takeaways=take,
                )
            )
    return pd.DataFrame(rows)


def _nfl_rows() -> list[dict]:
    return build_nfl_league_metrics_rows(
        2026,
        3,
        schedule=_nfl_schedule(),
        team_week_stats=_nfl_team_week_stats(),
        turnover_by_display={"Buffalo Bills": 1.5},
    )


def _row(rows: list[dict], team: str) -> dict:
    return next(r for r in rows if r["Team"] == team)


class TestNflLeagueMetrics:
    def test_header_and_team_count(self):
        rows = _nfl_rows()
        assert len(rows) == 4
        assert list(rows[0].keys()) == LEAGUE_METRICS_COLUMNS

    def test_per_game_values_and_totals(self):
        buf = _row(_nfl_rows(), "Buffalo Bills")
        assert buf["RY(O)"] == "150.0"
        assert buf["PY(O)"] == "250.0"
        assert buf["TY(O)"] == "400.0"
        assert buf["RY(D)"] == "90.0"
        assert buf["PY(D)"] == "210.0"
        assert buf["TY(D)"] == "300.0"

    def test_su_and_ats_records_from_schedule(self):
        rows = _nfl_rows()
        buf = _row(rows, "Buffalo Bills")
        # BUF: W 30-20 (line -7, margin +10 -> cover), W 27-20 away (line -2.5... )
        assert buf["SU"] == "2-0"
        assert buf["ATS"] == "2-0-0"
        nyj = _row(rows, "New York Jets")
        assert nyj["SU"] == "0-2"
        assert nyj["ATS"] == "0-2-0"

    def test_dense_ranks_share_tied_rank(self):
        rows = _nfl_rows()
        # Offense rushing: BUF 150, NYJ 150 (tie -> both rank 1), MIA 120 (2), NE 100 (3)
        assert _row(rows, "Buffalo Bills")["R(O)_RY"] == "1"
        assert _row(rows, "New York Jets")["R(O)_RY"] == "1"
        assert _row(rows, "Miami Dolphins")["R(O)_RY"] == "2"
        assert _row(rows, "New England Patriots")["R(O)_RY"] == "3"
        # Total offense: BUF 400 (1), NYJ 330 (2), MIA 320 & NE 320 tie (3)
        assert _row(rows, "Miami Dolphins")["R(O)_TY"] == "3"
        assert _row(rows, "New England Patriots")["R(O)_TY"] == "3"

    def test_teamrankings_value_overrides_computed_turnovers(self):
        rows = _nfl_rows()
        assert _row(rows, "Buffalo Bills")["TO"] == "1.5"  # injected TR value
        assert _row(rows, "Miami Dolphins")["TO"] == "0.0"  # computed fallback


def test_nfl_pf_pa_are_season_totals():
    rows = _nfl_rows()
    buf = _row(rows, "Buffalo Bills")
    assert buf["PF"] == 30 + 27
    assert buf["PA"] == 20 + 20


class TestNflProvider:
    def test_provider_uses_injected_fetchers_only(self):
        provider = NflStatsProvider(
            fetch_schedule=lambda season: _nfl_schedule(),
            fetch_team_week_stats=lambda season: _nfl_team_week_stats(),
            fetch_turnover_margin=lambda as_of_week=None: {"Buffalo Bills": 1.5},
        )
        mapping = provider.team_stats(2026, 3)
        token = NFL.merge_key("Buffalo Bills")
        assert mapping[token].ry_pg == 150.0
        assert mapping[token].rush_rank == 1
        assert mapping[token].su == "2-0"

    def test_as_of_week_is_forwarded(self):
        seen = {}

        def fetch_turnovers(as_of_week=None):
            seen["as_of_week"] = as_of_week
            return {}

        provider = NflStatsProvider(
            fetch_schedule=lambda season: _nfl_schedule(),
            fetch_team_week_stats=lambda season: _nfl_team_week_stats(),
            fetch_turnover_margin=fetch_turnovers,
        )
        provider.league_metrics_rows(2026, 3, as_of_week=2)
        assert seen["as_of_week"] == 2


def test_teamrankings_as_of_week_limitation_is_logged(caplog):
    table = pd.DataFrame({"Team": ["Buffalo"], "2026": [1.2]})
    with caplog.at_level(logging.WARNING):
        result = fetch_teamrankings_turnover_margin(read_html=lambda url: [table], as_of_week=4)
    assert any("as_of_week" in message for message in caplog.messages)
    assert result == {"Buffalo Bills": 1.2}


# ---------------------------------------------------------------------------
# CFB
# ---------------------------------------------------------------------------


def _cfb_fbs_teams() -> list[dict]:
    return [{"school": "Alpha"}, {"school": "Beta"}, {"school": "Gamma"}]


def _cfb_game(home: str, away: str, home_pts: int, away_pts: int, home_stats: dict, away_stats: dict) -> dict:
    def entry(team: str, points: int, stat_map: dict) -> dict:
        return {
            "team": team,
            "points": points,
            "stats": [{"category": key, "stat": str(value)} for key, value in stat_map.items()],
        }

    return {"teams": [entry(home, home_pts, home_stats), entry(away, away_pts, away_stats)]}


def _cfb_rows() -> list[dict]:
    games = [
        _cfb_game(
            "Alpha", "Beta", 35, 21,
            {"rushingYards": 200, "netPassingYards": 300, "totalYards": 500, "turnovers": 1},
            {"rushingYards": 100, "netPassingYards": 250, "totalYards": 350, "turnovers": 3},
        ),
        _cfb_game(
            "Gamma", "Alpha", 10, 24,
            {"rushingYards": 150, "netPassingYards": 150, "totalYards": 300, "turnovers": 2},
            {"rushingYards": 180, "netPassingYards": 220, "totalYards": 400, "turnovers": 0},
        ),
    ]
    return build_cfb_league_metrics_rows(2026, 3, fbs_teams=_cfb_fbs_teams(), team_game_stats=games)


class TestCfbLeagueMetrics:
    def test_per_game_averages_and_su(self):
        alpha = _row(_cfb_rows(), "Alpha")
        assert alpha["RY(O)"] == "190.0"  # (200 + 180) / 2
        assert alpha["TY(O)"] == "450.0"
        assert alpha["PF"] == "29.5"  # per-game for CFB (season-1 divergence)
        assert alpha["SU"] == "2-0"

    def test_ats_intentionally_blank(self):
        for row in _cfb_rows():
            assert row["ATS"] == ""

    def test_dense_ranks(self):
        rows = _cfb_rows()
        assert _row(rows, "Alpha")["R(O)_TY"] == "1"
        # Gamma played one game (300 TY); Beta 350 -> Beta 2, Gamma 3
        assert _row(rows, "Beta")["R(O)_TY"] == "2"
        assert _row(rows, "Gamma")["R(O)_TY"] == "3"

    def test_turnover_margin_per_game(self):
        alpha = _row(_cfb_rows(), "Alpha")
        # takeaways (3 + 2) - giveaways (1 + 0) = 4 over 2 games
        assert alpha["TO"] == "2.0"


class TestCfbProvider:
    def test_missing_api_key_fails_loud(self):
        provider = CfbStatsProvider(settings=Settings())
        with pytest.raises(ConfigError):
            provider.league_metrics_rows(2026, 3)

    def test_collects_weeks_below_build_week(self):
        calls = []

        def fetch_stats(season, week, key):
            calls.append(week)
            return []

        provider = CfbStatsProvider(
            settings=Settings(cfbd_api_key="test-key"),
            fetch_fbs_teams=lambda season, key: _cfb_fbs_teams(),
            fetch_team_game_stats=fetch_stats,
        )
        rows = provider.league_metrics_rows(2026, 4)
        assert calls == [1, 2, 3]
        assert len(rows) == 3

    def test_as_of_week_overrides_window(self):
        calls = []
        provider = CfbStatsProvider(
            settings=Settings(cfbd_api_key="test-key"),
            fetch_fbs_teams=lambda season, key: _cfb_fbs_teams(),
            fetch_team_game_stats=lambda season, week, key: calls.append(week) or [],
        )
        provider.league_metrics_rows(2026, 10, as_of_week=3)
        assert calls == [1, 2]


# ---------------------------------------------------------------------------
# Shared: CSV roundtrip + TeamStats parsing
# ---------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path):
    rows = _nfl_rows()
    path = tmp_path / "league_metrics_2026_3.csv"
    write_league_metrics_csv(NFL, 2026, 3, rows, path=path)
    assert path.exists()
    mapping = load_league_metrics_csv(NFL, 2026, 3, path=path)
    token = NFL.merge_key("Buffalo Bills")
    stats = mapping[token]
    assert stats.ry_pg == 150.0
    assert stats.tot_off_rank == 1
    assert stats.su == "2-0"
    assert stats.ats == "2-0-0"
    assert stats.pf_pg == 57.0  # NFL totals quirk survives the roundtrip


def test_team_stats_blank_values_parse_to_none():
    rows = [
        {
            "Team": "Alpha",
            "RY(O)": "",
            "R(O)_RY": "",
            "PY(O)": "",
            "R(O)_PY": "",
            "TY(O)": "",
            "R(O)_TY": "",
            "RY(D)": "",
            "R(D)_RY": "",
            "PY(D)": "",
            "R(D)_PY": "",
            "TY(D)": "",
            "R(D)_TY": "",
            "TO": "",
            "PF": "",
            "PA": "",
            "SU": "0-0",
            "ATS": "",
        }
    ]
    mapping = team_stats_from_metrics_rows(CFB, rows)
    stats = mapping[CFB.merge_key("Alpha")]
    assert stats.ry_pg is None
    assert stats.rush_rank is None
    assert stats.ats is None
    assert stats.su == "0-0"


def test_no_src_imports():
    """HARD RULE: the new package must never import from src.*."""
    import inspect

    source = inspect.getsource(stats_mod)
    assert "from src" not in source and "import src" not in source

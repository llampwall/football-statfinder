import pandas as pd
import pytest

from src.common.metrics import (
    compute_ats,
    compute_su,
    dense_rank,
    rating_diff,
    rating_vs_odds,
    team_centric_spread,
)


def test_rating_diff_with_hfa():
    assert pytest.approx(rating_diff(85.0, 82.5, 1.5)) == 4.0


def test_team_centric_spread_home_and_away():
    assert team_centric_spread(-3.5, "HOME") == pytest.approx(-3.5)
    assert team_centric_spread(-3.5, "AWAY") == pytest.approx(3.5)


def test_rating_vs_odds_alignment():
    diff = rating_diff(82.0, 79.0, 1.0)
    spread = team_centric_spread(-2.5, "HOME")
    assert pytest.approx(rating_vs_odds(diff, spread)) == pytest.approx(1.5)


def test_dense_rank_ties_dense_ordering():
    series = pd.Series({"A": 10, "B": 10, "C": 8})
    ranks = dense_rank(series, higher_is_better=True)
    assert ranks["A"] == 1
    assert ranks["B"] == 1
    assert ranks["C"] == 2


def test_compute_su_formats_w_l_t():
    games = [
        {"team_points": 21, "opp_points": 14},
        {"team_points": 10, "opp_points": 17},
        {"team_points": 24, "opp_points": 24},
    ]
    assert compute_su(games) == "1-1-1"


def test_compute_ats_w_l_p():
    games = [
        {"team_margin": 7, "team_line": -3.0},
        {"team_margin": 3, "team_line": -3.0},
        {"team_margin": -4, "team_line": 2.5},
    ]
    assert compute_ats(games) == "1-1-1"

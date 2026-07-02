"""Pure football metric helpers shared across modules.

Purpose:
    Hold the numeric helpers that can be unit tested in isolation and reused
    by league metrics, odds comparisons, and game view builders.
Inputs:
    Lightweight numeric parameters or in-memory game dictionaries.
Outputs:
    Derived metrics such as rating deltas, dense ranks, SU/ATS records.
Source(s) of truth:
    Existing business logic from the legacy gameview_weekpack pipeline.
Example:
    >>> rating_diff(84.2, 80.1, 1.5)
    5.599999999999994
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd


def rating_diff(pr_home: float, pr_away: float, hfa: float) -> float:
    """Calculate home advantage adjusted rating differential."""
    return (float(pr_home) + float(hfa)) - float(pr_away)


def team_centric_spread(home_relative_spread: float, side: Literal["HOME", "AWAY"]) -> float:
    """Transform a home-relative spread into the perspective of the given side."""
    return float(home_relative_spread) if side == "HOME" else -float(home_relative_spread)


def rating_vs_odds(rating_diff_val: float, team_centric_spread_val: float) -> float:
    """Compare model rating differential to a team-centric market line."""
    return float(rating_diff_val) - (-float(team_centric_spread_val))


def dense_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Dense rank a numeric series (ties share rank, next rank increments by 1)."""
    ordered = series.sort_values(ascending=not higher_is_better)
    ranks = {}
    rank = 0
    last_val: Optional[float] = None
    for index, value in ordered.items():
        if pd.isna(value):
            continue
        if last_val is None or value != last_val:
            rank += 1
            last_val = value
        ranks[index] = rank
    return pd.Series(ranks)


def compute_su(games: Iterable[dict]) -> str:
    """Return straight-up record W-L(-T) from game dictionaries."""
    w = l = t = 0
    for game in games:
        margin = float(game.get("team_points", 0) or 0) - float(game.get("opp_points", 0) or 0)
        if margin > 0:
            w += 1
        elif margin < 0:
            l += 1
        else:
            t += 1
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _ats_outcome(team_margin: float, team_line: Optional[float]) -> Optional[str]:
    if team_line is None or pd.isna(team_line):
        return None
    diff = float(team_margin) + float(team_line)
    if abs(diff) < 1e-9:
        return "P"
    return "W" if diff > 0 else "L"


def compute_ats(games: Iterable[dict]) -> str:
    """Return against-the-spread record W-L-P from team-centric lines."""
    w = l = p = 0
    for game in games:
        outcome = _ats_outcome(game.get("team_margin", 0.0), game.get("team_line"))
        if outcome == "W":
            w += 1
        elif outcome == "L":
            l += 1
        elif outcome == "P":
            p += 1
    return f"{w}-{l}-{p}"


__all__ = [
    "rating_diff",
    "team_centric_spread",
    "rating_vs_odds",
    "dense_rank",
    "compute_su",
    "compute_ats",
]

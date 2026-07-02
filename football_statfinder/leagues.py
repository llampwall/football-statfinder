"""League configuration objects.

The season-1 tree duplicated every pipeline stage as an NFL copy and a CFB
copy that differed only in constants and name normalization. This module is
the single place those differences live; every stage takes a ``League`` and
stays league-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .common import team_names, team_names_cfb


@dataclass(frozen=True)
class League:
    code: str  # lowercase id used in paths: "nfl" | "cfb"
    display: str  # uppercase id used in logs/NOTIFY lines: "NFL" | "CFB"
    odds_sport_key: str  # The Odds API sport key
    sagarin_url: str
    # Sagarin parse acceptance gate: exact count or inclusive (min, max) range.
    sagarin_expected_teams: Optional[int]
    sagarin_team_range: Optional[Tuple[int, int]]
    # Frozen frontend contract (season 1): NFL game_keys are
    # {ts}_{home}_{away}; CFB game_keys are {ts}_{away}_{home}.
    game_key_home_first: bool
    normalize_display: Callable[[str], str]
    merge_key: Callable[[str], str]
    # Normalizer for The Odds API participant names (CFB odds feeds use
    # different labels than schedules; NFL's display normalizer covers both).
    normalize_odds_name: Callable[[str], str]


NFL = League(
    code="nfl",
    display="NFL",
    odds_sport_key="americanfootball_nfl",
    sagarin_url="http://sagarin.com/sports/nflsend.htm",
    sagarin_expected_teams=32,
    sagarin_team_range=None,
    game_key_home_first=True,
    normalize_display=team_names.normalize_team_display,
    merge_key=team_names.team_merge_key,
    normalize_odds_name=team_names.normalize_team_display,
)

CFB = League(
    code="cfb",
    display="CFB",
    odds_sport_key="americanfootball_ncaaf",
    sagarin_url="http://sagarin.com/sports/cfsend.htm",
    sagarin_expected_teams=None,
    sagarin_team_range=(120, 140),
    game_key_home_first=False,
    normalize_display=team_names_cfb.normalize_team_name_cfb,
    merge_key=team_names_cfb.team_merge_key_cfb,
    normalize_odds_name=team_names_cfb.normalize_team_name_cfb_odds,
)

LEAGUES = {league.code: league for league in (NFL, CFB)}


def get_league(code: str) -> League:
    """Resolve a league by id (case-insensitive); raises on unknowns."""
    league = LEAGUES.get((code or "").strip().lower())
    if league is None:
        known = ", ".join(sorted(LEAGUES))
        raise ValueError(f"unknown league {code!r} (known: {known})")
    return league


__all__ = ["CFB", "LEAGUES", "League", "NFL", "get_league"]

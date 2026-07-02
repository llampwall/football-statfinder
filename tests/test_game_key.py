"""Tests for the canonical game_key builder and slug rules."""

from __future__ import annotations

from datetime import datetime, timezone

from football_statfinder.common.game_key import build_game_key, kickoff_stamp, slug
from football_statfinder.leagues import CFB, NFL

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)


def test_slug_basic():
    assert slug("Green Bay Packers") == "green_bay_packers"


def test_slug_ampersand_and_punctuation():
    assert slug("Texas A&M") == "texas_aandm"
    assert slug("Hawai'i") == "hawai_i"
    assert slug("Miami (OH)") == "miami_oh"


def test_slug_collapses_runs_and_trims():
    assert slug("  San   Jose  State  ") == "san_jose_state"
    assert slug("--49ers--") == "49ers"


def test_nfl_game_key_is_home_first():
    key = build_game_key(NFL, KICKOFF, "Green Bay Packers", "Chicago Bears")
    assert key == "20260913_1700_green_bay_packers_chicago_bears"


def test_cfb_game_key_is_away_first():
    key = build_game_key(CFB, KICKOFF, "Ohio State", "Texas A&M")
    assert key == "20260913_1700_texas_aandm_ohio_state"


def test_kickoff_stamp_coerces_to_utc():
    from datetime import timedelta, timezone as tz

    eastern = tz(timedelta(hours=-4))
    local = datetime(2026, 9, 13, 13, 0, tzinfo=eastern)
    assert kickoff_stamp(local) == "20260913_1700"


def test_kickoff_stamp_naive_treated_as_utc():
    naive = datetime(2026, 9, 13, 17, 0)
    assert kickoff_stamp(naive) == "20260913_1700"

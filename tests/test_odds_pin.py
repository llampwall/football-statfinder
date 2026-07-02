"""Tests for the league-parameterized odds pin stage (synthetic data, no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from football_statfinder import paths
from football_statfinder.common.game_key import build_game_key
from football_statfinder.config import OddsSettings, Settings
from football_statfinder.leagues import CFB, NFL, League
from football_statfinder.pipeline.odds_pin import (
    load_schedule_master,
    make_schedule_game,
    pin_to_schedule,
)

KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
KICKOFF_ISO = "2026-09-13T17:00:00Z"


def _settings(**odds_kwargs: Any) -> Settings:
    return Settings(the_odds_api_key="test-key", odds=OddsSettings(**odds_kwargs))


def _token(league: League, name: str) -> str:
    return league.merge_key(league.normalize_odds_name(name))


def _spread_record(
    league: League,
    home: str,
    away: str,
    *,
    event_start: Any = KICKOFF_ISO,
    fetch_ts: str = "2026-09-10T12:00:00Z",
    book: str = "fanduel",
    home_point: float = -3.5,
) -> Dict[str, Any]:
    """Raw staging record shaped like ``ingest_raw`` output (provider roles)."""
    home_token = _token(league, home)
    away_token = _token(league, away)
    return {
        "fetch_ts": fetch_ts,
        "event_id": "evt-1",
        "event_start": event_start,
        "book": book,
        "book_title": book.title(),
        "market": "spreads",
        "market_payload": {
            "key": "spreads",
            "last_update": fetch_ts,
            "outcomes": [
                {"name": home, "token": home_token, "price": -110, "point": home_point},
                {"name": away, "token": away_token, "price": -110, "point": -home_point},
            ],
        },
        "home_raw": home,
        "away_raw": away,
        "home_norm": league.normalize_odds_name(home),
        "away_norm": league.normalize_odds_name(away),
        "home_token": home_token,
        "away_token": away_token,
        "league": league.display,
        "source": "the-odds-api",
    }


def _nfl_game(week: int = 2):
    return make_schedule_game(
        NFL,
        season=2026,
        week=week,
        kickoff=KICKOFF,
        home="Green Bay Packers",
        away="Chicago Bears",
    )


def test_exact_match_pins_canonical_nfl_key(tmp_path: Path):
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears")
    result = pin_to_schedule(NFL, [record], [_nfl_game()], _settings(), out_root=tmp_path)

    assert result["counts"]["pinned"] == 1
    assert result["counts"]["unmatched"] == 0
    pinned = result["pinned_records"][0]
    # Bug 4 fix: NFL keys are {ts}_{home}_{away}; the legacy pin wrote away-first.
    assert pinned["game_key"] == "20260913_1700_green_bay_packers_chicago_bears"
    assert pinned["game_key"] == build_game_key(
        NFL, KICKOFF, "Green Bay Packers", "Chicago Bears"
    )
    assert pinned["role_swapped"] is False
    assert pinned["season"] == 2026 and pinned["week"] == 2
    assert pinned["line"]["spread_home_relative"] == -3.5
    assert pinned["line"]["favored_side"] == "HOME"
    ledger = paths.odds_pinned_jsonl("nfl", 2026, out_root=tmp_path)
    assert ledger.exists()
    assert result["pinned_paths"] == [ledger]


def test_cfb_pinned_key_is_away_first(tmp_path: Path):
    game = make_schedule_game(
        CFB, season=2026, week=3, kickoff=KICKOFF, home="Ohio State", away="Texas A&M"
    )
    record = _spread_record(CFB, "Ohio State", "Texas A&M")
    result = pin_to_schedule(CFB, [record], [game], _settings(), out_root=tmp_path)

    assert result["counts"]["pinned"] == 1
    pinned = result["pinned_records"][0]
    # CFB keys are away-first (frozen contract) built from normalized names.
    assert pinned["game_key"] == game.game_key
    assert pinned["game_key"] == build_game_key(CFB, KICKOFF, game.home_norm, game.away_norm)
    assert pinned["game_key"].startswith("20260913_1700_texas_aandm")


def test_role_swap_pins_with_home_line_intact(tmp_path: Path):
    # Provider lists Bears as home (+3.5); schedule says Packers host at -3.5.
    record = _spread_record(NFL, "Chicago Bears", "Green Bay Packers", home_point=3.5)
    result = pin_to_schedule(NFL, [record], [_nfl_game()], _settings(), out_root=tmp_path)

    assert result["counts"]["pinned"] == 1
    pinned = result["pinned_records"][0]
    assert pinned["role_swapped"] is True
    assert pinned["home_norm"] == "Green Bay Packers"
    # Legacy inverted the spread on role-swapped matches; the schedule home
    # team's own outcome must drive spread_home_relative.
    assert pinned["line"]["spread_home_relative"] == -3.5
    assert pinned["line"]["favored_side"] == "HOME"


def test_role_swap_disabled_quarantines(tmp_path: Path):
    record = _spread_record(NFL, "Chicago Bears", "Green Bay Packers")
    settings = _settings(role_swap_tolerance=False)
    result = pin_to_schedule(NFL, [record], [_nfl_game()], settings, out_root=tmp_path)

    assert result["counts"]["pinned"] == 0
    assert result["counts"]["unmatched"] == 1
    assert result["unmatched_records"][0]["why"] == "no_candidate"
    assert result["counts"]["unmatched_reasons"] == {"no_candidate": 1}


def test_kickoff_delta_bound_rejects_then_accepts(tmp_path: Path):
    event_start = (KICKOFF - timedelta(hours=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears", event_start=event_start)

    tight = pin_to_schedule(
        NFL, [record], [_nfl_game()], _settings(pin_max_kickoff_delta_hours=36.0),
        out_root=tmp_path,
    )
    assert tight["counts"]["pinned"] == 0
    assert tight["unmatched_records"][0]["why"] == "no_candidate"

    loose = pin_to_schedule(
        NFL, [record], [_nfl_game()], _settings(pin_max_kickoff_delta_hours=48.0),
        out_root=tmp_path,
    )
    assert loose["counts"]["pinned"] == 1


def test_invalid_event_time_reason_and_quarantine_file(tmp_path: Path):
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears", event_start=None)
    result = pin_to_schedule(NFL, [record], [_nfl_game()], _settings(), out_root=tmp_path)

    assert result["counts"]["unmatched"] == 1
    assert result["unmatched_records"][0]["why"] == "invalid_event_time"
    unmatched_path = result["unmatched_path"]
    assert unmatched_path is not None and unmatched_path.exists()
    quarantined = [json.loads(line) for line in unmatched_path.read_text("utf-8").splitlines()]
    assert quarantined[0]["why"] == "invalid_event_time"
    assert unmatched_path.parent == paths.odds_unmatched_dir("nfl", out_root=tmp_path)


def test_ambiguous_duplicate_schedule_rows(tmp_path: Path):
    games = [_nfl_game(week=2), _nfl_game(week=3)]  # same key/kickoff, twice
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears")
    result = pin_to_schedule(NFL, [record], games, _settings(), out_root=tmp_path)

    assert result["counts"]["pinned"] == 0
    assert result["unmatched_records"][0]["why"] == "ambiguous"
    assert result["counts"]["candidate_sets_multi"] == 1


def test_ledger_dedupes_across_two_pins(tmp_path: Path):
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears")
    settings = _settings()
    games = [_nfl_game()]

    first = pin_to_schedule(NFL, [record], games, settings, out_root=tmp_path)
    assert first["counts"]["appended"] == 1
    assert first["counts"]["duplicates_skipped"] == 0

    # Legacy appended duplicates every run (unbounded ledger growth).
    second = pin_to_schedule(NFL, [record], games, settings, out_root=tmp_path)
    assert second["counts"]["pinned"] == 1
    assert second["counts"]["appended"] == 0
    assert second["counts"]["duplicates_skipped"] == 1

    ledger = paths.odds_pinned_jsonl("nfl", 2026, out_root=tmp_path)
    assert len(ledger.read_text("utf-8").splitlines()) == 1

    # A genuinely new snapshot (different fetch_ts) still appends.
    fresher = _spread_record(
        NFL, "Green Bay Packers", "Chicago Bears", fetch_ts="2026-09-11T12:00:00Z"
    )
    third = pin_to_schedule(NFL, [fresher], games, settings, out_root=tmp_path)
    assert third["counts"]["appended"] == 1
    assert len(ledger.read_text("utf-8").splitlines()) == 2


def test_schedule_rows_as_mappings_and_master_loader(tmp_path: Path):
    csv_path = tmp_path / "nfl_schedule_master.csv"
    csv_path.write_text(
        "season,week,kickoff_iso_utc,home_team_norm,away_team_norm,neutral_site\n"
        f"2026,2,{KICKOFF_ISO},Green Bay Packers,Chicago Bears,False\n"
        "bad,row,not-a-time,,,\n",
        encoding="utf-8",
    )
    games = load_schedule_master(NFL, csv_path)
    assert len(games) == 1
    assert games[0].game_key == "20260913_1700_green_bay_packers_chicago_bears"

    # Mapping rows are accepted directly by pin_to_schedule.
    row = {
        "season": 2026,
        "week": 2,
        "kickoff_iso_utc": KICKOFF_ISO,
        "home_team_norm": "Green Bay Packers",
        "away_team_norm": "Chicago Bears",
        "neutral_site": False,
    }
    record = _spread_record(NFL, "Green Bay Packers", "Chicago Bears")
    result = pin_to_schedule(NFL, [record], [row], _settings(), out_root=tmp_path)
    assert result["counts"]["pinned"] == 1

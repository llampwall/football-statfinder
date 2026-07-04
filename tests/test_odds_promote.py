"""Tests for the league-parameterized odds promotion stage (synthetic data)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from football_statfinder import paths
from football_statfinder.common.game_key import build_game_key
from football_statfinder.config import OddsSettings, Settings
from football_statfinder.leagues import NFL, League
from football_statfinder.pipeline.odds_promote import pick_latest_before, promote_week

SEASON = 2026
WEEK = 2
KICKOFF = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
KICKOFF_ISO = "2026-09-13T17:00:00Z"

HOME = "Green Bay Packers"
AWAY = "Chicago Bears"
GAME_KEY = build_game_key(NFL, KICKOFF, HOME, AWAY)


def _settings(**odds_kwargs: Any) -> Settings:
    return Settings(odds=OddsSettings(**odds_kwargs))


def _week_row(
    league: League = NFL,
    *,
    game_key: str = GAME_KEY,
    kickoff_iso: Optional[str] = KICKOFF_ISO,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "season": SEASON,
        "week": WEEK,
        "game_key": game_key,
        "home_team_norm": HOME,
        "away_team_norm": AWAY,
    }
    if kickoff_iso is not None:
        row["kickoff_iso_utc"] = kickoff_iso
    return row


def _spread_line(spread: float) -> Dict[str, Any]:
    return {
        "spread_home_relative": spread,
        "favored_side": "HOME" if spread < 0 else "AWAY",
        "spread_favored_team": spread if spread < 0 else -abs(spread),
        "home_price": -110,
        "away_price": -110,
        "raw_outcomes": [],
    }


def _pinned(
    *,
    game_key: str = GAME_KEY,
    market: str = "spreads",
    book: str = "fanduel",
    fetch_ts: str,
    line: Optional[Dict[str, Any]] = None,
    season: int = SEASON,
    week: int = WEEK,
) -> Dict[str, Any]:
    return {
        "fetch_ts": fetch_ts,
        "source": "the-odds-api",
        "season": season,
        "week": week,
        "game_key": game_key,
        "market": market,
        "book": book,
        "line": line if line is not None else _spread_line(-3.5),
        "home_norm": HOME,
        "away_norm": AWAY,
        "kickoff_utc": KICKOFF_ISO,
        "role_swapped": False,
        "raw_event": {"event_id": "evt-1", "event_start": KICKOFF_ISO},
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stage(tmp_path: Path, rows: Iterable[Dict[str, Any]], pinned: Iterable[Dict[str, Any]]) -> None:
    _write_jsonl(paths.games_week_jsonl("nfl", SEASON, WEEK, out_root=tmp_path), rows)
    _write_jsonl(paths.odds_pinned_jsonl("nfl", SEASON, out_root=tmp_path), pinned)


def _read_rows(tmp_path: Path) -> list[dict]:
    path = paths.games_week_jsonl("nfl", SEASON, WEEK, out_root=tmp_path)
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_latest_policy_picks_freshest_snapshot(tmp_path: Path):
    _stage(
        tmp_path,
        [_week_row()],
        [
            _pinned(fetch_ts="2026-09-10T10:00:00Z", line=_spread_line(-3.5)),
            _pinned(fetch_ts="2026-09-12T10:00:00Z", line=_spread_line(-4.5)),
        ],
    )
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["promoted_games"] == 1
    assert result["policy"] == "latest_by_fetch_ts"
    assert result["coverage_ok"] is True
    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -4.5
    assert row["snapshot_at"] == "2026-09-12T10:00:00Z"
    assert row["is_closing"] is False
    assert row["raw_sources"]["odds_row"]["markets"]["spreads"]["book"] == "fanduel"


def test_latest_policy_across_books_takes_newest(tmp_path: Path):
    _stage(
        tmp_path,
        [_week_row()],
        [
            _pinned(book="fanduel", fetch_ts="2026-09-12T10:00:00Z", line=_spread_line(-3.5)),
            _pinned(book="draftkings", fetch_ts="2026-09-12T11:00:00Z", line=_spread_line(-4.0)),
        ],
    )
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["by_book"] == {"draftkings": 1}
    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -4.0
    assert row["odds_source"] == "draftkings"


def test_closing_policy_prefers_latest_pre_kickoff(tmp_path: Path):
    _stage(
        tmp_path,
        [_week_row()],
        [
            _pinned(book="fanduel", fetch_ts="2026-09-13T15:00:00Z", line=_spread_line(-3.5)),
            # Post-kickoff snapshot from another book would win latest_by_fetch_ts.
            _pinned(book="draftkings", fetch_ts="2026-09-13T18:00:00Z", line=_spread_line(-6.5)),
        ],
    )
    settings = _settings(select_policy="closing_pre_kickoff")
    result = promote_week(NFL, SEASON, WEEK, settings, out_root=tmp_path)

    assert result["policy"] == "closing_pre_kickoff"
    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -3.5
    assert row["is_closing"] is True
    assert row["snapshot_at"] == "2026-09-13T15:00:00Z"


def test_closing_policy_falls_back_when_all_post_kickoff(tmp_path: Path):
    _stage(
        tmp_path,
        [_week_row()],
        [_pinned(fetch_ts="2026-09-13T18:00:00Z", line=_spread_line(-6.5))],
    )
    settings = _settings(select_policy="closing_pre_kickoff")
    promote_week(NFL, SEASON, WEEK, settings, out_root=tmp_path)

    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -6.5
    assert row["is_closing"] is False


def test_kickoff_fallback_from_pinned_records(tmp_path: Path):
    # CFB keeper: row lacks a kickoff, pinned record's kickoff_utc drives the cutoff.
    _stage(
        tmp_path,
        [_week_row(kickoff_iso=None)],
        [
            _pinned(book="fanduel", fetch_ts="2026-09-13T15:00:00Z", line=_spread_line(-3.5)),
            _pinned(book="draftkings", fetch_ts="2026-09-13T18:00:00Z", line=_spread_line(-6.5)),
        ],
    )
    settings = _settings(select_policy="closing_pre_kickoff")
    promote_week(NFL, SEASON, WEEK, settings, out_root=tmp_path)

    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -3.5
    assert row["is_closing"] is True


def test_all_markets_merge_into_row(tmp_path: Path):
    _stage(
        tmp_path,
        [_week_row()],
        [
            _pinned(market="spreads", fetch_ts="2026-09-12T10:00:00Z", line=_spread_line(-3.5)),
            _pinned(
                market="totals",
                fetch_ts="2026-09-12T10:00:00Z",
                line={"total_points": 44.5, "over_price": -110, "under_price": -110, "raw_outcomes": []},
            ),
            _pinned(
                market="h2h",
                fetch_ts="2026-09-12T10:00:00Z",
                line={"moneyline_home": -180, "moneyline_away": 155, "raw_outcomes": []},
            ),
        ],
    )
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["by_market"] == {"spreads": 1, "totals": 1, "h2h": 1}
    row = _read_rows(tmp_path)[0]
    assert row["spread_home_relative"] == -3.5
    assert row["total"] == 44.5
    assert row["moneyline_home"] == -180
    assert row["moneyline_away"] == 155
    csv_path = paths.games_week_csv("nfl", SEASON, WEEK, out_root=tmp_path)
    assert csv_path.exists()


def test_coverage_gate_fails_when_pinned_keys_never_land(tmp_path: Path):
    # Bug-4 failure class: ledger says this week has odds, but its game_keys
    # match no week row, so promotion silently does nothing.
    dead_key = "20260913_1700_chicago_bears_green_bay_packers"  # away-first legacy key
    _stage(
        tmp_path,
        [_week_row()],
        [_pinned(game_key=dead_key, fetch_ts="2026-09-12T10:00:00Z")],
    )
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["promoted_games"] == 0
    assert result["coverage_ok"] is False
    assert dead_key in result["coverage"]["pinned_keys_missing_from_rows"]
    assert result["json_path"] is None  # outputs untouched when nothing promoted

    receipt_path = paths.odds_promotion_receipt_json("nfl", SEASON, WEEK, out_root=tmp_path)
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text("utf-8"))
    assert receipt["coverage"]["ok"] is False
    assert receipt["coverage"]["week_pinned_games"] == 1


def test_coverage_gate_passes_and_receipt_written(tmp_path: Path):
    _stage(tmp_path, [_week_row()], [_pinned(fetch_ts="2026-09-12T10:00:00Z")])
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["coverage_ok"] is True
    receipt = json.loads(Path(result["receipt_path"]).read_text("utf-8"))
    assert receipt["coverage"]["ok"] is True
    assert receipt["stats"]["promoted_games"] == 1
    assert receipt["samples"]["promoted_game_keys"] == [GAME_KEY]


def test_promotion_disabled_skips(tmp_path: Path):
    _stage(tmp_path, [_week_row()], [_pinned(fetch_ts="2026-09-12T10:00:00Z")])
    settings = _settings(promotion_enable=False)
    result = promote_week(NFL, SEASON, WEEK, settings, out_root=tmp_path)

    assert result["skipped_reason"] == "promotion_disabled"
    assert result["promoted_games"] == 0
    assert not paths.odds_promotion_receipt_json("nfl", SEASON, WEEK, out_root=tmp_path).exists()


def test_unsupported_policy_falls_back_to_latest(tmp_path: Path):
    _stage(tmp_path, [_week_row()], [_pinned(fetch_ts="2026-09-12T10:00:00Z")])
    result = promote_week(NFL, SEASON, WEEK, _settings(select_policy="bogus"), out_root=tmp_path)

    assert result["policy"] == "latest_by_fetch_ts"
    assert result["promoted_games"] == 1


def test_other_week_records_are_ignored(tmp_path: Path):
    other_key = build_game_key(NFL, datetime(2026, 9, 20, 17, 0, tzinfo=timezone.utc), HOME, AWAY)
    _stage(
        tmp_path,
        [_week_row()],
        [
            _pinned(fetch_ts="2026-09-12T10:00:00Z"),
            _pinned(game_key=other_key, week=WEEK + 1, fetch_ts="2026-09-19T10:00:00Z"),
        ],
    )
    result = promote_week(NFL, SEASON, WEEK, _settings(), out_root=tmp_path)

    assert result["season_records"] == 2
    assert result["current_week_records"] == 1
    assert result["other_week_records"] == 1
    assert result["promoted_games"] == 1


def test_pick_latest_before_breaks_fetch_ts_ties_by_book():
    # Same fetch captured several books: the greatest book label wins (legacy
    # promoters used max(..., key=(fetch_ts, book)); order-independence matters
    # because the ledger scan order is arbitrary).
    cutoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    fanduel = {"fetch_ts": "2026-09-13T16:00:00Z", "book": "fanduel"}
    williamhill = {"fetch_ts": "2026-09-13T16:00:00Z", "book": "williamhill_us"}
    assert pick_latest_before([fanduel, williamhill], cutoff) is williamhill
    assert pick_latest_before([williamhill, fanduel], cutoff) is williamhill
    # A strictly newer fetch still beats any book label.
    newer = {"fetch_ts": "2026-09-13T16:30:00Z", "book": "aaa_book"}
    assert pick_latest_before([williamhill, newer], cutoff) is newer

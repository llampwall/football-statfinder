"""Tests for raw odds ingestion (stubbed fetch, no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pytest

from football_statfinder import paths
from football_statfinder.config import ConfigError, OddsSettings, Settings
from football_statfinder.leagues import NFL
from football_statfinder.pipeline.odds_ingest import ingest_raw

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _event() -> dict:
    return {
        "id": "evt-1",
        "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "Green Bay Packers",
        "away_team": "Chicago Bears",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2026-09-10T11:59:00Z",
                        "outcomes": [
                            {"name": "Green Bay Packers", "price": -110, "point": -3.5},
                            {"name": "Chicago Bears", "price": -110, "point": 3.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": "2026-09-10T11:59:00Z",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 44.5},
                            {"name": "Under", "price": -110, "point": 44.5},
                        ],
                    },
                ],
            }
        ],
    }


def test_missing_api_key_fails_loud():
    # Bug 8 fix: the legacy twins returned an empty result and kept the run green.
    with pytest.raises(ConfigError):
        ingest_raw(NFL, Settings(), fetch_events=lambda *_: [])


def test_staging_disabled_is_explicit_noop():
    settings = Settings(odds=OddsSettings(staging_enable=False))  # no key needed
    result = ingest_raw(NFL, settings, fetch_events=lambda *_: [])
    assert result["skipped_reason"] == "staging_disabled"
    assert result["records"] == [] and result["path"] is None


def test_fetch_exceptions_propagate():
    def _boom(api_key: str, sport_key: str) -> List[dict]:
        raise RuntimeError("provider down")

    settings = Settings(the_odds_api_key="test-key")
    with pytest.raises(RuntimeError):
        ingest_raw(NFL, settings, fetch_events=_boom)


def test_stub_fetch_stages_normalized_records(tmp_path: Path):
    calls: List[Any] = []

    def _stub(api_key: str, sport_key: str) -> List[dict]:
        calls.append((api_key, sport_key))
        return [_event()]

    settings = Settings(the_odds_api_key="test-key")
    result = ingest_raw(NFL, settings, fetch_events=_stub, now=NOW, out_root=tmp_path)

    assert calls == [("test-key", "americanfootball_nfl")]
    assert result["fetch_ts"] == "2026-09-10T12:00:00Z"
    assert result["counts"] == {"books": {"fanduel": 1}, "markets": {"spreads": 1, "totals": 1}}

    expected_path = paths.odds_raw_dir("nfl", out_root=tmp_path) / "20260910T120000Z.jsonl"
    assert result["path"] == expected_path
    staged = [json.loads(line) for line in expected_path.read_text("utf-8").splitlines()]
    assert len(staged) == 2

    spreads = next(rec for rec in staged if rec["market"] == "spreads")
    assert spreads["league"] == "NFL"
    assert spreads["home_norm"] == "Green Bay Packers"
    assert spreads["event_start"] == "2026-09-13T17:00:00Z"
    outcome = spreads["market_payload"]["outcomes"][0]
    assert outcome["token"] == NFL.merge_key("Green Bay Packers")
    assert outcome["point"] == -3.5


def test_empty_events_write_nothing(tmp_path: Path):
    settings = Settings(the_odds_api_key="test-key")
    result = ingest_raw(NFL, settings, fetch_events=lambda *_: [], now=NOW, out_root=tmp_path)
    assert result["records"] == []
    assert result["path"] is None
    assert not paths.odds_raw_dir("nfl", out_root=tmp_path).exists()

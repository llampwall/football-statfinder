"""Tests for football_statfinder.pipeline.ats and the Odds API cache guards.

Every test passes ``out_root=tmp_path`` (the paths.py override) so nothing
touches the repo ``out/`` tree, and no test performs network I/O — the paid
tier is a stub, and the client tests inject an ``http_get`` that fails the
test if it is ever called.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from football_statfinder import paths
from football_statfinder.config import BackfillSettings, OddsSettings, Settings
from football_statfinder.leagues import NFL
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.pipeline import ats
from football_statfinder.sources.odds_api import OddsApiClient

SEASON = 2025
KICKOFF = "2025-10-05T17:00:00Z"


def make_settings(*, cache_only: bool = False, ats_source: str = "auto") -> Settings:
    return Settings(
        the_odds_api_key="test-key",
        odds=OddsSettings(cache_only=cache_only),
        backfill=BackfillSettings(ats_source=ats_source),
    )


def game_row(game_key, home, away, *, week, home_score=None, away_score=None,
             favored_side=None, spread=None, kickoff=KICKOFF, **extra):
    row = {
        "game_key": game_key,
        "season": SEASON,
        "week": week,
        "kickoff_iso_utc": kickoff,
        "home_team_norm": home,
        "away_team_norm": away,
        "home_score": home_score,
        "away_score": away_score,
        "favored_side": favored_side,
        "spread_favored_team": spread,
    }
    row.update(extra)
    return row


def write_week(out_root, week, rows):
    path = paths.games_week_jsonl(NFL.code, SEASON, week, out_root=out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def write_pinned(out_root, records):
    path = paths.odds_pinned_jsonl(NFL.code, SEASON, out_root=out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class StubApi:
    """Paid-tier stand-in recording every call."""

    def __init__(self, payload=None):
        self.payload = payload
        self.calls = []

    def resolve_closing_spread(self, season, week, game_row):
        self.calls.append((season, week, game_row.get("game_key")))
        return self.payload


# ---------------------------------------------------------------------------
# Blank sentinel (bug 19)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "—", "–", "-"])
def test_is_blank_true_for_sentinels_including_legacy_em_dash(value):
    assert ats.is_blank(value) is True


@pytest.mark.parametrize("value", ["3-2-0", "W", "0-0-1", 0, 1.5])
def test_is_blank_false_for_real_values(value):
    assert ats.is_blank(value) is False


# ---------------------------------------------------------------------------
# Cover math (W / L / P including push)
# ---------------------------------------------------------------------------


def test_compute_game_ats_home_favored_covers():
    payload = ats.compute_game_ats(30, 20, "HOME", 6.5)
    assert payload == {
        "home_ats": "W",
        "away_ats": "L",
        "to_margin_home": 3.5,
        "to_margin_away": -3.5,
    }


def test_compute_game_ats_push():
    payload = ats.compute_game_ats(27, 20, "HOME", 7)
    assert payload["home_ats"] == "P"
    assert payload["away_ats"] == "P"
    assert payload["to_margin_home"] == 0.0


def test_compute_game_ats_away_favored():
    payload = ats.compute_game_ats(20, 24, "AWAY", 3)
    assert payload["home_ats"] == "L"
    assert payload["away_ats"] == "W"
    assert payload["to_margin_away"] == 1.0


def test_compute_game_ats_spread_sign_is_magnitude():
    assert ats.compute_game_ats(30, 20, "HOME", -6.5) == ats.compute_game_ats(30, 20, "HOME", 6.5)


def test_compute_game_ats_rejects_garbage():
    assert ats.compute_game_ats(None, 20, "HOME", 3) is None
    assert ats.compute_game_ats(30, 20, "NEITHER", 3) is None
    assert ats.compute_game_ats(30, 20, "HOME", "n/a") is None


def test_build_team_ats_tallies_win_loss_push(tmp_path):
    # Week rows carry spread_favored_team in the promoted (negative) convention.
    write_week(tmp_path, 1, [
        game_row("g1", "Alpha", "Beta", week=1, home_score=30, away_score=20,
                 favored_side="HOME", spread=-3.0),
    ])
    write_week(tmp_path, 2, [
        # Away favorite pushes: margin -7, cover = 7 + (-7) = 0.
        game_row("g2", "Gamma", "Alpha", week=2, home_score=20, away_score=27,
                 favored_side="AWAY", spread=-7.0),
        # No spread: ignored entirely.
        game_row("g3", "Delta", "Echo", week=2, home_score=10, away_score=3),
    ])

    build = ats.build_team_ats(NFL, SEASON, 3, out_root=tmp_path)

    assert build.weeks_scanned == [1, 2]
    assert build.games_considered == 2
    alpha = build.stats[NFL.merge_key("Alpha")]
    assert (alpha.w, alpha.l, alpha.p) == (1, 0, 1)
    beta = build.stats[NFL.merge_key("Beta")]
    assert (beta.w, beta.l, beta.p) == (0, 1, 0)
    gamma = build.stats[NFL.merge_key("Gamma")]
    assert (gamma.w, gamma.l, gamma.p) == (0, 0, 1)
    assert NFL.merge_key("Delta") not in build.stats


def test_apply_ats_to_week_writes_records_and_normalizes_em_dash(tmp_path):
    write_week(tmp_path, 1, [
        game_row("g1", "Alpha", "Beta", week=1, home_score=30, away_score=20,
                 favored_side="HOME", spread=-3.0),
        game_row("g2", "Gamma", "Alpha", week=1, home_score=20, away_score=27,
                 favored_side="AWAY", spread=-7.0),
    ])
    # Current week rows carry the legacy em-dash placeholder from season 1.
    write_week(tmp_path, 2, [
        game_row("g3", "Alpha", "Beta", week=2, home_ats="—", away_ats="—"),
        game_row("g4", "Gamma", "Delta", week=2, home_ats="—", away_ats="—"),
    ])

    build = ats.build_team_ats(NFL, SEASON, 2, out_root=tmp_path)
    applied = ats.apply_ats_to_week(NFL, SEASON, 2, build, out_root=tmp_path)

    assert applied.rows_updated == 2
    assert applied.teams_in_week == 4
    assert applied.zero_lined == 1  # Delta has no lined games

    rows = read_jsonl(paths.games_week_jsonl(NFL.code, SEASON, 2, out_root=tmp_path)).rows
    by_key = {row["game_key"]: row for row in rows}
    assert by_key["g3"]["home_ats"] == "1-0-1"  # Alpha: W then push, always W-L-P
    assert by_key["g3"]["away_ats"] == "0-1-0"  # Beta
    assert by_key["g4"]["home_ats"] == "0-0-1"  # Gamma
    # Bug 19: unlined teams get the None sentinel, never a dash.
    assert by_key["g4"]["away_ats"] is None
    # The CSV artifact is rewritten alongside the JSONL.
    assert paths.games_week_csv(NFL.code, SEASON, 2, out_root=tmp_path).exists()


def test_apply_ats_to_week_no_rows_is_noop(tmp_path):
    applied = ats.apply_ats_to_week(NFL, SEASON, 5, ats.AtsBuildResult(), out_root=tmp_path)
    assert applied == ats.AtsApplyResult()


# ---------------------------------------------------------------------------
# Closing-spread resolution order (bug 4)
# ---------------------------------------------------------------------------


def _pinned_record(game_key, *, fetch_ts, spread_home_relative=-3.5, favored_side="HOME"):
    return {
        "market": "spreads",
        "game_key": game_key,
        "fetch_ts": fetch_ts,
        "book": "pinnacle",
        "line": {
            "spread_home_relative": spread_home_relative,
            "favored_side": favored_side,
            "spread_favored_team": spread_home_relative,
        },
        "home_norm": "Alpha",
        "away_norm": "Beta",
        "raw_event": {"event_id": "ev1"},
    }


def test_resolve_closing_spread_pinned_tier_wins_and_api_untouched(tmp_path):
    write_pinned(tmp_path, [
        _pinned_record("g1", fetch_ts="2025-10-05T16:00:00Z"),
    ])
    stub = StubApi(payload={"favored_team": "AWAY", "spread": 99.0})
    row = game_row("g1", "Alpha", "Beta", week=1)

    resolved = ats.resolve_closing_spread(
        NFL, SEASON, row, settings=make_settings(), api=stub, out_root=tmp_path
    )

    assert resolved == {
        "favored_team": "HOME",
        "spread": 3.5,
        "book": "pinnacle",
        "fetched_ts": "2025-10-05T16:00:00Z",
        "source": "pinned",
    }
    assert stub.calls == []


def test_resolve_closing_spread_falls_back_to_api_when_pinned_misses(tmp_path):
    stub = StubApi(payload={
        "favored_team": "AWAY", "spread": 6.0, "source": "history",
        "book": "fanduel", "fetched_ts": "2025-10-05T16:30:00Z",
    })
    row = game_row("g1", "Alpha", "Beta", week=1)

    resolved = ats.resolve_closing_spread(
        NFL, SEASON, row, settings=make_settings(), api=stub, out_root=tmp_path
    )

    assert resolved is not None
    assert resolved["source"] == "history"
    assert resolved["favored_team"] == "AWAY"
    assert resolved["spread"] == 6.0
    assert stub.calls == [(SEASON, 1, "g1")]


def test_resolve_closing_spread_ignores_post_kick_pinned_records(tmp_path):
    write_pinned(tmp_path, [
        _pinned_record("g1", fetch_ts="2025-10-05T18:00:00Z"),  # after kickoff
    ])
    stub = StubApi(payload=None)
    row = game_row("g1", "Alpha", "Beta", week=1)

    resolved = ats.resolve_closing_spread(
        NFL, SEASON, row, settings=make_settings(), api=stub, out_root=tmp_path
    )

    assert resolved is None
    assert stub.calls == [(SEASON, 1, "g1")]  # fell through to the paid tier


def test_resolve_closing_spread_source_pinned_never_calls_api(tmp_path):
    stub = StubApi(payload={"favored_team": "AWAY", "spread": 6.0})
    row = game_row("g1", "Alpha", "Beta", week=1)

    resolved = ats.resolve_closing_spread(
        NFL, SEASON, row,
        settings=make_settings(ats_source="pinned"),
        api=stub,
        out_root=tmp_path,
    )

    assert resolved is None
    assert stub.calls == []


def test_resolve_closing_spread_pick_em_from_raw_outcomes(tmp_path):
    record = _pinned_record("g1", fetch_ts="2025-10-05T16:00:00Z")
    record["line"] = {"raw_outcomes": [
        {"name": "Alpha", "point": 0.0},
        {"name": "Beta", "point": 0.0},
    ]}
    write_pinned(tmp_path, [record])
    row = game_row("g1", "Alpha", "Beta", week=1)

    resolved = ats.resolve_closing_spread(
        NFL, SEASON, row, settings=make_settings(), out_root=tmp_path
    )

    assert resolved is not None
    assert resolved["favored_team"] == "PICK"
    assert resolved["spread"] == 0.0


# ---------------------------------------------------------------------------
# The harvested API resolver end-to-end (pinned event id -> historical spread)
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for OddsApiClient; no network, records spread lookups."""

    league = NFL

    def __init__(self):
        self.hist_spread_calls = []

    def get_participants(self):
        teams = [{"name": f"Filler Team {i}"} for i in range(10)]
        return teams + [{"name": "Alpha"}, {"name": "Beta"}]

    def get_historical_events(self, snapshot_dt, *, commence_from=None,
                              commence_to=None, event_ids=None):
        return []

    def get_historical_spread(self, event_id, snapshot_iso, home, away, kickoff_dt):
        self.hist_spread_calls.append((event_id, home, away))
        return {
            "status": "ok",
            "favored_team": "HOME",
            "spread": 3.5,
            "book": "pinnacle",
            "fetched_ts": "2025-10-05T16:45:00Z",
            "source": "history",
        }


def test_ats_backfill_api_resolves_via_pinned_event_id(tmp_path):
    from football_statfinder.sources.ats_backfill_api import AtsBackfillApi

    write_pinned(tmp_path, [_pinned_record("g1", fetch_ts="2025-10-05T16:00:00Z")])
    client = FakeClient()
    api = AtsBackfillApi(NFL, make_settings(), client=client, out_root=tmp_path)
    row = game_row("g1", "Alpha", "Beta", week=5)

    resolved = api.resolve_closing_spread(SEASON, 5, row)

    assert resolved == {
        "favored_team": "HOME",
        "spread": 3.5,
        "source": "history",
        "book": "pinnacle",
        "fetched_ts": "2025-10-05T16:45:00Z",
    }
    # Event id came from the pinned ledger, not an events call.
    assert client.hist_spread_calls == [("ev1", "Alpha", "Beta")]


# ---------------------------------------------------------------------------
# cache_only blocks every paid call (settings.odds.cache_only)
# ---------------------------------------------------------------------------


def _no_network(*args, **kwargs):
    raise AssertionError("network call attempted with cache_only set")


def test_cache_only_blocks_all_paid_endpoints(tmp_path):
    client = OddsApiClient(
        NFL, make_settings(cache_only=True), out_root=tmp_path, http_get=_no_network
    )
    snapshot = "2025-10-05T17:00:00Z"

    assert client.get_historical_event_odds("ev1", snapshot, None) is None
    assert client.get_participants() is None
    assert client.get_historical_events(datetime(2025, 10, 5, tzinfo=timezone.utc)) is None
    assert client.get_current_spread("ev1", snapshot, "Alpha", "Beta") is None
    # And the wrapping spread call reports the cache miss instead of paying.
    payload = client.get_historical_spread(
        "ev1", snapshot, "Alpha", "Beta", datetime(2025, 10, 5, 17, tzinfo=timezone.utc)
    )
    assert payload["status"] != "ok"
    assert payload["reason"] == "cache_miss"


def test_cache_only_serves_from_disk_cache(tmp_path):
    snapshot = "2025-10-05T17:00:00Z"
    token = snapshot.replace(":", "").replace("/", "-")
    cache_dir = paths.hist_odds_cache_dir(NFL.code, out_root=tmp_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"ev1__{token}.json").write_text(
        json.dumps({"bookmakers": [{"key": "pinnacle", "markets": []}]}), encoding="utf-8"
    )
    client = OddsApiClient(
        NFL, make_settings(cache_only=True), out_root=tmp_path, http_get=_no_network
    )

    result = client.get_historical_event_odds("ev1", snapshot, None)

    assert result is not None
    bookmakers, _payload = result
    assert bookmakers == [{"key": "pinnacle", "markets": []}]


def test_missing_api_key_is_loud_not_empty(tmp_path):
    from football_statfinder.config import ConfigError

    settings = Settings(odds=OddsSettings(), backfill=BackfillSettings())  # no key
    client = OddsApiClient(NFL, settings, out_root=tmp_path, http_get=_no_network)
    with pytest.raises(ConfigError):
        client.get_participants()

"""Tests for football_statfinder.config: precedence, parsing, validation.

Every test passes an explicit ``environ`` mapping and a tmp ``env_file`` so
nothing here depends on the developer's real environment or the repo .env
(which the legacy src tree loads into os.environ as a side effect).
"""

from __future__ import annotations

import textwrap

import pytest

from football_statfinder.config import ConfigError, load_settings


def _write_env(tmp_path, body: str):
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent(body), encoding="utf-8")
    return env_file


def test_real_environment_beats_dotenv(tmp_path):
    env_file = _write_env(tmp_path, "ODDS_PIN_DAY_WINDOW=9\n")
    settings = load_settings(env_file=env_file, environ={"ODDS_PIN_DAY_WINDOW": "4"})
    assert settings.odds.pin_day_window == 4


def test_dotenv_used_when_env_unset(tmp_path):
    env_file = _write_env(tmp_path, "ODDS_PIN_DAY_WINDOW=9\n")
    settings = load_settings(env_file=env_file, environ={})
    assert settings.odds.pin_day_window == 9


def test_defaults_when_nothing_set(tmp_path):
    settings = load_settings(env_file=tmp_path / "missing.env", environ={})
    assert settings.the_odds_api_key is None
    assert settings.odds.staging_enable is True
    assert settings.odds.select_policy == "latest_by_fetch_ts"
    assert settings.odds.pin_day_window == 3
    assert settings.odds.pin_max_kickoff_delta_hours == 36.0
    assert settings.backfill.weeks == 2
    assert settings.backfill.promote_prev is False
    assert settings.week_force.applies_to("NFL") is False


def test_bool_parsing_rejects_garbage(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(
            env_file=tmp_path / "missing.env",
            environ={"ODDS_STAGING_ENABLE": "maybe"},
        )


def test_int_parsing_rejects_garbage(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(
            env_file=tmp_path / "missing.env",
            environ={"BACKFILL_WEEKS": "two"},
        )


def test_week_force_parsing_and_scope(tmp_path):
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={"WEEK_FORCE": "2026-3", "WEEK_FORCE_LEAGUE": "nfl"},
    )
    assert settings.week_force.season == 2026
    assert settings.week_force.week == 3
    assert settings.week_force.applies_to("NFL") is True
    assert settings.week_force.applies_to("CFB") is False


def test_week_force_all_scope(tmp_path):
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={"WEEK_FORCE": "2026:14", "WEEK_FORCE_LEAGUE": "ALL"},
    )
    assert settings.week_force.applies_to("NFL") is True
    assert settings.week_force.applies_to("CFB") is True


def test_week_force_garbage_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(
            env_file=tmp_path / "missing.env",
            environ={"WEEK_FORCE": "week three"},
        )


def test_require_raises_on_missing_secret(tmp_path):
    settings = load_settings(env_file=tmp_path / "missing.env", environ={})
    with pytest.raises(ConfigError, match="THE_ODDS_API_KEY"):
        settings.require("the_odds_api_key")


def test_require_passes_when_set(tmp_path):
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={"THE_ODDS_API_KEY": "k"},
    )
    settings.require("the_odds_api_key")


def test_banner_redacts_secrets(tmp_path):
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={"THE_ODDS_API_KEY": "supersecretvalue123"},
    )
    banner = settings.banner()
    assert "supersecretvalue123" not in banner
    assert "the_odds_api_key=set" in banner


def test_values_stripped_of_quotes_and_space(tmp_path):
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={"THE_ODDS_API_KEY": '  "abc123"  '},
    )
    assert settings.the_odds_api_key == "abc123"

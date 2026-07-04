"""Typed runtime configuration.

Replaces the ~20 loose ``getenv()`` flags of the season-1 tree. Differences
from the legacy behavior, all deliberate:

* Real environment variables beat ``.env`` (the legacy ``getenv`` loaded
  ``.env`` with ``override=True``, silently masking CI-provided values —
  REBUILD.md bug 17). ``os.environ`` is never mutated.
* Values are read once into a frozen dataclass and validated up front;
  missing secrets are a startup error at the stage that needs them, never an
  empty DataFrame (bug 8).
* The effective config prints as a redacted banner so every run records what
  it actually ran with.

Legacy flags that steered code this package does not carry (two Sagarin
generations, legacy odds snapshots, dry-run shims) are intentionally absent:
SAGARIN_STAGING_ENABLE, ODDS_LEGACY_JOIN_ENABLE, CFB_ATS_DRYRUN,
CFB_WRITE_DEBUG_SCHEDULE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

try:  # pragma: no cover - import guard mirrors requirements.txt
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover
    dotenv_values = None  # type: ignore[assignment]

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"0", "false", "no", "off", "disabled"}

_SECRET_FIELDS = {"the_odds_api_key", "cfbd_api_key", "discord_webhook_url"}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unparseable at startup."""


def _clean(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip().strip('"').strip("'").strip()
    return text or None


def _as_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    text = _clean(env.get(key))
    if text is None:
        return default
    lowered = text.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ConfigError(f"{key}={text!r} is not a boolean (use 1/0, true/false, on/off)")


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    text = _clean(env.get(key))
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise ConfigError(f"{key}={text!r} is not an integer") from exc


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    text = _clean(env.get(key))
    if text is None:
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise ConfigError(f"{key}={text!r} is not a number") from exc


def _as_path(env: Mapping[str, str], key: str) -> Optional[Path]:
    text = _clean(env.get(key))
    return Path(text) if text else None


def _parse_week_force(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse ``WEEK_FORCE`` shaped like ``2026-3`` / ``2026:3`` / ``2026 3``."""
    text = _clean(raw)
    if text is None:
        return None
    for sep in ("-", ":", ",", "/"):
        if sep in text:
            left, right = text.split(sep, 1)
            break
    else:
        parts = text.split()
        if len(parts) != 2:
            raise ConfigError(f"WEEK_FORCE={text!r} is not '<season>-<week>'")
        left, right = parts
    try:
        return int(left.strip()), int(right.strip())
    except ValueError as exc:
        raise ConfigError(f"WEEK_FORCE={text!r} is not '<season>-<week>'") from exc


@dataclass(frozen=True)
class OddsSettings:
    staging_enable: bool = True
    promotion_enable: bool = True
    select_policy: str = "latest_by_fetch_ts"
    cache_only: bool = False
    pin_day_window: int = 3
    pin_max_kickoff_delta_hours: float = 36.0
    role_swap_tolerance: bool = True


@dataclass(frozen=True)
class BackfillSettings:
    scores_enable: bool = True
    weeks: int = 2
    promote_prev: bool = False
    ats_enable: bool = True
    ats_source: str = "auto"
    ats_debug: bool = False


@dataclass(frozen=True)
class StorageSettings:
    """SQLite dual-write settings (Phase 2: ``football_statfinder/storage/``).

    The DB mirrors the flat-file outputs the pipeline already writes; disabling
    it (``enable=False``) makes the orchestrator perform zero DB touches.
    """

    enable: bool = True
    db_path: Optional[Path] = None


@dataclass(frozen=True)
class WeekForce:
    """Manual current-week override; applies when ``league`` matches or is ALL."""

    league: Optional[str] = None
    season: Optional[int] = None
    week: Optional[int] = None

    def applies_to(self, league_code: str) -> bool:
        if self.league is None or self.season is None or self.week is None:
            return False
        return self.league in {league_code.upper(), "ALL", "*"}


@dataclass(frozen=True)
class Settings:
    the_odds_api_key: Optional[str] = None
    cfbd_api_key: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    cfbd_refresh: bool = True
    odds: OddsSettings = field(default_factory=OddsSettings)
    backfill: BackfillSettings = field(default_factory=BackfillSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    week_force: WeekForce = field(default_factory=WeekForce)

    def require(self, *names: str) -> None:
        """Fail loud if any named secret is unset (call at stage startup)."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            raise ConfigError(f"missing required secret(s): {env_names}")

    def banner(self) -> str:
        """One-line effective config with secrets redacted to set/unset."""
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS:
                parts.append(f"{f.name}={'set' if value else 'unset'}")
            elif is_dataclass(value):
                inner = " ".join(
                    f"{g.name}={getattr(value, g.name)}" for g in fields(value)
                )
                parts.append(f"{f.name}[{inner}]")
            else:
                parts.append(f"{f.name}={value}")
        return "CONFIG " + " ".join(parts)


def load_settings(
    env_file: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Build Settings from ``.env`` (if present) overlaid by the real environment.

    ``environ`` defaults to ``os.environ``; tests pass an explicit mapping to
    stay hermetic.
    """
    from .paths import REPO_ROOT  # local import to avoid a cycle

    env_path = env_file if env_file is not None else REPO_ROOT / ".env"
    file_values: dict[str, str] = {}
    if dotenv_values is not None and env_path.exists():
        file_values = {
            key: value
            for key, value in dotenv_values(env_path).items()
            if value is not None
        }
    overlay = environ if environ is not None else os.environ
    env: dict[str, str] = {**file_values, **overlay}

    force_pair = _parse_week_force(env.get("WEEK_FORCE"))
    force_league = _clean(env.get("WEEK_FORCE_LEAGUE"))
    week_force = WeekForce(
        league=force_league.upper() if force_league else None,
        season=force_pair[0] if force_pair else None,
        week=force_pair[1] if force_pair else None,
    )

    return Settings(
        the_odds_api_key=_clean(env.get("THE_ODDS_API_KEY")),
        cfbd_api_key=_clean(env.get("CFBD_API_KEY")),
        discord_webhook_url=_clean(env.get("DISCORD_WEBHOOK_URL")),
        cfbd_refresh=_as_bool(env, "CFBD_REFRESH", True),
        odds=OddsSettings(
            staging_enable=_as_bool(env, "ODDS_STAGING_ENABLE", True),
            promotion_enable=_as_bool(env, "ODDS_PROMOTION_ENABLE", True),
            select_policy=_clean(env.get("ODDS_SELECT_POLICY")) or "latest_by_fetch_ts",
            cache_only=_as_bool(env, "ODDS_CACHE_ONLY", False),
            pin_day_window=_as_int(env, "ODDS_PIN_DAY_WINDOW", 3),
            pin_max_kickoff_delta_hours=_as_float(env, "ODDS_PIN_MAX_KICKOFF_DELTA_HOURS", 36.0),
            role_swap_tolerance=_as_bool(env, "ODDS_ROLE_SWAP_TOLERANCE", True),
        ),
        backfill=BackfillSettings(
            scores_enable=_as_bool(env, "SCORES_BACKFILL_ENABLE", True),
            weeks=_as_int(env, "BACKFILL_WEEKS", 2),
            promote_prev=_as_bool(env, "BACKFILL_PROMOTE_PREV", False),
            ats_enable=_as_bool(env, "ATS_BACKFILL_ENABLED", True),
            ats_source=_clean(env.get("ATS_BACKFILL_SOURCE")) or "auto",
            ats_debug=_as_bool(env, "ATS_DEBUG", False),
        ),
        storage=StorageSettings(
            enable=_as_bool(env, "STORAGE_ENABLE", True),
            db_path=_as_path(env, "STORAGE_DB_PATH"),
        ),
        week_force=week_force,
    )


_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide settings, loaded once on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def set_settings(settings: Optional[Settings]) -> None:
    """Override or reset (None) the cached settings; intended for tests."""
    global _SETTINGS
    _SETTINGS = settings


__all__ = [
    "BackfillSettings",
    "ConfigError",
    "OddsSettings",
    "Settings",
    "StorageSettings",
    "WeekForce",
    "get_settings",
    "load_settings",
    "set_settings",
]

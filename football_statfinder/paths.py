"""Repository-anchored filesystem layout.

One output-root convention for both leagues (REBUILD.md Phase 1):

    out/{league}/{season}_week{week}/

replacing season-1's three-way split (``out/{S}_week{W}`` for NFL,
``out/cfb/{S}_week{W}`` for CFB, and the stray ``out/nfl/`` tree that caused
bug 1). All paths are anchored to the repo root, never the CWD, and nothing
here creates directories at import time.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "out"
DATA_ROOT = REPO_ROOT / "data"

STATE_PATH = OUT_ROOT / "state" / "current_week.json"
MASTER_ROOT = OUT_ROOT / "master"
STAGING_ROOT = OUT_ROOT / "staging"
SAGARIN_RAW_ROOT = DATA_ROOT / "sagarin" / "raw"


def league_root(league_code: str) -> Path:
    return OUT_ROOT / league_code.lower()


def week_dir(league_code: str, season: int, week: int, *, create: bool = False) -> Path:
    """Per-week output directory under the unified convention."""
    directory = league_root(league_code) / f"{int(season)}_week{int(week)}"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def games_week_jsonl(league_code: str, season: int, week: int) -> Path:
    return week_dir(league_code, season, week) / f"games_week_{int(season)}_{int(week)}.jsonl"


def games_week_csv(league_code: str, season: int, week: int) -> Path:
    return week_dir(league_code, season, week) / f"games_week_{int(season)}_{int(week)}.csv"


def sidecar_dir(league_code: str, season: int, week: int) -> Path:
    return week_dir(league_code, season, week) / "game_schedules"


def sidecar_path(league_code: str, season: int, week: int, game_key: str) -> Path:
    return sidecar_dir(league_code, season, week) / f"{game_key}.json"


def schedule_master_csv(league_code: str) -> Path:
    return MASTER_ROOT / f"{league_code.lower()}_schedule_master.csv"


def sagarin_master_csv(league_code: str) -> Path:
    return MASTER_ROOT / f"{league_code.lower()}_sagarin_master.csv"


def sagarin_staging_dir(league_code: str) -> Path:
    return STAGING_ROOT / "sagarin_latest" / league_code.lower()


def odds_raw_dir(league_code: str) -> Path:
    return STAGING_ROOT / "odds_raw" / league_code.lower()


def odds_pinned_jsonl(league_code: str, season: int) -> Path:
    return STAGING_ROOT / "odds_pinned" / league_code.lower() / f"{int(season)}.jsonl"


def sagarin_raw_html_dir(league_code: str) -> Path:
    return SAGARIN_RAW_ROOT / league_code.lower()


__all__ = [
    "DATA_ROOT",
    "MASTER_ROOT",
    "OUT_ROOT",
    "REPO_ROOT",
    "SAGARIN_RAW_ROOT",
    "STAGING_ROOT",
    "STATE_PATH",
    "games_week_csv",
    "games_week_jsonl",
    "league_root",
    "odds_pinned_jsonl",
    "odds_raw_dir",
    "sagarin_master_csv",
    "sagarin_raw_html_dir",
    "sagarin_staging_dir",
    "schedule_master_csv",
    "sidecar_dir",
    "sidecar_path",
    "week_dir",
]

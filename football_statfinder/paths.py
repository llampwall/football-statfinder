"""Repository-anchored filesystem layout.

One output-root convention for both leagues (REBUILD.md Phase 1):

    out/{league}/{season}_week{week}/

replacing season-1's three-way split (``out/{S}_week{W}`` for NFL,
``out/cfb/{S}_week{W}`` for CFB, and the stray ``out/nfl/`` tree that caused
bug 1). All paths are anchored to the repo root, never the CWD, and nothing
here creates directories at import time.

Every helper accepts an optional ``out_root`` keyword so tests can point the
whole layout at a tmp directory without monkeypatching module globals; when
omitted, the repo-anchored ``OUT_ROOT`` applies. This keeps exactly one path
convention (the function bodies) while letting the pipeline stay hermetic in
tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "out"
DATA_ROOT = REPO_ROOT / "data"

STATE_PATH = OUT_ROOT / "state" / "current_week.json"
MASTER_ROOT = OUT_ROOT / "master"
STAGING_ROOT = OUT_ROOT / "staging"
SAGARIN_RAW_ROOT = DATA_ROOT / "sagarin" / "raw"


def _out(out_root: Optional[Path]) -> Path:
    return Path(out_root) if out_root is not None else OUT_ROOT


def _data(data_root: Optional[Path]) -> Path:
    return Path(data_root) if data_root is not None else DATA_ROOT


def staging_root(*, out_root: Optional[Path] = None) -> Path:
    return _out(out_root) / "staging"


def master_root(*, out_root: Optional[Path] = None) -> Path:
    return _out(out_root) / "master"


def league_root(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return _out(out_root) / league_code.lower()


def week_dir(
    league_code: str,
    season: int,
    week: int,
    *,
    create: bool = False,
    out_root: Optional[Path] = None,
) -> Path:
    """Per-week output directory under the unified convention."""
    directory = league_root(league_code, out_root=out_root) / f"{int(season)}_week{int(week)}"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def games_week_jsonl(
    league_code: str, season: int, week: int, *, out_root: Optional[Path] = None
) -> Path:
    return week_dir(league_code, season, week, out_root=out_root) / (
        f"games_week_{int(season)}_{int(week)}.jsonl"
    )


def games_week_csv(
    league_code: str, season: int, week: int, *, out_root: Optional[Path] = None
) -> Path:
    return week_dir(league_code, season, week, out_root=out_root) / (
        f"games_week_{int(season)}_{int(week)}.csv"
    )


def sidecar_dir(
    league_code: str, season: int, week: int, *, out_root: Optional[Path] = None
) -> Path:
    return week_dir(league_code, season, week, out_root=out_root) / "game_schedules"


def sidecar_path(
    league_code: str,
    season: int,
    week: int,
    game_key: str,
    *,
    out_root: Optional[Path] = None,
) -> Path:
    return sidecar_dir(league_code, season, week, out_root=out_root) / f"{game_key}.json"


def schedule_master_csv(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return master_root(out_root=out_root) / f"{league_code.lower()}_schedule_master.csv"


def sagarin_master_csv(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return master_root(out_root=out_root) / f"{league_code.lower()}_sagarin_master.csv"


def sagarin_staging_dir(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return staging_root(out_root=out_root) / "sagarin_latest" / league_code.lower()


def odds_raw_dir(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return staging_root(out_root=out_root) / "odds_raw" / league_code.lower()


def odds_pinned_jsonl(
    league_code: str, season: int, *, out_root: Optional[Path] = None
) -> Path:
    return staging_root(out_root=out_root) / "odds_pinned" / league_code.lower() / (
        f"{int(season)}.jsonl"
    )


def odds_unmatched_dir(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    return staging_root(out_root=out_root) / "odds_unmatched" / league_code.lower()


def odds_promotion_receipt_json(
    league_code: str, season: int, week: int, *, out_root: Optional[Path] = None
) -> Path:
    return week_dir(league_code, season, week, out_root=out_root) / (
        f"odds_promotion_receipt_{int(season)}_{int(week)}.json"
    )


def hist_odds_cache_dir(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    """Per-event historical-odds snapshot cache (legacy: ``out/debug/hist_odds/``)."""
    return staging_root(out_root=out_root) / "hist_odds" / league_code.lower()


def participants_cache_path(league_code: str, *, out_root: Optional[Path] = None) -> Path:
    """Odds API participants snapshot (legacy: ``out/cache/participants/``)."""
    return staging_root(out_root=out_root) / "participants" / f"{league_code.lower()}.json"


def sagarin_raw_html_dir(league_code: str) -> Path:
    return SAGARIN_RAW_ROOT / league_code.lower()


def db_path(*, data_root: Optional[Path] = None) -> Path:
    """Path to the SQLite storage DB (Phase 2): ``data/statfinder.sqlite3``.

    Anchored on :data:`DATA_ROOT`, not :data:`OUT_ROOT` (the storage DB is a
    development artifact, not a published output); ``data_root`` mirrors the
    other helpers' ``out_root`` test-injection keyword, just anchored on the
    other repo-root subtree.
    """
    return _data(data_root) / "statfinder.sqlite3"


__all__ = [
    "DATA_ROOT",
    "MASTER_ROOT",
    "OUT_ROOT",
    "REPO_ROOT",
    "SAGARIN_RAW_ROOT",
    "STAGING_ROOT",
    "STATE_PATH",
    "db_path",
    "games_week_csv",
    "games_week_jsonl",
    "hist_odds_cache_dir",
    "league_root",
    "master_root",
    "odds_pinned_jsonl",
    "odds_promotion_receipt_json",
    "odds_raw_dir",
    "odds_unmatched_dir",
    "participants_cache_path",
    "sagarin_master_csv",
    "sagarin_raw_html_dir",
    "sagarin_staging_dir",
    "schedule_master_csv",
    "sidecar_dir",
    "sidecar_path",
    "staging_root",
    "week_dir",
]

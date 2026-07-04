"""SQLite connection, schema, and transaction helper for Phase 2 storage.

Design decisions (frozen; see ``docs/PHASE2_SPEC.md`` Part 2 — "Design
decisions" and "Schema v1"):

* Document-style schema. Every table pairs a handful of indexed key columns
  with one ``payload`` JSON column holding the exact in-memory record the
  pipeline already produced. The flat-file contract stays the product; this
  is not a relational decomposition of record internals.
* Single writer. The orchestrator (:mod:`football_statfinder.refresh`) is the
  only writer; ``PRAGMA foreign_keys`` stays OFF (no FKs — cross-table
  integrity is the orchestrator's job, same as it is for the flat files
  today) and WAL mode is enabled for concurrent readers.
* Dual-write, files stay canonical in Phase 2. This module never reads the
  flat files back; :mod:`football_statfinder.storage.store` takes the
  in-memory objects the orchestrator already built.
* ``odds_raw`` (the append-only per-fetch JSONL ledger) stays on disk. It is
  already the right shape for its access pattern, and mirroring megabyte raw
  fetches into SQLite buys nothing in Phase 2 — only the *pinned* ledger
  (``odds_pinned``) is mirrored here.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schedule_games(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, kickoff_iso_utc TEXT,
  home_team_key TEXT, away_team_key TEXT,
  payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, game_key)
);
CREATE INDEX IF NOT EXISTS idx_schedule_week ON schedule_games(league, season, week);

CREATE TABLE IF NOT EXISTS sagarin_ratings(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  team_norm TEXT NOT NULL, fetch_ts TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY(league, season, week, team_norm, fetch_ts)
);

CREATE TABLE IF NOT EXISTS team_metrics(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  team TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, team)
);

CREATE TABLE IF NOT EXISTS odds_pinned(
  league TEXT NOT NULL, fetch_ts TEXT NOT NULL, game_key TEXT NOT NULL,
  market TEXT NOT NULL, book TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(league, fetch_ts, game_key, market, book)
);

CREATE TABLE IF NOT EXISTS games(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, game_key)
);

CREATE TABLE IF NOT EXISTS sidecars(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, game_key)
);
"""


class SchemaVersionError(RuntimeError):
    """Raised when an existing DB's ``meta.schema_version`` != this build's."""


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (creating file + schema on first use) the storage DB.

    ``db_path`` defaults to :func:`football_statfinder.paths.db_path` (the
    repo-anchored ``data/statfinder.sqlite3``); callers (tests, the
    orchestrator via ``StorageSettings.db_path``) can point elsewhere.

    Enables WAL mode, disables foreign keys (no FKs in this schema), and
    checks ``meta.schema_version``: a fresh DB gets stamped with
    :data:`SCHEMA_VERSION`; an existing DB with a different version raises
    :class:`SchemaVersionError` rather than silently running against a stale
    or newer layout.
    """
    if db_path is None:
        from .. import paths  # local import: avoid a paths<->storage import cycle at module load

        target = paths.db_path()
    else:
        target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    else:
        existing = int(row["value"])
        if existing != SCHEMA_VERSION:
            conn.close()
            raise SchemaVersionError(
                f"storage schema_version mismatch at {target}: db has {existing}, "
                f"code expects {SCHEMA_VERSION}"
            )
    logger.info("storage: connected %s (schema_version=%d)", target, SCHEMA_VERSION)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a block of writes in one transaction (commit on success, else rollback)."""
    with conn:
        yield conn


__all__ = ["SCHEMA_VERSION", "SchemaVersionError", "connect", "transaction"]

"""Upsert writers: the orchestrator's in-memory records into the storage DB.

Every function here takes objects the orchestrator already built for the
flat-file write (a schedule DataFrame, a games_week records list, ...) — none
of these re-read files from disk. Payloads are serialized with
``json.dumps(payload, ensure_ascii=False, sort_keys=True)``, the exact kwargs
:func:`football_statfinder.common.io_atomic.write_atomic_jsonl` uses for JSONL
lines, so a DB payload and the corresponding flat-file line encode identically
(load-bearing for the WP-D export byte-parity gate). ``updated_at`` is a
timezone-aware UTC ISO string with a ``Z`` suffix, matching
:class:`football_statfinder.run_summary.RunSummary`'s timestamp convention.

``odds_raw`` is not mirrored here (see ``storage/db.py`` docstring); only the
pinned ledger (``record_pinned_odds``) is.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from ..common.game_key import build_game_key
from ..leagues import League
from . import db

DataFrameOrRows = Any  # pandas.DataFrame or an iterable of row mappings


def _parse_kickoff(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _schedule_game_key(row: Mapping[str, Any], league: League) -> Optional[str]:
    """A row's ``game_key``, deriving it when absent.

    The schedule master (``sources/schedule_master.py`` ``KEEP`` columns)
    deliberately does not persist ``game_key`` — every downstream stage
    rebuilds it from kickoff + team names via the one constructor
    (``common.game_key.build_game_key``). ``record_schedule`` does the same so
    a master-sourced ``week_df`` (no ``game_key`` column) still gets a stable
    identity in ``schedule_games``.
    """
    game_key = row.get("game_key")
    if game_key:
        return str(game_key)
    kickoff = _parse_kickoff(row.get("kickoff_iso_utc"))
    home = row.get("home_team_norm")
    away = row.get("away_team_norm")
    if kickoff is None or not home or not away:
        return None
    return build_game_key(league, kickoff, str(home), str(away))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Recursively coerce pandas/numpy scalars to plain JSON-able values.

    Schedule rows arrive as ``DataFrame.to_dict(orient="records")`` output,
    which carries numpy scalar types and pandas NA/NaT; games/sidecars/metrics
    rows are already plain ``dict``s built by the pipeline and pass through
    unchanged.

    Plain ``float`` (including NaN — some flat-file rows carry a bare NaN,
    e.g. a games record's ``raw_sources.schedule_row`` passthrough of an
    un-coerced master row) is left untouched: ``json.dumps`` already
    serializes it exactly the way ``io_atomic.write_atomic_jsonl`` does, and
    touching it here would break payload/flat-file byte parity. Only values
    plain ``json.dumps`` cannot serialize at all — ``pandas.NA``/``NaT`` and
    numpy scalar types (``int64``, ``bool_``, ...; note ``numpy.float64`` is
    already a ``float`` subclass and is caught by the branch above) — get
    coerced, to ``None``/native Python respectively.
    """
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, float):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # not NA-checkable (e.g. an array-like slipped through) — leave as-is
    if hasattr(value, "item"):  # remaining numpy scalar types (int64/bool_/...)
        try:
            return _json_safe(value.item())
        except Exception:
            return value
    return value


def _dumps(payload: Mapping[str, Any]) -> str:
    """The one payload encoding, matching ``io_atomic.write_atomic_jsonl``."""
    return json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)


def _rows_of(data: DataFrameOrRows) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    return data


def record_schedule(conn: sqlite3.Connection, league: League, df: DataFrameOrRows) -> int:
    """Upsert one league's schedule rows (a season/week slice) into ``schedule_games``."""
    now = _now_iso()
    count = 0
    with db.transaction(conn):
        for row in _rows_of(df):
            game_key = _schedule_game_key(row, league)
            if not game_key:
                continue
            season = row.get("season")
            week = row.get("week")
            conn.execute(
                """
                INSERT INTO schedule_games
                    (league, season, week, game_key, kickoff_iso_utc,
                     home_team_key, away_team_key, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, season, game_key) DO UPDATE SET
                    week=excluded.week,
                    kickoff_iso_utc=excluded.kickoff_iso_utc,
                    home_team_key=excluded.home_team_key,
                    away_team_key=excluded.away_team_key,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    league.code,
                    int(season),
                    int(week),
                    game_key,
                    row.get("kickoff_iso_utc"),
                    row.get("home_team_key"),
                    row.get("away_team_key"),
                    _dumps(row),
                    now,
                ),
            )
            count += 1
    return count


def record_sagarin(
    conn: sqlite3.Connection,
    league: League,
    season: int,
    week: int,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Upsert one week's Sagarin snapshot rows into ``sagarin_ratings``.

    ``rows`` is the weekly staging JSONL content (``team_norm``, ``pr``,
    ``pr_rank``, ``sos``, ``sos_rank``, ``hfa``, plus provenance fields); the
    per-team fetch timestamp may be keyed ``fetch_ts`` or ``fetched_at``
    depending on the producer, both are honored.
    """
    count = 0
    with db.transaction(conn):
        for row in rows:
            team_norm = row.get("team_norm") or row.get("team")
            if not team_norm:
                continue
            fetch_ts = row.get("fetch_ts") or row.get("fetched_at") or ""
            conn.execute(
                """
                INSERT INTO sagarin_ratings
                    (league, season, week, team_norm, fetch_ts, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, season, week, team_norm, fetch_ts) DO UPDATE SET
                    payload=excluded.payload
                """,
                (league.code, int(season), int(week), str(team_norm), str(fetch_ts), _dumps(row)),
            )
            count += 1
    return count


def record_metrics(
    conn: sqlite3.Connection,
    league: League,
    season: int,
    week: int,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Upsert one week's league_metrics rows into ``team_metrics`` (keyed on ``Team``)."""
    now = _now_iso()
    count = 0
    with db.transaction(conn):
        for row in rows:
            team = row.get("Team") or row.get("team")
            if not team:
                continue
            conn.execute(
                """
                INSERT INTO team_metrics (league, season, week, team, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, season, week, team) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (league.code, int(season), int(week), str(team), _dumps(row), now),
            )
            count += 1
    return count


def record_pinned_odds(
    conn: sqlite3.Connection,
    league: League,
    records: Sequence[Mapping[str, Any]],
) -> int:
    """Upsert pinned odds ledger records into ``odds_pinned``.

    Identity mirrors the JSONL ledger's own dedupe key
    (``fetch_ts``, ``game_key``, ``market``, ``book``) — see
    ``pipeline/odds_pin.py``'s ``_dedupe_key``. No ``season``/``week``
    columns: those already live inside each record's payload.
    """
    count = 0
    with db.transaction(conn):
        for record in records:
            fetch_ts = record.get("fetch_ts")
            game_key = record.get("game_key")
            market = record.get("market")
            book = record.get("book")
            if not (fetch_ts and game_key and market and book):
                continue
            conn.execute(
                """
                INSERT INTO odds_pinned (league, fetch_ts, game_key, market, book, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, fetch_ts, game_key, market, book) DO UPDATE SET
                    payload=excluded.payload
                """,
                (
                    league.code,
                    str(fetch_ts),
                    str(game_key),
                    str(market),
                    str(book),
                    _dumps(record),
                ),
            )
            count += 1
    return count


def record_games(
    conn: sqlite3.Connection,
    league: League,
    season: int,
    week: int,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Upsert one week's games_week records into ``games``."""
    now = _now_iso()
    count = 0
    with db.transaction(conn):
        for row in rows:
            game_key = row.get("game_key")
            if not game_key:
                continue
            conn.execute(
                """
                INSERT INTO games (league, season, week, game_key, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, season, week, game_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (league.code, int(season), int(week), str(game_key), _dumps(row), now),
            )
            count += 1
    return count


def record_sidecars(
    conn: sqlite3.Connection,
    league: League,
    season: int,
    week: int,
    sidecar_payloads: Iterable[Mapping[str, Any]],
) -> int:
    """Upsert one week's per-game sidecar payloads into ``sidecars``.

    ``sidecar_payloads`` is the sequence of ``{game_key, home_ytd, away_ytd,
    home_prev, away_prev}`` dicts ``pipeline/sidecars.py`` writes to
    ``game_schedules/{game_key}.json`` (see ``build_sidecars``'s returned
    receipt ``"payloads"`` entry).
    """
    now = _now_iso()
    count = 0
    with db.transaction(conn):
        for payload in sidecar_payloads:
            game_key = payload.get("game_key")
            if not game_key:
                continue
            conn.execute(
                """
                INSERT INTO sidecars (league, season, week, game_key, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(league, season, week, game_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (league.code, int(season), int(week), str(game_key), _dumps(payload), now),
            )
            count += 1
    return count


__all__ = [
    "record_games",
    "record_metrics",
    "record_pinned_odds",
    "record_sagarin",
    "record_schedule",
    "record_sidecars",
]

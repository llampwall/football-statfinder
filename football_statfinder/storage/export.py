"""Export step: reconstruct flat-file artifacts from storage DB payloads.

Phase 2 WP-D (``docs/PHASE2_SPEC.md``, Part 2 — "WP-D — export step"). WP-C's
dual-write (``storage/store.py``) mirrors the pipeline's flat-file outputs
into SQLite; this module is the byte-parity proof that the mirror is
faithful. It writes the same artifacts a week's refresh would have written,
sourced entirely from DB payloads, by calling the pipeline's own
writer/ordering functions rather than re-implementing any serialization:

* :func:`football_statfinder.pipeline.gameview.order_records` /
  :func:`~football_statfinder.pipeline.gameview.write_games_week_files` for
  ``games_week_{S}_{W}.jsonl`` + ``.csv``.
* :func:`football_statfinder.sources.stats.write_league_metrics_csv` for
  ``league_metrics_{S}_{W}.csv``.
* :func:`football_statfinder.pipeline.sidecars.write_sidecar_json` for each
  ``game_schedules/{game_key}.json``.

Files stay canonical in Phase 2 (see ``storage/db.py``'s module docstring);
this is a reconstruction/verification step, not a new primary write path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import paths
from ..leagues import League
from ..pipeline import gameview as gameview_mod
from ..pipeline import sidecars as sidecars_mod
from ..sources import stats as stats_mod


class ExportError(RuntimeError):
    """Raised when a requested (league, season, week) has no rows in storage."""


def _payload_rows(
    conn: sqlite3.Connection, table: str, league: League, season: int, week: int
) -> List[Dict[str, Any]]:
    """A week's stored payloads for one table, in original insertion order.

    ``ORDER BY rowid`` reproduces the order the orchestrator's ``record_*``
    calls inserted rows in (upserts update in place, never reordering), which
    for ``games`` is already the builder's frozen kickoff/game_key order and
    for ``team_metrics`` is the stats provider's row order — the same order
    the pipeline's own writers received when it wrote the flat files.
    """
    cursor = conn.execute(
        f"SELECT payload FROM {table} WHERE league = ? AND season = ? AND week = ? ORDER BY rowid",
        (league.code, int(season), int(week)),
    )
    return [json.loads(row[0]) for row in cursor.fetchall()]


def export_week(
    conn: sqlite3.Connection,
    league: League,
    season: int,
    week: int,
    *,
    out_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Rebuild one week's flat-file artifacts from DB payloads.

    Writes ``games_week_{S}_{W}.jsonl`` + ``.csv``, ``league_metrics_{S}_{W}.csv``,
    and one ``game_schedules/{game_key}.json`` per stored sidecar, under the
    unified ``paths`` layout (or ``out_root`` when injected, matching every
    other stage's test-injection convention).

    Raises :class:`ExportError` naming the league/season/week when the
    ``games`` table has no rows for it — an absent week fails loudly rather
    than silently writing empty/partial artifacts.

    Returns a dict of the paths written: ``games_jsonl``, ``games_csv``,
    ``league_metrics_csv``, ``sidecars`` (a list of paths).
    """
    games = _payload_rows(conn, "games", league, season, week)
    if not games:
        raise ExportError(
            f"export_week: no games rows in storage for {league.display} "
            f"season={season} week={week}"
        )

    week_dir = paths.week_dir(league.code, season, week, out_root=out_root)

    ordered = gameview_mod.order_records(games)
    jsonl_path, csv_path = gameview_mod.write_games_week_files(
        league, season, week, ordered, out_dir=week_dir
    )

    metrics_rows = _payload_rows(conn, "team_metrics", league, season, week)
    metrics_path = stats_mod.write_league_metrics_csv(
        league,
        season,
        week,
        metrics_rows,
        path=week_dir / f"league_metrics_{int(season)}_{int(week)}.csv",
    )

    sidecar_rows = _payload_rows(conn, "sidecars", league, season, week)
    side_dir = week_dir / "game_schedules"
    sidecar_paths: List[Path] = []
    for payload in sidecar_rows:
        game_key = payload.get("game_key")
        if not game_key:
            continue
        side_path = side_dir / f"{game_key}.json"
        sidecars_mod.write_sidecar_json(side_path, payload)
        sidecar_paths.append(side_path)

    return {
        "games_jsonl": jsonl_path,
        "games_csv": csv_path,
        "league_metrics_csv": metrics_path,
        "sidecars": sidecar_paths,
    }


__all__ = ["ExportError", "export_week"]

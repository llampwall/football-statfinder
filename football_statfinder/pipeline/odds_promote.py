"""Promote pinned odds into the week's game rows (staging top layer).

One league-parameterized port of ``src/odds/nfl_promote_week.py`` and
``src/odds/cfb_promote_week.py``. Keeps, per REBUILD.md section 5:

* the shared selection/merge logic — freshest record per
  ``(game_key, market, book)``, then one record per market per game chosen by
  the selection policy;
* both legacy selection policies: ``latest_by_fetch_ts`` (legacy default) and
  ``closing_pre_kickoff`` (the season-1 ``.env`` setting) — policy now comes
  from ``settings.odds.select_policy``;
* NFL's atomic ``write_week_outputs`` (JSONL + CSV, stable sort, CSV column
  union with the existing file), now via ``common.io_atomic``;
* CFB's kickoff fallback: when a week row has no kickoff timestamp, the
  pinned record's ``kickoff_utc`` drives the closing-line cutoff;
* CFB's coverage gate and JSON debug receipt, folded into promotion: every
  run writes ``odds_promotion_receipt_{S}_{W}.json`` into the week dir, and
  the gate fails when the pinned ledger holds current-week games whose keys
  never land on week rows — exactly the dead-key failure class of bug 4.

Changes from the legacy behavior, all deliberate:

* ``promote_week(league, season, week, settings)`` reads and rewrites the
  week's ``games_week`` files itself (the legacy pair mutated caller-owned
  in-memory rows and only NFL had a writer); outputs are only rewritten when
  at least one game was promoted.
* Unsupported policies log a warning (once per process) instead of printing.
* JSONL reads go through the counted-skip reader (bug 10).
* ``pick_latest_before`` is ported from ``src/odds/ats_compute.py`` so no
  legacy import is needed.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from .. import paths
from ..common.io_atomic import write_atomic_csv, write_atomic_json, write_atomic_jsonl
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League

logger = logging.getLogger(__name__)

SUPPORTED_POLICIES = ("latest_by_fetch_ts", "closing_pre_kickoff")

# Coverage gate: at least this fraction of the ledger's current-week games
# must land on week rows (mirrors the legacy CFB MIN_MATCH_FRAC instinct).
MIN_PROMOTED_FRAC = 0.5
MAX_RECEIPT_SAMPLES = 30

_WARNED_POLICIES: set[str] = set()


def _parse_ts(value: Any) -> datetime:
    """Parse a fetch timestamp; unparseable values sort first (datetime.min)."""
    dt = _parse_utc(value)
    return dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)


def _parse_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def pick_latest_before(
    records: Iterable[Mapping[str, Any]], cutoff: datetime
) -> Optional[Mapping[str, Any]]:
    """Record with the greatest ``fetch_ts`` <= cutoff (missing/invalid skipped).

    Equal timestamps break by book label (greatest wins), matching the legacy
    promoters' ``max(..., key=(fetch_ts, book))`` — without this the choice is
    ledger-order-dependent when one fetch captured several books.
    """
    best: Optional[Tuple[Tuple[datetime, str], Mapping[str, Any]]] = None
    for record in records or []:
        when = _parse_utc(record.get("fetch_ts"))
        if when is None:
            continue
        rank = (when, str(record.get("book") or ""))
        if when <= cutoff and (best is None or rank > best[0]):
            best = (rank, record)
    return best[1] if best else None


def _select_latest(
    records: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    """Freshest record per (game_key, market, book)."""
    latest: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (record.get("game_key"), record.get("market"), record.get("book"))
        if None in key:
            continue
        existing = latest.get(key)  # type: ignore[arg-type]
        if existing is None or _parse_ts(record.get("fetch_ts")) > _parse_ts(
            existing.get("fetch_ts")
        ):
            latest[key] = record  # type: ignore[index]
    return latest


def _choose_best(records: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not records:
        return None
    return max(
        records,
        key=lambda rec: (_parse_ts(rec.get("fetch_ts")), rec.get("book") or ""),
    )


def _choose_by_policy(
    records: Sequence[Mapping[str, Any]],
    policy: str,
    kickoff: Optional[datetime],
) -> Tuple[Optional[Mapping[str, Any]], bool]:
    """Pick one record; second element is True when the closing rule applied."""
    if not records:
        return None, False
    if policy == "closing_pre_kickoff" and kickoff:
        pick = pick_latest_before(records, kickoff)
        if pick:
            return pick, True
    return _choose_best(records), False


def _merge_line(
    row: MutableMapping[str, Any], record: Mapping[str, Any], line: Mapping[str, Any]
) -> None:
    """Apply a selected line to a games_week row (frozen frontend fields)."""
    market = record.get("market")
    if market == "spreads":
        row["spread_home_relative"] = line.get("spread_home_relative")
        row["favored_side"] = line.get("favored_side")
        row["spread_favored_team"] = line.get("spread_favored_team")
    elif market == "totals":
        row["total"] = line.get("total_points")
    elif market == "h2h":
        row["moneyline_home"] = line.get("moneyline_home")
        row["moneyline_away"] = line.get("moneyline_away")


def _row_sort_key(row: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return (
        _safe_int(row.get("season")),
        _safe_int(row.get("week")),
        str(row.get("kickoff_iso_utc") or row.get("kickoff_iso") or ""),
        str(row.get("game_key") or ""),
    )


def _align_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = [col for col in columns if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + remainder)


def write_week_outputs(
    league: League,
    rows: Sequence[Mapping[str, Any]],
    season: int,
    week: int,
    *,
    out_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Atomically rewrite the week's games_week JSONL + CSV with sorted rows.

    Ported from the legacy NFL promote module; paths now come from
    ``football_statfinder.paths`` (one convention for both leagues).
    """
    json_path = paths.games_week_jsonl(league.code, season, week, out_root=out_root)
    csv_path = paths.games_week_csv(league.code, season, week, out_root=out_root)
    sorted_rows = sorted(rows, key=_row_sort_key)
    write_atomic_jsonl(json_path, sorted_rows)

    df = pd.DataFrame(sorted_rows)
    if not df.empty:
        sort_cols = [
            col for col in ("season", "week", "kickoff_iso_utc", "game_key") if col in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if csv_path.exists():
        existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        union_cols = list(dict.fromkeys(existing_cols + list(df.columns)))
        df = _align_dataframe(df, union_cols)
    write_atomic_csv(csv_path, df)
    return json_path, csv_path


def promote_week(
    league: League,
    season: int,
    week: int,
    settings: Settings,
    *,
    out_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote pinned odds into one week's game rows and persist them.

    Reads the season's pinned ledger and the week's ``games_week`` JSONL,
    merges selected lines into the rows, rewrites the JSONL/CSV atomically
    (only when something promoted), and writes a JSON receipt with counts and
    the coverage-gate verdict.

    Returns a summary dict: promoted_games, used_records, available_records,
    season_records, current_week_records, other_week_records, by_market,
    by_book, policy, coverage_ok, coverage, receipt_path, json_path,
    csv_path, skipped_reason.
    """
    summary: Dict[str, Any] = {
        "promoted_games": 0,
        "used_records": 0,
        "available_records": 0,
        "season_records": 0,
        "current_week_records": 0,
        "other_week_records": 0,
        "by_market": {},
        "by_book": {},
        "policy": None,
        "coverage_ok": True,
        "coverage": {},
        "receipt_path": None,
        "json_path": None,
        "csv_path": None,
        "skipped_reason": None,
    }
    if not settings.odds.promotion_enable:
        logger.info("%s odds promotion disabled by settings; skipping", league.display)
        summary["skipped_reason"] = "promotion_disabled"
        return summary

    policy = (settings.odds.select_policy or "latest_by_fetch_ts").strip() or "latest_by_fetch_ts"
    if policy not in SUPPORTED_POLICIES:
        if policy not in _WARNED_POLICIES:
            logger.warning(
                "unsupported odds selection policy %r; falling back to latest_by_fetch_ts",
                policy,
            )
            _WARNED_POLICIES.add(policy)
        policy = "latest_by_fetch_ts"
    summary["policy"] = policy

    week_result = read_jsonl(paths.games_week_jsonl(league.code, season, week, out_root=out_root))
    rows: List[MutableMapping[str, Any]] = list(week_result.rows)
    ledger_result = read_jsonl(paths.odds_pinned_jsonl(league.code, season, out_root=out_root))
    pinned_records = ledger_result.rows

    game_lookup: Dict[str, MutableMapping[str, Any]] = {}
    eligible_keys: set[str] = set()
    for row in rows:
        key = row.get("game_key")
        if not key:
            continue
        game_lookup[key] = row
        if row.get("season") == season and row.get("week") == week:
            eligible_keys.add(key)

    season_records = len(pinned_records)
    relevant_records = [rec for rec in pinned_records if rec.get("game_key") in eligible_keys]
    current_week_records = len(relevant_records)
    latest_map = _select_latest(pinned_records)

    per_game: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    used_record_total = 0
    for (game_key, market, _), record in latest_map.items():
        if game_key not in eligible_keys:
            continue
        per_game[game_key][market].append(record)
        used_record_total += 1

    by_market: Counter[str] = Counter()
    by_book: Counter[str] = Counter()
    promoted_games: set[str] = set()

    for game_key, market_records in per_game.items():
        row = game_lookup[game_key]
        kickoff_iso = (
            row.get("kickoff_iso_utc") or row.get("kickoff_iso") or row.get("kickoff_utc")
        )
        kickoff_dt = _parse_utc(kickoff_iso)
        if kickoff_dt is None:
            # CFB keeper: fall back to the pinned record's schedule kickoff.
            sample_records = next(iter(market_records.values()), [])
            if sample_records:
                kickoff_dt = _parse_utc(sample_records[0].get("kickoff_utc"))
        odds_payload: Dict[str, Any] = {
            "source": "staging",
            "season": season,
            "week": week,
            "markets": {},
        }
        primary_record: Optional[Mapping[str, Any]] = None
        closing_selected = False
        for market, records in market_records.items():
            best, used_closing = _choose_by_policy(records, policy, kickoff_dt)
            if not best:
                continue
            line = best.get("line") or {}
            _merge_line(row, best, line)
            odds_payload["markets"][market] = {
                "book": best.get("book"),
                "fetch_ts": best.get("fetch_ts"),
                "line": line,
            }
            by_market[market] += 1
            by_book[best.get("book") or "unknown"] += 1
            if primary_record is None or _parse_ts(best.get("fetch_ts")) > _parse_ts(
                primary_record.get("fetch_ts")
            ):
                primary_record = best
            if used_closing and market == "spreads":
                closing_selected = True
        if odds_payload["markets"]:
            promoted_games.add(game_key)
            row.setdefault("raw_sources", {})["odds_row"] = odds_payload
            if primary_record:
                row["odds_source"] = primary_record.get("book")
                row["snapshot_at"] = primary_record.get("fetch_ts")
                row["is_closing"] = (
                    closing_selected if policy == "closing_pre_kickoff" else False
                )

    # Coverage gate (folded in from the legacy CFB odds stage): the ledger's
    # own season/week stamps say how many current-week games have staged odds;
    # if those keys never land on week rows, promotion silently died (bug 4).
    week_pinned_keys = {
        rec.get("game_key")
        for rec in pinned_records
        if rec.get("season") == season and rec.get("week") == week and rec.get("game_key")
    }
    missing_keys = sorted(week_pinned_keys - set(game_lookup))
    coverage_ok = True
    required = 0
    if week_pinned_keys:
        required = max(1, math.ceil(MIN_PROMOTED_FRAC * len(week_pinned_keys)))
        coverage_ok = len(promoted_games) >= required
    coverage = {
        "ok": coverage_ok,
        "required_promoted": required,
        "week_pinned_games": len(week_pinned_keys),
        "promoted_games": len(promoted_games),
        "week_rows": len(rows),
        "eligible_rows": len(eligible_keys),
        "pinned_keys_missing_from_rows": missing_keys[:MAX_RECEIPT_SAMPLES],
    }

    json_path = None
    csv_path = None
    if promoted_games:
        json_path, csv_path = write_week_outputs(league, rows, season, week, out_root=out_root)

    receipt_path = None
    if rows or week_pinned_keys:
        receipt_path = paths.odds_promotion_receipt_json(
            league.code, season, week, out_root=out_root
        )
        write_atomic_json(
            receipt_path,
            {
                "league": league.display,
                "season": season,
                "week": week,
                "policy": policy,
                "stats": {
                    "season_records": season_records,
                    "current_week_records": current_week_records,
                    "other_week_records": season_records - current_week_records,
                    "used_records": used_record_total,
                    "promoted_games": len(promoted_games),
                    "week_rows_skipped_lines": week_result.skipped,
                    "ledger_skipped_lines": ledger_result.skipped,
                },
                "by_market": dict(by_market),
                "by_book": dict(by_book),
                "coverage": coverage,
                "samples": {"promoted_game_keys": sorted(promoted_games)[:MAX_RECEIPT_SAMPLES]},
            },
        )

    if not coverage_ok:
        logger.error(
            "%s odds promotion coverage below threshold for %s week %s: "
            "promoted %d of %d ledger game(s), required %d; see %s",
            league.display,
            season,
            week,
            len(promoted_games),
            len(week_pinned_keys),
            required,
            receipt_path,
        )
    else:
        logger.info(
            "%s odds promotion %s-%s: promoted=%d used=%d policy=%s",
            league.display,
            season,
            week,
            len(promoted_games),
            used_record_total,
            policy,
        )

    summary.update(
        {
            "promoted_games": len(promoted_games),
            "used_records": used_record_total,
            "available_records": current_week_records,
            "season_records": season_records,
            "current_week_records": current_week_records,
            "other_week_records": season_records - current_week_records,
            "by_market": dict(by_market),
            "by_book": dict(by_book),
            "coverage_ok": coverage_ok,
            "coverage": coverage,
            "receipt_path": receipt_path,
            "json_path": json_path,
            "csv_path": csv_path,
        }
    )
    return summary


__all__ = [
    "MIN_PROMOTED_FRAC",
    "SUPPORTED_POLICIES",
    "pick_latest_before",
    "promote_week",
    "write_week_outputs",
]

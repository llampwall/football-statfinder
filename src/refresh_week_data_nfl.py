"""NFL refresh wrapper with odds staging & promotion (legacy builder + staging hook).

Purpose & scope:
    Preserve the legacy NFL weekly refresh behaviour while adding
    append-only odds staging (raw ingest + schedule pinning) immediately
    after the legacy pipeline completes.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (B1–B2, F, H, I)

Notes:
    * CLI remains identical to `src.refresh_week_data` (season/week required).
    * The wrapper logs CurrentWeek(NFL) (read-only), invokes the legacy
      builder, then runs staging and emits a single diagnostics line plus
      a notify summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.current_week_service import get_current_week
from src.common.io_atomic import write_atomic_text
from src.common.io_utils import ensure_out_dir, getenv, write_atomic_json
from src.odds.nfl_ingest import ingest_nfl_odds_raw
from src.odds.nfl_pin_to_schedule import pin_nfl_odds
from src.odds.nfl_promote_week import (
    promote_week_odds,
    read_week_json,
    write_week_outputs,
    diff_game_rows,
)
from src.odds.ats_backfill_api import (
    compute_ats,
    load_pinned_event_index,
    resolve_event_id,
    select_closing_spread,
)
from src.odds.participants_cache import build_provider_map
from src.odds.odds_api_client import ODDS_API_USAGE, hist_odds_cache_exists
from src.ratings.sagarin_nfl_fetch import run_nfl_sagarin_staging
from src.scores.nfl_backfill import backfill_nfl_scores
from src.ats.nfl_ats import build_team_ats, apply_ats_to_week

_BLANK_SENTINELS = {None, "", "-", "\u2014"}
ATS_DEBUG = getenv("ATS_DEBUG", "0") == "1"


def _atsdbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg)


def _is_blank(value: Any) -> bool:
    if value in _BLANK_SENTINELS:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _merge_only(old: Any, new: Any) -> Any:
    return new if _is_blank(old) else old


def _parse_kickoff(game: Dict[str, Any]) -> Optional[datetime]:
    value = game.get("kickoff_ts") or (
        game.get("raw_sources", {}).get("schedule_row", {}).get("commence_time")
    )
    if not value:
        value = game.get("kickoff_iso_utc") or game.get("kickoff_iso")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _coerce_score_value(value: Any) -> Optional[int]:
    if _is_blank(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except Exception:
            return None


def _scores_from_sidecar(
    game_key: str, sidecar_map: Dict[str, Dict[str, Any]], season: int, week: int
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    record = sidecar_map.get(game_key) or {}

    def _scan(entries: Any) -> Tuple[Optional[int], Optional[int]]:
        for row in entries or []:
            if not isinstance(row, dict):
                continue
            try:
                if int(row.get("season")) != season or int(row.get("week")) != week:
                    continue
            except Exception:
                continue
            pf = _coerce_score_value(row.get("pf"))
            pa = _coerce_score_value(row.get("pa"))
            if pf is not None and pa is not None:
                return pf, pa
        return None, None

    home_pf, home_pa = _scan(record.get("home_ytd"))
    if home_pf is not None and home_pa is not None:
        return home_pf, home_pa, "home_ytd"

    away_pf, away_pa = _scan(record.get("away_ytd"))
    if away_pf is not None and away_pa is not None:
        return away_pa, away_pf, "away_ytd"

    return None, None, None


def _extract_scores(
    game: Dict[str, Any],
    sidecar_map: Dict[str, Dict[str, Any]],
    season: int,
    week: int,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    home_score = _coerce_score_value(game.get("home_score"))
    away_score = _coerce_score_value(game.get("away_score"))
    if home_score is not None and away_score is not None:
        return home_score, away_score, "game"

    schedule_row = (game.get("raw_sources") or {}).get("schedule_row") or {}
    if home_score is None:
        home_score = _coerce_score_value(schedule_row.get("home_score"))
    if away_score is None:
        away_score = _coerce_score_value(schedule_row.get("away_score"))
    if home_score is not None and away_score is not None:
        return home_score, away_score, "schedule_row"

    game_key = str(game.get("game_key") or "")
    sidecar_home, sidecar_away, source = _scores_from_sidecar(game_key, sidecar_map, season, week)
    if sidecar_home is not None and sidecar_away is not None:
        return sidecar_home, sidecar_away, source

    return None, None, None


def _week_sidecar_dir(league: str, season: int, week: int):
    root = ensure_out_dir()
    league_lower = league.lower()
    if league_lower == "nfl":
        return root / f"{season}_week{week}" / "game_schedules"
    return root / league_lower / f"{season}_week{week}" / "game_schedules"


def _load_sidecar_map(league: str, season: int, week: int) -> Dict[str, Dict[str, Any]]:
    side_dir = _week_sidecar_dir(league, season, week)
    mapping: Dict[str, Dict[str, Any]] = {}
    if not side_dir.exists():
        return mapping
    for path in side_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        game_key = payload.get("game_key")
        if isinstance(game_key, str):
            mapping[game_key] = payload
    return mapping


def _write_sidecar_files(
    league: str, season: int, week: int, sidecar_map: Dict[str, Dict[str, Any]], dirty_keys: Set[str]
) -> None:
    if not dirty_keys:
        return
    side_dir = _week_sidecar_dir(league, season, week)
    side_dir.mkdir(parents=True, exist_ok=True)
    for game_key in dirty_keys:
        payload = sidecar_map.get(game_key)
        if not isinstance(payload, dict):
            continue
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        write_atomic_text(side_dir / f"{game_key}.json", serialized)


def _ats_backfill_api(
    league: str,
    season: int,
    week: int,
    games: Optional[List[Dict[str, Any]]],
    sidecar_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    enabled = (os.getenv("ATS_BACKFILL_ENABLED", "0") == "1") or (
        os.getenv("ATS_BACKFILL_SOURCE", "") == "api"
    )
    if not enabled:
        return {"games_fixed": 0, "week_dirty": False, "sidecar_dirty": set()}

    debug_enabled = ATS_DEBUG
    pinned_index = load_pinned_event_index(league, season)
    source_counts: Counter[str] = Counter()
    resolver_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    book_counts: Counter[str] = Counter()
    total_probe_steps = 0
    games_fixed = 0
    week_dirty = False
    sidecar_dirty: Set[str] = set()
    league_tag = league.upper()
    debug_rows: List[Dict[str, Any]] = [] if debug_enabled else []

    team_labels: List[str] = []
    for candidate in games or []:
        if not isinstance(candidate, dict):
            continue
        team_labels.append(candidate.get("home_team_norm") or candidate.get("home_team_raw") or "")
        team_labels.append(candidate.get("away_team_norm") or candidate.get("away_team_raw") or "")
    provider_summary = build_provider_map(league, team_labels)
    print(
        "ATSDBG(PROVIDER-MAP): league={league} week={season}-{week} total={total} mapped={mapped} "
        "ambiguous={ambiguous} unknown={unknown}".format(
            league=league,
            season=season,
            week=week,
            total=provider_summary["total"],
            mapped=provider_summary["mapped"],
            ambiguous=provider_summary["ambiguous"],
            unknown=provider_summary["unknown"],
        ),
        flush=True,
    )

    contexts: List[Dict[str, Any]] = []
    for game in games or []:
        if not isinstance(game, dict):
            continue
        game_key = str(game.get("game_key") or "")
        if not game_key:
            continue

        debug_entry: Optional[Dict[str, Any]] = None
        if debug_enabled:
            debug_entry = {
                "game_key": game_key,
                "resolver": None,
                "event_id": None,
                "endpoint": None,
                "book": None,
                "favored_team": None,
                "spread": None,
                "kickoff": None,
                "scores": {"home": None, "away": None},
                "computed": None,
                "merged_week_fields": [],
                "patched_sidecar": {"home": False, "away": False},
                "reason": None,
                "probe_steps": 0,
            }

        context: Dict[str, Any] = {
            "game": game,
            "game_key": game_key,
            "debug_entry": debug_entry,
            "kickoff_dt": None,
            "kickoff_iso": None,
            "event_id": None,
            "resolver_used": None,
            "resolver_reason": None,
            "skip_reason": None,
            "debug_reason": None,
            "snapshot_iso": None,
            "cache_exists": None,
        }
        contexts.append(context)

        existing = (
            game.get("home_ats"),
            game.get("away_ats"),
            game.get("to_margin_home"),
            game.get("to_margin_away"),
        )
        if any(not _is_blank(field) for field in existing):
            context["skip_reason"] = "already_populated"
            context["debug_reason"] = "already_populated"
            if debug_enabled:
                debug_entry["reason"] = "already_populated"  # type: ignore[index]
            continue

        kickoff_dt = _parse_kickoff(game)
        kickoff_iso = kickoff_dt.isoformat() if kickoff_dt else (
            game.get("kickoff_iso_utc") or game.get("kickoff_iso")
        )
        context["kickoff_dt"] = kickoff_dt
        context["kickoff_iso"] = kickoff_iso
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=kickoff parsed={kickoff_iso or '-'}",
        )
        if debug_enabled:
            debug_entry["kickoff"] = kickoff_iso  # type: ignore[index]
        if kickoff_dt is None:
            context["skip_reason"] = "no_kickoff"
            context["debug_reason"] = "no_kickoff"
            continue

        event_id, resolver_used, resolver_reason = resolve_event_id(
            league,
            season,
            week,
            game,
            pinned_index=pinned_index,
        )
        resolver_counts[resolver_used] += 1
        context["event_id"] = event_id
        context["resolver_used"] = resolver_used
        context["resolver_reason"] = resolver_reason
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=resolve resolver={resolver_used} event_id={event_id or '-'}",
        )
        if debug_enabled:
            debug_entry["resolver"] = resolver_used  # type: ignore[index]
            debug_entry["event_id"] = event_id  # type: ignore[index]
            if resolver_reason:
                debug_entry["resolver_reason"] = resolver_reason  # type: ignore[index]

        if not event_id:
            context["skip_reason"] = "resolve_failed"
            context["debug_reason"] = resolver_reason or "resolve_failed"
            continue

        snapshot_dt = get_last_snapshot(league)
        if snapshot_dt is not None:
            snapshot_iso = snapshot_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            snapshot_iso = _normalize_kickoff(kickoff_dt, kickoff_iso)
        context["snapshot_iso"] = snapshot_iso
        if snapshot_iso:
            context["cache_exists"] = hist_odds_cache_exists(league, event_id, snapshot_iso)
        else:
            context["cache_exists"] = False

    eligible_contexts = [
        ctx for ctx in contexts if ctx.get("skip_reason") is None and ctx.get("event_id")
    ]
    snapshot_values = {
        str(ctx["snapshot_iso"])
        for ctx in eligible_contexts
        if ctx.get("snapshot_iso")
    }
    if len(snapshot_values) == 1:
        snapshot_label = next(iter(snapshot_values))
    elif len(snapshot_values) > 1:
        snapshot_label = "mixed"
    else:
        snapshot_label = "-"
    cache_available = sum(1 for ctx in eligible_contexts if ctx.get("cache_exists"))
    cache_missing = sum(
        1 for ctx in eligible_contexts if ctx.get("cache_exists") is False
    )
    print(
        f"HIST_ODDS_CACHE: league={league} week={season}-{week} snapshot={snapshot_label} "
        f"available={cache_available} missing={cache_missing}",
        flush=True,
    )

    for context in contexts:
        game = context["game"]
        game_key = context["game_key"]
        debug_entry = context["debug_entry"]
        skip_reason = context.get("skip_reason")

        if skip_reason:
            if skip_reason == "already_populated":
                counters["already_populated"] += 1
            elif skip_reason == "no_kickoff":
                counters["no_kickoff"] += 1
            elif skip_reason == "resolve_failed":
                counters["resolve_failed"] += 1
                resolver_reason = context.get("resolver_reason")
                if resolver_reason:
                    counters[resolver_reason] += 1
            if debug_enabled and debug_entry is not None:
                debug_entry["reason"] = context.get("debug_reason")  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        event_id = context.get("event_id")
        if not event_id:
            continue
        kickoff_iso = context.get("kickoff_iso") or ""

        selection = select_closing_spread(
            league=league,
            event_id=event_id,
            kickoff_iso=kickoff_iso,
            home_name=game.get("home_team_norm") or game.get("home_team_raw") or "",
            away_name=game.get("away_team_norm") or game.get("away_team_raw") or "",
        )
        selection = selection or {}
        try:
            probe_steps = int(selection.get("probe_steps", 0) or 0)
        except (TypeError, ValueError):
            probe_steps = 0
        total_probe_steps += probe_steps
        if debug_enabled and debug_entry is not None:
            debug_entry["probe_steps"] = probe_steps  # type: ignore[index]
        status = selection.get("status", "hist_odds_none")
        if status != "ok":
            counters[status] += 1
            selection_reason = selection.get("reason")
            if selection_reason:
                counters[selection_reason] += 1
            if debug_enabled and debug_entry is not None:
                debug_entry["reason"] = selection_reason or status  # type: ignore[index]
                debug_entry["books"] = {  # type: ignore[index]
                    "raw": selection.get("raw_book_count", 0),
                    "kept": selection.get("kept_book_count", 0),
                    "names": selection.get("kept_book_names", []),
                }
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        source_counts[selection.get("source", "history")] += 1
        endpoint_name = "historical_event_odds"
        snapshot_date = selection.get("snapshot_date")
        _atsdbg(
            debug_enabled,
            "ATSDBG({tag}): week={season}-{week} game={game} step=api endpoint={endpoint} "
            "book={book} favored={favored} spread={spread} ts={ts} snapshot={snapshot} raw_books={raw} kept_books={kept}".format(
                tag=league_tag,
                season=season,
                week=week,
                game=game_key,
                endpoint=endpoint_name,
                book=selection.get("book"),
                favored=selection.get("favored_team"),
                spread=selection.get("spread"),
                ts=selection.get("fetched_ts"),
                snapshot=snapshot_date,
                raw=selection.get("raw_book_count", 0),
                kept=selection.get("kept_book_count", 0),
            ),
        )
        if debug_enabled and debug_entry is not None:
            debug_entry["endpoint"] = endpoint_name  # type: ignore[index]
            debug_entry["book"] = selection.get("book")  # type: ignore[index]
            debug_entry["favored_team"] = selection.get("favored_team")  # type: ignore[index]
            debug_entry["spread"] = selection.get("spread")  # type: ignore[index]
            debug_entry["snapshot_date"] = snapshot_date  # type: ignore[index]
            debug_entry["snapshot_used"] = selection.get("snapshot_used", snapshot_date)  # type: ignore[index]
            debug_entry["books"] = {
                "raw": selection.get("raw_book_count", 0),
                "kept": selection.get("kept_book_count", 0),
                "names": selection.get("kept_book_names", []),
            }  # type: ignore[index]
        book_name = (selection.get("book") or "").strip()
        if book_name:
            book_counts[book_name.lower()] += 1
        favored = selection.get("favored_team")
        try:
            spread = float(selection["spread"])
        except (KeyError, TypeError, ValueError):
            counters["invalid_spread"] += 1
            if debug_enabled and debug_entry is not None:
                debug_entry["reason"] = "invalid_spread"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        home_score, away_score, score_source = _extract_scores(game, sidecar_map, season, week)
        if home_score is None or away_score is None:
            counters["no_scores"] += 1
            if debug_enabled and debug_entry is not None:
                debug_entry["scores"] = {"home": home_score, "away": away_score}  # type: ignore[index]
                debug_entry["score_source"] = score_source  # type: ignore[index]
                debug_entry["reason"] = "no_scores"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        hs = int(home_score)
        as_ = int(away_score)
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=scores home={hs} away={as_}",
        )
        if debug_enabled and debug_entry is not None:
            debug_entry["scores"] = {"home": hs, "away": as_}  # type: ignore[index]
            debug_entry["score_source"] = score_source  # type: ignore[index]

        ats_payload = compute_ats(hs, as_, str(favored or ""), spread)
        if not ats_payload:
            counters["no_scores"] += 1
            if debug_enabled and debug_entry is not None:
                debug_entry["reason"] = "compute_failed"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        _atsdbg(
            debug_enabled,
            "ATSDBG({tag}): week={season}-{week} game={game} step=compute home_ats={home_ats} "
            "away_ats={away_ats} tm_home={tm_home} tm_away={tm_away}".format(
                tag=league_tag,
                season=season,
                week=week,
                game=game_key,
                home_ats=ats_payload["home_ats"],
                away_ats=ats_payload["away_ats"],
                tm_home=ats_payload["to_margin_home"],
                tm_away=ats_payload["to_margin_away"],
            ),
        )
        if debug_enabled and debug_entry is not None:
            debug_entry["computed"] = ats_payload  # type: ignore[index]

        row_changed = False
        changed_fields: List[str] = []
        for field in ("home_ats", "away_ats", "to_margin_home", "to_margin_away"):
            merged = _merge_only(game.get(field), ats_payload[field])
            if merged != game.get(field):
                game[field] = merged
                row_changed = True
                changed_fields.append(field)

        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=merge_week changed={changed_fields}",
        )
        if row_changed:
            games_fixed += 1
            week_dirty = True
            counters["merged_week"] += 1
        if debug_enabled and debug_entry is not None:
            debug_entry["merged_week_fields"] = changed_fields  # type: ignore[index]

        sidecar_record = sidecar_map.get(game_key)

        def _patch_side(side: str, ats_value: Any, margin_value: Any) -> Tuple[bool, Optional[str]]:
            if not isinstance(sidecar_record, dict):
                return False, "missing_sidecar_record"
            key = f"{side}_ytd"
            rows = sidecar_record.get(key)
            if not isinstance(rows, list):
                if rows not in (None, []):
                    return False, "invalid_ytd_container"
                rows = []
                sidecar_record[key] = rows
            target = None
            for entry in rows:
                if not isinstance(entry, dict):
                    continue
                try:
                    if int(entry.get("season")) == season and int(entry.get("week")) == week:
                        target = entry
                        break
                except Exception:
                    continue
            created = False
            if target is None:
                target = {"season": season, "week": week, "ats": None, "to_margin": None}
                rows.append(target)
                created = True
            updated = False
            if ats_value is not None and _is_blank(target.get("ats")):
                target["ats"] = ats_value
                updated = True
            if margin_value is not None and _is_blank(target.get("to_margin")):
                try:
                    target["to_margin"] = float(margin_value)
                except Exception:
                    target["to_margin"] = margin_value
                updated = True
            if updated:
                return True, None
            if created:
                return False, "skeleton_created"
            return False, "already_filled"

        home_updated, home_reason = _patch_side(
            "home", ats_payload["home_ats"], ats_payload["to_margin_home"]
        )
        away_updated, away_reason = _patch_side(
            "away", ats_payload["away_ats"], ats_payload["to_margin_away"]
        )
        if home_updated:
            sidecar_dirty.add(game_key)
            counters["patched_sidecar"] += 1
        if away_updated:
            sidecar_dirty.add(game_key)
            counters["patched_sidecar"] += 1

        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=sidecar team=home updated={int(home_updated)}"
            + (f" reason={home_reason}" if home_reason else ""),
        )
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=sidecar team=away updated={int(away_updated)}"
            + (f" reason={away_reason}" if away_reason else ""),
        )
        if debug_enabled and debug_entry is not None:
            debug_entry["patched_sidecar"] = {  # type: ignore[index]
                "home": bool(home_updated),
                "away": bool(away_updated),
            }
            debug_rows.append(debug_entry)  # type: ignore[arg-type]

    used = ODDS_API_USAGE.get("used")
    remaining = ODDS_API_USAGE.get("remaining")
    if book_counts:
        book_summary_items = ",".join(f"{name}:{book_counts[name]}" for name in sorted(book_counts))
        book_summary = "{" + book_summary_items + "}"
    else:
        book_summary = "{}"
    summary_line = (
        f"ATS_BACKFILL(API {league_tag}): week={season}-{week} games_fixed={games_fixed} "
        f"sources={{history:{source_counts.get('history', 0)},current:{source_counts.get('current', 0)}}} "
        f"resolve={{pinned:{resolver_counts.get('pinned', 0)},events:{resolver_counts.get('events', 0)},failed:{resolver_counts.get('failed', 0)}}} "
        f"skips={{already_populated:{counters.get('already_populated', 0)},no_kickoff:{counters.get('no_kickoff', 0)},resolve_failed:{counters.get('resolve_failed', 0)},no_provider_map:{counters.get('no_provider_map', 0)},no_event_match:{counters.get('no_event_match', 0)},time_guard_miss:{counters.get('time_guard_miss', 0)},hist_odds_none:{counters.get('hist_odds_none', 0)},hist_odds_filtered:{counters.get('hist_odds_filtered', 0)},invalid_spread:{counters.get('invalid_spread', 0)},no_scores:{counters.get('no_scores', 0)}}} "
        f"writes={{merged_week:{counters.get('merged_week', 0)},patched_sidecar:{counters.get('patched_sidecar', 0)}}} "
        f"books={book_summary} probes_total={total_probe_steps} "
        f"usage=used:{used},remaining:{remaining}"
    )
    print(summary_line)

    if debug_enabled:
        debug_root = ensure_out_dir() / "debug" / "ats_backfill" / league.lower()
        debug_root.mkdir(parents=True, exist_ok=True)
        debug_path = debug_root / f"{season}_week{week}.json"
        write_atomic_json(debug_path, debug_rows)

    return {
        "games_fixed": games_fixed,
        "week_dirty": week_dirty,
        "sidecar_dirty": sidecar_dirty,
    }


def _run_odds_staging(season: int, week: int) -> Dict[str, Any]:
    ingest_result = ingest_nfl_odds_raw()
    raw_records = ingest_result.get("records", []) or []

    day_window = int(getenv("ODDS_PIN_DAY_WINDOW", "3") or "3")
    max_delta_hours = float(getenv("ODDS_PIN_MAX_KICKOFF_DELTA_HOURS", "36") or "36")
    role_swap_enabled = getenv("ODDS_ROLE_SWAP_TOLERANCE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "disabled",
    }

    pin_result = pin_nfl_odds(
        raw_records,
        day_window=day_window,
        max_delta_hours=max_delta_hours,
        role_swap_tolerance=role_swap_enabled,
    )
    counts = pin_result["counts"]
    markets_snapshot = counts.get("markets", {})
    books_snapshot = counts.get("books", {})
    log_line = (
        "NFL ODDS STAGING: "
        f"raw={counts.get('raw', 0)} "
        f"pinned={counts.get('pinned', 0)} "
        f"unmatched={counts.get('unmatched', 0)} "
        f"candidate_sets_zero={counts.get('candidate_sets_zero', 0)} "
        f"candidate_sets_multi={counts.get('candidate_sets_multi', 0)} "
        f"markets={markets_snapshot} "
        f"books={books_snapshot}"
    )
    if counts.get("raw", 0) > 0 and counts.get("pinned", 0) == 0:
        log_line += ' hint="provider ahead or schedule mismatch"'
    print(log_line)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    args, remaining = parser.parse_known_args(sys.argv[1:])

    if (args.season is None) != (args.week is None):
        raise SystemExit("NFL refresh: provide both --season and --week or neither.")

    current_week_info = None
    current_week_error = None
    try:
        current_week_info = get_current_week("NFL")
    except Exception as exc:
        current_week_error = exc

    if args.season is None:
        if current_week_info is None:
            raise SystemExit(f"Global Week Service unavailable: {current_week_error}")
        season, week, current_ts = current_week_info
        args_label = "auto"
    else:
        season = int(args.season)
        week = int(args.week)
        current_ts = current_week_info[2] if current_week_info else "unknown"
        args_label = "manual"

    if current_week_info:
        cur_season, cur_week, cur_ts = current_week_info
        print(f"CurrentWeek(NFL)={cur_season} W{cur_week} computed_at={cur_ts}; args={args_label}")
    else:
        print(f"CurrentWeek(NFL)=unavailable; args={args_label}")

    from src.refresh_week_data import main as legacy_main

    original_argv = sys.argv[:]
    legacy_argv = [original_argv[0], "--season", str(season), "--week", str(week), *remaining]
    sys.argv = legacy_argv
    try:
        exit_code = legacy_main()
    finally:
        sys.argv = original_argv
    if exit_code not in (None, 0):
        raise SystemExit(exit_code)

    staging_counts = _run_odds_staging(season, week)
    run_nfl_sagarin_staging()

    backfill_summary = backfill_nfl_scores(season, week)
    weeks_label = "[" + ",".join(backfill_summary.get("weeks", [])) + "]"
    print(
        f"Scores(NFL): weeks={weeks_label} updated={backfill_summary.get('updated', 0)} "
        f"skipped={backfill_summary.get('skipped', 0)}"
    )

    team_ats = build_team_ats(season, week)
    ats_rows_updated = apply_ats_to_week(season, week, team_ats)
    print(f"ATS(NFL): teams={apply_ats_to_week.teams_in_week} rows_updated={ats_rows_updated}")

    out_root = ensure_out_dir()
    week_dir = out_root / f"{season}_week{week}"
    json_path = week_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = week_dir / f"games_week_{season}_{week}.csv"

    games = read_week_json(json_path) or []
    sidecar_map = _load_sidecar_map("nfl", season, week)
    ats_state = _ats_backfill_api("nfl", season, week, games, sidecar_map)
    if ats_state["week_dirty"]:
        write_week_outputs(games, season, week)
        games = read_week_json(json_path)
    if ats_state["sidecar_dirty"]:
        _write_sidecar_files("nfl", season, week, sidecar_map, ats_state["sidecar_dirty"])

    promotion_info = None
    legacy_mismatch = 0
    promoted_total = 0

    if getenv("ODDS_PROMOTION_ENABLE", "1").strip().lower() not in {"0", "false", "off", "disabled"}:
        legacy_rows = games
        rows = copy.deepcopy(legacy_rows)
        policy = getenv("ODDS_SELECT_POLICY", "latest_by_fetch_ts") or "latest_by_fetch_ts"
        promotion_info = promote_week_odds(rows, season, week, policy=policy)
        promoted_total = promotion_info.get("promoted_games", 0)
        if promoted_total > 0:
            write_week_outputs(rows, season, week)
        legacy_flag = getenv("ODDS_LEGACY_JOIN_ENABLE", "0").strip().lower() not in {
            "0",
            "false",
            "off",
            "disabled",
        }
        if legacy_flag:
            diff_summary = diff_game_rows(rows, legacy_rows)
            legacy_mismatch = diff_summary.get("mismatched", 0)
        log_line = (
            f"NFL ODDS PROMOTION: week={season}-{week} promoted={promoted_total} "
            f"by_market={promotion_info.get('by_market', {})} "
            f"by_book={promotion_info.get('by_book', {})} "
            f"legacy_mismatch={legacy_mismatch}"
        )
        if promoted_total == 0 and promotion_info.get("other_week_records", 0):
            log_line += ' hint="provider ahead; staged"'
        print(log_line)
    else:
        print("NFL ODDS PROMOTION: disabled via ODDS_PROMOTION_ENABLE")

    rows_count = len(read_week_json(json_path))

    print(
        f"NOTIFY: NFL refresh complete week={season}-{week} rows={rows_count} "
        f"odds_promoted={promoted_total}."
    )

if __name__ == "__main__":
    main()

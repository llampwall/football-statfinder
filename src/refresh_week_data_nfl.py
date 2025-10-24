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
from typing import Any, Dict, Optional, Set

from src.common.current_week_service import get_current_week
from src.common.io_atomic import write_atomic_text
from src.common.io_utils import ensure_out_dir, getenv
from src.odds.nfl_ingest import ingest_nfl_odds_raw
from src.odds.nfl_pin_to_schedule import pin_nfl_odds
from src.odds.nfl_promote_week import (
    promote_week_odds,
    read_week_json,
    write_week_outputs,
    diff_game_rows,
)
from src.odds.ats_backfill_api import compute_ats, resolve_event_id, select_closing_spread
from src.odds.odds_api_client import ODDS_API_USAGE
from src.ratings.sagarin_nfl_fetch import run_nfl_sagarin_staging
from src.scores.nfl_backfill import backfill_nfl_scores
from src.ats.nfl_ats import build_team_ats, apply_ats_to_week

_BLANK_SENTINELS = {None, "", "-", "\u2014"}


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


def _week_sidecar_dir(league: str, season: int, week: int):
    root = ensure_out_dir()
    league_lower = league.lower()
    if league_lower == "nfl":
        return root / f"{season}_week{week}" / "game_schedules"
    return root / league_lower / f"{season}_week{week}" / "game_schedules"


def _load_pinned_index(league: str, season: int) -> Dict[str, str]:
    root = ensure_out_dir() / "staging" / "odds_pinned" / league.lower()
    path = root / f"{season}.jsonl"
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return mapping
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("market") != "spreads":
            continue
        game_key = record.get("game_key")
        event_id = (record.get("raw_event") or {}).get("event_id")
        if isinstance(game_key, str) and isinstance(event_id, str):
            mapping[game_key] = event_id
    return mapping


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
    games: Optional[list],
    sidecar_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    enable = (os.getenv("ATS_BACKFILL_ENABLED", "0") == "1") or (
        os.getenv("ATS_BACKFILL_SOURCE", "") == "api"
    )
    if not enable:
        return {"games_fixed": 0, "week_dirty": False, "sidecar_dirty": set()}

    pinned_index = _load_pinned_index(league, season)
    counts: Counter[str] = Counter()
    games_fixed = 0
    week_dirty = False
    sidecar_dirty: Set[str] = set()

    games_iter = games or []

    for game in games_iter:
        if not isinstance(game, dict):
            continue
        game_key = str(game.get("game_key") or "")
        if not game_key:
            continue

        week_fields = (
            game.get("home_ats"),
            game.get("away_ats"),
            game.get("to_margin_home"),
            game.get("to_margin_away"),
        )
        if any(not _is_blank(field) for field in week_fields):
            continue

        kickoff_dt = _parse_kickoff(game)
        event_id = pinned_index.get(game_key) or resolve_event_id(league, season, game)
        selection = select_closing_spread(
            league=league,
            season=season,
            event_id=event_id,
            home_name=game.get("home_team_norm") or game.get("home_team_raw"),
            away_name=game.get("away_team_norm") or game.get("away_team_raw"),
            kickoff=kickoff_dt,
        )
        if not selection:
            continue

        favored = selection.get("favored_team")
        try:
            spread = float(selection["spread"])
        except (KeyError, TypeError, ValueError):
            continue

        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if _is_blank(home_score) or _is_blank(away_score):
            schedule_row = (game.get("raw_sources") or {}).get("schedule_row") or {}
            if _is_blank(home_score):
                home_score = schedule_row.get("home_score")
            if _is_blank(away_score):
                away_score = schedule_row.get("away_score")
        try:
            home_score_int = int(home_score)
            away_score_int = int(away_score)
        except Exception:
            continue

        ats_payload = compute_ats(home_score_int, away_score_int, str(favored or ""), spread)
        if not ats_payload:
            continue

        row_changed = False
        for field in ("home_ats", "away_ats", "to_margin_home", "to_margin_away"):
            merged = _merge_only(game.get(field), ats_payload[field])
            if merged != game.get(field):
                game[field] = merged
                row_changed = True

        sidecar_record = sidecar_map.get(game_key)

        def _patch_side(side: str, ats_value: Any, margin_value: Any) -> bool:
            if not isinstance(sidecar_record, dict):
                return False
            key = f"{side}_ytd"
            rows = sidecar_record.get(key)
            if not isinstance(rows, list):
                if rows not in (None, []):
                    return False
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
            return created or updated

        if _patch_side("home", ats_payload["home_ats"], ats_payload["to_margin_home"]):
            sidecar_dirty.add(game_key)
        if _patch_side("away", ats_payload["away_ats"], ats_payload["to_margin_away"]):
            sidecar_dirty.add(game_key)

        if row_changed:
            games_fixed += 1
            week_dirty = True

        counts[selection.get("source", "api")] += 1

    used = ODDS_API_USAGE.get("used")
    remaining = ODDS_API_USAGE.get("remaining")
    print(
        f"ATS_BACKFILL(API {league.upper()}): week={season}-{week} "
        f"games_fixed={games_fixed} source_counts={dict(counts)} "
        f"usage=used:{used},remaining:{remaining}"
    )

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

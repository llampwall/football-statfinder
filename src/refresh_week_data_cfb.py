"""Weekly orchestrator for refreshing CFB outputs (mirrors NFL flow)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

OUT_ROOT = Path(__file__).resolve().parents[1] / "out"

from src.fetch_games_cfb import load_games, filter_week_reg
from src.ats.cfb_ats import apply_ats_to_week, build_team_ats
from src.common.current_week_service import get_current_week
from src.common.io_atomic import write_atomic_text
from src.common.io_utils import write_csv, write_jsonl, getenv, write_atomic_json
from src.odds.cfb_ingest import ingest_cfb_odds_raw
from src.odds.cfb_pin_to_schedule import pin_cfb_odds
from src.odds.cfb_promote_week import promote_week_odds, diff_game_rows
from src.odds.ats_backfill_api import (
    compute_ats,
    load_pinned_event_index,
    resolve_event_id,
    select_closing_spread,
)
from src.odds.participants_cache import build_provider_map
from src.odds.odds_api_client import ODDS_API_USAGE
from src.ratings.sagarin_cfb_fetch import run_cfb_sagarin_staging
from src.scores.cfb_backfill import backfill_cfb_scores
from src.schedule_master_cfb import (
    ensure_weeks_present as ensure_cfb_schedule_master,
    enrich_from_local_odds,
)

LEAGUE_MIN_TEAMS_ABS = 100
LEAGUE_MIN_TEAMS_FRAC = 0.70

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


def _week_sidecar_dir(season: int, week: int) -> Path:
    return OUT_ROOT / "cfb" / f"{season}_week{week}" / "game_schedules"


def _load_sidecar_map(season: int, week: int) -> Dict[str, Dict[str, Any]]:
    side_dir = _week_sidecar_dir(season, week)
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
    season: int, week: int, sidecar_map: Dict[str, Dict[str, Any]], dirty_keys: Set[str]
) -> None:
    if not dirty_keys:
        return
    side_dir = _week_sidecar_dir(season, week)
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
            }

        existing = (
            game.get("home_ats"),
            game.get("away_ats"),
            game.get("to_margin_home"),
            game.get("to_margin_away"),
        )
        if any(not _is_blank(field) for field in existing):
            counters["already_populated"] += 1
            if debug_enabled:
                debug_entry["reason"] = "already_populated"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        kickoff_dt = _parse_kickoff(game)
        kickoff_iso = kickoff_dt.isoformat() if kickoff_dt else (
            game.get("kickoff_iso_utc") or game.get("kickoff_iso")
        )
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=kickoff parsed={kickoff_iso or '-'}",
        )
        if debug_enabled:
            debug_entry["kickoff"] = kickoff_iso  # type: ignore[index]
        if kickoff_dt is None:
            counters["no_kickoff"] += 1

        event_id, resolver_used = resolve_event_id(
            league,
            season,
            week,
            game,
            pinned_index=pinned_index,
        )
        resolver_counts[resolver_used] += 1
        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=resolve resolver={resolver_used} event_id={event_id or '-'}",
        )
        if debug_enabled:
            debug_entry["resolver"] = resolver_used  # type: ignore[index]
            debug_entry["event_id"] = event_id  # type: ignore[index]

        if not event_id:
            counters["resolve_failed"] += 1
            if debug_enabled:
                debug_entry["reason"] = "resolve_failed"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        selection = select_closing_spread(
            league=league,
            event_id=event_id,
            kickoff_iso=kickoff_iso or "",
            home_name=game.get("home_team_norm") or game.get("home_team_raw") or "",
            away_name=game.get("away_team_norm") or game.get("away_team_raw") or "",
        )
        selection = selection or {}
        status = selection.get("status", "hist_odds_none")
        if status != "ok":
            counters[status] += 1
            if debug_enabled:
                debug_entry["reason"] = status  # type: ignore[index]
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
        if debug_enabled:
            debug_entry["endpoint"] = endpoint_name  # type: ignore[index]
            debug_entry["book"] = selection.get("book")  # type: ignore[index]
            debug_entry["favored_team"] = selection.get("favored_team")  # type: ignore[index]
            debug_entry["spread"] = selection.get("spread")  # type: ignore[index]
            debug_entry["snapshot_date"] = snapshot_date  # type: ignore[index]
            debug_entry["books"] = {
                "raw": selection.get("raw_book_count", 0),
                "kept": selection.get("kept_book_count", 0),
                "names": selection.get("kept_book_names", []),
            }  # type: ignore[index]

        favored = selection.get("favored_team")
        try:
            spread = float(selection["spread"])
        except (KeyError, TypeError, ValueError):
            counters["invalid_spread"] += 1
            if debug_enabled:
                debug_entry["reason"] = "invalid_spread"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
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
            hs = int(home_score)
            as_ = int(away_score)
        except Exception:
            counters["no_scores"] += 1
            if debug_enabled:
                debug_entry["scores"] = {"home": home_score, "away": away_score}  # type: ignore[index]
                debug_entry["reason"] = "no_scores"  # type: ignore[index]
                debug_rows.append(debug_entry)  # type: ignore[arg-type]
            continue

        _atsdbg(
            debug_enabled,
            f"ATSDBG({league_tag}): week={season}-{week} game={game_key} step=scores home={hs} away={as_}",
        )
        if debug_enabled:
            debug_entry["scores"] = {"home": hs, "away": as_}  # type: ignore[index]

        ats_payload = compute_ats(hs, as_, str(favored or ""), spread)
        if not ats_payload:
            counters["no_scores"] += 1
            if debug_enabled:
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
        if debug_enabled:
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
        if debug_enabled:
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
        if debug_enabled:
            debug_entry["patched_sidecar"] = {  # type: ignore[index]
                "home": bool(home_updated),
                "away": bool(away_updated),
            }
            debug_rows.append(debug_entry)  # type: ignore[arg-type]

    used = ODDS_API_USAGE.get("used")
    remaining = ODDS_API_USAGE.get("remaining")
    summary_line = (
        f"ATS_BACKFILL(API {league_tag}): week={season}-{week} games_fixed={games_fixed} "
        f"sources={{history:{source_counts.get('history', 0)},current:{source_counts.get('current', 0)}}} "
        f"resolve={{pinned:{resolver_counts.get('pinned', 0)},events:{resolver_counts.get('events', 0)},failed:{resolver_counts.get('failed', 0)}}} "
        f"skips={{already_populated:{counters.get('already_populated', 0)},no_kickoff:{counters.get('no_kickoff', 0)},resolve_failed:{counters.get('resolve_failed', 0)},hist_odds_none:{counters.get('hist_odds_none', 0)},hist_odds_filtered:{counters.get('hist_odds_filtered', 0)},invalid_spread:{counters.get('invalid_spread', 0)},no_scores:{counters.get('no_scores', 0)}}} "
        f"writes={{merged_week:{counters.get('merged_week', 0)},patched_sidecar:{counters.get('patched_sidecar', 0)}}} "
        f"usage=used:{used},remaining:{remaining}"
    )
    print(summary_line)

    if debug_enabled:
        debug_root = OUT_ROOT / "debug" / "ats_backfill" / league.lower()
        debug_root.mkdir(parents=True, exist_ok=True)
        debug_path = debug_root / f"{season}_week{week}.json"
        write_atomic_json(debug_path, debug_rows)

    return {
        "games_fixed": games_fixed,
        "week_dirty": week_dirty,
        "sidecar_dirty": sidecar_dirty,
    }


def run_module(
    module: str,
    season: int,
    week: int,
    *,
    include_week: bool = True,
    extra_args: Optional[List[str]] = None,
    check: bool = True,
) -> int:
    cmd = [sys.executable, "-m", module, "--season", str(season)]
    if include_week:
        cmd.extend(["--week", str(week)])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        # subtract header if present
        rows = sum(1 for _ in handle)
    return max(rows - 1, 0)


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def write_gaps_report(
    season: int,
    week: int,
    base_dir: Path,
    notes: List[str],
    critical_fields: List[str],
) -> None:
    report_path = base_dir / "orchestrator_gaps.json"
    cfb_path = base_dir / f"games_week_{season}_{week}.jsonl"
    cfb_records = read_jsonl(cfb_path)
    cfb_keys = {rec.get("game_key") for rec in cfb_records if isinstance(rec.get("game_key"), str)}

    nfl_path = OUT_ROOT / f"{season}_week{week}" / f"games_week_{season}_{week}.jsonl"
    nfl_records = read_jsonl(nfl_path)
    nfl_keys = {rec.get("game_key") for rec in nfl_records if isinstance(rec.get("game_key"), str)}

    report_notes = list(notes)
    if not nfl_keys:
        report_notes.append("NFL week baseline not found; skipping cross-league key comparison.")

    missing_keys = sorted(str(key) for key in nfl_keys - cfb_keys if key)
    missing_hits = {key: 1 for key in missing_keys}

    blank_keys: List[str] = []
    blank_hits: Dict[str, int] = {}
    for record in cfb_records:
        game_key = str(record.get("game_key") or "")
        if not game_key:
            continue
        blanks = 0
        for field in critical_fields:
            value = record.get(field)
            if value in (None, "", [], {}):
                blanks += 1
        if blanks > 0:
            blank_keys.append(game_key)
            blank_hits[game_key] = blanks

    payload = {
        "games_week_keys_missing": missing_keys,
        "games_week_keys_present_blank": sorted(blank_keys),
        "counts": {
            "rows": len(cfb_records),
            "missing_key_hits": missing_hits,
            "blank_key_hits": blank_hits,
        },
        "notes": report_notes,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _has_number(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def log_odds_join_audit(season: int, week: int, game_records: List[dict], base_dir: Path) -> None:
    num_games = len(game_records)
    num_with_spread = sum(1 for rec in game_records if _has_number(rec.get("spread_home_relative")))
    num_with_total = sum(1 for rec in game_records if _has_number(rec.get("total")))
    num_with_both = sum(
        1
        for rec in game_records
        if _has_number(rec.get("spread_home_relative")) and _has_number(rec.get("total"))
    )

    unmatched_names: List[str] = []
    debug_path = base_dir / "odds_match_debug.json"
    name_counter: Counter[str] = Counter()
    if debug_path.exists():
        try:
            debug_data = json.loads(debug_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: Unable to parse odds debug file ({debug_path})")
        else:
            unmatched_samples = (
                debug_data.get("samples", {}).get("unmatched") if isinstance(debug_data, dict) else None
            )
            if unmatched_samples and isinstance(unmatched_samples, list):
                for sample in unmatched_samples:
                    if not isinstance(sample, dict):
                        continue
                    for key in ("event_home", "event_away", "home_norm", "away_norm"):
                        name = sample.get(key)
                        if name:
                            name_counter[str(name)] += 1
    unmatched_names = [name for name, _ in name_counter.most_common(5)]

    print(
        f"CFB odds join: rows={num_games} spread={num_with_spread} total={num_with_total} "
        f"both={num_with_both} unmatched_names={unmatched_names}"
    )


def build_summary(season: int, week: int) -> Dict[str, int]:
    base = OUT_ROOT / "cfb" / f"{season}_week{week}"
    summary = {
        "games_jsonl": count_jsonl(base / f"games_week_{season}_{week}.jsonl"),
        "games_csv_rows": count_csv_rows(base / f"games_week_{season}_{week}.csv"),
        "league_rows": count_csv_rows(base / f"league_metrics_{season}_{week}.csv"),
        "odds_records": count_jsonl(base / f"odds_{season}_wk{week}.jsonl"),
        "sagarin_rows": count_csv_rows(base / f"sagarin_cfb_{season}_wk{week}.csv"),
        "sidecars": len(list((base / "game_schedules").glob("*.json"))) if (base / "game_schedules").exists() else 0,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh CFB weekly outputs.")
    parser.add_argument("--season", type=int, help="Season override (optional; defaults to Global Week service).")
    parser.add_argument("--week", type=int, help="Week override (optional; defaults to Global Week service).")
    parser.add_argument("--include-eoy", action="store_true", help="Also build prior-season CFB EOY metrics")
    parser.add_argument(
        "--odds-days-before",
        type=int,
        default=6,
        help="Extend odds window start by N days before earliest kickoff (default: 6).",
    )
    parser.add_argument(
        "--odds-days-after",
        type=int,
        default=6,
        help="Extend odds window end by N days after latest kickoff (default: 6).",
    )
    args = parser.parse_args()

    current_week_info = None
    try:
        current_week_info = get_current_week("CFB")
    except Exception as exc:
        current_week_info = None
        current_week_error = exc
    else:
        current_week_error = None

    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be provided together.")

    auto_mode = args.season is None and args.week is None
    if auto_mode:
        if current_week_info is None:
            raise SystemExit(f"FAIL: Global Week Service unavailable: {current_week_error}")
        season, week, cur_ts = current_week_info
        args_label = "auto"
    else:
        season = int(args.season)
        week = int(args.week)
        cur_ts = current_week_info[2] if current_week_info else "unavailable"
        args_label = "manual"

    if current_week_info:
        cur_season, cur_week, cur_ts_logged = current_week_info
        print(f"CurrentWeek(CFB)={cur_season} W{cur_week} computed_at={cur_ts_logged}; args={args_label}")
    else:
        print(f"CurrentWeek(CFB)=unavailable; args={args_label}")

    base_dir = OUT_ROOT / "cfb" / f"{season}_week{week}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=== CFB WEEKLY REFRESH ===")
    print(f"Season {season} Week {week}")

    notes: List[str] = []

    try:
        ensure_cfb_schedule_master([season, season - 1])
    except Exception as exc:
        print(f"WARNING: schedule master update failed: {exc}")
    critical_fields = [
        "spread_home_relative",
        "favored_side",
        "spread_favored_team",
        "rating_diff_favored_team",
        "rating_vs_odds_favored_team",
        "home_pf_pg",
        "away_pf_pg",
        "home_ry_pg",
        "away_ry_pg",
        "home_rush_rank",
        "away_rush_rank",
    ]

    # Schedule ingest
    print("\n>>> Running src.fetch_games_cfb .")
    schedule_rc = run_module("src.fetch_games_cfb", season, week, check=False)
    if schedule_rc != 0:
        print("FAIL: CFB schedule ingest command failed.")
        return schedule_rc or 1
    schedule_df = filter_week_reg(load_games(season), season, week)
    schedule_rows = int(schedule_df.shape[0])
    if schedule_rows == 0:
        print("FAIL: CFB schedule produced 0 normalized rows.")
        return 1
    print(f"PASS: CFB schedule rows={schedule_rows}")
    notes.append(f"Schedule rows={schedule_rows}")

    odds_flag = getenv("ODDS_STAGING_ENABLE", "1")
    staging_counts = {"raw": 0, "pinned": 0, "unmatched": 0}
    staging_examples: List[str] = []
    if odds_flag.strip().lower() not in {"0", "false", "off", "disabled"}:
        raw_payload = ingest_cfb_odds_raw()
        raw_records = raw_payload.get("records", []) or []
        day_window = int(getenv("ODDS_PIN_DAY_WINDOW", "3"))
        max_delta_hours = float(getenv("ODDS_PIN_MAX_KICKOFF_DELTA_HOURS", "36"))
        role_swap_enabled = getenv("ODDS_ROLE_SWAP_TOLERANCE", "1").strip().lower() not in {"0", "false", "off"}
        pin_result = pin_cfb_odds(
            raw_records,
            day_window=day_window,
            max_delta_hours=max_delta_hours,
            role_swap_tolerance=role_swap_enabled,
        )
        counts = pin_result.get("counts", {})
        staging_counts = {
            "raw": len(raw_records),
            "pinned": counts.get("pinned", 0),
            "unmatched": counts.get("unmatched", 0),
        }
        staging_examples = pin_result.get("examples_unmatched", []) or []
        staging_examples = list(dict.fromkeys(staging_examples))[:3]
        markets_snapshot = counts.get("markets", {}) or {}
        books_snapshot = counts.get("books", {}) or {}
        log_line = (
                "CFB ODDS STAGING: "
                f"raw={staging_counts['raw']} pinned={staging_counts['pinned']} "
                f"unmatched={staging_counts['unmatched']} new_raw={len(raw_records)} "
                f"new_pinned={staging_counts['pinned']} markets={markets_snapshot} "
                f"books={books_snapshot} window=+/-{day_window}d "
                f"candidate_sets_zero={staging_counts.get('candidate_sets_zero', 0)} "
                f"candidate_sets_multi={staging_counts.get('candidate_sets_multi', 0)} "
                f"max_delta={max_delta_hours}h examples_unmatched={staging_examples}"
            )
        if staging_counts["raw"] > 0 and staging_counts["pinned"] == 0:
            log_line += ' hint="provider ahead or schedule mismatch"'
        print(log_line)
        notes.append(
            f"Odds staging raw={staging_counts['raw']} pinned={staging_counts['pinned']} "
            f"unmatched={staging_counts['unmatched']}"
        )
    else:
        print("CFB ODDS STAGING: disabled via ODDS_STAGING_ENABLE")

    # League metrics
    print("\n>>> Running src.fetch_year_to_date_stats_cfb .")
    lm_rc = run_module("src.fetch_year_to_date_stats_cfb", season, week, check=False)
    league_path = base_dir / f"league_metrics_{season}_{week}.csv"
    league_debug_path = base_dir / "league_metrics_debug.json"
    if lm_rc != 0 or not league_path.exists():
        print(f"FAIL: CFB league metrics; see {league_debug_path}")
        return lm_rc or 1
    league_rows = count_csv_rows(league_path)
    league_stats = {}
    if league_debug_path.exists():
        try:
            league_stats = json.loads(league_debug_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            league_stats = {}
    teams_total = int(league_stats.get("teams_total") or 0)
    teams_meeting = int(league_stats.get("teams_meeting_threshold") or 0)
    rank_fraction = float(league_stats.get("rank_columns_ok_fraction") or 0.0)
    required = max(LEAGUE_MIN_TEAMS_ABS, math.ceil(LEAGUE_MIN_TEAMS_FRAC * teams_total)) if teams_total else 0
    coverage_msg = (
        f"{teams_meeting}/{teams_total} teams meeting threshold (required {required}); "
        f"ranks_ok={rank_fraction:.2f}"
        if teams_total
        else f"rows={league_rows}"
    )
    print(f"PASS: CFB league metrics coverage -> {coverage_msg}")
    notes.append(f"League metrics {coverage_msg}")

    # Prior-season finals (optional)
    if args.include_eoy:
        print("\n>>> Running src.fetch_last_year_stats_cfb .")
        prior = season - 1
        eoy_rc = run_module(
            "src.fetch_last_year_stats_cfb",
            season,
            week,
            include_week=False,
            check=False,
        )
        eoy_csv = OUT_ROOT / "cfb" / f"final_league_metrics_{prior}.csv"
        eoy_debug = OUT_ROOT / "cfb" / "final_league_metrics_debug.json"
        if eoy_rc != 0 or not eoy_csv.exists():
            print(f"FAIL: CFB final league metrics; see {eoy_debug}")
            return eoy_rc or 1
        eoy_stats = {}
        if eoy_debug.exists():
            try:
                eoy_stats = json.loads(eoy_debug.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                eoy_stats = {}
        eoy_total = int(eoy_stats.get("teams_total") or 0)
        eoy_meeting = int(eoy_stats.get("teams_meeting_threshold") or 0)
        eoy_rank = float(eoy_stats.get("rank_columns_ok_fraction") or 0.0)
        print(
            f"PASS: CFB final league metrics -> {eoy_csv} "
            f"(teams {eoy_meeting}/{eoy_total}, ranks_ok={eoy_rank:.2f})"
        )
        notes.append(f"EOY metrics {eoy_meeting}/{eoy_total} ranks_ok={eoy_rank:.2f}")

    # Odds snapshot
    print("\n>>> Running src.fetch_week_odds_cfb .")
    odds_extra_args = [
        "--odds-days-before",
        str(args.odds_days_before),
        "--odds-days-after",
        str(args.odds_days_after),
    ]
    odds_rc = run_module(
        "src.fetch_week_odds_cfb",
        season,
        week,
        extra_args=odds_extra_args,
        check=False,
    )
    odds_debug = base_dir / "odds_match_debug.json"
    if odds_rc != 0 or not odds_debug.exists():
        print(f"FAIL: CFB odds coverage; see {odds_debug}")
        return odds_rc or 1
    try:
        odds_stats = json.loads(odds_debug.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        odds_stats = {}
    stats = odds_stats.get("stats") or {}
    events_total = int(stats.get("events_total") or 0)
    events_in_window = int(stats.get("events_in_window") or stats.get("events_after_window_filter") or 0)
    matched = int(stats.get("matched") or 0)
    unmatched = int(stats.get("unmatched") or 0)
    reason_buckets = odds_stats.get("reason_buckets") or {}
    odds_path = base_dir / f"odds_{season}_wk{week}.jsonl"
    print(
        f"PASS: CFB odds matched {matched}/{events_in_window} in-window (total={events_total}); unmatched={unmatched} "
        f"({odds_path})"
    )
    if reason_buckets:
        print(f"      by_why={reason_buckets}")
    notes.append(
        f"Odds matched {matched}/{events_in_window} (total {events_total}); unmatched={unmatched}"
    )

    try:
        enrichment = enrich_from_local_odds(season, week)
        print(
            f"Master enrichment: updated {enrichment.get('rows_updated', 0)}/{enrichment.get('rows_considered', 0)} "
            f"rows for season={season} week={week}"
        )
        notes.append(
            f"Master enrichment {enrichment.get('rows_updated', 0)}/{enrichment.get('rows_considered', 0)}"
        )
    except Exception as exc:
        print(f"WARNING: master enrichment failed: {exc}")
        notes.append("Master enrichment failed")

    # Sagarin snapshot + master upsert
    sagarin_staging_enabled = (
        getenv("SAGARIN_STAGING_ENABLE", "1").strip().lower() not in {"0", "false", "off", "disabled"}
    )
    backfill_summary = {"weeks": [], "updated": 0, "skipped": 0, "files_rewritten": 0}
    if sagarin_staging_enabled:
        print("\n>>> Running Sagarin staging flow.")
        try:
            sagarin_summary = run_cfb_sagarin_staging(season, week)
        except Exception as exc:
            print(f"FAIL: CFB Sagarin staging failed: {exc}")
            return 1
        sag_csv = Path(sagarin_summary.get("weekly_csv"))
        sag_rows = count_csv_rows(sag_csv)
        if sag_rows == 0:
            print(f"FAIL: CFB Sagarin staging produced empty snapshot ({sag_csv}).")
            return 1
        master_rows = int(sagarin_summary.get("master_total", 0))
        teams_selected = int(sagarin_summary.get("teams_selected", sag_rows))
        wrote_master_rows = int(sagarin_summary.get("wrote_master_rows", teams_selected))
        latest_ts = sagarin_summary.get("latest_fetch_ts")
        log_line = (
            f"Sagarin(CFB): latest_fetch_ts={latest_ts} teams={teams_selected} "
            f"wrote_master_rows={wrote_master_rows} selected_for_build={teams_selected}"
        )
        print(log_line)
        notes.append(f"Sagarin rows={teams_selected}; master={master_rows}")
    else:
        print("\n>>> Running src.fetch_sagarin_week_cfb .")
        sag_rc = run_module("src.fetch_sagarin_week_cfb", season, week, check=False)
        sag_csv = base_dir / f"sagarin_cfb_{season}_wk{week}.csv"
        receipt_path = base_dir / f"sagarin_cfb_{season}_wk{week}_receipt.json"
        if sag_rc != 0:
            debug_hint = receipt_path if receipt_path.exists() else (
                base_dir / f"sagarin_cfb_{season}_wk{week}_raw.txt"
            )
            print(f"FAIL: CFB Sagarin fetch failed; see {debug_hint}")
            return sag_rc or 1
        sag_rows = count_csv_rows(sag_csv)
        if sag_rows == 0:
            debug_hint = receipt_path if receipt_path.exists() else sag_csv
            print(f"FAIL: CFB Sagarin snapshot empty; see {debug_hint}")
            return 1

        print("\n>>> Running src.sagarin_master_cfb .")
        master_rc = run_module("src.sagarin_master_cfb", season, week, check=False)
        if master_rc != 0:
            print("FAIL: CFB Sagarin master upsert failed.")
            return master_rc or 1
        master_csv = OUT_ROOT / "master" / "sagarin_cfb_master.csv"
        master_rows = count_csv_rows(master_csv)
        print(f"PASS: CFB Sagarin rows={sag_rows}; master_total={master_rows}")
        notes.append(f"Sagarin rows={sag_rows}; master={master_rows}")
    backfill_summary = backfill_cfb_scores(season, week)

    # Game view builder
    print("\n>>> Running src.gameview_build_cfb .")
    gameview_rc = run_module("src.gameview_build_cfb", season, week, check=False)
    gv_jsonl = base_dir / f"games_week_{season}_{week}.jsonl"
    gv_csv = base_dir / f"games_week_{season}_{week}.csv"
    if gameview_rc != 0 or not gv_jsonl.exists() or not gv_csv.exists():
        print("FAIL: CFB Game View build failed.")
        return gameview_rc or 1
    gv_records = read_jsonl(gv_jsonl)
    if not gv_records:
        print("FAIL: CFB Game View JSONL is empty.")
        return 1
    favorite_fields = [
        "spread_home_relative",
        "favored_side",
        "spread_favored_team",
        "rating_diff_favored_team",
        "rating_vs_odds_favored_team",
    ]
    missing_favorite_fields = [field for field in favorite_fields if field not in gv_records[0]]
    if missing_favorite_fields:
        print(f"FAIL: CFB Game View missing fields: {', '.join(missing_favorite_fields)}")
        return 1

    # Sidecar timelines
    print("\n>>> Running src.build_team_timelines_cfb .")
    timelines_rc = run_module("src.build_team_timelines_cfb", season, week, check=False)
    if timelines_rc != 0:
        print("FAIL: CFB team timelines step failed.")
        return timelines_rc or 1
    sidecar_dir = base_dir / "game_schedules"
    if not sidecar_dir.exists():
        print("FAIL: CFB sidecar directory missing.")
        return 1
    sidecar_count = len(list(sidecar_dir.glob("*.json")))
    print(f"PASS: CFB sidecars written -> {sidecar_count} files (schedule rows={schedule_rows})")
    notes.append(f"Sidecars {sidecar_count}/{schedule_rows}")
    sagarin_receipt_path = base_dir / "schedules_sagarin_receipt.json"
    if sagarin_receipt_path.exists():
        try:
            sagarin_data = json.loads(sagarin_receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: Unable to parse Sagarin enrichment receipt ({sagarin_receipt_path})")
        else:
            rows_considered = int(sagarin_data.get("rows_considered") or 0)
            coverage = float(sagarin_data.get("coverage_fraction") or 0.0)
            opp_cov = float(sagarin_data.get("opp_coverage_fraction") or 0.0)
            print(
                f"PASS: Sagarin enrichment rows={rows_considered} "
                f"(coverage {coverage:.0%}, opp {opp_cov:.0%}) -> {sagarin_receipt_path}"
            )
            notes.append(f"Sagarin enrichment {coverage:.0%}/{opp_cov:.0%}")
    else:
        print(f"WARNING: Sagarin enrichment receipt missing ({sagarin_receipt_path})")

    # Game view builder (again - might be necessary)
    print("\n>>> Running src.gameview_build_cfb .")
    gameview_rc = run_module("src.gameview_build_cfb", season, week, check=False)
    gv_jsonl = base_dir / f"games_week_{season}_{week}.jsonl"
    gv_csv = base_dir / f"games_week_{season}_{week}.csv"
    if gameview_rc != 0 or not gv_jsonl.exists() or not gv_csv.exists():
        print("FAIL: CFB Game View build failed.")
        return gameview_rc or 1
    gv_records = read_jsonl(gv_jsonl)
    if not gv_records:
        print("FAIL: CFB Game View JSONL is empty.")
        return 1
    favorite_fields = [
        "spread_home_relative",
        "favored_side",
        "spread_favored_team",
        "rating_diff_favored_team",
        "rating_vs_odds_favored_team",
    ]
    missing_favorite_fields = [field for field in favorite_fields if field not in gv_records[0]]
    if missing_favorite_fields:
        print(f"FAIL: CFB Game View missing fields: {', '.join(missing_favorite_fields)}")
        return 1

    sidecar_map = _load_sidecar_map(season, week)
    ats_state = _ats_backfill_api("cfb", season, week, gv_records, sidecar_map)
    if ats_state["week_dirty"]:
        write_jsonl(gv_records, gv_jsonl)
        write_csv(gv_records, gv_csv)
        gv_records = read_jsonl(gv_jsonl)
    if ats_state["sidecar_dirty"]:
        _write_sidecar_files(season, week, sidecar_map, ats_state["sidecar_dirty"])

    legacy_rows = None
    legacy_flag = getenv("ODDS_LEGACY_JOIN_ENABLE", "0").strip().lower() not in {"0", "false", "off", "disabled"}
    if legacy_flag:
        legacy_rows = copy.deepcopy(gv_records)

    promotion_info = None
    if getenv("ODDS_PROMOTION_ENABLE", "1").strip().lower() not in {"0", "false", "off", "disabled"}:
        policy = getenv("ODDS_SELECT_POLICY", "latest_by_fetch_ts")
        promotion_info = promote_week_odds(gv_records, season, week, policy=policy)
        if promotion_info["promoted_games"] > 0:
            write_jsonl(gv_records, gv_jsonl)
            write_csv(gv_records, gv_csv)
    else:
        promotion_info = None

    print(f"PASS: CFB Game View rows={len(gv_records)} ({gv_jsonl})")
    log_odds_join_audit(season, week, gv_records, base_dir)
    legacy_mismatch = 0
    if promotion_info is not None:
        if legacy_flag and legacy_rows is not None:
            diff_summary = diff_game_rows(gv_records, legacy_rows)
            print(
                "ODDS DIFF: promoted_only="
                f"{diff_summary['promoted_only']} legacy_only={diff_summary['legacy_only']} "
                f"both_equal={diff_summary['both_equal']} mismatched={diff_summary['mismatched']}"
            )
            legacy_mismatch = diff_summary['mismatched']
        log_line = (
            f"CFB ODDS PROMOTION: week={season}-{week} promoted={promotion_info['promoted_games']} "
            f"by_market={promotion_info['by_market']} by_book={promotion_info['by_book']} "
            f"legacy_mismatch={legacy_mismatch}"
        )
        if promotion_info['promoted_games'] == 0 and promotion_info.get('other_week_records', 0) > 0:
            log_line += ' hint="provider ahead; staged"'
        print(log_line)
        notes.append(
            "Odds promotion "
            f"promoted={promotion_info['promoted_games']} "
            f"current_week={promotion_info.get('current_week_records', 0)} "
            f"other_weeks={promotion_info.get('other_week_records', 0)} "
            f"mismatch={legacy_mismatch}"
        )
    else:
        print("CFB ODDS PROMOTION: disabled via ODDS_PROMOTION_ENABLE")

    team_ats = build_team_ats(season, week)
    rows_updated = apply_ats_to_week(season, week, team_ats)
    weeks_meta = getattr(build_team_ats, "meta", {})
    scanned_weeks = weeks_meta.get("weeks_scanned") if isinstance(weeks_meta, dict) else []
    zero_lined = getattr(apply_ats_to_week, "zero_lined", 0)
    teams_with_data = len(team_ats)
    weeks_label = (
        "[" + ",".join(f"W{wk}" for wk in backfill_summary.get("weeks", [])) + "]"
        if backfill_summary.get("weeks")
        else "[]"
    )
    scores_log = (
        f"Scores(CFB): weeks={weeks_label} updated={backfill_summary.get('updated', 0)} "
        f"skipped={backfill_summary.get('skipped', 0)}; ATS: teams={teams_with_data} "
        f"rows_updated={rows_updated}"
    )
    print(scores_log)
    notes.append(
        f"Scores backfill {weeks_label} "
        f"updated={backfill_summary.get('updated', 0)} skipped={backfill_summary.get('skipped', 0)} "
        f"files={backfill_summary.get('files_rewritten', 0)}"
    )
    notes.append(
        f"ATS rows={rows_updated} teams={teams_with_data} zero={zero_lined} weeks_scanned={scanned_weeks}"
    )
    notes.append(f"Game View rows={len(gv_records)}")

    # Gaps report
    write_gaps_report(season, week, base_dir, notes, critical_fields)

    summary = build_summary(season, week)
    print("\n=== SUMMARY ===")
    print(f"Games JSONL records : {summary['games_jsonl']}")
    print(f"Games CSV rows      : {summary['games_csv_rows']}")
    print(f"League metric rows  : {summary['league_rows']}")
    print(f"Odds records        : {summary['odds_records']}")
    print(f"Sagarin rows        : {summary['sagarin_rows']}")
    print(f"Sidecar files       : {summary['sidecars']}")
    print("Done.")
    promoted_total = promotion_info["promoted_games"] if promotion_info is not None else 0
    print(
        f"NOTIFY: CFB refresh complete week={season}-{week} "
        f"rows={summary['games_jsonl']} odds_promoted={promoted_total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

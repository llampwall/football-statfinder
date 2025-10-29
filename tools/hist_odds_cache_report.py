"""Report historical odds cache coverage for a given league/week."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.odds.ats_backfill_api import (
    load_pinned_event_index,
    resolve_event_id,
    _normalize_kickoff,
)
from src.odds.historical_events import get_last_snapshot
from src.odds.odds_api_client import hist_odds_cache_exists
from src.odds.participants_cache import build_provider_map


OUT_ROOT = Path(__file__).resolve().parents[1] / "out"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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


def _parse_kickoff(game: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[str]]:
    kickoff = (
        game.get("kickoff_ts")
        or game.get("kickoff_iso_utc")
        or game.get("kickoff_iso")
        or (game.get("raw_sources", {}).get("schedule_row", {}).get("commence_time"))
    )
    if not kickoff:
        return None, None
    kickoff_str = str(kickoff)
    try:
        kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        kickoff_dt = None
    return kickoff_dt, kickoff_str


def _league_paths(league: str, season: int, week: int) -> Path:
    league_lower = league.lower()
    if league_lower == "cfb":
        return OUT_ROOT / "cfb" / f"{season}_week{week}" / f"games_week_{season}_{week}.jsonl"
    if league_lower == "nfl":
        return OUT_ROOT / f"{season}_week{week}" / f"games_week_{season}_{week}.jsonl"
    raise ValueError(f"Unsupported league '{league}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="List cached historical odds for a league/week.")
    parser.add_argument("--league", required=True, choices=["cfb", "nfl"], help="League code (cfb or nfl).")
    parser.add_argument("--season", type=int, required=True, help="Season year.")
    parser.add_argument("--week", type=int, required=True, help="Week number.")
    args = parser.parse_args()

    league = args.league.lower()
    season = args.season
    week = args.week

    week_path = _league_paths(league, season, week)
    games = _read_jsonl(week_path)
    if not games:
        print(f"HIST_ODDS_CACHE: league={league} week={season}-{week} snapshot=- available=0 missing=0 unresolved=0 (no games)")
        return

    team_labels: List[str] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        team_labels.append(game.get("home_team_norm") or game.get("home_team_raw") or game.get("home_team") or "")
        team_labels.append(game.get("away_team_norm") or game.get("away_team_raw") or game.get("away_team") or "")

    build_provider_map(league, team_labels)
    pinned_index = load_pinned_event_index(league, season)

    results: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for game in games:
        if not isinstance(game, dict):
            continue
        game_key = str(game.get("game_key") or "")
        if not game_key:
            continue

        kickoff_dt, kickoff_iso = _parse_kickoff(game)
        event_id, resolver_used, resolver_reason = resolve_event_id(
            league,
            season,
            week,
            game,
            pinned_index=pinned_index,
        )

        if not event_id:
            counters["unresolved"] += 1
            results.append(
                {
                    "game_key": game_key,
                    "status": resolver_reason or "resolve_failed",
                    "event_id": None,
                    "snapshot": None,
                    "resolver": resolver_used,
                }
            )
            continue

        snapshot_dt = get_last_snapshot(league)
        if snapshot_dt is not None:
            snapshot_iso = snapshot_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            snapshot_iso = _normalize_kickoff(kickoff_dt, kickoff_iso)

        cache_hit = False
        if snapshot_iso:
            cache_hit = hist_odds_cache_exists(league, event_id, snapshot_iso)

        status = "cached" if cache_hit else "missing"
        counters[status] += 1
        results.append(
            {
                "game_key": game_key,
                "status": status,
                "event_id": event_id,
                "snapshot": snapshot_iso,
                "resolver": resolver_used,
            }
        )

    snapshot_label = "-"
    snapshots = {entry["snapshot"] for entry in results if entry.get("snapshot")}
    if len(snapshots) == 1:
        snapshot_label = snapshots.pop() or "-"
    elif len(snapshots) > 1:
        snapshot_label = "mixed"

    print(
        f"HIST_ODDS_CACHE: league={league} week={season}-{week} snapshot={snapshot_label} "
        f"available={counters.get('cached', 0)} missing={counters.get('missing', 0)} "
        f"unresolved={counters.get('unresolved', 0)}"
    )

    for entry in sorted(results, key=lambda item: item["game_key"]):
        print(
            "  {game_key} status={status} event_id={event_id} snapshot={snapshot} resolver={resolver}".format(
                game_key=entry["game_key"],
                status=entry["status"],
                event_id=entry.get("event_id") or "-",
                snapshot=entry.get("snapshot") or "-",
                resolver=entry.get("resolver") or "-",
            )
        )


if __name__ == "__main__":
    # Ensure consistent output without backfill side-effects.
    os.environ.setdefault("ATS_DEBUG", "0")
    main()

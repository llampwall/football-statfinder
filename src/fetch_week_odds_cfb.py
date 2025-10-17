"""Fetch CFB odds (The Odds API) and write NFL-compatible JSONL outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from src.common.io_utils import ensure_out_dir, read_env, write_jsonl
from src.common.team_names_cfb import normalize_team_name_cfb
from src.fetch_games_cfb import get_schedule_df

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"


def cfb_week_dir(season: int, week: int) -> Path:
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def canonical_time(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        dt = None
    if dt is None or pd.isna(dt):
        return None
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    return None


def norm_team(value: Optional[str]) -> str:
    return normalize_team_name_cfb(value).lower()


def fetch_theoddsapi_events(api_key: str) -> Optional[List[dict]]:
    params = {
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    resp = requests.get(THE_ODDS_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return None


def build_schedule_index(schedule: pd.DataFrame, week: int) -> Tuple[Dict[Tuple[str, str, Optional[str]], pd.Series], Dict[Tuple[str, str], List[pd.Series]]]:
    weekly = schedule[pd.to_numeric(schedule.get("week"), errors="coerce") == week].copy()
    exact: Dict[Tuple[str, str, Optional[str]], pd.Series] = {}
    by_teams: Dict[Tuple[str, str], List[pd.Series]] = {}
    for _, row in weekly.iterrows():
        home_norm = norm_team(row.get("home_team_norm") or row.get("home_team"))
        away_norm = norm_team(row.get("away_team_norm") or row.get("away_team"))
        kick = canonical_time(row.get("kickoff_iso_utc"))
        exact[(home_norm, away_norm, kick)] = row
        by_teams.setdefault((home_norm, away_norm), []).append(row)
    return exact, by_teams


def select_bookmaker(event: dict) -> Optional[dict]:
    bookmakers = event.get("bookmakers") or []
    if not bookmakers:
        return None
    preferred = next((b for b in bookmakers if (b.get("key") or "").lower() == "pinnacle" or (b.get("title") or "").lower() == "pinnacle"), None)
    return preferred or bookmakers[0]


def extract_market(markets: Iterable[dict], key: str) -> Optional[dict]:
    for market in markets:
        if market.get("key") == key:
            return market
    return None


def float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def int_or_none(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def build_record(row: pd.Series, event: dict, bookmaker: dict) -> Dict[str, Any]:
    home_norm = norm_team(row.get("home_team_norm") or row.get("home_team"))
    away_norm = norm_team(row.get("away_team_norm") or row.get("away_team"))

    markets = bookmaker.get("markets") or []
    spreads = extract_market(markets, "spreads") or {}
    totals = extract_market(markets, "totals") or {}
    moneyline = extract_market(markets, "h2h") or {}

    spread_home_relative = None
    favored_side = None
    spread_favored_team = None

    outcomes = spreads.get("outcomes") or []
    home_point = None
    away_point = None
    for outcome in outcomes:
        team_norm = norm_team(outcome.get("name"))
        point = float_or_none(outcome.get("point"))
        if team_norm == home_norm:
            home_point = point
        elif team_norm == away_norm:
            away_point = point
    if home_point is None and away_point is not None:
        home_point = -away_point
    spread_home_relative = home_point

    if spread_home_relative is not None:
        if spread_home_relative < 0:
            favored_side = "HOME"
            spread_favored_team = spread_home_relative
        elif spread_home_relative > 0:
            favored_side = "AWAY"
            spread_favored_team = -abs(spread_home_relative)

    total = None
    for outcome in totals.get("outcomes") or []:
        if outcome.get("name", "").lower().startswith("over"):
            total = float_or_none(outcome.get("point"))
            break

    moneyline_home = None
    moneyline_away = None
    for outcome in moneyline.get("outcomes") or []:
        team_norm = norm_team(outcome.get("name"))
        price = int_or_none(outcome.get("price"))
        if team_norm == home_norm:
            moneyline_home = price
        elif team_norm == away_norm:
            moneyline_away = price

    snapshot = bookmaker.get("last_update")
    if snapshot:
        snapshot_at = canonical_time(snapshot[:-1] if snapshot.endswith("Z") else snapshot)
        if snapshot_at:
            dt_snapshot = pd.to_datetime(snapshot_at, utc=True, errors="coerce")
            if isinstance(dt_snapshot, pd.Timestamp):
                snapshot_at = dt_snapshot.to_pydatetime().strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            snapshot_at = snapshot
    else:
        snapshot_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "source": "the-odds-api",
        "league": "CFB",
        "season": int(row.get("season")),
        "week": int(row.get("week")),
        "game_key": row.get("game_key"),
        "kickoff_iso": row.get("kickoff_iso_utc"),
        "home_team_raw": row.get("home_team"),
        "away_team_raw": row.get("away_team"),
        "home_team_norm": row.get("home_team_norm"),
        "away_team_norm": row.get("away_team_norm"),
        "rotation_home": None,
        "rotation_away": None,
        "spread_home_relative": spread_home_relative,
        "total": total,
        "moneyline_home": moneyline_home,
        "moneyline_away": moneyline_away,
        "market_scope": "full_game",
        "is_closing": False,
        "book_label": bookmaker.get("title") or bookmaker.get("key"),
        "snapshot_at": snapshot_at,
        "raw_payload": {
            "event_id": event.get("id"),
            "bookmaker_key": bookmaker.get("key"),
        },
        "favored_side": favored_side,
        "spread_favored_team": spread_favored_team,
        "rating_diff_favored_team": None,
        "rating_vs_odds": None,
    }
    return record


def match_schedule_row(
    event: dict,
    schedule_exact: Dict[Tuple[str, str, Optional[str]], pd.Series],
    schedule_by_team: Dict[Tuple[str, str], List[pd.Series]],
    used_keys: set,
) -> Optional[pd.Series]:
    home_norm = norm_team(event.get("home_team"))
    away_norm = norm_team(event.get("away_team"))
    commence = canonical_time(event.get("commence_time"))

    row = schedule_exact.get((home_norm, away_norm, commence))
    if row is not None and row.get("game_key") not in used_keys:
        return row

    candidates = schedule_by_team.get((home_norm, away_norm), [])
    for candidate in candidates:
        if candidate.get("game_key") not in used_keys:
            return candidate
    return None


def write_odds(season: int, week: int, records: List[Dict[str, Any]]) -> Path:
    out_dir = cfb_week_dir(season, week)
    odds_path = out_dir / f"odds_{season}_wk{week}.jsonl"
    write_jsonl(records, odds_path)
    return odds_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFB odds and write JSONL output.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    env = read_env(["THE_ODDS_API_KEY"])
    api_key = env.get("THE_ODDS_API_KEY")
    schedule = get_schedule_df(args.season)

    if schedule.empty:
        print("CFB odds: schedule empty, writing empty output.")
        write_odds(args.season, args.week, [])
        return 0

    schedule_exact, schedule_by_team = build_schedule_index(schedule, args.week)

    if not api_key:
        print("CFB odds: THE_ODDS_API_KEY missing, writing empty output.")
        write_odds(args.season, args.week, [])
        return 0

    try:
        events = fetch_theoddsapi_events(api_key) or []
    except Exception as exc:
        print(f"CFB odds: API error ({exc}); writing empty output.")
        write_odds(args.season, args.week, [])
        return 0

    if not events:
        print("CFB odds: API returned no events; writing empty output.")
        write_odds(args.season, args.week, [])
        return 0

    used_game_keys: set = set()
    records: List[Dict[str, Any]] = []
    unmatched = 0

    for event in events:
        row = match_schedule_row(event, schedule_exact, schedule_by_team, used_game_keys)
        if row is None:
            unmatched += 1
            continue
        bookmaker = select_bookmaker(event)
        if not bookmaker:
            unmatched += 1
            continue
        record = build_record(row, event, bookmaker)
        records.append(record)
        used_game_keys.add(row.get("game_key"))

    odds_path = write_odds(args.season, args.week, records)
    print(f"CFB odds: matched {len(records)} games; unmatched events {unmatched}; file={odds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch CFB odds (The Odds API) and emit NFL-compatible JSONL plus debug receipts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from collections import Counter

import pandas as pd
import requests

from src.common.io_utils import ensure_out_dir, read_env, write_jsonl
from src.common.team_names_cfb import normalize_team_name_cfb_odds
from src.fetch_games_cfb import get_schedule_df

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"

MIN_ABS_MATCHED = 10
MIN_MATCH_FRAC = 0.50
MAX_UNMATCH_FRAC = 0.60
KICKOFF_TOLERANCE_MINUTES = 5
NEAR_KICKOFF_TOLERANCE_MINUTES = 120
MAX_DEBUG_SAMPLES = 30
ALIAS_DEBT_LIMIT = 50


def cfb_week_dir(season: int, week: int) -> Path:
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
        return None
    return dt.to_pydatetime().astimezone(timezone.utc)


def canonical_time(value: Optional[str]) -> Optional[str]:
    dt = parse_utc(value)
    return dt.strftime("%Y-%m-%dT%H:%M") if isinstance(dt, datetime) else None


def norm_team(value: Optional[str]) -> str:
    return normalize_team_name_cfb_odds(value or "").lower()


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


def build_schedule_index(
    schedule: pd.DataFrame, week: int
) -> Tuple[
    pd.DataFrame,
    Dict[Tuple[str, str, Optional[str]], pd.Series],
    Dict[Tuple[str, str], List[pd.Series]],
    Dict[frozenset, List[pd.Series]],
    Optional[datetime],
    Optional[datetime],
]:
    weekly = schedule[pd.to_numeric(schedule.get("week"), errors="coerce") == week].copy()
    if weekly.empty:
        return weekly, {}, {}, {}, None, None

    def kickoff_dt(series: pd.Series) -> Optional[datetime]:
        value = series.get("kickoff_dt_utc")
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        iso = series.get("kickoff_iso_utc")
        return parse_utc(iso)

    weekly["__kickoff_dt"] = weekly.apply(kickoff_dt, axis=1)
    weekly["__kickoff_date"] = weekly["__kickoff_dt"].apply(
        lambda dt: dt.date() if isinstance(dt, datetime) else None
    )
    weekly["__kickoff_token"] = weekly["kickoff_iso_utc"].apply(canonical_time)
    weekly["__home_tokens"] = weekly.apply(
        lambda r: sorted({
            norm_team(r.get("home_team_norm")),
            norm_team(r.get("home_team")),
        } - {""}),
        axis=1,
    )
    weekly["__away_tokens"] = weekly.apply(
        lambda r: sorted({
            norm_team(r.get("away_team_norm")),
            norm_team(r.get("away_team")),
        } - {""}),
        axis=1,
    )

    exact: Dict[Tuple[str, str, Optional[str]], pd.Series] = {}
    ordered: Dict[Tuple[str, str], List[pd.Series]] = {}
    unordered: Dict[frozenset, List[pd.Series]] = {}

    for _, row in weekly.iterrows():
        kickoff_token = row.get("__kickoff_token")
        for home in row["__home_tokens"]:
            for away in row["__away_tokens"]:
                if not home or not away:
                    continue
                exact.setdefault((home, away, kickoff_token), row)
                ordered.setdefault((home, away), []).append(row)
                unordered.setdefault(frozenset({home, away}), []).append(row)

    min_dt = weekly["__kickoff_dt"].dropna().min()
    max_dt = weekly["__kickoff_dt"].dropna().max()
    return weekly, exact, ordered, unordered, min_dt, max_dt


def select_bookmaker(event: dict) -> Optional[dict]:
    bookmakers = event.get("bookmakers") or []
    if not bookmakers:
        return None
    preferred = next(
        (
            b
            for b in bookmakers
            if (b.get("key") or "").lower() == "pinnacle"
            or (b.get("title") or "").lower() == "pinnacle"
        ),
        None,
    )
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
        if team_norm in row["__home_tokens"]:
            home_point = point
        elif team_norm in row["__away_tokens"]:
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
        else:
            favored_side = "PICK"
            spread_favored_team = 0.0

    total = None
    total_outcomes = totals.get("outcomes") or []
    for outcome in total_outcomes:
        if outcome.get("name", "").lower().startswith("over"):
            total = float_or_none(outcome.get("point"))
            break
    if total is None and total_outcomes:
        total = float_or_none(total_outcomes[0].get("point"))

    moneyline_home = None
    moneyline_away = None
    for outcome in moneyline.get("outcomes") or []:
        team_norm = norm_team(outcome.get("name"))
        price = int_or_none(outcome.get("price"))
        if team_norm in row["__home_tokens"]:
            moneyline_home = price
        elif team_norm in row["__away_tokens"]:
            moneyline_away = price

    snapshot = parse_utc(bookmaker.get("last_update"))
    if isinstance(snapshot, datetime):
        snapshot_str = snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        snapshot_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
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
        "odds_source": "the-odds-api",
        "is_closing": False,
        "book_label": bookmaker.get("title") or bookmaker.get("key") or "Composite",
        "snapshot_at": snapshot_str,
        "raw_payload": {
            "event_id": event.get("id"),
            "bookmaker_key": bookmaker.get("key"),
        },
        "favored_side": favored_side,
        "spread_favored_team": spread_favored_team,
        "rating_diff_favored_team": None,
        "rating_vs_odds": None,
    }


def _select_candidate(
    candidates: Iterable[pd.Series],
    commence_dt: Optional[datetime],
    used_keys: set,
    tolerance_seconds: int,
    require_same_date: bool,
) -> Optional[pd.Series]:
    if not isinstance(commence_dt, datetime):
        return None
    best_row: Optional[pd.Series] = None
    best_diff: Optional[float] = None
    for candidate in candidates or []:
        if not isinstance(candidate, pd.Series):
            continue
        game_key = candidate.get("game_key")
        if not game_key or game_key in used_keys:
            continue
        kickoff_dt = candidate.get("__kickoff_dt")
        if not isinstance(kickoff_dt, datetime):
            continue
        diff_seconds = abs((kickoff_dt - commence_dt).total_seconds())
        if diff_seconds > tolerance_seconds:
            continue
        if require_same_date:
            kickoff_date = candidate.get("__kickoff_date")
            if kickoff_date is None or kickoff_date != commence_dt.date():
                continue
        if best_diff is None or diff_seconds < best_diff:
            best_diff = diff_seconds
            best_row = candidate
    return best_row


def match_schedule_row(
    event: dict,
    schedule_by_team: Dict[Tuple[str, str], List[pd.Series]],
    schedule_unordered: Dict[frozenset, List[pd.Series]],
    used_keys: set,
) -> Tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    home_norm = event.get("_home_norm") or norm_team(event.get("home_team"))
    away_norm = event.get("_away_norm") or norm_team(event.get("away_team"))
    commence_dt = event.get("_commence_dt")

    if not home_norm or not away_norm:
        return None, None, "name_mismatch"
    if not isinstance(commence_dt, datetime):
        return None, None, "invalid_commence_time"

    ordered_key = (home_norm, away_norm)
    unordered_key = frozenset({home_norm, away_norm})
    exact_tolerance = KICKOFF_TOLERANCE_MINUTES * 60
    near_tolerance = NEAR_KICKOFF_TOLERANCE_MINUTES * 60

    candidate = _select_candidate(
        schedule_by_team.get(ordered_key, []),
        commence_dt,
        used_keys,
        exact_tolerance,
        require_same_date=False,
    )
    if candidate is not None:
        return candidate, "ordered_exact", None

    candidate = _select_candidate(
        schedule_unordered.get(unordered_key, []),
        commence_dt,
        used_keys,
        exact_tolerance,
        require_same_date=False,
    )
    if candidate is not None:
        return candidate, "unordered_exact", None

    candidate = _select_candidate(
        schedule_by_team.get(ordered_key, []),
        commence_dt,
        used_keys,
        near_tolerance,
        require_same_date=True,
    )
    if candidate is not None:
        return candidate, "ordered_near", None

    candidate = _select_candidate(
        schedule_unordered.get(unordered_key, []),
        commence_dt,
        used_keys,
        near_tolerance,
        require_same_date=True,
    )
    if candidate is not None:
        return candidate, "unordered_near", None

    return None, None, "no_schedule_candidate"


def write_odds(season: int, week: int, records: List[Dict[str, Any]]) -> Path:
    out_dir = cfb_week_dir(season, week)
    odds_path = out_dir / f"odds_{season}_wk{week}.jsonl"
    write_jsonl(records, odds_path)
    return odds_path


def write_debug(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_alias_debt(
    path: Path,
    season: int,
    week: int,
    unmatched_samples: List[Dict[str, Any]],
) -> None:
    debt_counter: Counter[str] = Counter()
    for sample in unmatched_samples:
        reason = sample.get("why")
        if reason not in {"no_schedule_candidate", "name_mismatch"}:
            continue
        for key in ("home_norm", "away_norm"):
            token = str(sample.get(key) or "").strip()
            if token:
                debt_counter[token] += 1
    top_tokens = [
        {"token": token, "count": count}
        for token, count in debt_counter.most_common(ALIAS_DEBT_LIMIT)
    ]
    payload = {
        "season": season,
        "week": week,
        "top_unmatched_tokens": top_tokens,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFB odds and write JSONL output.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    schedule = get_schedule_df(args.season)
    weekly, _schedule_exact, schedule_by_team, schedule_unordered, min_dt, max_dt = build_schedule_index(schedule, args.week)

    out_dir = cfb_week_dir(args.season, args.week)
    debug_path = out_dir / "odds_match_debug.json"
    alias_debt_path = out_dir / "odds_alias_debt.json"

    if weekly.empty:
        write_odds(args.season, args.week, [])
        write_debug(
            debug_path,
            {
                "season": args.season,
                "week": args.week,
                "stats": {
                    "events_total": 0,
                    "events_after_window_filter": 0,
                    "matched": 0,
                    "unmatched": 0,
                },
                "samples": {"matched": [], "unmatched": []},
                "window": None,
            },
        )
        write_alias_debt(alias_debt_path, args.season, args.week, [])
        print("CFB odds: schedule empty, writing empty output.")
        print(f"Debug: {debug_path}")
        print(f"Alias debt: {alias_debt_path}")
        return 0

    window_start = min_dt - timedelta(days=1) if isinstance(min_dt, datetime) else None
    window_end = max_dt + timedelta(days=1) if isinstance(max_dt, datetime) else None
    if window_start and window_end:
        window_label = f"{window_start.strftime('%Y-%m-%dT%H:%M:%SZ')} -> {window_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    else:
        window_label = "unknown"

    env = read_env(["THE_ODDS_API_KEY"])
    api_key = env.get("THE_ODDS_API_KEY")

    if not api_key:
        write_odds(args.season, args.week, [])
        write_debug(
            debug_path,
            {
                "season": args.season,
                "week": args.week,
                "stats": {
                    "events_total": 0,
                    "events_after_window_filter": 0,
                    "matched": 0,
                    "unmatched": 0,
                },
                "samples": {"matched": [], "unmatched": []},
                "window": window_label,
            },
        )
        write_alias_debt(alias_debt_path, args.season, args.week, [])
        print("CFB odds: THE_ODDS_API_KEY missing, writing empty output.")
        print(f"Debug: {debug_path}")
        print(f"Alias debt: {alias_debt_path}")
        return 0

    try:
        events = fetch_theoddsapi_events(api_key) or []
    except Exception as exc:
        write_odds(args.season, args.week, [])
        write_debug(
            debug_path,
            {
                "season": args.season,
                "week": args.week,
                "stats": {
                    "events_total": 0,
                    "events_after_window_filter": 0,
                    "matched": 0,
                    "unmatched": 0,
                },
                "samples": {"matched": [], "unmatched": [{"event_home": None, "event_away": None, "event_time": None, "why": f"api_error:{exc}"}]},
                "window": window_label,
            },
        )
        write_alias_debt(alias_debt_path, args.season, args.week, [])
        print(f"CFB odds: API error ({exc}); writing empty output.")
        print(f"Debug: {debug_path}")
        print(f"Alias debt: {alias_debt_path}")
        return 0

    total_events = len(events)
    filtered_events: List[dict] = []
    unmatched_samples: List[Dict[str, Any]] = []

    for event in events:
        commence_dt = parse_utc(event.get("commence_time"))
        home_norm = norm_team(event.get("home_team"))
        away_norm = norm_team(event.get("away_team"))
        event["_commence_dt"] = commence_dt
        event["_home_norm"] = home_norm
        event["_away_norm"] = away_norm
        if not isinstance(commence_dt, datetime):
            unmatched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "why": "invalid_commence_time",
                }
            )
            continue
        if window_start and commence_dt < window_start:
            unmatched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "why": "outside_window",
                }
            )
            continue
        if window_end and commence_dt > window_end:
            unmatched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "why": "outside_window",
                }
            )
            continue
        filtered_events.append(event)

    used_game_keys: set = set()
    records: List[Dict[str, Any]] = []
    matched_samples: List[Dict[str, Any]] = []
    unmatched_post_filter = 0

    for event in filtered_events:
        row, match_mode, failure_reason = match_schedule_row(
            event,
            schedule_by_team,
            schedule_unordered,
            used_game_keys,
        )
        home_norm = event.get("_home_norm") or norm_team(event.get("home_team"))
        away_norm = event.get("_away_norm") or norm_team(event.get("away_team"))
        if row is None:
            unmatched_post_filter += 1
            unmatched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "why": failure_reason or "no_schedule_candidate",
                }
            )
            continue

        bookmaker = select_bookmaker(event)
        if not bookmaker:
            unmatched_post_filter += 1
            unmatched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "why": "no_bookmaker",
                }
            )
            continue

        record = build_record(row, event, bookmaker)
        records.append(record)
        used_game_keys.add(row.get("game_key"))
        if len(matched_samples) < MAX_DEBUG_SAMPLES:
            matched_samples.append(
                {
                    "event_home": event.get("home_team"),
                    "event_away": event.get("away_team"),
                    "event_time": event.get("commence_time"),
                    "home_norm": home_norm,
                    "away_norm": away_norm,
                    "game_key": row.get("game_key"),
                    "kickoff_iso_utc": row.get("kickoff_iso_utc"),
                    "mode": match_mode or "unknown",
                }
            )

    odds_path = write_odds(args.season, args.week, records)

    debug_data = {
        "season": args.season,
        "week": args.week,
        "window": window_label,
        "stats": {
            "events_total": total_events,
            "events_after_window_filter": len(filtered_events),
            "matched": len(records),
            "unmatched": unmatched_post_filter,
        },
        "samples": {
            "matched": matched_samples,
            "unmatched": unmatched_samples[:MAX_DEBUG_SAMPLES],
        },
    }
    write_debug(debug_path, debug_data)
    write_alias_debt(alias_debt_path, args.season, args.week, unmatched_samples)

    has_output_lines = False
    if odds_path.exists():
        with odds_path.open("r", encoding="utf-8") as handle:
            for _ in handle:
                has_output_lines = True
                break
    if not has_output_lines:
        print(f"FAIL: CFB odds output missing or empty at {odds_path}")
        print(f"See {debug_path}")
        print(f"Alias debt: {alias_debt_path}")
        return 1

    events_after_filter = len(filtered_events)
    if events_after_filter <= 0:
        required_matched = 0
        max_unmatched_allowed = 0
    else:
        required_matched = max(MIN_ABS_MATCHED, math.ceil(MIN_MATCH_FRAC * events_after_filter))
        max_unmatched_allowed = math.ceil(MAX_UNMATCH_FRAC * events_after_filter)

    matched_count = len(records)
    unmatched_count = unmatched_post_filter

    print(
        f"Window: {window_label} | events {len(filtered_events)} / matched {matched_count} / unmatched {unmatched_count}"
    )

    if matched_count < required_matched or unmatched_count > max_unmatched_allowed:
        print(
            f"FAIL: CFB odds coverage below threshold (matched {matched_count} < {required_matched} or unmatched {unmatched_count} > {max_unmatched_allowed}). See {debug_path}"
        )
        print(f"Alias debt: {alias_debt_path}")
        return 1

    print(f"PASS: CFB odds matched {matched_count}; unmatched {unmatched_count}; wrote {odds_path}")
    print(f"Debug: {debug_path}")
    print(f"Alias debt: {alias_debt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch weekly NFL odds from external feeds and emit JSONL files.

Purpose:
    Normalize The Odds API or Don Best responses into the house odds schema.
Inputs:
    Source provider (--source), season/week, optional bookmaker scope.
Outputs:
    /out/odds_{season}_wk{week}.jsonl
Source(s) of truth:
    the-odds-api.com v4 endpoints, Don Best XML v2 feed.
Example:
    python -m src.fetch_week_odds_nfl --source theoddsapi --season 2025 --week 6 --book Pinnacle
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from src.common.io_utils import ensure_out_dir, read_env, write_jsonl
from src.common.team_names import normalize_team_display

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DONBEST_BASE = "https://xml.donbest.com/v2"
SPORT_KEY_ODDSAPI = "americanfootball_nfl"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def to_float_or_none(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def first_non_none(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None

# ---------- HTTP helpers ----------

def _http_get_json(url: str, params: Dict[str, str]) -> Any:
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"
    req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0 (odds-pack-cli)"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (odds-pack-cli)"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read()

# ---------- SOURCE: The Odds API ----------

def fetch_odds_theoddsapi(season: int, week: int, api_key: str, book_pref: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Uses The Odds API v4 'odds' endpoint for NFL:
      /v4/sports/americanfootball_nfl/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey=...
    Docs: https://the-odds-api.com/liveapi/guides/v4/ (sport key, params, oddsFormat)
    """
    # Note: This endpoint returns **live and upcoming games** with bookmaker arrays.
    # It does not filter by NFL "week" on the server; we include the target week in the emitted JSON.
    url = f"{THE_ODDS_API_BASE}/sports/{SPORT_KEY_ODDSAPI}/odds"
    params = {
        "regions": "us",               # US books
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    data = _http_get_json(url, params)

    out: List[Dict[str, Any]] = []
    snap = now_utc_iso()

    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        kickoff_iso = ev.get("commence_time")  # already ISO8601 in UTC per docs
        bookmakers = ev.get("bookmakers", []) or []

        # pick a bookmaker: prefer title match; else first
        chosen = None
        if book_pref:
            for bm in bookmakers:
                title = bm.get("title") or bm.get("key")
                if title and book_pref.lower() in title.lower():
                    chosen = bm
                    break
        if not chosen and bookmakers:
            chosen = bookmakers[0]

        ml_home = ml_away = None
        total = None
        spread_hr = None

        if chosen:
            for m in chosen.get("markets", []):
                key = m.get("key")
                outcomes = m.get("outcomes") or []
                if key == "h2h":
                    for o in outcomes:
                        if o.get("name") == home:
                            ml_home = to_int_or_none(o.get("price"))
                        if o.get("name") == away:
                            ml_away = to_int_or_none(o.get("price"))
                elif key == "spreads":
                    # Each side has a "point" (home positive means home +X). Our schema wants **home-relative spread**.
                    for o in outcomes:
                        if o.get("name") == home:
                            p = o.get("point")
                            spread_hr = to_float_or_none(p) if p is not None else spread_hr
                elif key == "totals":
                    # outcomes: Over/Under with the same "point"; read once
                    if outcomes:
                        total = to_float_or_none(outcomes[0].get("point"))

        record = {
            "source": "the-odds-api",
            "league": "NFL",
            "season": season,
            "week": week,
            "kickoff_iso": kickoff_iso,
            "home_team_raw": home,
            "away_team_raw": away,
            "home_team_norm": normalize_team_display(home or ""),
            "away_team_norm": normalize_team_display(away or ""),
            "rotation_home": None,
            "rotation_away": None,
            "spread_home_relative": spread_hr,
            "total": total,
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "market_scope": "full_game",
            "is_closing": False,
            "book_label": (chosen.get("title") if chosen else "Composite"),
            "snapshot_at": snap,
            "raw_payload": {},  # keep empty/minimal for audit
        }

        # Skip if we somehow lack teams
        if record["home_team_raw"] and record["away_team_raw"]:
            out.append(record)

    return out

# ---------- SOURCE: Don Best XML v2 ----------

def fetch_odds_donbest(season: int, week: int, token: str, scope: str = "odds") -> List[Dict[str, Any]]:
    """
    Don Best XML (v2) for NFL:
      Current lines: /v2/odds/NFL/?token=TOKEN
      Opening lines: /v2/open/NFL/?token=TOKEN
      Closing lines: /v2/close/NFL/?token=TOKEN
    Notes: REST feed is for *current contests*; not historical.
    """
    scope = scope.lower().strip()
    if scope not in ("odds", "open", "close"):
        scope = "odds"
    url = f"{DONBEST_BASE}/{scope}/NFL/?token={urllib.parse.quote(token)}"
    data = _http_get_bytes(url)
    root = ET.fromstring(data)

    out: List[Dict[str, Any]] = []
    snap = now_utc_iso()

    # The exact XML shape can vary by account; we map conservatively.
    # Common fields seen in docs/examples: <event>, <home_team>, <away_team>, <rot_home>, <rot_away>,
    # period/line blocks with spread/total/moneyline nodes.
    for ev in root.findall(".//event"):
        home = ev.findtext("home_team")
        away = ev.findtext("away_team")
        rot_home = to_int_or_none(ev.findtext("rot_home") or ev.findtext(".//rotation/home"))
        rot_away = to_int_or_none(ev.findtext("rot_away") or ev.findtext(".//rotation/away"))

        # kickoff: some feeds provide UTC explicitly; fall back to local if needed
        kickoff_iso = first_non_none(
            ev.findtext("start_time_utc"),
            ev.findtext("start_time"),
            ev.get("start_time_utc"),
            ev.get("start_time"),
        )

        # try to read lines (home/away numeric points)
        spread_hr = None
        total = None
        ml_home = None
        ml_away = None

        # Spread: if XML gives separate <spread><home> -3.5 </home> and <away> +3.5 </away>
        sp_home = ev.findtext(".//spread/home")
        sp_away = ev.findtext(".//spread/away")
        if sp_home is not None:
            spread_hr = to_float_or_none(sp_home)
        elif sp_away is not None:
            # away +X means home-relative is -X
            v = to_float_or_none(sp_away)
            spread_hr = (-v) if v is not None else None

        # Total: a single <total><points>41.5</points>
        total = to_float_or_none(ev.findtext(".//total/points"))

        # Moneylines: <moneyline><home>-145</home><away>+125</away>
        ml_home = to_int_or_none(ev.findtext(".//moneyline/home"))
        ml_away = to_int_or_none(ev.findtext(".//moneyline/away"))

        record = {
            "source": "donbest",
            "league": "NFL",
            "season": season,
            "week": week,
            "kickoff_iso": kickoff_iso,
            "home_team_raw": home,
            "away_team_raw": away,
            "home_team_norm": normalize_team_display(home or ""),
            "away_team_norm": normalize_team_display(away or ""),
            "rotation_home": rot_home,
            "rotation_away": rot_away,
            "spread_home_relative": spread_hr,
            "total": total,
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "market_scope": "full_game",
            "is_closing": (scope == "close"),
            "book_label": "Don Best (composite)",
            "snapshot_at": snap,
            "raw_payload": {},
        }
        if record["home_team_raw"] and record["away_team_raw"]:
            out.append(record)

    return out

# ---------- CLI ----------


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch weekly NFL odds into JSONL.")
    parser.add_argument("--source", choices=["theoddsapi", "donbest"], required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--book", type=str, default=None, help="Preferred bookmaker title for The Odds API.")
    parser.add_argument("--scope", type=str, default="odds", help="Don Best scope: odds|open|close.")
    parser.add_argument("--key", type=str, help="Override API key/token.")
    parser.add_argument("--out", type=str, help="Optional output path; defaults to /out/odds_{season}_wk{week}.jsonl")
    args = parser.parse_args()

    env = read_env(["THE_ODDS_API_KEY", "DONBEST_TOKEN"])
    if args.source == "theoddsapi":
        key = args.key or env.get("THE_ODDS_API_KEY")
        if not key:
            raise SystemExit("Missing The Odds API key. Provide --key or set THE_ODDS_API_KEY.")
        records = fetch_odds_theoddsapi(args.season, args.week, key, book_pref=args.book)
    else:
        key = args.key or env.get("DONBEST_TOKEN")
        if not key:
            raise SystemExit("Missing Don Best token. Provide --key or set DONBEST_TOKEN.")
        records = fetch_odds_donbest(args.season, args.week, key, scope=args.scope)

    out_path = Path(args.out) if args.out else ensure_out_dir() / f"odds_{args.season}_wk{args.week}.jsonl"
    write_jsonl(records, out_path)

    total = len(records)
    have_spread = sum(1 for r in records if r.get("spread_home_relative") is not None)
    have_total = sum(1 for r in records if r.get("total") is not None)
    have_ml = sum(
        1
        for r in records
        if r.get("moneyline_home") is not None or r.get("moneyline_away") is not None
    )

    print(f"Wrote: {out_path}")
    print(f"Games scraped: {total}")
    print(f"Fields present -> spread:{have_spread} total:{have_total} moneyline:{have_ml}")
    if records:
        sample = json.dumps(records[:2], indent=2, ensure_ascii=False)
        print("Sample:", sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

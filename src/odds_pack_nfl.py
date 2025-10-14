#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot NFL full-game odds pack (standalone; no DB).

Sources:
  - theoddsapi  (JSON, free-tier friendly)  -> requires THE_ODDS_API_KEY or --key
  - donbest     (XML v2 current/open/close) -> requires DONBEST_TOKEN   or --key

Output: one JSON object per game with home-relative spread, total, MLs, snapshot time.

Usage examples:
  # The Odds API (recommended today)
  python odds_pack_nfl.py --source theoddsapi --season 2025 --week 6 --out odds_2025_wk6.jsonl --book Pinnacle

  # Don Best XML (when you have a token)
  python odds_pack_nfl.py --source donbest --season 2025 --week 6 --scope odds --key <DONBEST_TOKEN>
"""

from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# 100% stdlib (requests is nice but not required); using urllib for portability
import urllib.request
import urllib.parse
import ssl
import xml.etree.ElementTree as ET

# ---------- constants ----------
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DONBEST_BASE = "https://xml.donbest.com/v2"

SPORT_KEY_ODDSAPI = "americanfootball_nfl"  # The Odds API sport key

# ---------- small helpers (pure) ----------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def to_int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None

def to_float_or_none(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def normalize_team_name(s: str) -> str:
    # conservative pass-through with a couple light normalizations
    if not s:
        return s
    s2 = " ".join(s.replace(".", "").replace(",", "").split())
    # a tiny synonyms map can live here if you want, but keep conservative
    syn = {
        "la rams": "Los Angeles Rams",
        "la chargers": "Los Angeles Chargers",
        "ny jets": "New York Jets",
        "ny giants": "New York Giants",
        "sf 49ers": "San Francisco 49ers",
    }
    return syn.get(s2.lower(), s.strip())

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
            "home_team_norm": normalize_team_name(home or ""),
            "away_team_norm": normalize_team_name(away or ""),
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
            "home_team_norm": normalize_team_name(home or ""),
            "away_team_norm": normalize_team_name(away or ""),
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

def main():
    ap = argparse.ArgumentParser(description="NFL Game Odds Pack (The Odds API or Don Best XML)")
    ap.add_argument("--source", choices=["theoddsapi", "donbest"], required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="Write JSONL to this path")
    ap.add_argument("--key", type=str, help="API key (The Odds API) or token (Don Best). If omitted, env is used.")
    ap.add_argument("--book", type=str, default=None, help="(theoddsapi) prefer a bookmaker title (e.g., 'Pinnacle')")
    ap.add_argument("--scope", type=str, default="odds", help="(donbest) odds|open|close")
    args = ap.parse_args()

    try:
        if args.source == "theoddsapi":
            key = args.key or os.getenv("THE_ODDS_API_KEY")
            if not key:
                raise RuntimeError("Missing The Odds API key. Set THE_ODDS_API_KEY or pass --key.")
            recs = fetch_odds_theoddsapi(args.season, args.week, key, book_pref=args.book)
        else:
            key = args.key or os.getenv("DONBEST_TOKEN")
            if not key:
                raise RuntimeError("Missing Don Best token. Set DONBEST_TOKEN or pass --key.")
            recs = fetch_odds_donbest(args.season, args.week, key, scope=args.scope)

        # write
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # summary
        total = len(recs)
        have_spread = sum(1 for r in recs if r.get("spread_home_relative") is not None)
        have_total  = sum(1 for r in recs if r.get("total") is not None)
        have_ml     = sum(1 for r in recs if (r.get("moneyline_home") is not None or r.get("moneyline_away") is not None))
        print(f"Games scraped: {total}")
        print(f"Fields present -> spread:{have_spread} total:{have_total} moneyline:{have_ml}")
        if recs[:2]:
            print("\nSample:", json.dumps(recs[:2], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

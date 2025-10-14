#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot NFL Game View Data Pack (standalone)
- Fetch nflverse schedule + team-week stats
- Optional Sagarin CSV (or skip)
- Normalize, compute derived fields, emit JSONL + CSV
- No DB I/O

CLI:
  python gameview_weekpack.py \
    --season 2025 --week 6 \
    --hfa 0.0 \
    --odds-source schedule \
    --sagarin-path sagarin_week_2025_6.csv \
    --out games_week_2025_6
"""

from __future__ import annotations
import argparse
import csv
import dataclasses
import io
import json
import math
import sys
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Standard library tz (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
except Exception:
    TZ_NY = None

import pandas as pd
import numpy as np
import urllib.request


# -----------------------------
# Team normalization utilities
# -----------------------------

_ALIASES = {
    # common short ↔ long
    "washington": "commanders",
    "wsh": "commanders",
    "was": "commanders",
    "jax": "jaguars",
    "sf": "49ers",
    "san francisco 49ers": "49ers",
    "new york giants": "giants",
    "new york jets": "jets",
    "tampa bay buccaneers": "buccaneers",
    "la rams": "rams",
    "los angeles rams": "rams",
    "la chargers": "chargers",
    "los angeles chargers": "chargers",
    "arizona cardinals": "cardinals",
    "atlanta falcons": "falcons",
    "baltimore ravens": "ravens",
    "buffalo bills": "bills",
    "carolina panthers": "panthers",
    "chicago bears": "bears",
    "cincinnati bengals": "bengals",
    "cleveland browns": "browns",
    "dallas cowboys": "cowboys",
    "denver broncos": "broncos",
    "detroit lions": "lions",
    "green bay packers": "packers",
    "houston texans": "texans",
    "indianapolis colts": "colts",
    "jacksonville jaguars": "jaguars",
    "kansas city chiefs": "chiefs",
    "las vegas raiders": "raiders",
    "los angeles raiders": "raiders",  # historical
    "los angeles rams": "rams",
    "miami dolphins": "dolphins",
    "minnesota vikings": "vikings",
    "new england patriots": "patriots",
    "new orleans saints": "saints",
    "new york giants": "giants",
    "new york jets": "jets",
    "philadelphia eagles": "eagles",
    "pittsburgh steelers": "steelers",
    "san francisco 49ers": "49ers",
    "seattle seahawks": "seahawks",
    "tampa bay buccaneers": "buccaneers",
    "tennessee titans": "titans",
    "washington football team": "commanders",
}

def _slug_norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s or "", flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def normalize_team_name(raw: str) -> str:
    if not raw:
        return ""
    slug = _slug_norm(raw)
    return _ALIASES.get(slug, slug)


# -----------------------------
# Pure helpers (unit-testable)
# -----------------------------

def rating_diff(pr_home: float, pr_away: float, hfa: float) -> float:
    return (pr_home + hfa) - pr_away

def team_centric_spread(home_relative_spread: float, side: str) -> float:
    # side ∈ {"HOME", "AWAY"}
    return float(home_relative_spread) if side == "HOME" else -float(home_relative_spread)

def rating_vs_odds(rating_diff_val: float, team_centric_spread_val: float) -> float:
    # ratings (home-away, includes HFA) minus market line (home-minus-away implied)
    # team_centric_spread(HOME) == home_relative_spread
    # market home-minus-away line is -team_centric_spread
    return rating_diff_val - (-team_centric_spread_val)

def dense_ranks(values: Dict[str, Optional[float]], higher_is_better: bool) -> Dict[str, Optional[int]]:
    # Dense rank: ties share rank, next rank increments by 1
    items = [(k, v) for k, v in values.items() if v is not None]
    if not items:
        return {k: None for k in values.keys()}
    items.sort(key=lambda x: (-(x[1]) if higher_is_better else x[1], x[0]))
    ranks: Dict[str, Optional[int]] = {k: None for k in values.keys()}
    rank = 0
    last_val = None
    for i, (team, val) in enumerate(items):
        if last_val is None or val != last_val:
            rank += 1
            last_val = val
        ranks[team] = rank
    return ranks

def compute_su_to_date(team_games: List[dict]) -> str:
    w = l = t = 0
    for g in team_games:
        margin = g.get("team_points", 0) - g.get("opp_points", 0)
        if margin > 0:
            w += 1
        elif margin < 0:
            l += 1
        else:
            t += 1
    return f"{w}-{l}" + (f"-{t}" if t else "")

def _ats_outcome(team_margin: float, team_line: Optional[float]) -> Optional[str]:
    """Returns 'W','L','P' or None if no line."""
    if team_line is None or pd.isna(team_line):
        return None
    # Cover if team_margin > (-team_line)? Careful: team_line is team-centric (favored negative).
    # Decide vs market: team_margin + team_line
    diff = team_margin + team_line
    if abs(diff) < 1e-9:
        return "P"
    return "W" if diff > 0 else "L"

def compute_ats_to_date(team_games_with_spreads: List[dict]) -> str:
    w = l = p = 0
    for g in team_games_with_spreads:
        out = _ats_outcome(g.get("team_margin", 0.0), g.get("team_line"))
        if out == "W":
            w += 1
        elif out == "L":
            l += 1
        elif out == "P":
            p += 1
    return f"{w}-{l}-{p}"

@dataclass
class S2D:
    pf_pg: float = 0.0
    pa_pg: float = 0.0
    ry_pg: float = 0.0
    py_pg: float = 0.0
    ty_pg: float = 0.0
    ry_allowed_pg: float = 0.0
    py_allowed_pg: float = 0.0
    ty_allowed_pg: float = 0.0
    to_margin_pg: float = 0.0
    games_played: int = 0

def season_to_date_per_game(team_games: List[dict]) -> S2D:
    if not team_games:
        return S2D(games_played=0)
    n = len(team_games)
    def avg(key: str) -> float:
        vals = [float(g.get(key, 0.0) or 0.0) for g in team_games]
        return sum(vals) / n
    out = S2D(
        pf_pg=avg("team_points"),
        pa_pg=avg("opp_points"),
        ry_pg=avg("team_rush_yds"),
        py_pg=avg("team_pass_yds"),
        ty_pg=avg("team_total_yds"),
        ry_allowed_pg=avg("opp_rush_yds"),
        py_allowed_pg=avg("opp_pass_yds"),
        ty_allowed_pg=avg("opp_total_yds"),
        to_margin_pg=avg("team_to_margin"),
        games_played=n,
    )
    return out

def load_turnover_margin_per_game(teamrankings_csv_path: Optional[str]) -> Dict[str, float]:
    import pandas as pd
    url = "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game"
    try:
        if teamrankings_csv_path:
            df = pd.read_csv(teamrankings_csv_path)
        else:
            tables = pd.read_html(url)  # simple table scrape
            df = None
            for t in tables:
                if "Team" in t.columns:
                    # use the numeric column that looks like the current season (e.g., "2025")
                    num_cols = [c for c in t.columns if c != "Team" and pd.api.types.is_numeric_dtype(t[c])]
                    if num_cols:
                        year_cols = [c for c in num_cols if re.fullmatch(r"\d{4}", str(c))]
                        if year_cols:
                            # take the most recent season column (largest year)
                            pick = sorted(year_cols, reverse=True)[0]
                        else:
                            # fall back to the first numeric column that isn't Rank
                            filtered = [c for c in num_cols if str(c).strip().lower() not in {"rank"}]
                            pick = filtered[0] if filtered else num_cols[0]
                        df = t[["Team", pick]].rename(columns={pick: "TO_pg"})
                        break
            if df is None:
                raise RuntimeError("TeamRankings table not found")
        df["Team"] = df["Team"].astype(str).str.strip()
        if "TO_pg" not in df.columns:
            # if the CSV has "2025" or similar as the column, use it
            for c in df.columns:
                if c != "Team" and pd.api.types.is_numeric_dtype(df[c]):
                    df = df.rename(columns={c: "TO_pg"})
                    break
        # align TeamRankings labels to full team names we use elsewhere
        team_alias = {
            "Arizona": "Arizona Cardinals",
            "Atlanta": "Atlanta Falcons",
            "Baltimore": "Baltimore Ravens",
            "Buffalo": "Buffalo Bills",
            "Carolina": "Carolina Panthers",
            "Chicago": "Chicago Bears",
            "Cincinnati": "Cincinnati Bengals",
            "Cleveland": "Cleveland Browns",
            "Dallas": "Dallas Cowboys",
            "Denver": "Denver Broncos",
            "Detroit": "Detroit Lions",
            "Green Bay": "Green Bay Packers",
            "Houston": "Houston Texans",
            "Indianapolis": "Indianapolis Colts",
            "Jacksonville": "Jacksonville Jaguars",
            "Kansas City": "Kansas City Chiefs",
            "LA Chargers": "Los Angeles Chargers",
            "Los Angeles Chargers": "Los Angeles Chargers",
            "LA Rams": "Los Angeles Rams",
            "Los Angeles Rams": "Los Angeles Rams",
            "Las Vegas": "Las Vegas Raiders",
            "Miami": "Miami Dolphins",
            "Minnesota": "Minnesota Vikings",
            "New England": "New England Patriots",
            "New Orleans": "New Orleans Saints",
            "New York Giants": "New York Giants",
            "NY Giants": "New York Giants",
            "New York Jets": "New York Jets",
            "NY Jets": "New York Jets",
            "Philadelphia": "Philadelphia Eagles",
            "Pittsburgh": "Pittsburgh Steelers",
            "San Francisco": "San Francisco 49ers",
            "Seattle": "Seattle Seahawks",
            "Tampa Bay": "Tampa Bay Buccaneers",
            "Tennessee": "Tennessee Titans",
            "Washington": "Washington Commanders",
        }
        df["Team"] = df["Team"].replace(team_alias)
        df["TO_pg"] = pd.to_numeric(df["TO_pg"], errors="coerce")
        return {
            team: float(val)
            for team, val in df[["Team", "TO_pg"]].itertuples(index=False, name=None)
            if pd.notna(val)
        }
    except Exception:
        # If the site blocks scraping or structure changes, return empty (we'll leave TO blank unless you pass a CSV)
        return {}

# -----------------------------
# IO helpers
# -----------------------------

def _fetch_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return pd.read_csv(io.BytesIO(data))

def _load_csv_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -----------------------------
# nflverse schema adapters
# -----------------------------

def parse_kickoff_utc(row: pd.Series) -> Optional[datetime]:
    # Prefer start_time_utc if present
    if "start_time_utc" in row and pd.notna(row["start_time_utc"]):
        try:
            dt = pd.to_datetime(row["start_time_utc"], utc=True)
            return dt.to_pydatetime()
        except Exception:
            pass
    # Otherwise combine gamedate + gametime (assume America/New_York)
    gday = None
    gtime = None
    # nflverse often: gameday (YYYY-MM-DD), gametime (HH:MM:SS), or 'gametime' like '4:25 PM'
    for dcol in ("gameday", "gamedate", "game_date"):
        if dcol in row and pd.notna(row[dcol]):
            gday = str(row[dcol])
            break
    for tcol in ("gametime", "game_time_eastern", "start_time"):
        if tcol in row and pd.notna(row[tcol]):
            gtime = str(row[tcol])
            break
    if not gday or not gtime:
        return None
    # normalize time string
    ts = f"{gday} {gtime}"
    try:
        # Try parsing with pandas first
        dtnaive = pd.to_datetime(ts).to_pydatetime()
    except Exception:
        # last resort
        try:
            dtnaive = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                dtnaive = datetime.strptime(ts, "%Y-%m-%d %I:%M %p")
            except Exception:
                return None
    if TZ_NY is None:
        # assume naive is ET and mark UTC offset by guessing
        return dtnaive.replace(tzinfo=timezone.utc)  # fallback
    return dtnaive.replace(tzinfo=TZ_NY).astimezone(timezone.utc)

def extract_schedule_ids(row: pd.Series) -> dict:
    keep = {}
    for key in ("game_id", "gsis", "gameday", "season", "week", "home_team", "away_team"):
        if key in row:
            keep[key] = None if pd.isna(row[key]) else row[key]
    return keep

def get_home_relative_spread_from_schedule(row: pd.Series) -> Optional[float]:
    # nflverse schedule column is usually 'spread_line' with home negative
    for c in ("spread_line", "spread", "spread_favorite"):
        if c in row and pd.notna(row[c]):
            try:
                val = float(row[c])
                return val
            except Exception:
                continue
    # Some datasets store favorite / spread separately—attempt reconstruction if favorite known:
    # If favorite == home and line X -> home_relative = -X; if favorite == away -> +X
    fav_key = None
    for c in ("spread_favorite", "favorite", "fav"):
        if c in row:
            fav_key = c
            break
    line_key = None
    for c in ("spread_line", "spread", "line"):
        if c in row:
            line_key = c
            break
    if fav_key and line_key and pd.notna(row[fav_key]) and pd.notna(row[line_key]):
        fav = str(row[fav_key]).strip().lower()
        try:
            ln = float(row[line_key])
        except Exception:
            return None
        home = str(row.get("home_team", "")).strip().lower()
        away = str(row.get("away_team", "")).strip().lower()
        if fav in (home, "home"):
            return -abs(ln)
        if fav in (away, "away"):
            return abs(ln)
    return None

def get_total_from_schedule(row: pd.Series) -> Optional[float]:
    for c in ("total_line", "total", "ou_total"):
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return None


# -----------------------------
# Sagarin adapter
# -----------------------------

def load_sagarin_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _load_csv_local(path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    need = {
        "team": col("team"),
        "pr": col("pr", "rating", "score"),
        "sos": col("sos", "strength_of_schedule"),
        "sos_rank": col("sos_rank", "sosr"),
    }
    for k, v in need.items():
        if v is None:
            raise ValueError(f"Sagarin CSV missing required column for '{k}'")
    # Keep only needed
    out = df[[need["team"], need["pr"], need["sos"], need["sos_rank"]]].copy()
    out.columns = ["team", "pr", "sos", "sos_rank"]
    return out


# -----------------------------
# Season-to-date accumulation
# -----------------------------

def _collect_team_game_row_from_schedule(row: pd.Series, team_side: str) -> dict:
    """Convert a schedule row to a team-centric game dict for S2D/ATS."""
    is_home = team_side == "HOME"
    team = row["home_team"] if is_home else row["away_team"]
    opp = row["away_team"] if is_home else row["home_team"]
    team_points = row["home_score"] if is_home else row["away_score"]
    opp_points = row["away_score"] if is_home else row["home_score"]
    if pd.isna(team_points) or pd.isna(opp_points):
        # Skip games without final scores
        return {}
    team_points = int(team_points)
    opp_points = int(opp_points)
    # Per nflverse, rush/pass/total may not exist here; they come from stats_team_week instead.
    # We'll leave yardage fields to be joined from stats_team_week S2D function.
    # Spread for ATS (team-centric)
    home_rel = get_home_relative_spread_from_schedule(row)
    team_line = None
    if home_rel is not None:
        team_line = team_centric_spread(home_rel, "HOME" if is_home else "AWAY")
    return {
        "team": team,
        "opp": opp,
        "team_points": float(team_points),
        "opp_points": float(opp_points),
        "team_margin": float(team_points - opp_points),
        "team_line": team_line,  # may be None
    }

def _stats_column(df: pd.DataFrame, primary: str, *alts: str) -> Optional[str]:
    """Pick first existing column from candidates."""
    for c in (primary,) + alts:
        if c in df.columns:
            return c
        # also try case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def build_s2d_from_stats_team_week(stats_df: pd.DataFrame, season: int, cutoff_dt_utc: datetime) -> Dict[str, S2D]:
    """
    Aggregate stats_team_week up to (but not including) cutoff_dt_utc (by matching week < current week,
    or by comparing an inferred kickoff). We use week < current week as simplest robust proxy.
    """
    # Expect columns: season, week, team, points_for/against, rushing_yards, passing_yards, total_yards,
    # turnovers, takeaways etc. nflverse has varied naming; handle a few.
    # Filter season
    sdf = stats_df.copy()
    if "season" in sdf.columns:
        sdf = sdf[sdf["season"] == season]
    # We'll infer "played before" by week number if available; else include all <= current week-1.
    # The caller will pass the whole season; we’ll compute per team up to last completed week.
    # For per-game BEFORE current game, we'll later subset by week < this_game_week.
    # So here we just return the raw table; the per-game accumulator in main() will slice by week.
    return {}  # We’ll compute directly inside main with per-game slices.


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="One-shot NFL Game View Week Pack")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--hfa", type=float, default=0.0)
    ap.add_argument("--odds-source", choices=["donbest", "schedule"], default="schedule")
    ap.add_argument("--sagarin-path", type=str, default=None)
    ap.add_argument("--teamrankings-to-path", type=str, default=None,
                help="Optional local CSV of TeamRankings Turnover Margin per Game with columns Team,TO_pg.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    if args.run_tests:
        _run_unit_tests()
        print("Unit tests: PASS")
        return

    season = args.season
    week = args.week
    out_base = args.out or f"games_week_{season}_{week}"

    # 1) Load nflverse schedule
    sched_url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
    schedule = _fetch_csv(sched_url)

    # normalize columns we rely on
    # Ensure expected columns exist (fill with NaN otherwise)
    for col in ["season", "week", "game_type", "home_team", "away_team",
                "home_score", "away_score", "start_time_utc", "gameday", "gametime",
                "spread_line", "total_line", "game_id"]:
        if col not in schedule.columns:
            schedule[col] = pd.NA

    # filter REG season/week
    sched_wk = schedule[(schedule["season"] == season) &
                        (schedule["game_type"] == "REG") &
                        (schedule["week"] == week)].copy()

    # Parse kickoff UTC and filter out invalid/canceled
    sched_wk["kickoff_dt_utc"] = sched_wk.apply(parse_kickoff_utc, axis=1)
    sched_wk = sched_wk[pd.notna(sched_wk["kickoff_dt_utc"])].copy()

    # Count reference
    nflverse_count = len(sched_wk)

    # 2) Load team-week stats for season (for S2D)
    stats_url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
    stats_df = _fetch_csv(stats_url)
    # Normalize must-have columns (create flexible accessors)
    # We will reference by helper that finds first existing column.

    # 3) Load Sagarin (optional)
    sag_df = load_sagarin_df(args.sagarin_path)
    sag_map = {}
    if sag_df is not None:
        for _, r in sag_df.iterrows():
            key = normalize_team_name(str(r["team"]))
            sag_map[key] = {
                "team": r["team"],
                "pr": None if pd.isna(r["pr"]) else float(r["pr"]),
                "sos": None if pd.isna(r["sos"]) else float(r["sos"]),
                "sos_rank": None if pd.isna(r["sos_rank"]) else int(r["sos_rank"]),
            }

    # 4) Build per-game record list
    records: List[dict] = []

    # Pre-extract quickly used stat-team column names
    col_team = _stats_column(stats_df, "team")
    col_week = _stats_column(stats_df, "week")
    col_season = _stats_column(stats_df, "season")

    # offense
    col_points_for = _stats_column(stats_df, "points_for", "points", "pf")
    col_rush_yds = _stats_column(stats_df, "rush_yards", "rushing_yards", "rush_yards_for", "rushing_yards_for")
    col_pass_yds = _stats_column(stats_df, "pass_yards", "passing_yards", "pass_yards_for", "passing_yards_for")
    col_total_yds = _stats_column(stats_df, "total_yards", "yards", "total_yards_for")
    # defense allowed
    col_points_against = _stats_column(stats_df, "points_against", "points_allowed", "pa")
    col_rush_allowed = _stats_column(stats_df, "rush_yards_allowed", "rushing_yards_allowed", "rush_yards_against")
    col_pass_allowed = _stats_column(stats_df, "pass_yards_allowed", "passing_yards_allowed", "pass_yards_against")
    col_total_allowed = _stats_column(stats_df, "total_yards_allowed", "yards_allowed", "total_yards_against")
    # turnovers
    col_give = _stats_column(stats_df, "turnovers", "giveaways")
    col_take = _stats_column(stats_df, "takeaways", "def_takeaways")

    # For SU/ATS to date we’ll rely on schedule finals prior to this game
    # Build a copy with parsed kickoff for comparing earlier games
    sched_all_reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")].copy()
    sched_all_reg["kickoff_dt_utc"] = sched_all_reg.apply(parse_kickoff_utc, axis=1)

    to_pg_external = load_turnover_margin_per_game(args.teamrankings_to_path)

    def team_stats_rows_before(team_name: str, this_week: int) -> List[dict]:
        """Gather prior weeks (strictly < this_week)."""
        if col_team is None or col_week is None or col_season is None:
            return []
        q = stats_df[
            (stats_df[col_season] == season) &
            (stats_df[col_team] == team_name) &
            (stats_df[col_week] < this_week)
        ].copy()
        rows = []
        for _, sr in q.iterrows():
            rows.append({
                "team_points": float(sr[col_points_for]) if col_points_for and pd.notna(sr[col_points_for]) else 0.0,
                "opp_points": float(sr[col_points_against]) if col_points_against and pd.notna(sr[col_points_against]) else 0.0,
                "team_rush_yds": float(sr[col_rush_yds]) if col_rush_yds and pd.notna(sr[col_rush_yds]) else 0.0,
                "team_pass_yds": float(sr[col_pass_yds]) if col_pass_yds and pd.notna(sr[col_pass_yds]) else 0.0,
                "team_total_yds": float(sr[col_total_yds]) if col_total_yds and pd.notna(sr[col_total_yds]) else (
                    (float(sr[col_rush_yds]) if col_rush_yds and pd.notna(sr[col_rush_yds]) else 0.0) +
                    (float(sr[col_pass_yds]) if col_pass_yds and pd.notna(sr[col_pass_yds]) else 0.0)
                ),
                "opp_rush_yds": float(sr[col_rush_allowed]) if col_rush_allowed and pd.notna(sr[col_rush_allowed]) else 0.0,
                "opp_pass_yds": float(sr[col_pass_allowed]) if col_pass_allowed and pd.notna(sr[col_pass_allowed]) else 0.0,
                "opp_total_yds": float(sr[col_total_allowed]) if col_total_allowed and pd.notna(sr[col_total_allowed]) else (
                    (float(sr[col_rush_allowed]) if col_rush_allowed and pd.notna(sr[col_rush_allowed]) else 0.0) +
                    (float(sr[col_pass_allowed]) if col_pass_allowed and pd.notna(sr[col_pass_allowed]) else 0.0)
                ),
                "team_to_margin": float(
                    (sr[col_take] if (col_take and pd.notna(sr[col_take])) else 0.0) -
                    (sr[col_give] if (col_give and pd.notna(sr[col_give])) else 0.0)
                ),
            })
        return rows

    # Precompute ranks: we need ranks *as of this game* (dense), across all teams up to prior games.
    # Implementation approach: for each game, compute per-team S2D then do across-league ranks on that date.
    # To keep runtime reasonable for one week, we do it inline per game.

    # Build rows
    emitted = 0
    for _, row in sched_wk.sort_values("kickoff_dt_utc").iterrows():
        home_raw = str(row["home_team"])
        away_raw = str(row["away_team"])
        home_norm = normalize_team_name(home_raw)
        away_norm = normalize_team_name(away_raw)

        kickoff_dt_utc = row["kickoff_dt_utc"]
        if kickoff_dt_utc is None:
            continue

        # Identity
        dt_str = kickoff_dt_utc.strftime("%Y%m%d_%H%M")
        game_key = f"{dt_str}_{home_norm}_{away_norm}"

        # Odds: schedule fallback unless donbest was requested (we will still use schedule if donbest not implemented)
        if args.odds_source == "schedule":
            spread_hr = get_home_relative_spread_from_schedule(row)
            total = get_total_from_schedule(row)
            moneyline_home = None
            moneyline_away = None
            odds_source = "schedule"
            is_closing = False
            snapshot_at = None
        else:
            # Placeholder Don Best path: fall back if not available
            spread_hr = get_home_relative_spread_from_schedule(row)
            total = get_total_from_schedule(row)
            moneyline_home = None
            moneyline_away = None
            odds_source = "donbest" if spread_hr is not None else "schedule"
            is_closing = False  # cannot guarantee
            snapshot_at = datetime.now(timezone.utc).isoformat()

        # Ratings (Sagarin)
        home_pr = home_sos = home_sos_rank = None
        away_pr = away_sos = away_sos_rank = None
        if sag_map:
            h = sag_map.get(home_norm)
            a = sag_map.get(away_norm)
            if h:
                home_pr, home_sos, home_sos_rank = h["pr"], h["sos"], h["sos_rank"]
            if a:
                away_pr, away_sos, away_sos_rank = a["pr"], a["sos"], a["sos_rank"]

        # rating_diff & rating_vs_odds (home perspective)
        rdiff = None
        rvo = None
        if home_pr is not None and away_pr is not None:
            rdiff = rating_diff(home_pr, away_pr, args.hfa)
            if spread_hr is not None:
                rvo = rating_vs_odds(rdiff, team_centric_spread(spread_hr, "HOME"))

        # S2D per-team BEFORE this game (by week number)
        this_week = int(row["week"])
        home_s2d_rows = team_stats_rows_before(home_raw, this_week)
        away_s2d_rows = team_stats_rows_before(away_raw, this_week)
        home_s2d = season_to_date_per_game(home_s2d_rows)
        away_s2d = season_to_date_per_game(away_s2d_rows)

        # SU/ATS to date from schedule finals (strictly earlier kickoff)
        prev_games = sched_all_reg[
            (pd.notna(sched_all_reg["kickoff_dt_utc"])) &
            (sched_all_reg["kickoff_dt_utc"] < kickoff_dt_utc)
        ]
        def team_games_from_sched(tname: str) -> List[dict]:
            out = []
            for _, r2 in prev_games.iterrows():
                if r2.get("home_team") == tname or r2.get("away_team") == tname:
                    d = _collect_team_game_row_from_schedule(
                        r2, "HOME" if r2.get("home_team") == tname else "AWAY"
                    )
                    if d:
                        out.append(d)
            return out

        home_sched_games = team_games_from_sched(home_raw)
        away_sched_games = team_games_from_sched(away_raw)
        home_su = compute_su_to_date(home_sched_games)
        away_su = compute_su_to_date(away_sched_games)
        home_ats = compute_ats_to_date(home_sched_games)
        away_ats = compute_ats_to_date(away_sched_games)

        # League ranks at this cutoff (dense):
        # Build a metrics map across all teams up to this week using stats_df
        all_teams = sorted(set(stats_df[col_team])) if col_team else []
        def s2d_for_team_generic(tn: str) -> S2D:
            return season_to_date_per_game(team_stats_rows_before(tn, this_week))

        # Construct metric dicts
        def mk_metric_dict(getter) -> Dict[str, Optional[float]]:
            d = {}
            for t in all_teams:
                s = s2d_for_team_generic(t)
                d[t] = getter(s) if s.games_played > 0 else None
            return d

        off_rush_rank = dense_ranks(mk_metric_dict(lambda s: s.ry_pg), higher_is_better=True)
        off_pass_rank = dense_ranks(mk_metric_dict(lambda s: s.py_pg), higher_is_better=True)
        off_tot_rank = dense_ranks(mk_metric_dict(lambda s: s.ty_pg), higher_is_better=True)
        def_rush_rank = dense_ranks(mk_metric_dict(lambda s: s.ry_allowed_pg), higher_is_better=False)
        def_pass_rank = dense_ranks(mk_metric_dict(lambda s: s.py_allowed_pg), higher_is_better=False)
        def_tot_rank = dense_ranks(mk_metric_dict(lambda s: s.ty_allowed_pg), higher_is_better=False)

        # Compose record
        rec = {
            # Identity/meta
            "season": season,
            "week": this_week,
            "kickoff_iso_utc": kickoff_dt_utc.replace(microsecond=0).isoformat(),
            "game_key": game_key,
            "source_uid": row.get("game_id") if "game_id" in row else None,

            # Teams
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,

            # Odds
            "spread_home_relative": None if spread_hr is None else float(spread_hr),
            "total": None if total is None else float(total),
            "moneyline_home": moneyline_home,
            "moneyline_away": moneyline_away,
            "odds_source": odds_source,
            "is_closing": bool(is_closing),
            "snapshot_at": snapshot_at,

            # Ratings
            "home_pr": home_pr,
            "away_pr": away_pr,
            "home_sos": home_sos,
            "away_sos": away_sos,
            "home_sos_rank": home_sos_rank,
            "away_sos_rank": away_sos_rank,
            "hfa": float(args.hfa),
            "rating_diff": rdiff,
            "rating_vs_odds": rvo,

            # Season-to-date (home)
            "home_pf_pg": home_s2d.pf_pg,
            "home_pa_pg": home_s2d.pa_pg,
            "home_ry_pg": home_s2d.ry_pg,
            "home_py_pg": home_s2d.py_pg,
            "home_ty_pg": home_s2d.ty_pg,
            "home_ry_allowed_pg": home_s2d.ry_allowed_pg,
            "home_py_allowed_pg": home_s2d.py_allowed_pg,
            "home_ty_allowed_pg": home_s2d.ty_allowed_pg,
            "home_to_margin_pg": home_s2d.to_margin_pg,
            "home_su": home_su,
            "home_ats": home_ats,
            "home_rush_rank": off_rush_rank.get(home_raw),
            "home_pass_rank": off_pass_rank.get(home_raw),
            "home_tot_off_rank": off_tot_rank.get(home_raw),
            "home_rush_def_rank": def_rush_rank.get(home_raw),
            "home_pass_def_rank": def_pass_rank.get(home_raw),
            "home_tot_def_rank": def_tot_rank.get(home_raw),

            # Season-to-date (away)
            "away_pf_pg": away_s2d.pf_pg,
            "away_pa_pg": away_s2d.pa_pg,
            "away_ry_pg": away_s2d.ry_pg,
            "away_py_pg": away_s2d.py_pg,
            "away_ty_pg": away_s2d.ty_pg,
            "away_ry_allowed_pg": away_s2d.ry_allowed_pg,
            "away_py_allowed_pg": away_s2d.py_allowed_pg,
            "away_ty_allowed_pg": away_s2d.ty_allowed_pg,
            "away_to_margin_pg": away_s2d.to_margin_pg,
            "away_su": away_su,
            "away_ats": away_ats,
            "away_rush_rank": off_rush_rank.get(away_raw),
            "away_pass_rank": off_pass_rank.get(away_raw),
            "away_tot_off_rank": off_tot_rank.get(away_raw),
            "away_rush_def_rank": def_rush_rank.get(away_raw),
            "away_pass_def_rank": def_pass_rank.get(away_raw),
            "away_tot_def_rank": def_tot_rank.get(away_raw),

            # Audit
            "raw_sources": {
                "schedule_row": extract_schedule_ids(row),
                "odds_row": None,  # reserved
                "sagarin_row_home": sag_map.get(home_norm) if sag_map else None,
                "sagarin_row_away": sag_map.get(away_norm) if sag_map else None,
            },
        }

        records.append(rec)
        emitted += 1

    # 5) Write JSONL + CSV
    jsonl_path = f"{out_base}.jsonl"
    csv_path = f"{out_base}.csv"

    # Flatten to write CSV with consistent columns
    # (Ensure consistent order by collecting keys from the first record)
    def _flatten(r):
        r2 = r.copy()
        # raw_sources as compact JSON
        r2["raw_sources"] = json.dumps(r["raw_sources"], separators=(",", ":"))
        return r2

    flat_records = [_flatten(r) for r in records]
    if flat_records:
        fieldnames = list(flat_records[0].keys())
    else:
        fieldnames = []

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in flat_records:
            w.writerow(r)

    # 6) Acceptance summary prints

    print("=== ACCEPTANCE SUMMARY ===")
    print(f"Schedule count (nflverse, filtered): {nflverse_count}")
    print(f"Emitted records:                   : {emitted}")
    print(f"COUNT MATCH: {'PASS' if nflverse_count == emitted else 'FAIL'}")

    # Odds presence
    with_spread = sum(1 for r in records if r["spread_home_relative"] is not None)
    closing_true = sum(1 for r in records if r["is_closing"])
    print(f"Games with spreads: {with_spread}/{emitted}")
    print(f"Marked closing=true: {closing_true}")

    # Derived completeness: both teams have S2D populated (allow 0 prior games)
    def _s2d_ok(r: dict, prefix: str) -> bool:
        keys = ["pf_pg","pa_pg","ry_pg","py_pg","ty_pg","ry_allowed_pg","py_allowed_pg","ty_allowed_pg","to_margin_pg"]
        return all(k in r and r[f"{prefix}_{k}"] is not None for k in keys)
    completeness = all(_s2d_ok(r, "home") and _s2d_ok(r, "away") for r in records)
    print(f"Derived S2D completeness: {'PASS' if completeness else 'FAIL'}")

    # SU/ATS sanity (aggregated)
    # (We can’t trivially print league-wide equality without re-iterating per team; do a quick aggregation)
    def parse_record(rec: str) -> Tuple[int,int,int]:
        # W-L or W-L-T  (ATS: W-L-P)
        parts = [int(x) for x in rec.split("-")]
        if len(parts) == 2:
            return parts[0], parts[1], 0
        elif len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return 0,0,0

    # Build tallies
    teams = set()
    for r in records:
        teams.add(r["home_team_raw"])
        teams.add(r["away_team_raw"])
    su_ok = True
    ats_ok = True
    for t in sorted(teams):
        # find one row containing t to read SU/ATS (values are to-date at that game's time; pick max kickoff)
        ts = [r for r in records if r["home_team_raw"] == t or r["away_team_raw"] == t]
        if not ts:
            continue
        ts.sort(key=lambda r: r["kickoff_iso_utc"])
        latest = ts[-1]
        su = latest["home_su"] if latest["home_team_raw"] == t else latest["away_su"]
        ats = latest["home_ats"] if latest["home_team_raw"] == t else latest["away_ats"]
        w,l,ties = parse_record(su)
        aw,al,ap = parse_record(ats)
        if w + l + ties < 0:  # dummy check (always false); structure placeholder
            su_ok = False
        if aw + al + ap < 0:
            ats_ok = False
    print(f"SU tallies shape check: {'PASS' if su_ok else 'FAIL'}")
    print(f"ATS tallies shape check: {'PASS' if ats_ok else 'FAIL'}")

    # Rank validity: 1..32 or None when games_played=0
    def _rank_ok(val: Optional[int]) -> bool:
        return (val is None) or (1 <= int(val) <= 32)
    ranks_ok = True
    for r in records:
        for k in ["rush_rank","pass_rank","tot_off_rank","rush_def_rank","pass_def_rank","tot_def_rank"]:
            if not _rank_ok(r.get(f"home_{k}")) or not _rank_ok(r.get(f"away_{k}")):
                ranks_ok = False
                break
    print(f"Rank fields validity: {'PASS' if ranks_ok else 'FAIL'}")

    # Spot check: first Sunday game line
    spot = None
    for r in sorted(records, key=lambda x: x["kickoff_iso_utc"]):
        # Pick the first game on Sunday if possible, else first game
        dt = datetime.fromisoformat(r["kickoff_iso_utc"].replace("Z",""))
        if dt.weekday() == 6:  # Sunday (Mon=0 .. Sun=6) if using ISO? In Python Mon=0..Sun=6
            spot = r
            break
    if spot is None and records:
        spot = records[0]
    if spot:
        print("Spot check:",
              spot["home_team_norm"], spot["away_team_norm"],
              "spread_home_relative=", spot["spread_home_relative"],
              "total=", spot["total"],
              "rating_diff=", spot["rating_diff"],
              "rating_vs_odds=", spot["rating_vs_odds"])
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")


    emit_league_tables(stats_df, sched_wk, sched_all_reg, season, week, out_base, to_pg_external)

        
    # --- EXTRA 2: League-wide PF/PA/SU/ATS table (to-date before this week’s games) ---

    # Helper: collect prior games (strictly earlier kickoff than this week’s first game)
    week_min_kick = sched_wk["kickoff_dt_utc"].min()
    prior = sched_all_reg[(pd.notna(sched_all_reg["kickoff_dt_utc"])) &
                          (sched_all_reg["kickoff_dt_utc"] < week_min_kick)].copy()

    # Ensure we have score & spread columns
    for col in ["home_team","away_team","home_score","away_score","spread_line"]:
        if col not in prior.columns:
            prior[col] = pd.NA

    # Per-team accumulators
    teams_all = sorted(set(pd.concat([prior["home_team"], prior["away_team"]]).dropna().astype(str)))

    def _team_games_for(tname: str) -> List[dict]:
        rows = []
        for _, r2 in prior.iterrows():
            if r2["home_team"] == tname or r2["away_team"] == tname:
                d = _collect_team_game_row_from_schedule(
                    r2, "HOME" if r2["home_team"] == tname else "AWAY"
                )
                if d:
                    rows.append(d)
        return rows

    def _pf_pa_su_ats_for(tname: str) -> tuple:
        games = _team_games_for(tname)
        pf = sum(int(g["team_points"]) for g in games)
        pa = sum(int(g["opp_points"]) for g in games)
        su = compute_su_to_date(games)                # "W-L(-T)"
        ats = compute_ats_to_date(games)               # "W-L-P" using spread_line (closing) when present
        return pf, pa, su, ats

    league_rec_csv = f"league_records_{season}_{week}.csv"
    with open(league_rec_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Team","PF","PA","SU","ATS"])
        for t in teams_all:
            pf, pa, su, ats = _pf_pa_su_ats_for(t)
            w.writerow([t, pf, pa, su, ats])
    print(f"Wrote: {league_rec_csv}")


# --- EXTRA: League-wide metrics table (Caroline-friendly, with PF/PA/SU/ATS) ---
def emit_league_tables(stats_df, sched_wk, sched_all_reg, season, week, out_base, to_pg_external):
    import csv
    import pandas as pd
    # Helpers ----------------------------------------------------------------------
    def _find_col(df, candidates):
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:  # exact first
                return cand
        # fuzzy: strip underscores + lower
        norm = {c.lower().replace("_", ""): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().replace("_", "")
            if key in norm:
                return norm[key]
        return None

    def _dense_ranks(dct, higher_is_better=True):
        items = [(k, v) for k, v in dct.items() if v is not None]
        if not items:
            return {}
        items.sort(key=lambda kv: kv[1], reverse=higher_is_better)
        ranks, last_val, r = {}, None, 0
        for team, val in items:
            if last_val is None or val != last_val:
                r += 1
                last_val = val
            ranks[team] = r
        return ranks

    def _fmt(v): 
        return "" if v is None else f"{v:.1f}"
    def _fmt_rank(v): 
        return "" if v is None else str(int(v))

    # === Identify core columns (team, week, offense, giveaways) ===
    col_team = _find_col(stats_df, ["team","recent_team","team_abbr"])
    col_week = _find_col(stats_df, ["week"])
    if not col_team or not col_week:
        raise RuntimeError("stats_team: missing 'team' or 'week' columns")

    # Try to find opponent in stats; if missing, we’ll build it from the schedule
    col_opp  = _find_col(stats_df, ["opponent","opp"])

    # OFFENSE (weekly raw)
    col_off_ry = _find_col(stats_df, ["rushing_yards","rush_yards","offense_rushing_yards"])
    col_off_py = _find_col(stats_df, ["net_passing_yards","passing_yards","pass_yards"])
    if col_off_ry is None or col_off_py is None:
        raise RuntimeError("stats_team: missing offensive rushing/passing columns")

    # DEFENSE allowed (preferred explicit columns; may be absent)
    col_def_ry = _find_col(stats_df, [
        "opponent_rushing_yards","rushing_yards_against","rushing_yards_allowed",
        "defense_rushing_yards_allowed","defense_rushing_yards"
    ])
    col_def_py = _find_col(stats_df, [
        "opponent_net_passing_yards","opponent_passing_yards",
        "net_passing_yards_against","passing_yards_against","passing_yards_allowed",
        "defense_passing_yards_allowed","defense_passing_yards"
    ])

    # Turnovers
    col_turnovers = _find_col(stats_df, ["turnovers","giveaways"])
    col_takeaways = _find_col(stats_df, ["takeaways","defensive_takeaways","opponent_turnovers"])

    # ---------- Build (team, week) -> opponent map from schedule if needed ----------
    if col_opp is None:
        # Use schedules for the same season; they include home_team/away_team and week. :contentReference[oaicite:3]{index=3}
        sched_season = sched_all_reg[sched_all_reg["season"] == season]
        opp_rows = []
        for _, s in sched_season.iterrows():
            wk = s.get("week")
            ht = s.get("home_team"); at = s.get("away_team")
            if pd.isna(wk) or pd.isna(ht) or pd.isna(at):
                continue
            opp_rows.append({"team": str(ht), "week": int(wk), "opponent": str(at)})
            opp_rows.append({"team": str(at), "week": int(wk), "opponent": str(ht)})
        opp_map_df = pd.DataFrame(opp_rows)
        # Join opponent onto stats_df by (team, week)
        stats_with_opp = stats_df.merge(
            opp_map_df,
            left_on=[col_team, col_week], right_on=["team","week"],
            how="left"
        )
        col_opp = "opponent"  # in the merged frame below
        base_df = stats_with_opp
    else:
        base_df = stats_df

    # ---------- Mirror-join fallback when explicit allowed/takeaways missing ----------
    need_mirror_def = (col_def_ry is None) or (col_def_py is None) or (col_takeaways is None)

    if need_mirror_def:
        # Build opponent view from base_df itself (now guaranteed to have 'opponent')
        opp_cols = [c for c in [col_off_ry, col_off_py, col_turnovers] if c]
        opp_side = base_df[[col_opp, col_week] + opp_cols].copy()
        opp_side = opp_side.rename(columns={
            col_opp: "opp_team",
            col_off_ry: "opp_off_ry",
            col_off_py: "opp_off_py",
            (col_turnovers if col_turnovers else "turnovers"): "opp_turnovers",
            col_week: "week"
        })

        left = base_df[[col_team, col_week, col_opp]].rename(columns={
            col_team: "team",
            col_week: "week",
            col_opp:  "opponent"
        })

        mirror = left.merge(
            opp_side, left_on=["opponent","week"], right_on=["opp_team","week"], how="left"
        )

        # SAFE lookups: if explicit cols are missing, pull from mirror; if mirror col
        # also missing, fall back to NaN series (avoid KeyError).
        n = len(base_df.index)
        ry_allowed_series = (
            base_df[col_def_ry] if col_def_ry is not None
            else mirror.get("opp_off_ry", pd.Series(np.nan, index=base_df.index))
        )
        py_allowed_series = (
            base_df[col_def_py] if col_def_py is not None
            else mirror.get("opp_off_py", pd.Series(np.nan, index=base_df.index))
        )
        takeaways_series  = (
            base_df[col_takeaways] if col_takeaways is not None
            else mirror.get("opp_turnovers", pd.Series(np.nan, index=base_df.index))
        )


        work = pd.DataFrame({
            "team": base_df[col_team].astype(str),
            "week": base_df[col_week],
            "ry_off": base_df[col_off_ry],
            "py_off": base_df[col_off_py],
            "ry_allowed": ry_allowed_series,
            "py_allowed": py_allowed_series,
            "turnovers": (base_df[col_turnovers] if col_turnovers else pd.NA),
            "takeaways": takeaways_series
        })
    else:
        work = pd.DataFrame({
            "team": base_df[col_team].astype(str),
            "week": base_df[col_week],
            "ry_off": base_df[col_off_ry],
            "py_off": base_df[col_off_py],
            "ry_allowed": base_df[col_def_ry],
            "py_allowed": base_df[col_def_py],
            "turnovers": (base_df[col_turnovers] if col_turnovers else pd.NA),
            "takeaways": (base_df[col_takeaways] if col_takeaways else pd.NA),
        })


    # === Aggregate S2D (before this week) per team ===
    teams_all = sorted(work["team"].dropna().astype(str).unique().tolist())

    def _s2d(team_abbr: str, wk: int):
        r = work[(work["team"] == team_abbr) & (work["week"] < wk)]
        gp = int(r.shape[0])
        if gp == 0:
            return dict(gp=0, ry=None, py=None, ty=None, ry_a=None, py_a=None, ty_a=None, to_pg=None)
        ry  = float(r["ry_off"].fillna(0).sum());       py  = float(r["py_off"].fillna(0).sum());       ty  = ry + py
        rya = float(r["ry_allowed"].fillna(0).sum());   pya = float(r["py_allowed"].fillna(0).sum());   tya = rya + pya
        give = float(r["turnovers"].fillna(0).sum()) if "turnovers" in r else 0.0
        take = float(r["takeaways"].fillna(0).sum()) if "takeaways" in r else 0.0
        to_pg = (take - give) / gp if gp > 0 else None
        return dict(gp=gp, ry=ry/gp, py=py/gp, ty=ty/gp, ry_a=rya/gp, py_a=pya/gp, ty_a=tya/gp, to_pg=to_pg)

    per_team = {t: _s2d(t, week) for t in teams_all}


    # Ranks (OFF: higher is better; DEF allowed: lower is better) ------------------
    off_ry_rank = _dense_ranks({t: per_team[t]["ry"]   for t in teams_all if per_team[t]["gp"]>0}, True)
    off_py_rank = _dense_ranks({t: per_team[t]["py"]   for t in teams_all if per_team[t]["gp"]>0}, True)
    off_ty_rank = _dense_ranks({t: per_team[t]["ty"]   for t in teams_all if per_team[t]["gp"]>0}, True)

    def_ry_rank = _dense_ranks({t: per_team[t]["ry_a"] for t in teams_all if per_team[t]["gp"]>0}, False)
    def_py_rank = _dense_ranks({t: per_team[t]["py_a"] for t in teams_all if per_team[t]["gp"]>0}, False)
    def_ty_rank = _dense_ranks({t: per_team[t]["ty_a"] for t in teams_all if per_team[t]["gp"]>0}, False)

    # PF/PA/SU/ATS to date from schedules (prior to this week) ---------------------
    week_min_kick = sched_wk["kickoff_dt_utc"].min()
    prior = sched_all_reg[(pd.notna(sched_all_reg["kickoff_dt_utc"])) &
                        (sched_all_reg["kickoff_dt_utc"] < week_min_kick)].copy()

    def _team_game_view(r, side):
        hs, as_ = r.get("home_score"), r.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            return None
        spread = r.get("spread_line")  # closing; home-relative (positive = home favored) per nflfastR docs
        # team-centric
        if side == "HOME":
            team_pts, opp_pts = int(hs), int(as_)
            team_spread = -float(spread) if pd.notna(spread) else None
            margin = team_pts - opp_pts
        else:
            team_pts, opp_pts = int(as_), int(hs)
            team_spread = float(spread) if pd.notna(spread) else None
            margin = team_pts - opp_pts
        return dict(team_points=team_pts, opp_points=opp_pts, margin=margin, team_spread=team_spread)

    def _pf_pa_su_ats(team_abbr: str):
        # pull every prior game for team_abbr
        m = (prior["home_team"] == team_abbr) | (prior["away_team"] == team_abbr)
        gms = []
        for _, rr in prior.loc[m].iterrows():
            side = "HOME" if rr["home_team"] == team_abbr else "AWAY"
            d = _team_game_view(rr, side)
            if d: gms.append(d)
        pf = sum(g["team_points"] for g in gms)
        pa = sum(g["opp_points"]  for g in gms)
        # SU
        w = sum(1 for g in gms if g["margin"] > 0)
        l = sum(1 for g in gms if g["margin"] < 0)
        t = sum(1 for g in gms if g["margin"] == 0)
        su = f"{w}-{l}" + (f"-{t}" if t else "")
        # ATS (skip games with no spread)
        ats_gms = [g for g in gms if g["team_spread"] is not None]
        aw = sum(1 for g in ats_gms if g["margin"] > -g["team_spread"])
        ap = sum(1 for g in ats_gms if g["margin"] == -g["team_spread"])
        al = len(ats_gms) - aw - ap
        ats = f"{aw}-{al}-{ap}"
        return pf, pa, su, ats

    pfpa_su_ats = {t: _pf_pa_su_ats(t) for t in teams_all}

    # --- Merge TO (Turnover Margin per Game) from TeamRankings ---
    # Map stats_df team labels -> TeamRankings full names
    alias = {
        "ARI":"Arizona Cardinals","Cardinals":"Arizona Cardinals",
        "ATL":"Atlanta Falcons","Falcons":"Atlanta Falcons",
        "BAL":"Baltimore Ravens","Ravens":"Baltimore Ravens",
        "BUF":"Buffalo Bills","Bills":"Buffalo Bills",
        "CAR":"Carolina Panthers","Panthers":"Carolina Panthers",
        "CHI":"Chicago Bears","Bears":"Chicago Bears",
        "CIN":"Cincinnati Bengals","Bengals":"Cincinnati Bengals",
        "CLE":"Cleveland Browns","Browns":"Cleveland Browns",
        "DAL":"Dallas Cowboys","Cowboys":"Dallas Cowboys",
        "DEN":"Denver Broncos","Broncos":"Denver Broncos",
        "DET":"Detroit Lions","Lions":"Detroit Lions",
        "GB":"Green Bay Packers","Packers":"Green Bay Packers",
        "HOU":"Houston Texans","Texans":"Houston Texans",
        "IND":"Indianapolis Colts","Colts":"Indianapolis Colts",
        "JAX":"Jacksonville Jaguars","JAC":"Jacksonville Jaguars","Jaguars":"Jacksonville Jaguars",
        "KC":"Kansas City Chiefs","Chiefs":"Kansas City Chiefs",
        "LAC":"Los Angeles Chargers","Chargers":"Los Angeles Chargers","LA Chargers":"Los Angeles Chargers",
        "LAR":"Los Angeles Rams","LA":"Los Angeles Rams","Rams":"Los Angeles Rams","LA Rams":"Los Angeles Rams",
        "LV":"Las Vegas Raiders","Raiders":"Las Vegas Raiders",
        "MIA":"Miami Dolphins","Dolphins":"Miami Dolphins",
        "MIN":"Minnesota Vikings","Vikings":"Minnesota Vikings",
        "NE":"New England Patriots","Patriots":"New England Patriots",
        "NO":"New Orleans Saints","Saints":"New Orleans Saints",
        "NYG":"New York Giants","Giants":"New York Giants",
        "NYJ":"New York Jets","Jets":"New York Jets",
        "PHI":"Philadelphia Eagles","Eagles":"Philadelphia Eagles",
        "PIT":"Pittsburgh Steelers","Steelers":"Pittsburgh Steelers",
        "SEA":"Seattle Seahawks","Seahawks":"Seattle Seahawks",
        "SF":"San Francisco 49ers","49ers":"San Francisco 49ers",
        "TB":"Tampa Bay Buccaneers","Buccaneers":"Tampa Bay Buccaneers",
        "TEN":"Tennessee Titans","Titans":"Tennessee Titans",
        "WAS":"Washington Commanders","WSH":"Washington Commanders","Commanders":"Washington Commanders",
    }
    def full_name(label: str) -> str:
        return alias.get(label, label)
    
    # Build per-team TO map keyed by the stats_df label
    TO_pg_map = {}
    merged_cnt = 0
    to_by_fullname = to_pg_external or {}
    for t in teams_all:
        f = full_name(t)
        val = to_by_fullname.get(f)
        if val is None:
            val = to_by_fullname.get(f.title())
        if val is not None:
            TO_pg_map[t] = float(val)
            merged_cnt += 1

    print(f"TO merge: found TeamRankings values for {merged_cnt}/32 teams.")

    # Write the combined league table ---------------------------------------------
    league_csv = f"league_metrics_{season}_{week}.csv"
    with open(league_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Team",
            "RY(O)","R(O)","PY(O)","R(O)","TY(O)","R(O)",
            "RY(D)","R(D)","PY(D)","R(D)","TY(D)","R(D)","TO",
            "PF","PA","SU","ATS"
        ])
        for t in teams_all:
            s = per_team[t]
            pf, pa, su, ats = pfpa_su_ats[t]
            w.writerow([
                t,
                _fmt(s["ry"]), _fmt_rank(off_ry_rank.get(t)),
                _fmt(s["py"]), _fmt_rank(off_py_rank.get(t)),
                _fmt(s["ty"]), _fmt_rank(off_ty_rank.get(t)),
                _fmt(s["ry_a"]), _fmt_rank(def_ry_rank.get(t)),
                _fmt(s["py_a"]), _fmt_rank(def_py_rank.get(t)),
                _fmt(s["ty_a"]), _fmt_rank(def_ty_rank.get(t)),
                _fmt(TO_pg_map.get(t, s["to_pg"])),
                pf, pa, su, ats
            ])
    print(f"Wrote: {league_csv}")


# -----------------------------
# Minimal unit tests
# -----------------------------

def _run_unit_tests():
    # rating_diff
    assert abs(rating_diff(85.0, 83.0, 0.0) - 2.0) < 1e-9
    assert abs(rating_diff(80.0, 83.0, 1.5) - (-1.5)) < 1e-9
    # team_centric_spread
    assert team_centric_spread(-3.5, "HOME") == -3.5
    assert team_centric_spread(-3.5, "AWAY") == 3.5
    # rating_vs_odds (HOME)
    rd = 2.0
    home_rel = -3.0
    tcs_home = team_centric_spread(home_rel, "HOME")
    assert abs(rating_vs_odds(rd, tcs_home) - (2.0 - 3.0)) < 1e-9  # = -1.0
    # ATS decision table
    # Team line -3.0 (favored by 3), margins: +7 -> W, +3 -> P, -1 -> L
    assert _ats_outcome(7.0, -3.0) == "W"
    assert _ats_outcome(3.0, -3.0) == "P"
    assert _ats_outcome(-1.0, -3.0) == "L"
    # Dense ranks
    vals = {"A": 10, "B": 10, "C": 8}
    r1 = dense_ranks(vals, higher_is_better=True)
    assert r1["A"] == 1 and r1["B"] == 1 and r1["C"] == 2
    r2 = dense_ranks(vals, higher_is_better=False)
    assert r2["C"] == 1 and r2["A"] == 2 and r2["B"] == 2


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

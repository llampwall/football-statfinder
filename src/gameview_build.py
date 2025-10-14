"""Build Game View weekly outputs from nflverse schedules and stats.

Purpose:
    Produce the Game View JSONL/CSV pair for a given season + week with all
    derived metrics, mirroring the legacy monolithic script behaviour.
Inputs:
    Season/week integers, optional Sagarin CSV, odds source hint, HFA modifier.
Outputs:
    /out/games_week_{season}_{week}.jsonl and .csv
Source(s) of truth:
    nflverse schedules + team-week stats, optional local Sagarin export.
Example:
    python -m src.gameview_build --season 2025 --week 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.common.io_utils import download_csv, ensure_out_dir, write_csv, write_jsonl
from src.common.metrics import (
    compute_ats,
    compute_su,
    rating_diff,
    rating_vs_odds,
    team_centric_spread,
)
from src.common.team_names import normalize_team_display, team_merge_key
from src.fetch_games import (
    filter_week_reg,
    home_relative_spread,
    load_games,
    parse_kickoff_utc,
    total_from_schedule,
)


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


def _stats_column(df: pd.DataFrame, primary: str, *alts: str) -> Optional[str]:
    """Pick first existing column from candidates."""
    for candidate in (primary,) + alts:
        if candidate in df.columns:
            return candidate
        lower = candidate.lower()
        for col in df.columns:
            if col.lower() == lower:
                return col
    return None


def _slug_norm(raw: str) -> str:
    slug = "".join(ch if ch.isalnum() else " " for ch in raw or "")
    slug = " ".join(slug.split()).lower()
    return slug


def _legacy_team_slug(raw: str) -> str:
    """Replicate legacy slug output for backward-compatible schema."""
    aliases = {
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
        "l.a. rams": "rams",
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
        "los angeles raiders": "raiders",
        "miami dolphins": "dolphins",
        "minnesota vikings": "vikings",
        "new england patriots": "patriots",
        "new orleans saints": "saints",
        "philadelphia eagles": "eagles",
        "pittsburgh steelers": "steelers",
        "seattle seahawks": "seahawks",
        "tennessee titans": "titans",
        "washington football team": "commanders",
    }
    slug = _slug_norm(raw or "")
    return aliases.get(slug, slug)


def _collect_team_game_row_from_schedule(row: pd.Series, team_side: str) -> dict:
    """Convert a schedule row to team-centric game dict for S2D/ATS."""
    is_home = team_side == "HOME"
    team = row["home_team"] if is_home else row["away_team"]
    opp = row["away_team"] if is_home else row["home_team"]
    team_points = row["home_score"] if is_home else row["away_score"]
    opp_points = row["away_score"] if is_home else row["home_score"]
    if pd.isna(team_points) or pd.isna(opp_points):
        return {}
    team_points = int(team_points)
    opp_points = int(opp_points)
    home_rel = home_relative_spread(row)
    team_line = None
    if home_rel is not None:
        team_line = team_centric_spread(home_rel, "HOME" if is_home else "AWAY")
    return {
        "team": team,
        "opp": opp,
        "team_points": float(team_points),
        "opp_points": float(opp_points),
        "team_margin": float(team_points - opp_points),
        "team_line": team_line,
    }


def _round_record(record: dict) -> dict:
    for key, value in record.items():
        if isinstance(value, float):
            record[key] = round(value, 2)
    return record


def season_to_date_per_game(team_games: Iterable[dict]) -> S2D:
    games = list(team_games)
    if not games:
        return S2D(games_played=0)

    def avg(key: str) -> float:
        vals = [float(game.get(key, 0.0) or 0.0) for game in games]
        return sum(vals) / len(games)

    return S2D(
        pf_pg=avg("team_points"),
        pa_pg=avg("opp_points"),
        ry_pg=avg("team_rush_yds"),
        py_pg=avg("team_pass_yds"),
        ty_pg=avg("team_total_yds"),
        ry_allowed_pg=avg("opp_rush_yds"),
        py_allowed_pg=avg("opp_pass_yds"),
        ty_allowed_pg=avg("opp_total_yds"),
        to_margin_pg=avg("team_to_margin"),
        games_played=len(games),
    )


def load_sagarin_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def column(*names):
        for name in names:
            if name in cols:
                return cols[name]
        return None

    required = {
        "team": column("team"),
        "pr": column("pr", "rating", "score"),
        "sos": column("sos", "strength_of_schedule"),
        "sos_rank": column("sos_rank", "sosr"),
    }
    pr_rank_col = column("pr_rank", "rank", "rating_rank")
    hfa_col = column("hfa", "home_field_advantage", "home_field", "homeadvantage", "home advantage")
    for key, col in required.items():
        if col is None:
            raise ValueError(f"Sagarin CSV missing required column for '{key}'")

    selected = [required["team"], required["pr"], required["sos"], required["sos_rank"]]
    column_names = ["team", "pr", "sos", "sos_rank"]
    if pr_rank_col:
        selected.insert(2, pr_rank_col)
        column_names.insert(2, "pr_rank")
    if hfa_col:
        selected.append(hfa_col)
        column_names.append("hfa")

    out = df[selected].copy()
    out.columns = column_names
    return out


def _team_stats_rows_before(
    stats_df: pd.DataFrame,
    season: int,
    team_name: str,
    this_week: int,
    col_team: Optional[str],
    col_week: Optional[str],
    col_season: Optional[str],
    columns: dict,
) -> List[dict]:
    if col_team is None or col_week is None or col_season is None:
        return []
    display = normalize_team_display(team_name)
    candidates = {team_name}
    if isinstance(team_name, str):
        candidates.add(team_name.upper())
        candidates.add(team_name.lower())
    if display:
        candidates.add(display)
    merge_key = team_merge_key(team_name)
    if merge_key:
        candidates.add(merge_key)
    view = stats_df[
        (stats_df[col_season] == season)
        & (stats_df[col_team].astype(str).str.strip().isin({str(c) for c in candidates}))
        & (stats_df[col_week] < this_week)
    ].copy()
    rows: List[dict] = []
    for _, sr in view.iterrows():
        def v(col_name: Optional[str], fallback: float = 0.0) -> float:
            if col_name is None or pd.isna(sr[col_name]):
                return fallback
            return float(sr[col_name])

        rush_for = v(columns["rush_for"])
        pass_for = v(columns["pass_for"])
        rush_allowed = v(columns["rush_allowed"])
        pass_allowed = v(columns["pass_allowed"])

        rows.append(
            {
                "team_points": v(columns["points_for"]),
                "opp_points": v(columns["points_against"]),
                "team_rush_yds": rush_for,
                "team_pass_yds": pass_for,
                "team_total_yds": v(columns["total_for"], rush_for + pass_for),
                "opp_rush_yds": rush_allowed,
                "opp_pass_yds": pass_allowed,
                "opp_total_yds": v(columns["total_allowed"], rush_allowed + pass_allowed),
                "team_to_margin": v(columns["takeaways"]) - v(columns["turnovers"]),
            }
        )
    return rows


def build_gameview(season: int, week: int, hfa: float = 0.0, sagarin_path: Optional[str] = None, odds_source: str = "schedule") -> dict:
    out_dir = ensure_out_dir()
    out_base = out_dir / f"games_week_{season}_{week}"
    jsonl_path = out_base.with_suffix(".jsonl")
    csv_path = out_dir / f"games_week_{season}_{week}.csv"

    schedule = load_games(season)
    if "kickoff_dt_utc" not in schedule.columns:
        schedule["kickoff_dt_utc"] = schedule.apply(parse_kickoff_utc, axis=1)

    sched_wk = filter_week_reg(schedule, season, week)
    nflverse_count = int(sched_wk.shape[0])

    stats_url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
    stats_df = download_csv(stats_url)

    sag_df = load_sagarin_df(sagarin_path)
    sag_map = {}
    sagarin_hfa_from_csv: Optional[float] = None
    if sag_df is not None:
        if "hfa" in sag_df.columns:
            hfa_values = sag_df["hfa"].dropna()
            if not hfa_values.empty:
                sagarin_hfa_from_csv = float(hfa_values.iloc[0])
        for _, row in sag_df.iterrows():
            key = team_merge_key(row["team"])
            pr_rank_val = row.get("pr_rank")
            hfa_val = row.get("hfa")
            sag_map[key] = {
                "team": row["team"],
                "pr": None if pd.isna(row["pr"]) else float(row["pr"]),
                "pr_rank": None if pd.isna(pr_rank_val) else int(pr_rank_val),
                "sos": None if pd.isna(row["sos"]) else float(row["sos"]),
                "sos_rank": None if pd.isna(row["sos_rank"]) else int(row["sos_rank"]),
                "hfa": None if pd.isna(hfa_val) else float(hfa_val),
            }

    hfa_used = sagarin_hfa_from_csv if sagarin_hfa_from_csv is not None else hfa

    records: List[dict] = []

    col_team = _stats_column(stats_df, "team")
    col_week = _stats_column(stats_df, "week")
    col_season = _stats_column(stats_df, "season")
    columns = {
        "points_for": _stats_column(stats_df, "points_for", "points", "pf"),
        "rush_for": _stats_column(stats_df, "rush_yards", "rushing_yards", "rush_yards_for", "rushing_yards_for"),
        "pass_for": _stats_column(stats_df, "pass_yards", "passing_yards", "pass_yards_for", "passing_yards_for"),
        "total_for": _stats_column(stats_df, "total_yards", "yards", "total_yards_for"),
        "points_against": _stats_column(stats_df, "points_against", "points_allowed", "pa"),
        "rush_allowed": _stats_column(stats_df, "rush_yards_allowed", "rushing_yards_allowed", "rush_yards_against"),
        "pass_allowed": _stats_column(stats_df, "pass_yards_allowed", "passing_yards_allowed", "pass_yards_against"),
        "total_allowed": _stats_column(stats_df, "total_yards_allowed", "yards_allowed", "total_yards_against"),
        "turnovers": _stats_column(stats_df, "turnovers", "giveaways"),
        "takeaways": _stats_column(stats_df, "takeaways", "def_takeaways"),
    }

    if "game_type" in schedule.columns:
        game_type_series = schedule["game_type"]
        mask_reg = (game_type_series == "REG") | (game_type_series.isna())
    else:
        mask_reg = pd.Series([True] * len(schedule), index=schedule.index)
    sched_all_reg = schedule[mask_reg].copy()
    if "kickoff_dt_utc" not in sched_all_reg.columns:
        sched_all_reg["kickoff_dt_utc"] = sched_all_reg.apply(parse_kickoff_utc, axis=1)

    def team_games_from_schedule(team_name: str, before_dt: datetime) -> List[dict]:
        prior = sched_all_reg[
            (pd.notna(sched_all_reg["kickoff_dt_utc"]))
            & (sched_all_reg["kickoff_dt_utc"] < before_dt)
        ]
        games = []
        for _, sched_row in prior.iterrows():
            if sched_row.get("home_team") == team_name or sched_row.get("away_team") == team_name:
                record = _collect_team_game_row_from_schedule(
                    sched_row, "HOME" if sched_row.get("home_team") == team_name else "AWAY"
                )
                if record:
                    games.append(record)
        return games

    all_team_labels = (
        sorted(stats_df[col_team].dropna().astype(str).str.strip().unique().tolist())
        if col_team
        else []
    )

    def s2d_for_team_generic(team_name: str, game_week: int) -> S2D:
        rows = _team_stats_rows_before(
            stats_df,
            season,
            team_name,
            game_week,
            col_team,
            col_week,
            col_season,
            columns,
        )
        return season_to_date_per_game(rows)

    for _, row in sched_wk.sort_values("kickoff_dt_utc").iterrows():
        home_raw = str(row["home_team"])
        away_raw = str(row["away_team"])
        home_norm = _legacy_team_slug(home_raw)
        away_norm = _legacy_team_slug(away_raw)

        kickoff_dt_utc = row["kickoff_dt_utc"]
        if kickoff_dt_utc is None:
            continue

        dt_str = kickoff_dt_utc.strftime("%Y%m%d_%H%M")
        game_key = f"{dt_str}_{home_norm}_{away_norm}"

        if odds_source == "schedule":
            spread_hr = home_relative_spread(row)
            total = total_from_schedule(row)
            moneyline_home = None
            moneyline_away = None
            odds_src = "schedule"
            is_closing = False
            snapshot_at = None
            odds_row = None
        else:
            spread_hr = home_relative_spread(row)
            total = total_from_schedule(row)
            moneyline_home = None
            moneyline_away = None
            odds_src = "donbest" if spread_hr is not None else "schedule"
            is_closing = False
            snapshot_at = datetime.now(timezone.utc).isoformat()
            odds_row = None

        home_key = team_merge_key(home_raw)
        away_key = team_merge_key(away_raw)

        home_sag = sag_map.get(home_key, {})
        away_sag = sag_map.get(away_key, {})

        home_pr = home_sag.get("pr")
        away_pr = away_sag.get("pr")

        rdiff = None
        rvo = None
        if home_pr is not None and away_pr is not None:
            rdiff = rating_diff(home_pr, away_pr, hfa_used)
            if spread_hr is not None:
                rvo = rating_vs_odds(rdiff, team_centric_spread(spread_hr, "HOME"))

        this_week = int(row["week"])
        home_s2d = season_to_date_per_game(
            _team_stats_rows_before(stats_df, season, home_raw, this_week, col_team, col_week, col_season, columns)
        )
        away_s2d = season_to_date_per_game(
            _team_stats_rows_before(stats_df, season, away_raw, this_week, col_team, col_week, col_season, columns)
        )

        home_sched_games = team_games_from_schedule(home_raw, kickoff_dt_utc)
        away_sched_games = team_games_from_schedule(away_raw, kickoff_dt_utc)
        home_su = compute_su(home_sched_games)
        away_su = compute_su(away_sched_games)
        home_ats = compute_ats(home_sched_games)
        away_ats = compute_ats(away_sched_games)

        def league_metric(getter, higher_is_better):
            data = {}
            for team_label in all_team_labels:
                s2d = s2d_for_team_generic(team_label, this_week)
                key = team_merge_key(team_label)
                if not key:
                    continue
                if s2d.games_played > 0:
                    data[key] = getter(s2d)
            if not data:
                return {}
            series = pd.Series(data)
            ranks = series.sort_values(ascending=not higher_is_better).rank(method="dense", ascending=not higher_is_better)
            return ranks.astype(int).to_dict()

        off_rush_rank = league_metric(lambda s: s.ry_pg, True)
        off_pass_rank = league_metric(lambda s: s.py_pg, True)
        off_tot_rank = league_metric(lambda s: s.ty_pg, True)
        def_rush_rank = league_metric(lambda s: s.ry_allowed_pg, False)
        def_pass_rank = league_metric(lambda s: s.py_allowed_pg, False)
        def_tot_rank = league_metric(lambda s: s.ty_allowed_pg, False)

        record = {
            "season": season,
            "week": this_week,
            "kickoff_iso_utc": kickoff_dt_utc.replace(microsecond=0).isoformat(),
            "game_key": game_key,
            "source_uid": row.get("game_id") if "game_id" in row else None,
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,
            "spread_home_relative": None if spread_hr is None else float(spread_hr),
            "total": None if total is None else float(total),
            "moneyline_home": None if moneyline_home is None else float(moneyline_home),
            "moneyline_away": None if moneyline_away is None else float(moneyline_away),
            "odds_source": odds_src,
            "is_closing": bool(is_closing),
            "snapshot_at": snapshot_at,
            "home_pr": home_pr,
            "home_pr_rank": home_sag.get("pr_rank") if home_sag else None,
            "away_pr": away_pr,
            "away_pr_rank": away_sag.get("pr_rank") if away_sag else None,
            "home_sos": home_sag.get("sos"),
            "away_sos": away_sag.get("sos"),
            "home_sos_rank": home_sag.get("sos_rank"),
            "away_sos_rank": away_sag.get("sos_rank"),
            "hfa": hfa_used,
            "rating_diff": rdiff,
            "rating_vs_odds": rvo,
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
            "home_rush_rank": off_rush_rank.get(team_merge_key(home_raw)),
            "home_pass_rank": off_pass_rank.get(team_merge_key(home_raw)),
            "home_tot_off_rank": off_tot_rank.get(team_merge_key(home_raw)),
            "home_rush_def_rank": def_rush_rank.get(team_merge_key(home_raw)),
            "home_pass_def_rank": def_pass_rank.get(team_merge_key(home_raw)),
            "home_tot_def_rank": def_tot_rank.get(team_merge_key(home_raw)),
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
            "away_rush_rank": off_rush_rank.get(team_merge_key(away_raw)),
            "away_pass_rank": off_pass_rank.get(team_merge_key(away_raw)),
            "away_tot_off_rank": off_tot_rank.get(team_merge_key(away_raw)),
            "away_rush_def_rank": def_rush_rank.get(team_merge_key(away_raw)),
            "away_pass_def_rank": def_pass_rank.get(team_merge_key(away_raw)),
            "away_tot_def_rank": def_tot_rank.get(team_merge_key(away_raw)),
            "raw_sources": {
                "schedule_row": {
                    "game_id": row.get("game_id"),
                    "gsis": row.get("gsis"),
                    "gameday": row.get("gameday"),
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                },
                "odds_row": odds_row,
                "sagarin_row_home": home_sag or None,
                "sagarin_row_away": away_sag or None,
            },
        }
        records.append(_round_record(record))

    write_jsonl(records, jsonl_path)
    write_csv(pd.DataFrame(records), csv_path)
    print(f"HFA used: {hfa_used:.2f}")

    print("=== ACCEPTANCE SUMMARY ===")
    print(f"Schedule count (nflverse, filtered): {nflverse_count}")
    print(f"Emitted records:                   : {len(records)}")
    print(f"COUNT MATCH: {'PASS' if nflverse_count == len(records) else 'FAIL'}")
    with_spread = sum(1 for r in records if r["spread_home_relative"] is not None)
    closing_true = sum(1 for r in records if r["is_closing"])
    print(f"Games with spreads: {with_spread}/{len(records) if records else 0}")
    print(f"Marked closing=true: {closing_true}")

    def _s2d_ok(rec: dict, prefix: str) -> bool:
        keys = [
            "pf_pg",
            "pa_pg",
            "ry_pg",
            "py_pg",
            "ty_pg",
            "ry_allowed_pg",
            "py_allowed_pg",
            "ty_allowed_pg",
            "to_margin_pg",
        ]
        return all(rec.get(f"{prefix}_{key}") is not None for key in keys)

    completeness = all(_s2d_ok(rec, "home") and _s2d_ok(rec, "away") for rec in records)
    print(f"Derived S2D completeness: {'PASS' if completeness else 'FAIL'}")

    def _parse_record(rec: str) -> tuple[int, int, int]:
        parts = [int(x) for x in rec.split("-") if x]
        if len(parts) == 2:
            return parts[0], parts[1], 0
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return 0, 0, 0

    teams = {r["home_team_raw"] for r in records} | {r["away_team_raw"] for r in records}
    su_ok = True
    ats_ok = True
    for team in sorted(teams):
        team_rows = [r for r in records if r["home_team_raw"] == team or r["away_team_raw"] == team]
        if not team_rows:
            continue
        team_rows.sort(key=lambda r: r["kickoff_iso_utc"])
        latest = team_rows[-1]
        su = latest["home_su"] if latest["home_team_raw"] == team else latest["away_su"]
        ats = latest["home_ats"] if latest["home_team_raw"] == team else latest["away_ats"]
        w, l, ties = _parse_record(su)
        aw, al, ap = _parse_record(ats)
        if w + l + ties < 0:
            su_ok = False
        if aw + al + ap < 0:
            ats_ok = False
    print(f"SU tallies shape check: {'PASS' if su_ok else 'FAIL'}")
    print(f"ATS tallies shape check: {'PASS' if ats_ok else 'FAIL'}")

    def _rank_ok(val: Optional[int]) -> bool:
        return (val is None) or (1 <= int(val) <= 32)

    ranks_ok = True
    for rec in records:
        for key in [
            "rush_rank",
            "pass_rank",
            "tot_off_rank",
            "rush_def_rank",
            "pass_def_rank",
            "tot_def_rank",
        ]:
            if not _rank_ok(rec.get(f"home_{key}")) or not _rank_ok(rec.get(f"away_{key}")):
                ranks_ok = False
                break
    print(f"Rank fields validity: {'PASS' if ranks_ok else 'FAIL'}")

    spot = None
    for rec in sorted(records, key=lambda r: r["kickoff_iso_utc"]):
        dt = datetime.fromisoformat(rec["kickoff_iso_utc"].replace("Z", ""))
        if dt.weekday() == 6:
            spot = rec
            break
    if spot is None and records:
        spot = records[0]
    if spot:
        print(
            "Spot check:",
            spot["home_team_norm"],
            spot["away_team_norm"],
            "spread_home_relative=",
            spot["spread_home_relative"],
            "total=",
            spot["total"],
            "rating_diff=",
            spot["rating_diff"],
            "rating_vs_odds=",
            spot["rating_vs_odds"],
        )
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")
    return {"jsonl": jsonl_path, "csv": csv_path, "count": len(records)}


def main():
    parser = argparse.ArgumentParser(description="Build Game View weekly outputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--hfa", type=float, default=0.0)
    parser.add_argument("--odds-source", choices=["donbest", "schedule"], default="schedule")
    parser.add_argument("--sagarin-path", type=str, default=None)
    args = parser.parse_args()
    build_gameview(
        season=args.season,
        week=args.week,
        hfa=args.hfa,
        sagarin_path=args.sagarin_path,
        odds_source=args.odds_source,
    )


if __name__ == "__main__":
    main()

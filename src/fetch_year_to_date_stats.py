"""Generate in-season league metrics table up to a given NFL week.

Purpose:
    Reproduce the legacy league_metrics_{season}_{week}.csv output using
    nflverse stats and schedules with minimal behavioural changes.
Inputs:
    Season/week integers, optional TeamRankings turnover CSV override.
Outputs:
    /out/league_metrics_{season}_{week}.csv
Source(s) of truth:
    nflverse stats_team_week release + schedules, TeamRankings turnover margin.
Example:
    python -m src.fetch_year_to_date_stats --season 2025 --week 6
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.common.io_utils import download_csv, week_out_dir, write_csv
from src.common.metrics import compute_ats, compute_su, dense_rank
from src.common.team_names import normalize_team_display, team_merge_key
from src.fetch_games import filter_week_reg, load_games, parse_kickoff_utc

TEAMRANKINGS_TURNOVER_URL = "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game"
STATS_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"


def load_turnover_margin_per_game(csv_path: Optional[str]) -> Dict[str, float]:
    """Fetch Turnover Margin per Game from TeamRankings."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        tables = pd.read_html(TEAMRANKINGS_TURNOVER_URL)
        df = None
        for table in tables:
            if "Team" not in table.columns:
                continue
            numeric_cols = [c for c in table.columns if c != "Team" and pd.api.types.is_numeric_dtype(table[c])]
            if not numeric_cols:
                continue
            year_cols = [c for c in numeric_cols if re.fullmatch(r"\d{4}", str(c))]
            if year_cols:
                keep_col = sorted(year_cols, reverse=True)[0]
            else:
                filtered = [c for c in numeric_cols if str(c).strip().lower() not in {"rank"}]
                keep_col = filtered[0] if filtered else numeric_cols[0]
            df = table[["Team", keep_col]].rename(columns={keep_col: "TO_pg"})
            break
        if df is None:
            return {}
    if "TO_pg" not in df.columns:
        for column in df.columns:
            if column != "Team" and pd.api.types.is_numeric_dtype(df[column]):
                df = df.rename(columns={column: "TO_pg"})
                break
    df["Team"] = df["Team"].astype(str).str.strip()
    df["TO_pg"] = pd.to_numeric(df["TO_pg"], errors="coerce")
    return {
        normalize_team_display(team): float(value)
        for team, value in df[["Team", "TO_pg"]].itertuples(index=False, name=None)
        if pd.notna(value)
    }


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = cand.lower()
        if key in cols:
            return cols[key]
    norm = {c.lower().replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    return None


def _build_opponent_map(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = []
    sched_season = schedule[schedule["season"] == season]
    for _, row in sched_season.iterrows():
        if pd.isna(row.get("week")) or pd.isna(row.get("home_team")) or pd.isna(row.get("away_team")):
            continue
        week = int(row["week"])
        rows.append({"team": str(row["home_team"]), "week": week, "opponent": str(row["away_team"])})
        rows.append({"team": str(row["away_team"]), "week": week, "opponent": str(row["home_team"])})
    return pd.DataFrame(rows)


def _collect_prior_games(schedule: pd.DataFrame, cutoff_dt: pd.Timestamp) -> pd.DataFrame:
    prior = schedule[
        (pd.notna(schedule["kickoff_dt_utc"]))
        & (schedule["kickoff_dt_utc"] < cutoff_dt)
    ].copy()
    return prior


def _team_games(prior: pd.DataFrame, team_abbr: str) -> list[dict]:
    records = []
    mask = (prior["home_team"] == team_abbr) | (prior["away_team"] == team_abbr)
    for _, row in prior.loc[mask].iterrows():
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        side = "HOME" if row["home_team"] == team_abbr else "AWAY"
        team_points = int(row["home_score"] if side == "HOME" else row["away_score"])
        opp_points = int(row["away_score"] if side == "HOME" else row["home_score"])
        spread = row.get("spread_line")
        if pd.isna(spread):
            team_line = None
        else:
            team_line = -float(spread) if side == "HOME" else float(spread)
        records.append(
            {
                "team_points": team_points,
                "opp_points": opp_points,
                "team_margin": team_points - opp_points,
                "team_line": team_line,
            }
        )
    return records


def generate_league_metrics(
    season: int,
    week: int,
    teamrankings_csv: Optional[str] = None,
) -> pd.DataFrame:
    stats_df = download_csv(STATS_URL_TEMPLATE.format(season=season))
    schedule = load_games(season)
    if "kickoff_dt_utc" not in schedule.columns:
        schedule["kickoff_dt_utc"] = schedule.apply(parse_kickoff_utc, axis=1)
    week_games = filter_week_reg(schedule, season, week)
    if week_games.empty:
        raise RuntimeError(f"No regular season games found for season {season} week {week}.")

    col_team = _find_col(stats_df, ["team", "recent_team", "team_abbr"])
    col_week = _find_col(stats_df, ["week"])
    if not col_team or not col_week:
        raise RuntimeError("stats_team_week missing required team/week columns.")

    col_opp = _find_col(stats_df, ["opponent", "opp"])
    col_off_ry = _find_col(stats_df, ["rushing_yards", "rush_yards", "offense_rushing_yards"])
    col_off_py = _find_col(stats_df, ["net_passing_yards", "passing_yards", "pass_yards"])
    if col_off_ry is None or col_off_py is None:
        raise RuntimeError("stats_team_week missing offensive rushing/passing columns.")

    col_def_ry = _find_col(
        stats_df,
        [
            "opponent_rushing_yards",
            "rushing_yards_against",
            "rushing_yards_allowed",
            "defense_rushing_yards_allowed",
            "defense_rushing_yards",
        ],
    )
    col_def_py = _find_col(
        stats_df,
        [
            "opponent_net_passing_yards",
            "opponent_passing_yards",
            "net_passing_yards_against",
            "passing_yards_against",
            "passing_yards_allowed",
            "defense_passing_yards_allowed",
            "defense_passing_yards",
        ],
    )
    col_turnovers = _find_col(stats_df, ["turnovers", "giveaways"])
    col_takeaways = _find_col(stats_df, ["takeaways", "defensive_takeaways", "opponent_turnovers"])

    if col_opp is None:
        opp_df = _build_opponent_map(schedule, season)
        stats_df = stats_df.merge(
            opp_df,
            left_on=[col_team, col_week],
            right_on=["team", "week"],
            how="left",
        )
        col_opp = "opponent"

    base_df = stats_df.copy()

    need_mirror_def = (col_def_ry is None) or (col_def_py is None) or (col_takeaways is None)
    if need_mirror_def:
        opp_cols = [c for c in [col_off_ry, col_off_py, col_turnovers] if c]
        opp_side = base_df[[col_opp, col_week] + opp_cols].copy()
        rename_map = {
            col_opp: "opp_team",
            col_off_ry: "opp_off_ry",
            col_off_py: "opp_off_py",
            (col_turnovers if col_turnovers else "turnovers"): "opp_turnovers",
            col_week: "week",
        }
        opp_side = opp_side.rename(columns=rename_map)
        left = base_df[[col_team, col_week, col_opp]].rename(
            columns={col_team: "team", col_week: "week", col_opp: "opponent"}
        )
        mirror = left.merge(
            opp_side,
            left_on=["opponent", "week"],
            right_on=["opp_team", "week"],
            how="left",
        )
        ry_allowed_series = (
            base_df[col_def_ry]
            if col_def_ry is not None
            else mirror.get("opp_off_ry", pd.Series(np.nan, index=base_df.index))
        )
        py_allowed_series = (
            base_df[col_def_py]
            if col_def_py is not None
            else mirror.get("opp_off_py", pd.Series(np.nan, index=base_df.index))
        )
        takeaways_series = (
            base_df[col_takeaways]
            if col_takeaways is not None
            else mirror.get("opp_turnovers", pd.Series(np.nan, index=base_df.index))
        )
    else:
        ry_allowed_series = base_df[col_def_ry]
        py_allowed_series = base_df[col_def_py]
        takeaways_series = base_df[col_takeaways] if col_takeaways else pd.Series(np.nan, index=base_df.index)

    work = pd.DataFrame(
        {
            "team": base_df[col_team].astype(str),
            "week": base_df[col_week],
            "ry_off": base_df[col_off_ry],
            "py_off": base_df[col_off_py],
            "ry_allowed": ry_allowed_series,
            "py_allowed": py_allowed_series,
            "turnovers": base_df[col_turnovers] if col_turnovers else pd.NA,
            "takeaways": takeaways_series,
        }
    )

    teams = sorted(work["team"].dropna().astype(str).unique().tolist())

    def _s2d(team_label: str):
        subset = work[(work["team"] == team_label) & (work["week"] < week)]
        gp = int(subset.shape[0])
        if gp == 0:
            return dict(gp=0, ry=None, py=None, ty=None, ry_a=None, py_a=None, ty_a=None, to_pg=None)
        ry = float(subset["ry_off"].fillna(0).sum())
        py = float(subset["py_off"].fillna(0).sum())
        ty = ry + py
        ry_a = float(subset["ry_allowed"].fillna(0).sum())
        py_a = float(subset["py_allowed"].fillna(0).sum())
        ty_a = ry_a + py_a
        give = float(subset["turnovers"].fillna(0).sum()) if "turnovers" in subset else 0.0
        take = float(subset["takeaways"].fillna(0).sum()) if "takeaways" in subset else 0.0
        to_pg = (take - give) / gp if gp > 0 else None
        return dict(gp=gp, ry=ry / gp, py=py / gp, ty=ty / gp, ry_a=ry_a / gp, py_a=py_a / gp, ty_a=ty_a / gp, to_pg=to_pg)

    per_team = {team: _s2d(team) for team in teams}

    def build_rank(metric_key: str, higher_is_better: bool):
        data = {team_merge_key(team): stats[metric_key] for team, stats in per_team.items() if stats["gp"] > 0}
        if not data:
            return {}
        series = pd.Series(data)
        ranked = dense_rank(series, higher_is_better=higher_is_better)
        return ranked.astype(int).to_dict()

    off_ry_rank = build_rank("ry", True)
    off_py_rank = build_rank("py", True)
    off_ty_rank = build_rank("ty", True)
    def_ry_rank = build_rank("ry_a", False)
    def_py_rank = build_rank("py_a", False)
    def_ty_rank = build_rank("ty_a", False)

    week_min_kick = week_games["kickoff_dt_utc"].min()
    prior_games = _collect_prior_games(schedule, week_min_kick)
    pfpa_su_ats = {}
    for team in teams:
        games = _team_games(prior_games, team)
        pf = sum(g["team_points"] for g in games)
        pa = sum(g["opp_points"] for g in games)
        su = compute_su(games)
        ats = compute_ats(games)
        pfpa_su_ats[team] = (pf, pa, su, ats)

    turnover_map = load_turnover_margin_per_game(teamrankings_csv)
    to_pg_map = {}
    for team in teams:
        display = normalize_team_display(team)
        val = turnover_map.get(display)
        if val is None and display:
            val = turnover_map.get(display.title())
        if val is not None:
            to_pg_map[team] = float(val)

    print(f"TO merge: found TeamRankings values for {len(to_pg_map)}/32 teams.")

    rows = []
    for team in teams:
        display = normalize_team_display(team) or team
        stats = per_team[team]
        pf, pa, su, ats = pfpa_su_ats[team]
        key = team_merge_key(team)
        row = {
            "Team": display,
            "RY(O)": "" if stats["ry"] is None else f"{stats['ry']:.1f}",
            "R(O)_RY": "" if stats["gp"] == 0 else str(off_ry_rank.get(key, "")),
            "PY(O)": "" if stats["py"] is None else f"{stats['py']:.1f}",
            "R(O)_PY": "" if stats["gp"] == 0 else str(off_py_rank.get(key, "")),
            "TY(O)": "" if stats["ty"] is None else f"{stats['ty']:.1f}",
            "R(O)_TY": "" if stats["gp"] == 0 else str(off_ty_rank.get(key, "")),
            "RY(D)": "" if stats["ry_a"] is None else f"{stats['ry_a']:.1f}",
            "R(D)_RY": "" if stats["gp"] == 0 else str(def_ry_rank.get(key, "")),
            "PY(D)": "" if stats["py_a"] is None else f"{stats['py_a']:.1f}",
            "R(D)_PY": "" if stats["gp"] == 0 else str(def_py_rank.get(key, "")),
            "TY(D)": "" if stats["ty_a"] is None else f"{stats['ty_a']:.1f}",
            "R(D)_TY": "" if stats["gp"] == 0 else str(def_ty_rank.get(key, "")),
            "TO": "" if team not in to_pg_map and stats["to_pg"] is None else f"{to_pg_map.get(team, stats['to_pg']):.1f}",
            "PF": pf,
            "PA": pa,
            "SU": su,
            "ATS": ats,
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = df_out[
        [
            "Team",
            "RY(O)",
            "R(O)_RY",
            "PY(O)",
            "R(O)_PY",
            "TY(O)",
            "R(O)_TY",
            "RY(D)",
            "R(D)_RY",
            "PY(D)",
            "R(D)_PY",
            "TY(D)",
            "R(D)_TY",
            "TO",
            "PF",
            "PA",
            "SU",
            "ATS",
        ]
    ]
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Generate league metrics CSV through a given week.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--teamrankings-csv", type=str, default=None)
    args = parser.parse_args()
    df = generate_league_metrics(args.season, args.week, args.teamrankings_csv)
    out_dir = week_out_dir(args.season, args.week)
    output_path = out_dir / f"league_metrics_{args.season}_{args.week}.csv"
    write_csv(df, output_path)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

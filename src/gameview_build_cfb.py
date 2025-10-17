"""Stub CFB game view builder that writes shapeful empty outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl

GAMEVIEW_COLUMNS: List[str] = [
    "season",
    "week",
    "kickoff_iso_utc",
    "game_key",
    "source_uid",
    "league",
    "home_team_raw",
    "home_team_norm",
    "away_team_raw",
    "away_team_norm",
    "spread_home_relative",
    "total",
    "moneyline_home",
    "moneyline_away",
    "odds_source",
    "is_closing",
    "snapshot_at",
    "home_pr",
    "home_pr_rank",
    "away_pr",
    "away_pr_rank",
    "home_sos",
    "away_sos",
    "home_sos_rank",
    "away_sos_rank",
    "hfa",
    "rating_diff",
    "rating_vs_odds",
    "favored_side",
    "spread_favored_team",
    "rating_diff_favored_team",
    "rating_vs_odds_favored_team",
    "home_pf_pg",
    "home_pa_pg",
    "home_ry_pg",
    "home_py_pg",
    "home_ty_pg",
    "home_ry_allowed_pg",
    "home_py_allowed_pg",
    "home_ty_allowed_pg",
    "home_to_margin_pg",
    "home_su",
    "home_ats",
    "home_rush_rank",
    "home_pass_rank",
    "home_tot_off_rank",
    "home_rush_def_rank",
    "home_pass_def_rank",
    "home_tot_def_rank",
    "away_pf_pg",
    "away_pa_pg",
    "away_ry_pg",
    "away_py_pg",
    "away_ty_pg",
    "away_ry_allowed_pg",
    "away_py_allowed_pg",
    "away_ty_allowed_pg",
    "away_to_margin_pg",
    "away_su",
    "away_ats",
    "away_rush_rank",
    "away_pass_rank",
    "away_tot_off_rank",
    "away_rush_def_rank",
    "away_pass_def_rank",
    "away_tot_def_rank",
    "raw_sources",
]


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_empty_gameview(season: int, week: int) -> tuple[Path, Path]:
    out_dir = cfb_week_dir(season, week)
    jsonl_path = out_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = out_dir / f"games_week_{season}_{week}.csv"

    write_jsonl([], jsonl_path)
    df = pd.DataFrame(columns=GAMEVIEW_COLUMNS)
    write_csv(df, csv_path)
    return jsonl_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB game view builder.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: gameview_build_cfb season={args.season} week={args.week}")
    jsonl_path, csv_path = write_empty_gameview(args.season, args.week)
    print(f"PASS: wrote shapeful empty Game View JSONL to {jsonl_path}")
    print(f"PASS: wrote shapeful empty Game View CSV to {csv_path}")
    print("Info: future implementation will join CFB schedules, ratings, metrics, and odds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stub CFB game view builder that writes shapeful empty outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl
from src.fetch_games_cfb import get_schedule_df

DEFAULT_COLUMNS: List[str] = [
    "season",
    "week",
    "kickoff_iso_utc",
    "game_key",
    "source_uid",
    "home_team_raw",
    "away_team_raw",
    "home_team_norm",
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


def infer_columns(season: int, week: int) -> List[str]:
    nfl_dir = ensure_out_dir() / f"{season}_week{week}"
    nfl_csv = nfl_dir / f"games_week_{season}_{week}.csv"
    if nfl_csv.exists():
        cols = pd.read_csv(nfl_csv, nrows=0).columns.tolist()
        if cols:
            return cols
    return DEFAULT_COLUMNS.copy()


def build_records(season: int, week: int) -> List[Dict[str, Any]]:
    schedule = get_schedule_df(season)
    if schedule.empty:
        return []
    weekly = schedule[pd.to_numeric(schedule.get("week"), errors="coerce") == week].copy()
    if weekly.empty:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in weekly.iterrows():
        home_raw = row.get("home_team")
        away_raw = row.get("away_team")
        record: Dict[str, Any] = {
            "season": row.get("season"),
            "week": row.get("week"),
            "kickoff_iso_utc": row.get("kickoff_iso_utc"),
            "game_key": row.get("game_key"),
            "source_uid": row.get("id"),
            "home_team_raw": home_raw,
            "home_team_norm": row.get("home_team_norm"),
            "away_team_raw": away_raw,
            "away_team_norm": row.get("away_team_norm"),
            "spread_home_relative": None,
            "total": None,
            "moneyline_home": None,
            "moneyline_away": None,
            "odds_source": None,
            "is_closing": False,
            "snapshot_at": None,
            "home_pr": None,
            "home_pr_rank": None,
            "away_pr": None,
            "away_pr_rank": None,
            "home_sos": None,
            "away_sos": None,
            "home_sos_rank": None,
            "away_sos_rank": None,
            "hfa": None,
            "rating_diff": None,
            "rating_vs_odds": None,
            "favored_side": None,
            "spread_favored_team": None,
            "rating_diff_favored_team": None,
            "rating_vs_odds_favored_team": None,
            "home_pf_pg": None,
            "home_pa_pg": None,
            "home_ry_pg": None,
            "home_py_pg": None,
            "home_ty_pg": None,
            "home_ry_allowed_pg": None,
            "home_py_allowed_pg": None,
            "home_ty_allowed_pg": None,
            "home_to_margin_pg": None,
            "home_su": None,
            "home_ats": None,
            "home_rush_rank": None,
            "home_pass_rank": None,
            "home_tot_off_rank": None,
            "home_rush_def_rank": None,
            "home_pass_def_rank": None,
            "home_tot_def_rank": None,
            "away_pf_pg": None,
            "away_pa_pg": None,
            "away_ry_pg": None,
            "away_py_pg": None,
            "away_ty_pg": None,
            "away_ry_allowed_pg": None,
            "away_py_allowed_pg": None,
            "away_ty_allowed_pg": None,
            "away_to_margin_pg": None,
            "away_su": None,
            "away_ats": None,
            "away_rush_rank": None,
            "away_pass_rank": None,
            "away_tot_off_rank": None,
            "away_rush_def_rank": None,
            "away_pass_def_rank": None,
            "away_tot_def_rank": None,
            "raw_sources": {
                "schedule_row": {
                    "id": row.get("id"),
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "kickoff_iso_utc": row.get("kickoff_iso_utc"),
                    "home_team": home_raw,
                    "away_team": away_raw,
                    "venue": row.get("venue"),
                    "conference_game": row.get("conference_game"),
                }
            },
        }
        records.append(record)
    return records


def write_gameview(season: int, week: int, records: List[Dict[str, Any]]) -> tuple[Path, Path]:
    out_dir = cfb_week_dir(season, week)
    jsonl_path = out_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = out_dir / f"games_week_{season}_{week}.csv"

    columns = infer_columns(season, week)
    df = pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)
    write_jsonl(records, jsonl_path)
    write_csv(df, csv_path)
    return jsonl_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB game view builder.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: gameview_build_cfb season={args.season} week={args.week}")
    records = build_records(args.season, args.week)
    jsonl_path, csv_path = write_gameview(args.season, args.week, records)
    print(f"PASS: wrote Game View JSONL to {jsonl_path} (rows={len(records)})")
    print(f"PASS: wrote Game View CSV to {csv_path} (rows={len(records)})")
    favored_present = sum(1 for rec in records if rec.get("favored_side"))
    print(f"Favorite metrics coverage: {favored_present}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

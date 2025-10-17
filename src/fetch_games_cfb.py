"""CFB schedule ingestion (CFBD API) without producing games_week outputs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.common.cfb_source import fetch_cfbd_week_games
from src.common.io_utils import ensure_out_dir, read_env
from src.common.team_names_cfb import normalize_team_name_cfb, team_merge_key_cfb

SCHEDULE_COLUMNS: List[str] = [
    "season",
    "week",
    "game_id",
    "game_key",
    "kickoff_iso_utc",
    "home_team_raw",
    "home_team_norm",
    "home_team_key",
    "away_team_raw",
    "away_team_norm",
    "away_team_key",
    "venue",
    "conference_game",
]


def ensure_week_dirs(season: int, week: int) -> Path:
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "game_schedules").mkdir(parents=True, exist_ok=True)
    return out_dir


def sanitize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def build_game_key(kickoff_iso: Optional[str], home_norm: str, away_norm: str, game_id: Optional[str]) -> str:
    home_key = sanitize_key(home_norm)
    away_key = sanitize_key(away_norm)
    if kickoff_iso:
        dt_part = kickoff_iso.replace(":", "").replace("-", "")
        return f"{dt_part}_{away_key}_{home_key}"
    if game_id:
        return f"{game_id}_{away_key}_{home_key}"
    return f"{home_key}_{away_key}"


def normalize_schedule(records: List[dict], season: int, week: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    df = pd.DataFrame(records)
    df["season"] = season
    df["week"] = week
    def column_series(names: List[str]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([None] * len(df), index=df.index)

    start_series = column_series(["start_date", "startDate", "kickoff", "start_time"])
    kickoff = pd.to_datetime(start_series, utc=True, errors="coerce")
    df["kickoff_iso_utc"] = kickoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["home_team_raw"] = column_series(["home_team", "homeTeam"]).astype(str)
    df["away_team_raw"] = column_series(["away_team", "awayTeam"]).astype(str)
    df["home_team_norm"] = df["home_team_raw"].map(normalize_team_name_cfb)
    df["away_team_norm"] = df["away_team_raw"].map(normalize_team_name_cfb)
    df["home_team_key"] = df["home_team_norm"].map(team_merge_key_cfb)
    df["away_team_key"] = df["away_team_norm"].map(team_merge_key_cfb)
    df["venue"] = column_series(["venue", "venue_name"]).astype(str)
    conf_series = column_series(["conference_game", "conferenceGame"])
    df["conference_game"] = conf_series.fillna(False).astype(bool)
    df["game_key"] = df.apply(
        lambda row: build_game_key(
            row.get("kickoff_iso_utc"),
            row.get("home_team_norm", "") or str(row.get("home_team_raw", "")),
            row.get("away_team_norm", "") or str(row.get("away_team_raw", "")),
            str(row.get("id") or row.get("game_id") or ""),
        ),
        axis=1,
    )
    df["game_id"] = column_series(["id", "game_id"]).astype(str)
    return df[SCHEDULE_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFB weekly schedule from CFBD API.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"CFB schedule fetch: season={args.season} week={args.week}")
    out_dir = ensure_week_dirs(args.season, args.week)

    env = read_env(["CFBD_API_KEY"])
    api_key = env.get("CFBD_API_KEY")
    if not api_key:
        print("CFB schedule unavailable (CFBD_API_KEY missing). Skipping.")
        return 0

    try:
        records = fetch_cfbd_week_games(args.season, args.week, api_key) or []
    except Exception as exc:
        print(f"CFB schedule unavailable (error: {exc}). Skipping.")
        return 0

    if not records:
        print("CFB schedule API returned no games. Nothing to write.")
        return 0

    schedule_df = normalize_schedule(records, args.season, args.week)
    if schedule_df.empty:
        print("CFB schedule normalization produced no rows.")
        return 0

    schedule_path = out_dir / "_schedule_norm.csv"
    schedule_df.to_csv(schedule_path, index=False)
    print(f"PASS: wrote normalized schedule to {schedule_path} (rows={len(schedule_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

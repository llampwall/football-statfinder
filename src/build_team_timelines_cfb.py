"""Build CFB game timelines (YTD + prior-season) for Game View sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.common.io_utils import ensure_out_dir
from src.common.team_names_cfb import team_merge_key_cfb
from src.schedule_master_cfb import (
    ensure_weeks_present as ensure_schedule_master,
    load_master as load_schedule_master,
)


SIDE_COLUMNS = [
    "season",
    "week",
    "date",
    "opp",
    "site",
    "pf",
    "pa",
    "result",
    "pr",
    "pr_rank",
    "sos",
    "sos_rank",
    "opp_pr",
    "opp_pr_rank",
    "opp_sos",
    "opp_sos_rank",
]


def _cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    (base / "game_schedules").mkdir(parents=True, exist_ok=True)
    return base


def _dateutc(iso: str | None) -> str | None:
    return iso.split("T")[0] if isinstance(iso, str) and "T" in iso else None


def _result(pf, pa) -> str | None:
    if pd.isna(pf) or pd.isna(pa):
        return None
    if pf > pa:
        return "W"
    if pf < pa:
        return "L"
    return "T"


def _slice_team(
    schedule_df: pd.DataFrame,
    team_key: str,
    season: int,
    cutoff_iso: str | None = None,
    cutoff_week: int | None = None,
) -> pd.DataFrame:
    work = schedule_df[schedule_df["season"] == season].copy()
    work = work[
        (work["home_team_key"] == team_key)
        | (work["away_team_key"] == team_key)
    ]
    if cutoff_iso is not None:
        work = work[work["kickoff_iso_utc"] < cutoff_iso]
    if cutoff_week is not None:
        work = work[work["week"] < cutoff_week]

    rows = []
    for _, row in work.iterrows():
        is_home = row["home_team_key"] == team_key
        opp_norm = row["away_team_norm"] if is_home else row["home_team_norm"]
        opp_key = row["away_team_key"] if is_home else row["home_team_key"]
        pf = row["home_score"] if is_home else row["away_score"]
        pa = row["away_score"] if is_home else row["home_score"]
        rows.append(
            {
                "season": row["season"],
                "week": row.get("week"),
                "date": _dateutc(row.get("kickoff_iso_utc")),
                "opp": opp_norm,
                "opp_key": opp_key,
                "site": "H" if is_home else "A",
                "pf": None if pd.isna(pf) else int(pf),
                "pa": None if pd.isna(pa) else int(pa),
                "result": _result(pf, pa),
                "team_key": team_key,
            }
        )
    return pd.DataFrame(rows)


def _augment_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in SIDE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def _project_rows(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    df = df.sort_values(["season", "week", "date"], na_position="last").copy()
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "season": int(row.season) if pd.notna(row.season) else None,
                "week": int(row.week) if pd.notna(row.week) else None,
                "date": row.date,
                "opp": row.opp,
                "site": row.site,
                "pf": row.pf,
                "pa": row.pa,
                "result": row.result,
                "pr": row.pr,
                "pr_rank": row.pr_rank,
                "sos": row.sos,
                "sos_rank": row.sos_rank,
                "opp_pr": row.opp_pr,
                "opp_pr_rank": row.opp_pr_rank,
                "opp_sos": row.opp_sos,
                "opp_sos_rank": row.opp_sos_rank,
            }
        )
    return records


def _load_games_json(games_path: Path) -> pd.DataFrame:
    if not games_path.exists():
        raise FileNotFoundError(f"games jsonl missing at {games_path}")
    return pd.read_json(games_path, lines=True)


def build_sidecars(season: int, week: int) -> dict:
    out_dir = _cfb_week_dir(season, week)
    games_path = out_dir / f"games_week_{season}_{week}.jsonl"
    sidecar_dir = out_dir / "game_schedules"

    games_df = _load_games_json(games_path)
    games_total = len(games_df)

    ensure_schedule_master([season, season - 1])
    schedule_master = load_schedule_master()
    schedule_master = schedule_master[
        (schedule_master["league"].astype(str).str.upper() == "CFB")
        & (schedule_master["game_type"].astype(str).str.upper() == "REG")
        & (schedule_master["season"].isin([season, season - 1]))
    ].copy()
    if schedule_master.empty:
        raise RuntimeError("CFB schedule master is empty for requested seasons.")

    schedule_master["week"] = pd.to_numeric(schedule_master["week"], errors="coerce").astype("Int64")
    schedule_master["kickoff_iso_utc"] = schedule_master["kickoff_iso_utc"].astype(str)

    sidecars_written = 0
    teams_with_ytd = 0
    teams_missing_ytd: List[str] = []
    teams_missing_prev: List[str] = []
    join_issues: List[dict] = []

    for _, game in games_df.iterrows():
        game_key = game.get("game_key")
        kickoff_iso = game.get("kickoff_iso_utc")
        game_week = int(game.get("week")) if pd.notna(game.get("week")) else None
        home_norm = game.get("home_team_norm") or game.get("home_team_raw")
        away_norm = game.get("away_team_norm") or game.get("away_team_raw")
        home_key = team_merge_key_cfb(home_norm)
        away_key = team_merge_key_cfb(away_norm)

        mask = (
            (schedule_master["season"] == season)
            & (schedule_master["home_team_key"] == home_key)
            & (schedule_master["away_team_key"] == away_key)
        )
        if game_week is not None:
            mask &= schedule_master["week"] == game_week
        if not mask.any():
            join_issues.append({"game_key": game_key, "team": "both", "reason": "missing_schedule"})
            continue

        home_ytd = _slice_team(
            schedule_master,
            home_key,
            season,
            cutoff_iso=kickoff_iso,
            cutoff_week=game_week,
        )
        away_ytd = _slice_team(
            schedule_master,
            away_key,
            season,
            cutoff_iso=kickoff_iso,
            cutoff_week=game_week,
        )
        home_prev = _slice_team(schedule_master, home_key, season - 1)
        away_prev = _slice_team(schedule_master, away_key, season - 1)

        home_ytd = _augment_columns(home_ytd)
        away_ytd = _augment_columns(away_ytd)
        home_prev = _augment_columns(home_prev)
        away_prev = _augment_columns(away_prev)

        home_ytd_list = _project_rows(home_ytd)
        away_ytd_list = _project_rows(away_ytd)
        home_prev_list = _project_rows(home_prev)
        away_prev_list = _project_rows(away_prev)

        if home_ytd_list:
            teams_with_ytd += 1
        else:
            teams_missing_ytd.append(str(home_norm))
        if away_ytd_list:
            teams_with_ytd += 1
        else:
            teams_missing_ytd.append(str(away_norm))

        if not home_prev_list:
            teams_missing_prev.append(str(home_norm))
        if not away_prev_list:
            teams_missing_prev.append(str(away_norm))

        payload = {
            "game_key": game_key,
            "home_ytd": home_ytd_list,
            "away_ytd": away_ytd_list,
            "home_prev": home_prev_list,
            "away_prev": away_prev_list,
        }

        side_path = sidecar_dir / f"{game_key}.json"
        side_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        sidecars_written += 1

    receipt = {
        "season": season,
        "week": week,
        "games_total": games_total,
        "sidecars_written": sidecars_written,
        "teams_with_ytd": teams_with_ytd,
        "teams_missing_ytd": sorted(set(teams_missing_ytd)),
        "teams_missing_prev": sorted(set(teams_missing_prev)),
        "join_issues": join_issues,
    }

    receipt_path = out_dir / "sidecars_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    if sidecars_written != games_total:
        raise RuntimeError(
            f"Sidecar build incomplete: wrote {sidecars_written}/{games_total} files."
        )

    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CFB timelines for Game View sidecars.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    receipt = build_sidecars(args.season, args.week)
    print(
        f"PASS: CFB sidecars written -> out/cfb/{args.season}_week{args.week}/game_schedules "
        f"({receipt['sidecars_written']}/{receipt['games_total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

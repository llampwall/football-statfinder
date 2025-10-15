"""Build per-game Sagarin PR/SoS timelines for Game View sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.common.team_names import team_merge_key
from src.schedule_master import load_master as load_schedule_master


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
    sched_all: pd.DataFrame,
    team_key: str,
    season: int,
    cutoff_iso: str | None = None,
) -> pd.DataFrame:
    df = sched_all[sched_all["season"] == season].copy()
    df = df[(df["home_team_key"] == team_key) | (df["away_team_key"] == team_key)]
    if cutoff_iso is not None:
        df = df[df["kickoff_iso_utc"] < cutoff_iso]
    rows = []
    for _, row in df.iterrows():
        is_home = row["home_team_key"] == team_key
        opp = row["away_team_norm"] if is_home else row["home_team_norm"]
        opp_key = row["away_team_key"] if is_home else row["home_team_key"]
        pf = row.get("home_score") if is_home else row.get("away_score")
        pa = row.get("away_score") if is_home else row.get("home_score")
        rows.append(
            {
                "season": row["season"],
                "week": row.get("week"),
                "date": _dateutc(row.get("kickoff_iso_utc")),
                "opp": opp,
                "opp_key": opp_key,
                "site": "H" if is_home else "A",
                "pf": None if pd.isna(pf) else int(pf),
                "pa": None if pd.isna(pa) else int(pa),
                "result": _result(pf, pa),
                "team_key": team_key,
            }
        )
    return pd.DataFrame(rows)


def build_timelines(
    season: int,
    week: int,
    pr_master: pd.DataFrame,
    outdir: Path,
    games_df: pd.DataFrame,
) -> Tuple[Dict[str, Path], Dict[str, dict]]:
    seasons_needed = [season, season - 1]
    schedule_master = load_schedule_master()
    schedule_master = schedule_master[
        (schedule_master["league"].astype(str).str.upper() == "NFL")
        & (schedule_master["game_type"] == "REG")
        & (schedule_master["season"].isin(seasons_needed))
    ].copy()

    if schedule_master.empty:
        raise RuntimeError("schedule master empty for requested seasons")

    schedule_master["week"] = pd.to_numeric(schedule_master["week"], errors="coerce").astype("Int64")
    pr_master = pr_master[pr_master["league"].astype(str).str.upper() == "NFL"].copy()
    pr_master["week"] = pd.to_numeric(pr_master["week"], errors="coerce").astype("Int64")

    pr_master = pr_master.copy()
    pr_master["team_key"] = pr_master["team_norm"].map(team_merge_key)
    pr_key_set = {
        (int(row.season), int(row.week), str(row.team_key))
        for row in pr_master.itertuples()
        if pd.notna(row.season) and pd.notna(row.week) and pd.notna(row.team_key)
    }

    pr_team = pr_master[["season", "week", "team_key", "pr", "pr_rank", "sos", "sos_rank"]].copy()
    opp_pr = pr_team.rename(
        columns={
            "team_key": "opp_team_key",
            "pr": "opp_pr",
            "pr_rank": "opp_pr_rank",
            "sos": "opp_sos",
            "sos_rank": "opp_sos_rank",
        }
    )

    side_dir = outdir / "game_schedules"
    side_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    details: Dict[str, dict] = {}

    missing_schedule = []

    for _, game in games_df.iterrows():
        game_key = game["game_key"]
        kickoff_iso = game.get("kickoff_iso_utc")
        home_slug = game.get("home_team_norm")
        away_slug = game.get("away_team_norm")
        home_key = team_merge_key(home_slug)
        away_key = team_merge_key(away_slug)
        game_week = game.get("week")

        mask = (
            (schedule_master["season"] == season)
            & (schedule_master["week"] == game_week)
            & (schedule_master["home_team_key"] == home_key)
            & (schedule_master["away_team_key"] == away_key)
        )
        if not mask.any():
            missing_schedule.append(game_key)
            continue

        home_ytd = _slice_team(schedule_master, home_key, season, cutoff_iso=kickoff_iso)
        away_ytd = _slice_team(schedule_master, away_key, season, cutoff_iso=kickoff_iso)
        home_prev = _slice_team(schedule_master, home_key, season - 1, cutoff_iso=None)
        away_prev = _slice_team(schedule_master, away_key, season - 1, cutoff_iso=None)

        def _augment(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df = df.merge(pr_team, how="left", left_on=["season", "week", "team_key"], right_on=["season", "week", "team_key"])
            df = df.merge(opp_pr, how="left", left_on=["season", "week", "opp_key"], right_on=["season", "week", "opp_team_key"])
            return df

        home_ytd = _augment(home_ytd)
        away_ytd = _augment(away_ytd)
        home_prev = _augment(home_prev)
        away_prev = _augment(away_prev)

        def _project(df: pd.DataFrame, team_key_value: str) -> Tuple[list[dict], int, int, int, int]:
            if df.empty:
                return [], 0, 0, 0, 0
            df = df.sort_values(["season", "week", "date"], na_position="last").copy()
            keep = [
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
            for col in keep:
                if col not in df.columns:
                    df[col] = None
            timeline = df[keep].to_dict(orient="records")
            last_key = None
            for entry in timeline:
                key = (
                    entry.get("season") if entry.get("season") is not None else -1,
                    entry.get("week") if entry.get("week") is not None else -1,
                )
                if last_key and key < last_key:
                    raise RuntimeError(f"Timeline not sorted for team key {team_key_value}")
                last_key = key
            expected_team = sum(
                1
                for row in timeline
                if row["season"] is not None
                and row["week"] is not None
                and (int(row["season"]), int(row["week"]), team_key_value) in pr_key_set
            )
            present_team = sum(1 for row in timeline if row.get("pr") is not None)
            expected_opp = sum(
                1
                for row in timeline
                if row["season"] is not None
                and row["week"] is not None
                and row.get("opp_key") is not None
                and (int(row["season"]), int(row["week"]), str(row.get("opp_key"))) in pr_key_set
            )
            present_opp = sum(1 for row in timeline if row.get("opp_pr") is not None)
            return timeline, expected_team, present_team, expected_opp, present_opp

        home_ytd_list, home_ytd_exp_t, home_ytd_pres_t, home_ytd_exp_o, home_ytd_pres_o = _project(
            home_ytd, home_key
        )
        away_ytd_list, away_ytd_exp_t, away_ytd_pres_t, away_ytd_exp_o, away_ytd_pres_o = _project(
            away_ytd, away_key
        )
        home_prev_list, home_prev_exp_t, home_prev_pres_t, home_prev_exp_o, home_prev_pres_o = _project(
            home_prev, home_key
        )
        away_prev_list, away_prev_exp_t, away_prev_pres_t, away_prev_exp_o, away_prev_pres_o = _project(
            away_prev, away_key
        )

        payload = {
            "game_key": game_key,
            "home_ytd": home_ytd_list,
            "away_ytd": away_ytd_list,
            "home_prev": home_prev_list,
            "away_prev": away_prev_list,
        }

        side_path = side_dir / f"{game_key}.json"
        side_path.write_text(json.dumps(payload, ensure_ascii=False))
        written[game_key] = side_path

        details[game_key] = {
            "path": side_path,
            "home_ytd_len": len(home_ytd_list),
            "away_ytd_len": len(away_ytd_list),
            "home_prev_len": len(home_prev_list),
            "away_prev_len": len(away_prev_list),
            "team_expected_pr": home_ytd_exp_t + away_ytd_exp_t + home_prev_exp_t + away_prev_exp_t,
            "team_present_pr": home_ytd_pres_t + away_ytd_pres_t + home_prev_pres_t + away_prev_pres_t,
            "opp_expected_pr": home_ytd_exp_o + away_ytd_exp_o + home_prev_exp_o + away_prev_exp_o,
            "opp_present_pr": home_ytd_pres_o + away_ytd_pres_o + home_prev_pres_o + away_prev_pres_o,
        }

    details["_missing"] = missing_schedule

    return written, details


__all__ = ["build_timelines"]

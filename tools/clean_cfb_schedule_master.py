"""Deduplicate cfb_schedule_master.csv by matchup with kickoff preference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

MASTER_PATH = Path("out/master/cfb_schedule_master.csv")
MATCHUP_KEY: List[str] = [
    "league",
    "season",
    "week",
    "game_type",
    "home_team_key",
    "away_team_key",
]


def _kickoff_rank(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 2
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate CFB schedule master by matchup.")
    parser.parse_args()

    if not MASTER_PATH.exists():
        print(f"SKIP: master file missing at {MASTER_PATH}")
        return

    df = pd.read_csv(MASTER_PATH)
    if df.empty:
        print("SKIP: master file contains no rows.")
        return

    original_count = len(df)

    df["_kickoff_dt"] = pd.to_datetime(df.get("kickoff_iso_utc"), errors="coerce", utc=True)
    df["_kickoff_rank"] = df["_kickoff_dt"].apply(_kickoff_rank)
    df["_score_present"] = (
        df.get("home_score").notna() & df.get("away_score").notna()
    ).astype(int)
    source_priority = {"seed": 0, "cfbd": 1}
    df["_source_priority"] = df.get("source", "cfbd").map(source_priority).fillna(0).astype(int)

    sort_keys = MATCHUP_KEY + ["_score_present", "_source_priority", "_kickoff_rank", "_kickoff_dt"]
    ascending = [True] * len(MATCHUP_KEY) + [True, True, False, True]
    df = (
        df.sort_values(sort_keys, ascending=ascending)
        .drop_duplicates(MATCHUP_KEY, keep="last")
        .drop(columns=["_kickoff_dt", "_kickoff_rank", "_score_present", "_source_priority"])
        .reset_index(drop=True)
    )

    cleaned_count = len(df)
    removed = original_count - cleaned_count
    dup_groups = (
        df.duplicated(MATCHUP_KEY, keep=False).sum()
    )

    df.to_csv(MASTER_PATH, index=False)
    print(
        f"CLEANUP: rows_before={original_count} rows_after={cleaned_count} "
        f"removed={removed} remaining_duplicates={dup_groups}"
    )


if __name__ == "__main__":
    main()

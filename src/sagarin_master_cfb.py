"""Maintain the College Football Sagarin master snapshot table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

OUT_DIR = Path("out")
MASTER_DIR = OUT_DIR / "master"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = MASTER_DIR / "sagarin_cfb_master.csv"
LEAGUE = "CFB"
MASTER_COLUMNS = [
    "league",
    "season",
    "week",
    "team_norm",
    "team_raw",
    "pr",
    "rank",
    "sos",
    "sos_rank",
]
KEY_COLUMNS = ["season", "week", "team_norm"]


def _load_weekly_csv(season: int, week: int) -> Tuple[Path, pd.DataFrame]:
    week_dir = OUT_DIR / "cfb" / f"{season}_week{week}"
    csv_path = week_dir / f"sagarin_cfb_{season}_wk{week}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Weekly Sagarin CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    return csv_path, df


def _normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "team_norm", "team_raw", "pr", "pr_rank", "sos", "sos_rank"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Weekly CSV missing required columns: {', '.join(missing)}")

    work = df.copy()
    work["league"] = LEAGUE
    work["rank"] = pd.to_numeric(work["pr_rank"], errors="coerce")
    work["pr"] = pd.to_numeric(work["pr"], errors="coerce")
    work["sos"] = pd.to_numeric(work["sos"], errors="coerce")
    work["sos_rank"] = pd.to_numeric(work["sos_rank"], errors="coerce")
    work["season"] = pd.to_numeric(work["season"], errors="coerce").astype("Int64")
    work["week"] = pd.to_numeric(work["week"], errors="coerce").astype("Int64")

    normalized = work[MASTER_COLUMNS].copy()
    normalized = normalized.sort_values(["rank", "team_norm"]).reset_index(drop=True)
    return normalized


def _load_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
        for col in MASTER_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[MASTER_COLUMNS].copy()
    return pd.DataFrame(columns=MASTER_COLUMNS)


def upsert_master(week_df: pd.DataFrame) -> Tuple[int, int]:
    master_df = _load_master()
    before = len(master_df)
    combined = pd.concat([master_df, week_df], ignore_index=True)
    combined = combined.drop_duplicates(KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["season", "week", "team_norm"]).reset_index(drop=True)
    combined[MASTER_COLUMNS].to_csv(MASTER_CSV, index=False)
    after = len(combined)
    return before, after


def process_week(season: int, week: int) -> Tuple[int, int, Path]:
    csv_path, weekly_df = _load_weekly_csv(season, week)
    normalized = _normalize_weekly(weekly_df)
    before, after = upsert_master(normalized)
    return before, after, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert weekly CFB Sagarin ratings into master.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    try:
        before, after, csv_path = process_week(args.season, args.week)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:
        print(f"FAIL: unable to update CFB Sagarin master -> {exc}")
        return 1

    delta = after - before
    print(",".join(MASTER_COLUMNS))
    print(f"Upsert complete: before={before} after={after} delta={delta} ({csv_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

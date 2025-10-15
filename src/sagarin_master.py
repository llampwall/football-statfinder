"""Maintain a master Sagarin table for historical PR/SoS snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

MASTER_DIR = Path("out/master")
MASTER_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = MASTER_DIR / "sagarin_nfl_master.csv"

KEEP = [
    "league",
    "season",
    "week",
    "team_norm",
    "team_raw",
    "pr",
    "pr_rank",
    "sos",
    "sos_rank",
]

KEY = ["league", "season", "week", "team_norm"]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["league"] = df.get("league", "NFL").fillna("NFL").astype(str)
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    for col in ("pr", "sos"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in ("pr_rank", "sos_rank"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").astype("Int64")
    return df


def append_week(sag_week_csv: Path, league: str = "NFL") -> Tuple[int, int]:
    """Append a weekly Sagarin CSV into the master table. Returns (before, after) row counts."""

    wdf = pd.read_csv(sag_week_csv).copy()
    if "league" not in wdf.columns:
        wdf["league"] = league

    rename_map = {"rank": "pr_rank"}
    for src, dest in rename_map.items():
        if src in wdf.columns and dest not in wdf.columns:
            wdf.rename(columns={src: dest}, inplace=True)

    for col in KEEP:
        if col not in wdf.columns:
            wdf[col] = None
    wdf = wdf[KEEP]
    wdf = _coerce_types(wdf)

    if MASTER_CSV.exists():
        mdf = pd.read_csv(MASTER_CSV)
    else:
        mdf = pd.DataFrame(columns=KEEP)

    before = len(mdf)
    mdf = pd.concat([mdf, wdf], ignore_index=True)
    mdf = mdf.sort_values(KEY).drop_duplicates(KEY, keep="last").reset_index(drop=True)
    after = len(mdf)

    mdf.to_csv(MASTER_CSV, index=False)

    return before, after


def replace_master_from_csv(src_csv: Path) -> int:
    df = pd.read_csv(src_csv).copy()
    for col in KEEP:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP]
    df = _coerce_types(df)
    df = df.sort_values(KEY).drop_duplicates(KEY, keep="last").reset_index(drop=True)
    df.to_csv(MASTER_CSV, index=False)
    return len(df)


def load_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV)
    return pd.DataFrame(columns=KEEP)


__all__ = ["append_week", "replace_master_from_csv", "load_master", "MASTER_CSV"]

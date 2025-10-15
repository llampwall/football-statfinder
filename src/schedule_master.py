"""Maintain the NFL schedule master table."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.common.team_names import normalize_team_display, team_merge_key
from src.fetch_games import get_schedule_df

MASTER_DIR = Path("out/master")
MASTER_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = MASTER_DIR / "nfl_schedule_master.csv"

KEEP = [
    "league",
    "season",
    "week",
    "game_type",
    "kickoff_iso_utc",
    "home_team_norm",
    "away_team_norm",
    "home_team_key",
    "away_team_key",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "source",
]

KEY = [
    "league",
    "season",
    "week",
    "game_type",
    "home_team_key",
    "away_team_key",
    "kickoff_iso_utc",
]


def _log(message: str) -> None:
    print(f"[schedule_master] {message}")


def _parse_kickoff(row: pd.Series) -> str | None:
    iso = row.get("kickoff_iso_utc")
    if isinstance(iso, str) and "T" in iso:
        try:
            dt = pd.to_datetime(iso, utc=True)
            return dt.replace(microsecond=0).isoformat()
        except Exception:
            pass
    date_candidates = [row.get(c) for c in ("gameday", "gamedate", "game_date", "date")]
    time_candidates = [row.get(c) for c in ("gametime", "start_time", "kickoff_time")]
    date_val = next((d for d in date_candidates if isinstance(d, str) and d.strip()), None)
    time_val = next((t for t in time_candidates if isinstance(t, str) and t.strip()), "00:00")
    if not date_val:
        return None
    try:
        dt = pd.to_datetime(f"{date_val} {time_val}", utc=False)
        if dt.tzinfo is None:
            dt = dt.tz_localize("America/New_York")
        return dt.tz_convert("UTC").replace(microsecond=0).isoformat()
    except Exception:
        return None


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["league"] = df.get("league", "NFL").fillna("NFL").astype(str)
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    if "kickoff_iso_utc" in df.columns:
        kickoff = pd.to_datetime(df["kickoff_iso_utc"], errors="coerce", utc=True)
        df["kickoff_iso_utc"] = kickoff.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace("+0000", "+00:00")
    for col in ("home_team_norm", "away_team_norm"):
        values = df.get(col)
        values = values.where(pd.notna(values), None)
        values = values.apply(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = values
    if "home_team_key" not in df.columns or df["home_team_key"].isna().all():
        df["home_team_key"] = df["home_team_norm"].map(team_merge_key)
    if "away_team_key" not in df.columns or df["away_team_key"].isna().all():
        df["away_team_key"] = df["away_team_norm"].map(team_merge_key)
    df["home_team_key"] = df["home_team_key"].astype(str)
    df["away_team_key"] = df["away_team_key"].astype(str)
    for col in ("home_team_key", "away_team_key"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ("home_score", "away_score", "spread_line", "total_line"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["source"] = df.get("source", "seed").fillna("seed").astype(str)
    return df


def _normalize_schedule(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["league"] = "NFL"
    df["game_type"] = "REG"
    df["kickoff_iso_utc"] = df.apply(_parse_kickoff, axis=1)
    if "home_team_norm" not in df.columns or df["home_team_norm"].isna().all():
        df["home_team_norm"] = df["home_team"].astype(str).map(normalize_team_display)
    if "away_team_norm" not in df.columns or df["away_team_norm"].isna().all():
        df["away_team_norm"] = df["away_team"].astype(str).map(normalize_team_display)
    df["home_team_key"] = df["home_team_norm"].map(team_merge_key)
    df["away_team_key"] = df["away_team_norm"].map(team_merge_key)
    for col in KEEP:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP]
    return _coerce_types(df)


def load_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
        return _coerce_types(df)
    return pd.DataFrame(columns=KEEP)


def upsert_rows(df: pd.DataFrame) -> Tuple[int, int]:
    df = _coerce_types(df)
    mdf = load_master()
    if df.empty:
        count = len(mdf)
        return count, count
    combined = pd.concat([mdf, df], ignore_index=True)
    combined["score_present"] = (
        combined["home_score"].notna() & combined["away_score"].notna()
    ).astype(int)
    source_priority = {"seed": 0, "nflverse": 1}
    combined["source_priority"] = combined["source"].map(source_priority).fillna(0).astype(int)
    combined = (
        combined.sort_values(KEY + ["score_present", "source_priority"])
        .drop_duplicates(KEY, keep="last")
        .reset_index(drop=True)
    )
    combined = combined.drop(columns=["score_present", "source_priority"])
    before = len(mdf)
    after = len(combined)
    dups = combined[combined.duplicated(KEY, keep=False)]
    if not dups.empty:
        _log(f"Duplicate keys detected after upsert: {dups.head().to_dict(orient='records')}")
        raise RuntimeError("Schedule master still has duplicate keys")
    combined.to_csv(MASTER_CSV, index=False)
    _log(f"Upsert complete: before={before}, after={after}, delta={after-before}")
    return before, after


def upsert_from_seed(seed_csv: Path) -> int:
    df = pd.read_csv(seed_csv)
    normalized = _normalize_schedule(df, source="seed")
    before, after = upsert_rows(normalized)
    delta = after - before
    _log(f"Seed upsert from {seed_csv} -> delta {delta}")
    return delta


def ensure_weeks_present(seasons: Iterable[int]) -> None:
    seasons = list(seasons)
    frames = []
    for season in seasons:
        _log(f"Fetching nflverse schedule for season {season}")
        schedule_df = get_schedule_df(season)
        frames.append(_normalize_schedule(schedule_df, source="nflverse"))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        before, after = upsert_rows(combined)
        _log(f"ensure_weeks_present seasons={seasons} -> delta {after-before}")


__all__ = [
    "MASTER_CSV",
    "load_master",
    "upsert_rows",
    "upsert_from_seed",
    "ensure_weeks_present",
]

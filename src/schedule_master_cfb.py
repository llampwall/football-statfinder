
"""Maintain the CFB schedule master table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.common.cfb_source import fetch_cfbd_games
from src.common.io_utils import read_env
from src.common.team_names_cfb import normalize_team_name_cfb_stats, team_merge_key_cfb

__all__ = [
    "MASTER_CSV",
    "load_master",
    "upsert_rows",
    "ensure_weeks_present",
    "enrich_from_local_odds",
]

MASTER_DIR = Path("out/master")
MASTER_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = MASTER_DIR / "cfb_schedule_master.csv"

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
    print(f"[schedule_master_cfb] {message}")


def _parse_kickoff(row: pd.Series) -> str | None:
    iso = row.get("startDate") or row.get("kickoff_iso_utc")
    if isinstance(iso, str) and iso.strip():
        try:
            dt = pd.to_datetime(iso, utc=True)
            if pd.isna(dt):
                raise ValueError
            return dt.tz_convert("UTC").replace(microsecond=0).isoformat().replace("+00:00", "+00:00")
        except Exception:
            pass
    date_val = row.get("start_date") or row.get("gameday")
    time_val = row.get("start_time") or row.get("kickoff_time") or "00:00"
    if not isinstance(date_val, str) or not date_val.strip():
        return None
    try:
        dt = pd.to_datetime(f"{date_val} {time_val}", utc=False, errors="coerce")
        if pd.isna(dt):
            return None
        if dt.tzinfo is None:
            dt = dt.tz_localize("America/New_York")
        return dt.tz_convert("UTC").replace(microsecond=0).isoformat().replace("+00:00", "+00:00")
    except Exception:
        return None


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["league"] = df.get("league", "CFB").fillna("CFB").astype(str)
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    if "kickoff_iso_utc" in df.columns:
        kickoff = pd.to_datetime(df["kickoff_iso_utc"], errors="coerce", utc=True)
        df["kickoff_iso_utc"] = kickoff.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace("+0000", "+00:00")
    for col in ("home_team_norm", "away_team_norm"):
        values = df.get(col)
        values = values.where(pd.notna(values), None)
        df[col] = values.apply(lambda v: v.strip() if isinstance(v, str) else v)
    if "home_team_key" not in df.columns or df["home_team_key"].isna().all():
        df["home_team_key"] = df["home_team_norm"].map(team_merge_key_cfb)
    if "away_team_key" not in df.columns or df["away_team_key"].isna().all():
        df["away_team_key"] = df["away_team_norm"].map(team_merge_key_cfb)
    for col in ("home_team_key", "away_team_key"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ("home_score", "away_score", "spread_line", "total_line"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["source"] = df.get("source", "cfbd").fillna("cfbd").astype(str)
    df["game_type"] = df.get("game_type", "REG").fillna("REG").astype(str)
    return df


def _normalize_schedule(df: pd.DataFrame, source: str = "cfbd") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEEP)
    work = df.copy()
    work["league"] = "CFB"
    work["game_type"] = work.get("seasonType", "").map(
        lambda s: "POST" if isinstance(s, str) and s.lower() == "postseason" else "REG"
    )
    work["kickoff_iso_utc"] = work.apply(_parse_kickoff, axis=1)
    work["home_team_norm"] = work["homeTeam"].astype(str).map(normalize_team_name_cfb_stats)
    work["away_team_norm"] = work["awayTeam"].astype(str).map(normalize_team_name_cfb_stats)
    work["home_team_key"] = work["home_team_norm"].map(team_merge_key_cfb)
    work["away_team_key"] = work["away_team_norm"].map(team_merge_key_cfb)
    work["home_score"] = pd.to_numeric(work.get("homePoints"), errors="coerce")
    work["away_score"] = pd.to_numeric(work.get("awayPoints"), errors="coerce")
    work["spread_line"] = pd.to_numeric(work.get("spread"), errors="coerce")
    work["total_line"] = pd.to_numeric(work.get("overUnder"), errors="coerce")
    work["season"] = pd.to_numeric(work.get("season"), errors="coerce")
    work["week"] = pd.to_numeric(work.get("week"), errors="coerce")
    work["source"] = source
    for col in KEEP:
        if col not in work.columns:
            work[col] = None
    work = work[KEEP]
    return _coerce_types(work)


def load_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
        return _coerce_types(df)
    return pd.DataFrame(columns=KEEP)


def upsert_rows(df: pd.DataFrame) -> Tuple[int, int]:
    df = _coerce_types(df)
    master_df = load_master()
    if df.empty:
        count = len(master_df)
        return count, count
    combined = pd.concat([master_df, df], ignore_index=True)
    combined["score_present"] = (
        combined["home_score"].notna() & combined["away_score"].notna()
    ).astype(int)
    source_priority = {"seed": 0, "cfbd": 1}
    combined["source_priority"] = combined["source"].map(source_priority).fillna(0).astype(int)
    combined = (
        combined.sort_values(KEY + ["score_present", "source_priority"])
        .drop_duplicates(KEY, keep="last")
        .reset_index(drop=True)
    )
    combined = combined.drop(columns=["score_present", "source_priority"])
    before = len(master_df)
    after = len(combined)
    duplicates = combined[combined.duplicated(KEY, keep=False)]
    if not duplicates.empty:
        _log(f"Duplicate keys detected after upsert: {duplicates.head().to_dict(orient='records')}")
        raise RuntimeError("CFB schedule master still has duplicate keys")
    combined.to_csv(MASTER_CSV, index=False)
    _log(f"Upsert complete: before={before}, after={after}, delta={after-before}")
    return before, after


def _fetch_cfbd_schedule(season: int) -> pd.DataFrame:
    env = read_env(["CFBD_API_KEY"])
    api_key = env.get("CFBD_API_KEY")
    if not api_key:
        _log("CFBD_API_KEY missing; returning empty schedule.")
        return pd.DataFrame(columns=KEEP)
    try:
        records = fetch_cfbd_games(season, None, api_key) or []
    except Exception as exc:
        _log(f"Error fetching CFBD schedule for season {season}: {exc}")
        return pd.DataFrame(columns=KEEP)
    if not records:
        _log(f"CFBD returned no schedule rows for season {season}.")
        return pd.DataFrame(columns=KEEP)
    raw = pd.DataFrame(records)
    raw["season"] = season
    normalized = _normalize_schedule(raw, source="cfbd")
    return normalized


def ensure_weeks_present(seasons: Iterable[int]) -> None:
    seasons = list(seasons)
    frames = []
    for season in seasons:
        _log(f"Fetching CFBD schedule for season {season}")
        season_df = _fetch_cfbd_schedule(season)
        if season_df.empty:
            continue
        frames.append(season_df)
    if not frames:
        _log(f"No schedule rows fetched for seasons={seasons}")
        return
    combined = pd.concat(frames, ignore_index=True)
    before, after = upsert_rows(combined)
    _log(f"ensure_weeks_present seasons={seasons} -> delta {after-before}")


def _team_token(value: str | None) -> str:
    token = (value or "").lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in token)
    token = "_".join(filter(None, token.split("_")))
    return token or "unknown"


def _build_game_key_from_row(row: pd.Series) -> str | None:
    kickoff = row.get("kickoff_iso_utc")
    if not isinstance(kickoff, str) or "T" not in kickoff:
        return None
    try:
        dt = pd.to_datetime(kickoff, utc=True)
        if pd.isna(dt):
            return None
    except Exception:
        return None
    yyyymmdd = dt.strftime("%Y%m%d")
    hhmm = dt.strftime("%H%M")
    away_token = _team_token(row.get("away_team_norm"))
    home_token = _team_token(row.get("home_team_norm"))
    return f"{yyyymmdd}_{hhmm}_{away_token}_{home_token}"


def _load_odds_records(season: int, week: int) -> list[dict]:
    odds_path = Path(f"out/cfb/{season}_week{week}/odds_{season}_wk{week}.jsonl")
    if not odds_path.exists():
        return []
    records: list[dict] = []
    with odds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def enrich_from_local_odds(season: int, week: int) -> dict:
    odds_records = _load_odds_records(season, week)
    base_dir = Path(f"out/cfb/{season}_week{week}")
    base_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = base_dir / "master_enrichment_receipt.json"
    if not odds_records:
        _log(f"SKIP: no local odds file for season={season} week={week}")
        receipt_path.write_text(json.dumps({"season": season, "week": week, "rows_considered": 0, "rows_updated": 0}), encoding="utf-8")
        return {"season": season, "week": week, "rows_considered": 0, "rows_updated": 0}

    odds_by_key: dict[str, dict] = {}
    fallback_records: list[dict] = []
    for record in odds_records:
        try:
            spread = record.get("spread_home_relative")
            total = record.get("total")
            if spread is None and total is None:
                continue
            game_key = record.get("game_key")
            kickoff = record.get("kickoff_iso") or record.get("kickoff_iso_utc")
            home_norm = record.get("home_team_norm")
            away_norm = record.get("away_team_norm")
            home_key = team_merge_key_cfb(home_norm)
            away_key = team_merge_key_cfb(away_norm)
            odds_row = {
                "spread": float(spread) if spread is not None else None,
                "total": float(total) if total is not None else None,
                "kickoff": kickoff,
                "home_key": home_key,
                "away_key": away_key,
            }
            if isinstance(game_key, str) and game_key:
                odds_by_key[game_key] = odds_row
            fallback_records.append(odds_row)
        except Exception:
            continue

    master_df = load_master()
    mask = (master_df["season"] == season) & (master_df["week"] == week)
    indices = master_df.index[mask]
    rows_considered = len(indices)
    if rows_considered == 0:
        _log(f"SKIP: no master rows found for season={season} week={week}")
        receipt_path.write_text(json.dumps({"season": season, "week": week, "rows_considered": 0, "rows_updated": 0}), encoding="utf-8")
        return {"season": season, "week": week, "rows_considered": 0, "rows_updated": 0}

    rows_updated = 0
    for idx in indices:
        row = master_df.loc[idx]
        need_spread = pd.isna(row.get("spread_line"))
        need_total = pd.isna(row.get("total_line"))
        if not need_spread and not need_total:
            continue

        candidate = None
        game_key = _build_game_key_from_row(row)
        if game_key and game_key in odds_by_key:
            candidate = odds_by_key[game_key]
        else:
            kickoff = row.get("kickoff_iso_utc")
            if isinstance(kickoff, str):
                try:
                    kickoff_dt = pd.to_datetime(kickoff, utc=True)
                except Exception:
                    kickoff_dt = None
            else:
                kickoff_dt = None
            for odds_row in fallback_records:
                if not odds_row["home_key"] or not odds_row["away_key"]:
                    continue
                if odds_row["home_key"] != row.get("home_team_key") or odds_row["away_key"] != row.get("away_team_key"):
                    continue
                if kickoff_dt is not None and odds_row["kickoff"]:
                    try:
                        odds_dt = pd.to_datetime(odds_row["kickoff"], utc=True)
                    except Exception:
                        odds_dt = None
                    if odds_dt is None:
                        continue
                    delta = abs((kickoff_dt - odds_dt).total_seconds())
                    if delta > 600:
                        continue
                candidate = odds_row
                break

        if not candidate:
            continue

        updated = False
        if need_spread and candidate["spread"] is not None:
            master_df.at[idx, "spread_line"] = round(candidate["spread"], 1)
            updated = True
        if need_total and candidate["total"] is not None:
            master_df.at[idx, "total_line"] = round(candidate["total"], 1)
            updated = True
        if updated:
            rows_updated += 1

    if rows_updated > 0:
        master_df = _coerce_types(master_df)
        master_df.to_csv(MASTER_CSV, index=False)
    _log(f"Master enrichment season={season} week={week}: updated {rows_updated}/{rows_considered}")
    receipt = {"season": season, "week": week, "rows_considered": rows_considered, "rows_updated": rows_updated}
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Maintain the CFB schedule master table.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Season(s) to fetch from CFBD.")
    args = parser.parse_args()
    ensure_weeks_present(args.seasons)
    if MASTER_CSV.exists():
        columns = pd.read_csv(MASTER_CSV, nrows=0).columns.tolist()
        _log(f"Master columns: {columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

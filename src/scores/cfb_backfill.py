"""CFB scores backfill helpers (prior weeks before ATS).

Purpose & scope:
    Load recent College Football week files, hydrate missing final scores from
    the schedule master, and rewrite the week JSONL/CSV outputs atomically so
    the ATS step operates on complete historical data.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (D, E, F, I)

Invariants:
    * Only past weeks are touched; current week is never mutated.
    * Weekly files are rewritten atomically when updates occur.
    * Schedule master remains the source of truth for authoritative finals.

Side effects:
    * Rewrites ``out/cfb/{season}_week{w}/games_week_{season}_{w}.jsonl`` and
      matching CSV when scores are backfilled.

Do not:
    * Attempt to infer scores when the schedule master lacks both totals.
    * Modify weeks beyond the configured lookback window.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir, getenv
from src.schedule_master_cfb import load_master

OUT_ROOT = ensure_out_dir()
CFB_ROOT = OUT_ROOT / "cfb"


def _team_token(value: Optional[str]) -> str:
    text = (value or "").lower()
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    parts = [part for part in token.split("_") if part]
    return "_".join(parts)


def _build_game_key_from_row(row: pd.Series) -> Optional[str]:
    kickoff = row.get("kickoff_iso_utc")
    if not isinstance(kickoff, str) or "T" not in kickoff:
        return None
    try:
        dt = pd.to_datetime(kickoff, utc=True)
    except Exception:
        return None
    if pd.isna(dt):
        return None
    away_token = _team_token(row.get("away_team_norm"))
    home_token = _team_token(row.get("home_team_norm"))
    yyyymmdd = dt.strftime("%Y%m%d")
    hhmm = dt.strftime("%H%M")
    return f"{yyyymmdd}_{hhmm}_{away_token}_{home_token}"


def _load_week_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def _is_missing_score(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, str):
        return not value.strip()
    return False


def _build_score_lookup(season: int) -> Dict[str, Tuple[int, int]]:
    master_df = load_master()
    if master_df.empty:
        return {}
    season_df = master_df[master_df["season"] == season].copy()
    if season_df.empty:
        return {}
    season_df["game_key"] = season_df.apply(_build_game_key_from_row, axis=1)
    season_df = season_df.dropna(subset=["game_key", "home_score", "away_score"])
    lookup: Dict[str, Tuple[int, int]] = {}
    for row in season_df.itertuples(index=False):
        game_key = getattr(row, "game_key", None)
        if not isinstance(game_key, str):
            continue
        home_score = getattr(row, "home_score", None)
        away_score = getattr(row, "away_score", None)
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        lookup[game_key] = (int(round(float(home_score))), int(round(float(away_score))))
    return lookup


def _derive_weeks_to_backfill(current_week: int, include_weeks_back: int) -> List[int]:
    weeks: List[int] = []
    for offset in range(1, include_weeks_back + 1):
        target = current_week - offset
        if target >= 1:
            weeks.append(target)
    return sorted(weeks)


def backfill_cfb_scores(
    season: int,
    week: int,
    *,
    include_weeks_back: Optional[int] = None,
) -> Dict[str, object]:
    """Backfill missing scores for prior weeks using the schedule master.

    Args:
        season: Target season identifier.
        week: Current week (used to bound the lookback).
        include_weeks_back: Optional override for how many weeks to scan.

    Returns:
        Dict summarising the operation with keys: ``weeks`` (list of ints),
        ``updated`` (count), ``skipped`` (missing finals), ``files_rewritten``.
    """
    if include_weeks_back is None:
        include_weeks_back = int(getenv("REFRESH_INCLUDE_WEEKS_BACK", "2") or "2")
    if include_weeks_back <= 0 or week <= 1:
        return {"weeks": [], "updated": 0, "skipped": 0, "files_rewritten": 0}

    weeks_to_scan = _derive_weeks_to_backfill(week, include_weeks_back)
    if not weeks_to_scan:
        return {"weeks": [], "updated": 0, "skipped": 0, "files_rewritten": 0}

    score_lookup = _build_score_lookup(season)
    updated = 0
    skipped = 0
    files_rewritten = 0

    for prior_week in weeks_to_scan:
        week_dir = CFB_ROOT / f"{season}_week{prior_week}"
        json_path = week_dir / f"games_week_{season}_{prior_week}.jsonl"
        csv_path = week_dir / f"games_week_{season}_{prior_week}.csv"
        records = _load_week_json(json_path)
        if not records:
            continue

        changed = False
        record_map = {row.get("game_key"): idx for idx, row in enumerate(records) if isinstance(row.get("game_key"), str)}
        csv_df = pd.read_csv(csv_path) if csv_path.exists() else None
        csv_index: Dict[str, int] = {}
        if csv_df is not None and "game_key" in csv_df.columns:
            csv_index = {str(row.game_key): idx for idx, row in csv_df.iterrows()}

        for idx, row in enumerate(records):
            game_key = row.get("game_key")
            if not isinstance(game_key, str):
                continue
            home_score_present = not _is_missing_score(row.get("home_score"))
            away_score_present = not _is_missing_score(row.get("away_score"))
            if home_score_present and away_score_present:
                continue
            scores = score_lookup.get(game_key)
            if not scores:
                if not home_score_present or not away_score_present:
                    skipped += 1
                continue

            records[idx]["home_score"], records[idx]["away_score"] = scores
            changed = True
            updated += 1

            if csv_df is not None:
                csv_idx = csv_index.get(game_key)
                if csv_idx is not None:
                    csv_df.at[csv_idx, "home_score"] = scores[0]
                    csv_df.at[csv_idx, "away_score"] = scores[1]

        if changed:
            write_atomic_jsonl(json_path, records)
            if csv_df is not None:
                write_atomic_csv(csv_path, csv_df)
            files_rewritten += 1

    return {
        "weeks": weeks_to_scan,
        "updated": updated,
        "skipped": skipped,
        "files_rewritten": files_rewritten,
    }


__all__ = ["backfill_cfb_scores"]

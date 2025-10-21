"""NFL ATS aggregation helpers (season-to-date).

Purpose & scope:
    Build season-to-date against-the-spread tallies from finalized, lined
    NFL games and apply the results to the current week's Game View rows.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (C, E, F, H)

Invariants:
    * Only prior weeks contribute to ATS tallies.
    * Current week JSONL/CSV files are rewritten atomically.
    * Games lacking spreads or final scores are ignored.

Side effects:
    * Rewrites `out/nfl/{season}_week{week}/games_week_{season}_{week}.{jsonl,csv}`
      when ATS fields change.

Do not:
    * Derive ATS from incomplete (non-final) games.
    * Modify historical weeks beyond the aggregate computation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.common.io_atomic import write_atomic_csv, write_atomic_jsonl
from src.common.io_utils import ensure_out_dir, getenv

OUT_ROOT = ensure_out_dir()
NFL_ROOT = OUT_ROOT / "nfl"
EPSILON = 1e-6


def _week_path(season: int, week: int) -> Tuple[Path, Path]:
    week_dir = NFL_ROOT / f"{season}_week{week}"
    return (
        week_dir / f"games_week_{season}_{week}.jsonl",
        week_dir / f"games_week_{season}_{week}.csv",
    )


def _load_week_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


def _to_number(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _score_from_row(row: Mapping[str, object], side: str) -> Optional[float]:
    key = f"{side}_score"
    value = row.get(key)
    if value is not None:
        return _to_number(value)
    schedule_row = row.get("raw_sources", {}).get("schedule_row") if isinstance(row.get("raw_sources"), dict) else None
    if isinstance(schedule_row, dict):
        return _to_number(schedule_row.get(key))
    return None


def _format_ats(entry: Mapping[str, int]) -> Optional[str]:
    wins = int(entry.get("w", 0))
    losses = int(entry.get("l", 0))
    pushes = int(entry.get("p", 0))
    if wins + losses + pushes == 0:
        return None
    base = f"{wins}-{losses}"
    if pushes > 0:
        base += f"-{pushes}"
    return base


def build_team_ats(season: int, week: int) -> Dict[str, Dict[str, int]]:
    """Return season-to-date ATS tallies for weeks < current."""
    stats: Dict[str, Dict[str, int]] = {}
    weeks_scanned: List[int] = []
    games_considered = 0

    if week <= 1 or os.getenv("ATS_ENABLE", "1").strip().lower() in {"0", "false", "off", "disabled"}:
        build_team_ats.meta = {"weeks_scanned": weeks_scanned, "games_considered": games_considered}
        return stats

    for prior_week in range(1, week):
        json_path, _ = _week_path(season, prior_week)
        rows = _load_week_rows(json_path)
        if not rows:
            continue
        weeks_scanned.append(prior_week)
        for row in rows:
            favored_side = str(row.get("favored_side") or "").upper()
            if favored_side not in {"HOME", "AWAY"}:
                continue
            spread = _to_number(row.get("spread_favored_team"))
            if spread is None:
                continue
            home_score = _score_from_row(row, "home")
            away_score = _score_from_row(row, "away")
            if home_score is None or away_score is None:
                continue

            home_team = str(row.get("home_team_norm") or "").lower()
            away_team = str(row.get("away_team_norm") or "").lower()
            if not home_team or not away_team:
                continue

            margin = float(home_score) - float(away_score)
            if favored_side == "HOME":
                cover_score = margin + float(spread)
                favored = home_team
                dog = away_team
            else:
                cover_score = (-margin) + float(spread)
                favored = away_team
                dog = home_team

            stats.setdefault(favored, {"w": 0, "l": 0, "p": 0})
            stats.setdefault(dog, {"w": 0, "l": 0, "p": 0})
            games_considered += 1

            if cover_score > EPSILON:
                stats[favored]["w"] += 1
                stats[dog]["l"] += 1
            elif cover_score < -EPSILON:
                stats[favored]["l"] += 1
                stats[dog]["w"] += 1
            else:
                stats[favored]["p"] += 1
                stats[dog]["p"] += 1

    build_team_ats.meta = {"weeks_scanned": sorted(weeks_scanned), "games_considered": games_considered}
    return stats


def apply_ats_to_week(season: int, week: int, team_ats: Mapping[str, Mapping[str, int]]) -> int:
    """Apply ATS tallies to the current week's Game View rows."""
    if os.getenv("ATS_ENABLE", "1").strip().lower() in {"0", "false", "off", "disabled"}:
        apply_ats_to_week.teams_in_week = 0
        return 0

    json_path, csv_path = _week_path(season, week)
    rows = _load_week_rows(json_path)
    if not rows:
        apply_ats_to_week.teams_in_week = 0
        return 0

    rows_updated = 0
    teams_in_week = set()

    updated_rows: List[dict] = []
    for row in rows:
        row_changed = False
        for side in ("home", "away"):
            team_norm = str(row.get(f"{side}_team_norm") or "").lower()
            if team_norm:
                teams_in_week.add(team_norm)
            stats = team_ats.get(team_norm) or {}
            ats_value = _format_ats(stats)
            ats_string = ats_value if ats_value is not None else "—"
            if row.get(f"{side}_ats") != ats_string:
                row[f"{side}_ats"] = ats_string
                row_changed = True
        if row_changed:
            rows_updated += 1
        updated_rows.append(row)

    if rows_updated > 0:
        write_atomic_jsonl(json_path, updated_rows)
        df = pd.DataFrame(updated_rows)
        write_atomic_csv(csv_path, df)

    apply_ats_to_week.teams_in_week = len(teams_in_week)
    return rows_updated


apply_ats_to_week.teams_in_week = 0  # type: ignore[attr-defined]

__all__ = ["build_team_ats", "apply_ats_to_week"]

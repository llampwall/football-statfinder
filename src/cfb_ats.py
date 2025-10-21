"""Helpers for computing College Football ATS records from prior weeks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.common.io_utils import write_jsonl, getenv

OUT_ROOT = Path(__file__).resolve().parents[1] / "out"
CFB_ROOT = OUT_ROOT / "cfb"
EPSILON = 1e-6


def _week_path(season: int, week: int) -> Path:
    return CFB_ROOT / f"{season}_week{week}" / f"games_week_{season}_{week}.jsonl"


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_number(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
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


def _score_from_row(row: dict, side: str) -> Optional[float]:
    candidates = [
        f"{side}_score",
        f"{side}_points",
        f"{side}_final",
        f"{side}Score",
        f"score_{side}",
    ]
    for key in candidates:
        num = _to_number(row.get(key))
        if num is not None:
            return num
    sources = row.get("raw_sources", {})
    schedule_row = sources.get("schedule_row") if isinstance(sources, dict) else None
    if isinstance(schedule_row, dict):
        for key in candidates:
            num = _to_number(schedule_row.get(key))
            if num is not None:
                return num
    return None


def _team_key(row: dict, side: str) -> Optional[str]:
    primary = row.get(f"{side}_team_norm")
    fallback = row.get(f"{side}_team_raw")
    for candidate in (primary, fallback):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _ensure_team(stats: Dict[str, Dict[str, int]], team: str) -> Dict[str, int]:
    return stats.setdefault(team, {"w": 0, "l": 0, "p": 0})


def _is_valid_ats_row(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    if _score_from_row(row, "home") is None or _score_from_row(row, "away") is None:
        return False
    favored_side = str(row.get("favored_side") or "").upper()
    if favored_side not in {"HOME", "AWAY"}:
        return False
    spread = _to_number(row.get("spread_favored_team"))
    if spread is None or not math.isfinite(spread):
        return False
    if not _team_key(row, "home") or not _team_key(row, "away"):
        return False
    return True


def build_team_ats(season: int, week: int) -> Dict[str, Dict[str, int]]:
    """Return {team_norm: {w: int, l: int, p: int}} using weeks 1..week-1."""
    team_stats: Dict[str, Dict[str, int]] = {}
    weeks_scanned: List[int] = []
    games_considered = 0

    if week <= 1:
        build_team_ats.meta = {"weeks_scanned": weeks_scanned, "games_considered": games_considered}
        return team_stats

    for prior_week in range(1, week):
        path = _week_path(season, prior_week)
        if not path.exists():
            continue
        weeks_scanned.append(prior_week)
        for row in _iter_jsonl(path):
            if not _is_valid_ats_row(row):
                continue
            home_team = _team_key(row, "home")
            away_team = _team_key(row, "away")
            if not home_team or not away_team:
                continue
            home_score = _score_from_row(row, "home")
            away_score = _score_from_row(row, "away")
            spread = _to_number(row.get("spread_favored_team"))
            favored_side = str(row.get("favored_side")).upper()
            if None in (home_score, away_score, spread):
                continue

            margin = float(home_score) - float(away_score)
            if favored_side == "HOME":
                cover_score = margin + float(spread)
                favored, dog = home_team, away_team
            else:
                cover_score = (-margin) + float(spread)
                favored, dog = away_team, home_team

            _ensure_team(team_stats, favored)
            _ensure_team(team_stats, dog)
            games_considered += 1
            if cover_score > EPSILON:
                team_stats[favored]["w"] += 1
                team_stats[dog]["l"] += 1
            elif cover_score < -EPSILON:
                team_stats[favored]["l"] += 1
                team_stats[dog]["w"] += 1
            else:
                team_stats[favored]["p"] += 1
                team_stats[dog]["p"] += 1

    weeks_scanned = sorted(set(weeks_scanned))
    build_team_ats.meta = {"weeks_scanned": weeks_scanned, "games_considered": games_considered}
    return team_stats


def _format_ats(entry: Dict[str, int]) -> Optional[str]:
    wins = int(entry.get("w", 0))
    losses = int(entry.get("l", 0))
    pushes = int(entry.get("p", 0))
    total = wins + losses + pushes
    if total <= 0:
        return None
    return f"{wins}-{losses}-{pushes}"


def apply_ats_to_week(season: int, week: int, team_ats: Dict[str, Dict[str, int]]) -> int:
    """Update games_week JSONL with ATS strings and split counts."""
    path = _week_path(season, week)
    if not path.exists():
        apply_ats_to_week.teams_in_week = 0
        apply_ats_to_week.zero_lined = 0
        return 0

    dry_run = getenv("CFB_ATS_DRYRUN", "").strip() == "1"
    updated_rows = 0
    rows: List[dict] = []
    teams_this_week: set[str] = set()

    for row in _iter_jsonl(path):
        changed = False
        for side in ("home", "away"):
            team = _team_key(row, side)
            if team:
                teams_this_week.add(team)
            stats = team_ats.get(team) if team else None
            total_games = (
                (stats.get("w", 0) + stats.get("l", 0) + stats.get("p", 0)) if stats else 0
            )
            ats_field = f"{side}_ats"

            if stats and total_games > 0:
                ats_value = _format_ats(stats)
                if ats_value is not None and row.get(ats_field) != ats_value:
                    row[ats_field] = ats_value
                    changed = True
                for suffix in ("w", "l", "p"):
                    field_name = f"{side}_ats_{suffix}"
                    if field_name in row:
                        value = int(stats.get(suffix, 0))
                        if row.get(field_name) != value:
                            row[field_name] = value
                            changed = True
            else:
                if row.get(ats_field) is not None:
                    row[ats_field] = None
                    changed = True
                for suffix in ("w", "l", "p"):
                    field_name = f"{side}_ats_{suffix}"
                    if field_name in row and row.get(field_name) is not None:
                        row[field_name] = None
                        changed = True

        if changed:
            updated_rows += 1
        rows.append(row)

    zero_lined = sum(
        1
        for team in teams_this_week
        if team not in team_ats
        or (team_ats[team]["w"] + team_ats[team]["l"] + team_ats[team]["p"] == 0)
    )
    apply_ats_to_week.teams_in_week = len(teams_this_week)
    apply_ats_to_week.zero_lined = zero_lined

    if dry_run:
        return 0

    tmp_path = path.with_suffix(".jsonl.tmp")
    write_jsonl(rows, tmp_path)
    tmp_path.replace(path)
    return updated_rows


apply_ats_to_week.teams_in_week = 0  # type: ignore[attr-defined]
apply_ats_to_week.zero_lined = 0  # type: ignore[attr-defined]

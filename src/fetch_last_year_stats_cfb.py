
"""Build prior-season College Football end-of-year league metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from src.common.cfb_source import (
    fetch_cfbd_fbs_teams,
    fetch_cfbd_team_game_stats,
)
from src.common.io_utils import ensure_out_dir, read_env, write_csv
from src.common.metrics import dense_rank
from src.common.team_names_cfb import normalize_team_name_cfb_stats, team_merge_key_cfb

REQUIRED_FIELDS = 14
FIELDS_HIT_THRESHOLD = 10
MIN_TEAMS_ABS = 100
MIN_TEAMS_FRAC = 0.70
RANK_OK_THRESHOLD = 0.75
ATS_NOTE = "ATS unavailable for CFB S2D; intentionally blank"

CSV_COLUMNS = [
    "Team",
    "RY(O)",
    "R(O)_RY",
    "PY(O)",
    "R(O)_PY",
    "TY(O)",
    "R(O)_TY",
    "RY(D)",
    "R(D)_RY",
    "PY(D)",
    "R(D)_PY",
    "TY(D)",
    "R(D)_TY",
    "TO",
    "PF",
    "PA",
    "SU",
    "ATS",
]

NUMERIC_FIELDS: List[str] = [
    "RY(O)",
    "PY(O)",
    "TY(O)",
    "RY(D)",
    "PY(D)",
    "TY(D)",
    "R(O)_RY",
    "R(O)_PY",
    "R(O)_TY",
    "R(D)_RY",
    "R(D)_PY",
    "R(D)_TY",
    "TO",
    "PF",
    "PA",
]

RANK_FIELDS = ["R(O)_RY", "R(O)_PY", "R(O)_TY", "R(D)_RY", "R(D)_PY", "R(D)_TY"]


def out_dir_cfb() -> Path:
    base = ensure_out_dir() / "cfb"
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_debug(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "-" in text:
            head = text.split("-", 1)[0]
            try:
                return float(head)
            except ValueError:
                return None
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    result = _safe_float(value)
    if result is None:
        return None
    return int(round(result))


def _extract_stat(stats: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in stats:
            return _safe_float(stats.get(key))
    return None


def _build_team_baseline(fbs_payload: List[dict]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    for entry in fbs_payload:
        raw_name = entry.get("school") or entry.get("team")
        if not raw_name:
            continue
        display = normalize_team_name_cfb_stats(raw_name)
        token = team_merge_key_cfb(display)
        if not token:
            continue
        teams.setdefault(
            token,
            {
                "team_display": display,
                "team_raws": set(),
                "games": 0,
                "ry": 0.0,
                "py": 0.0,
                "ty": 0.0,
                "ry_allowed": 0.0,
                "py_allowed": 0.0,
                "ty_allowed": 0.0,
                "pf": 0.0,
                "pa": 0.0,
                "giveaways": 0.0,
                "takeaways": 0.0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
            },
        )
    return teams


def _parse_team_entry(team_entry: dict) -> Tuple[str, dict]:
    team_name = team_entry.get("team") or ""
    display = normalize_team_name_cfb_stats(team_name)
    token = team_merge_key_cfb(display)
    stats_map = {
        item.get("category"): item.get("stat")
        for item in team_entry.get("stats") or []
        if isinstance(item, dict) and item.get("category")
    }
    rushing = _extract_stat(stats_map, ("rushingYards",))
    passing = _extract_stat(stats_map, ("netPassingYards", "passingYards"))
    total = _extract_stat(stats_map, ("totalYards",))
    turnovers = _extract_stat(stats_map, ("turnovers",))
    points = _safe_int(team_entry.get("points"))
    return token, {
        "team_name": team_name,
        "team_display": display,
        "points": points,
        "rushing": rushing or 0.0,
        "passing": passing or 0.0,
        "total": total,
        "turnovers": turnovers or 0.0,
    }


def _collect_season_games(season: int, api_key: str) -> Tuple[List[dict], List[str]]:
    games: List[dict] = []
    notes: List[str] = []
    for season_type in ("regular", "postseason"):
        weeks_seen: List[int] = []
        empty_streak = 0
        for week in range(1, 21):
            try:
                payload = fetch_cfbd_team_game_stats(
                    season,
                    week,
                    api_key,
                    season_type=season_type,
                ) or []
            except requests.RequestException as exc:  # pragma: no cover - network guard
                raise RuntimeError(
                    f"CFBD games/teams fetch failed for {season_type} week {week}: {exc}"
                ) from exc
            if not payload:
                empty_streak += 1
                if empty_streak >= 2:
                    break
                continue
            empty_streak = 0
            weeks_seen.append(week)
            games.extend(payload)
        if weeks_seen:
            notes.append(
                f"cfbd endpoints used: /games/teams ({season_type} weeks {weeks_seen[0]}-{weeks_seen[-1]})"
            )
    return games, notes


def _aggregate_games(
    games_payload: List[dict],
    teams: Dict[str, dict],
) -> None:
    for game in games_payload:
        team_entries = game.get("teams") or []
        if len(team_entries) < 1:
            continue
        parsed = []
        for entry in team_entries:
            token, parsed_entry = _parse_team_entry(entry)
            parsed_entry["token"] = token
            parsed.append(parsed_entry)
        if not parsed:
            continue
        for idx, team_info in enumerate(parsed):
            token = team_info["token"]
            if token not in teams:
                continue
            opponent_info = None
            if len(parsed) == 2:
                opponent_info = parsed[1 - idx]
            else:
                for j, candidate in enumerate(parsed):
                    if j != idx:
                        opponent_info = candidate
                        break
            if opponent_info is None:
                continue
            record = teams[token]
            record["team_raws"].add(team_info["team_name"])
            record["games"] += 1
            team_points = team_info["points"]
            opp_points = opponent_info.get("points")
            record["pf"] += float(team_points or 0)
            record["pa"] += float(opp_points or 0)
            record["ry"] += float(team_info["rushing"] or 0.0)
            record["py"] += float(team_info["passing"] or 0.0)
            team_total = (
                float(team_info["total"])
                if team_info["total"] is not None
                else float(team_info["rushing"] or 0.0) + float(team_info["passing"] or 0.0)
            )
            opp_total = (
                float(opponent_info["total"])
                if opponent_info["total"] is not None
                else float(opponent_info["rushing"] or 0.0) + float(opponent_info["passing"] or 0.0)
            )
            record["ty"] += team_total
            record["ry_allowed"] += float(opponent_info["rushing"] or 0.0)
            record["py_allowed"] += float(opponent_info["passing"] or 0.0)
            record["ty_allowed"] += opp_total
            record["giveaways"] += float(team_info["turnovers"] or 0.0)
            record["takeaways"] += float(opponent_info["turnovers"] or 0.0)
            if team_points is not None and opp_points is not None:
                if team_points > opp_points:
                    record["wins"] += 1
                elif team_points < opp_points:
                    record["losses"] += 1
                else:
                    record["ties"] += 1


def _compute_ranks(per_team: Dict[str, dict]) -> Dict[str, Dict[str, Optional[int]]]:
    valid = {token: data for token, data in per_team.items() if data["games"] > 0}

    def _series(metric: str) -> pd.Series:
        return pd.Series(
            {token: data[metric] for token, data in valid.items() if data.get(metric) is not None},
            dtype=float,
        )

    ranks: Dict[str, Dict[str, Optional[int]]] = {token: {} for token in per_team}
    series_map = {
        "ry": True,
        "py": True,
        "ty": True,
        "ry_allowed": False,
        "py_allowed": False,
        "ty_allowed": False,
    }
    for metric, higher_is_better in series_map.items():
        series = _series(metric)
        if series.empty:
            continue
        rank_series = dense_rank(series, higher_is_better=higher_is_better).astype(int)
        for token, value in rank_series.to_dict().items():
            if metric == "ry":
                ranks[token]["R(O)_RY"] = value
            elif metric == "py":
                ranks[token]["R(O)_PY"] = value
            elif metric == "ty":
                ranks[token]["R(O)_TY"] = value
            elif metric == "ry_allowed":
                ranks[token]["R(D)_RY"] = value
            elif metric == "py_allowed":
                ranks[token]["R(D)_PY"] = value
            elif metric == "ty_allowed":
                ranks[token]["R(D)_TY"] = value
    return ranks


def _format_su(wins: int, losses: int, ties: int) -> str:
    if ties:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def build_rows(teams: Dict[str, dict]) -> List[dict]:
    ranks = _compute_ranks(teams)
    rows: List[dict] = []
    for token, data in sorted(teams.items(), key=lambda item: item[1]["team_display"]):
        games = data["games"]
        ry_pg = (data["ry"] / games) if games else None
        py_pg = (data["py"] / games) if games else None
        ty_pg = (data["ty"] / games) if games else None
        ry_allow_pg = (data["ry_allowed"] / games) if games else None
        py_allow_pg = (data["py_allowed"] / games) if games else None
        ty_allow_pg = (data["ty_allowed"] / games) if games else None
        to_margin_pg = ((data["takeaways"] - data["giveaways"]) / games) if games else None
        pf_pg = (data["pf"] / games) if games else None
        pa_pg = (data["pa"] / games) if games else None
        rows.append(
            {
                "_token": token,
                "Team": data["team_display"],
                "RY(O)": ry_pg,
                "R(O)_RY": ranks.get(token, {}).get("R(O)_RY"),
                "PY(O)": py_pg,
                "R(O)_PY": ranks.get(token, {}).get("R(O)_PY"),
                "TY(O)": ty_pg,
                "R(O)_TY": ranks.get(token, {}).get("R(O)_TY"),
                "RY(D)": ry_allow_pg,
                "R(D)_RY": ranks.get(token, {}).get("R(D)_RY"),
                "PY(D)": py_allow_pg,
                "R(D)_PY": ranks.get(token, {}).get("R(D)_PY"),
                "TY(D)": ty_allow_pg,
                "R(D)_TY": ranks.get(token, {}).get("R(D)_TY"),
                "TO": to_margin_pg,
                "PF": pf_pg,
                "PA": pa_pg,
                "SU": _format_su(data["wins"], data["losses"], data["ties"]),
                "ATS": "",
                "_games": games,
            }
        )
    return rows


def _format_csv_rows(rows: List[dict]) -> List[dict]:
    formatted: List[dict] = []
    for row in rows:
        formatted.append(
            {
                "Team": row["Team"],
                "RY(O)": f"{row['RY(O)']:.1f}" if row["RY(O)"] is not None else "",
                "R(O)_RY": "" if row["R(O)_RY"] is None else str(int(row["R(O)_RY"])),
                "PY(O)": f"{row['PY(O)']:.1f}" if row["PY(O)"] is not None else "",
                "R(O)_PY": "" if row["R(O)_PY"] is None else str(int(row["R(O)_PY"])),
                "TY(O)": f"{row['TY(O)']:.1f}" if row["TY(O)"] is not None else "",
                "R(O)_TY": "" if row["R(O)_TY"] is None else str(int(row["R(O)_TY"])),
                "RY(D)": f"{row['RY(D)']:.1f}" if row["RY(D)"] is not None else "",
                "R(D)_RY": "" if row["R(D)_RY"] is None else str(int(row["R(D)_RY"])),
                "PY(D)": f"{row['PY(D)']:.1f}" if row["PY(D)"] is not None else "",
                "R(D)_PY": "" if row["R(D)_PY"] is None else str(int(row["R(D)_PY"])),
                "TY(D)": f"{row['TY(D)']:.1f}" if row["TY(D)"] is not None else "",
                "R(D)_TY": "" if row["R(D)_TY"] is None else str(int(row["R(D)_TY"])),
                "TO": f"{row['TO']:.1f}" if row["TO"] is not None else "",
                "PF": f"{row['PF']:.1f}" if row["PF"] is not None else "",
                "PA": f"{row['PA']:.1f}" if row["PA"] is not None else "",
                "SU": row["SU"],
                "ATS": row["ATS"],
            }
        )
    return formatted


def _compute_coverage(rows: List[dict]) -> Tuple[int, int, int, float, float, Dict[str, List[str]]]:
    teams_total = len(rows)
    teams_with_metrics = sum(1 for row in rows if row["_games"] > 0)
    required_teams = max(MIN_TEAMS_ABS, math.ceil(MIN_TEAMS_FRAC * teams_total)) if teams_total else 0
    teams_meeting_threshold = 0
    missing_fields: Dict[str, List[str]] = {}
    for row in rows:
        games = row.get("_games", 0)
        present = 0
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if games == 0:
                value_check = None
            else:
                value_check = value
            if value_check not in (None, "") and not pd.isna(value_check):
                present += 1
        if present >= FIELDS_HIT_THRESHOLD:
            teams_meeting_threshold += 1
        else:
            token = row["_token"]
            missing = []
            for field in NUMERIC_FIELDS:
                value = row.get(field)
                if games == 0 or value in (None, "") or pd.isna(value):
                    missing.append(field)
            if missing:
                missing_fields[token] = missing
    rank_ok_count = 0
    for row in rows:
        ok = True
        for field in RANK_FIELDS:
            value = row.get(field)
            if value in (None, ""):
                ok = False
                break
            try:
                intval = int(value)
            except (TypeError, ValueError):
                ok = False
                break
            if intval < 1:
                ok = False
                break
        if ok:
            rank_ok_count += 1
    rank_fraction = rank_ok_count / teams_total if teams_total else 0.0
    covered_fraction = teams_with_metrics / teams_total if teams_total else 0.0
    return (
        teams_total,
        teams_with_metrics,
        teams_meeting_threshold,
        covered_fraction,
        rank_fraction,
        missing_fields,
    )


def _sample_missing_fields(missing: Dict[str, List[str]], limit: int = 10) -> Dict[str, List[str]]:
    sample: Dict[str, List[str]] = {}
    for token, fields in missing.items():
        sample[token] = fields[:]
        if len(sample) >= limit:
            break
    return sample


def run(season: int) -> int:
    prior_season = season - 1
    out_dir = out_dir_cfb()
    csv_path = out_dir / f"final_league_metrics_{prior_season}.csv"
    debug_path = out_dir / "final_league_metrics_debug.json"

    env = read_env(["CFBD_API_KEY"])
    api_key = env.get("CFBD_API_KEY")
    if not api_key:
        payload = {
            "season": prior_season,
            "teams_total": 0,
            "teams_with_metrics": 0,
            "covered_fraction": 0.0,
            "fields_required_per_team": REQUIRED_FIELDS,
            "fields_hit_threshold": FIELDS_HIT_THRESHOLD,
            "teams_meeting_threshold": 0,
            "rank_columns_ok_fraction": 0.0,
            "missing_fields_per_team_sample": {},
            "source_notes": [ATS_NOTE],
            "error": "CFBD_API_KEY missing",
        }
        write_debug(debug_path, payload)
        print(f"FAIL: CFB final league metrics missing CFBD_API_KEY. See {debug_path}")
        return 1

    try:
        fbs_payload = fetch_cfbd_fbs_teams(prior_season, api_key) or []
    except requests.RequestException as exc:
        payload = {
            "season": prior_season,
            "teams_total": 0,
            "teams_with_metrics": 0,
            "covered_fraction": 0.0,
            "fields_required_per_team": REQUIRED_FIELDS,
            "fields_hit_threshold": FIELDS_HIT_THRESHOLD,
            "teams_meeting_threshold": 0,
            "rank_columns_ok_fraction": 0.0,
            "missing_fields_per_team_sample": {},
            "source_notes": ["cfbd endpoint error: /teams/fbs", ATS_NOTE],
            "error": str(exc),
        }
        write_debug(debug_path, payload)
        print(f"FAIL: CFB final league metrics unable to load FBS teams. See {debug_path}")
        return 1

    if not fbs_payload:
        payload = {
            "season": prior_season,
            "teams_total": 0,
            "teams_with_metrics": 0,
            "covered_fraction": 0.0,
            "fields_required_per_team": REQUIRED_FIELDS,
            "fields_hit_threshold": FIELDS_HIT_THRESHOLD,
            "teams_meeting_threshold": 0,
            "rank_columns_ok_fraction": 0.0,
            "missing_fields_per_team_sample": {},
            "source_notes": ["cfbd endpoints used: /teams/fbs (empty)", ATS_NOTE],
            "error": "CFBD returned no FBS teams",
        }
        write_debug(debug_path, payload)
        print(f"FAIL: CFB final league metrics FBS roster empty. See {debug_path}")
        return 1

    teams = _build_team_baseline(fbs_payload)
    try:
        games_payload, notes = _collect_season_games(prior_season, api_key)
    except RuntimeError as exc:
        payload = {
            "season": prior_season,
            "teams_total": len(teams),
            "teams_with_metrics": 0,
            "covered_fraction": 0.0,
            "fields_required_per_team": REQUIRED_FIELDS,
            "fields_hit_threshold": FIELDS_HIT_THRESHOLD,
            "teams_meeting_threshold": 0,
            "rank_columns_ok_fraction": 0.0,
            "missing_fields_per_team_sample": {},
            "source_notes": notes + [ATS_NOTE],
            "error": str(exc),
        }
        write_debug(debug_path, payload)
        print(f"FAIL: CFB final league metrics failed to collect games. See {debug_path}")
        return 1

    _aggregate_games(games_payload, teams)
    league_rows = build_rows(teams)
    csv_rows = _format_csv_rows(league_rows)
    df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
    write_csv(df, csv_path)

    (
        teams_total,
        teams_with_metrics,
        teams_meeting_threshold,
        covered_fraction,
        rank_fraction,
        missing_fields,
    ) = _compute_coverage(league_rows)

    source_notes = [f"cfbd endpoints used: /teams/fbs (season={prior_season})"] + notes + [ATS_NOTE]

    debug_payload = {
        "season": prior_season,
        "teams_total": teams_total,
        "teams_with_metrics": teams_with_metrics,
        "covered_fraction": covered_fraction,
        "fields_required_per_team": REQUIRED_FIELDS,
        "fields_hit_threshold": FIELDS_HIT_THRESHOLD,
        "teams_meeting_threshold": teams_meeting_threshold,
        "rank_columns_ok_fraction": rank_fraction,
        "missing_fields_per_team_sample": _sample_missing_fields(missing_fields),
        "source_notes": source_notes,
    }
    write_debug(debug_path, debug_payload)

    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    required_teams = max(MIN_TEAMS_ABS, math.ceil(MIN_TEAMS_FRAC * teams_total)) if teams_total else 0

    failure = False
    if not csv_exists:
        failure = True
    if teams_meeting_threshold < required_teams:
        failure = True
    if rank_fraction < RANK_OK_THRESHOLD:
        failure = True

    if failure:
        print(
            f"FAIL: CFB final league metrics coverage insufficient "
            f"(teams_meeting_threshold={teams_meeting_threshold}, required={required_teams}, "
            f"rank_fraction={rank_fraction:.2f}). See {debug_path}"
        )
        return 1

    print(
        f"PASS: CFB final league metrics teams {teams_with_metrics}/{teams_total}; "
        f"ranks_ok={rank_fraction:.2f}; wrote {csv_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build prior-season CFB final league metrics.")
    parser.add_argument("--season", type=int, required=True, help="Current season (prior season will be used).")
    args = parser.parse_args()
    return run(args.season)


if __name__ == "__main__":
    raise SystemExit(main())

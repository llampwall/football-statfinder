"""Build College Football Game View outputs (mirrors NFL builder flow)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl
from src.common.team_names_cfb import team_merge_key_cfb
from src.fetch_games_cfb import load_games

if Path(__file__).read_text(encoding="utf-8").count("STUB:") > 1:
    raise RuntimeError("gameview_build_cfb still contains a stub marker; aborting.")

OUTPUT_COLUMNS: List[str] = [
    "season",
    "week",
    "kickoff_iso_utc",
    "game_key",
    "source_uid",
    "home_team_raw",
    "home_team_norm",
    "away_team_raw",
    "away_team_norm",
    "spread_home_relative",
    "total",
    "moneyline_home",
    "moneyline_away",
    "odds_source",
    "is_closing",
    "snapshot_at",
    "home_pr",
    "home_pr_rank",
    "away_pr",
    "away_pr_rank",
    "home_sos",
    "away_sos",
    "home_sos_rank",
    "away_sos_rank",
    "hfa",
    "rating_diff",
    "rating_vs_odds",
    "favored_side",
    "spread_favored_team",
    "rating_diff_favored_team",
    "rating_vs_odds_favored_team",
    "home_pf_pg",
    "home_pa_pg",
    "home_ry_pg",
    "home_py_pg",
    "home_ty_pg",
    "home_ry_allowed_pg",
    "home_py_allowed_pg",
    "home_ty_allowed_pg",
    "home_to_margin_pg",
    "home_su",
    "home_ats",
    "home_rush_rank",
    "home_pass_rank",
    "home_tot_off_rank",
    "home_rush_def_rank",
    "home_pass_def_rank",
    "home_tot_def_rank",
    "away_pf_pg",
    "away_pa_pg",
    "away_ry_pg",
    "away_py_pg",
    "away_ty_pg",
    "away_ry_allowed_pg",
    "away_py_allowed_pg",
    "away_ty_allowed_pg",
    "away_to_margin_pg",
    "away_su",
    "away_ats",
    "away_rush_rank",
    "away_pass_rank",
    "away_tot_off_rank",
    "away_rush_def_rank",
    "away_pass_def_rank",
    "away_tot_def_rank",
    "raw_sources",
]

SAGARIN_TOKEN_OVERRIDES: Dict[str, str] = {
    "appalachianstate": "appstate",
    "armywestpoint": "army",
    "centralfloridaucf": "ucf",
    "connecticut": "uconn",
    "flainternational": "floridainternational",
    "louisianalafayette": "louisiana",
    "louisianamonroeulm": "ulmonroe",
    "miamiflorida": "miami",
    "miamiohio": "miamioh",
    "mississippi": "olemiss",
    "samhoustonstate": "samhouston",
    "sanjosestate": "sanjosstate",
    "southerncalifornia": "usctrojans",
}

RECEIPT_NAME = "gameview_build_receipt.json"


def _cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _round_float(value: Any, digits: int = 1) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return round(val, digits)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        return None
    return val


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if pd.isna(value):
        return None
    return str(value)


def _clean_structure(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_structure(v) for v in obj]
    if isinstance(obj, (pd.Series,)):
        return _clean_structure(obj.to_dict())
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - best effort conversion
            pass
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def _round_tenth(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return None


def _f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _compute_rating_edge(row: Dict[str, Any], hfa: Any) -> Optional[float]:
    favored = (row.get("favored_side") or "").upper()
    home_pr = _f(row.get("home_pr"))
    away_pr = _f(row.get("away_pr"))
    if favored not in ("HOME", "AWAY") or home_pr is None or away_pr is None:
        return None
    hfa_val = _f(hfa) or 0.0
    if favored == "HOME":
        favored_pr, unfav_pr = home_pr + hfa_val, away_pr
    else:
        favored_pr, unfav_pr = away_pr, home_pr + hfa_val
    return _round_tenth(favored_pr - unfav_pr)


def _compute_rating_vs_odds(row: Dict[str, Any], hfa: Any) -> Tuple[Optional[float], Optional[float]]:
    edge = _compute_rating_edge(row, hfa)
    spread = _f(row.get("spread_favored_team"))
    if edge is None or spread is None:
        return None, edge
    return _round_tenth(edge + spread), edge


def _read_schedule(season: int, week: int, base_dir: Path) -> pd.DataFrame:
    schedule_path = base_dir / "_schedule_norm.csv"
    if schedule_path.exists():
        schedule_df = pd.read_csv(schedule_path)
    else:
        schedule_df = load_games(season)
    if schedule_df.empty:
        return schedule_df.copy()
    work = schedule_df.copy()
    work["week"] = pd.to_numeric(work.get("week"), errors="coerce")
    work = work[work["week"] == week]
    if work.empty:
        return work
    for column in ("season", "week"):
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("Int64")
    if "conference_game" in work.columns:
        work["conference_game"] = work["conference_game"].map(
            lambda val: bool(str(val).strip().lower() == "true")
        )
    else:
        work["conference_game"] = False
    return work.reset_index(drop=True)


def _load_metrics_map(metrics_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(metrics_path)
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        token = team_merge_key_cfb(row.get("Team"))
        metrics_payload = {
            "pf_pg": _round_float(row.get("PF"), 1),
            "pa_pg": _round_float(row.get("PA"), 1),
            "ry_pg": _round_float(row.get("RY(O)"), 1),
            "py_pg": _round_float(row.get("PY(O)"), 1),
            "ty_pg": _round_float(row.get("TY(O)"), 1),
            "ry_allowed_pg": _round_float(row.get("RY(D)"), 1),
            "py_allowed_pg": _round_float(row.get("PY(D)"), 1),
            "ty_allowed_pg": _round_float(row.get("TY(D)"), 1),
            "to_margin_pg": _round_float(row.get("TO"), 1),
            "rush_rank": _int_or_none(row.get("R(O)_RY")),
            "pass_rank": _int_or_none(row.get("R(O)_PY")),
            "tot_off_rank": _int_or_none(row.get("R(O)_TY")),
            "rush_def_rank": _int_or_none(row.get("R(D)_RY")),
            "pass_def_rank": _int_or_none(row.get("R(D)_PY")),
            "tot_def_rank": _int_or_none(row.get("R(D)_TY")),
            "su": _clean_text(row.get("SU")),
            "ats": _clean_text(row.get("ATS")),
        }
        mapping[token] = {
            "metrics": metrics_payload,
            "raw": _clean_structure(row.to_dict()),
        }
    return mapping


def _load_odds_map(odds_path: Path) -> Dict[str, Dict[str, Any]]:
    if not odds_path.exists():
        return {}
    odds_map: Dict[str, Dict[str, Any]] = {}
    with odds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            game_key = payload.get("game_key")
            if not game_key:
                continue
            odds_map[game_key] = payload
    return odds_map


def _sagarin_token(name: str) -> str:
    base = team_merge_key_cfb(name)
    return SAGARIN_TOKEN_OVERRIDES.get(base, base)


def _load_sagarin_map(sagarin_path: Path) -> Dict[str, Dict[str, Any]]:
    if not sagarin_path.exists():
        return {}
    df = pd.read_json(sagarin_path, lines=True)
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        token = _sagarin_token(row.get("team_norm"))
        mapping[token] = {
            "pr": _round_float(row.get("pr"), 2),
            "pr_rank": _int_or_none(row.get("pr_rank")),
            "sos": _round_float(row.get("sos"), 2),
            "sos_rank": _int_or_none(row.get("sos_rank")),
            "hfa": _round_float(row.get("hfa"), 2),
            "raw": _clean_structure(row.to_dict()),
        }
    return mapping


def _schedule_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": _int_or_none(row.get("id")),
        "season": _int_or_none(row.get("season")),
        "week": _int_or_none(row.get("week")),
        "kickoff_iso_utc": row.get("kickoff_iso_utc"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "venue": row.get("venue"),
        "conference_game": bool(row.get("conference_game")),
    }


def _game_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "game_key": record.get("game_key"),
        "home": record.get("home_team_norm"),
        "away": record.get("away_team_norm"),
        "kickoff_iso_utc": record.get("kickoff_iso_utc"),
    }


def _compute_rating_vectors(
    home_pr: Optional[float],
    away_pr: Optional[float],
    favored_side: Optional[str],
    spread_favored_team: Optional[float],
    spread_home_relative: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if home_pr is None or away_pr is None:
        return None, None, None, None
    rating_delta = round(home_pr - away_pr, 2)
    rating_vs_odds_value: Optional[float] = None
    rating_diff_favored_team: Optional[float] = None
    rating_vs_odds_favored_team: Optional[float] = None

    if favored_side in ("HOME", "AWAY") and spread_favored_team is not None:
        spread_mag = abs(spread_favored_team)
        home_line = spread_mag if favored_side == "HOME" else -spread_mag
        rating_vs_odds_value = round(rating_delta - home_line, 2)
        favored_diff = rating_delta if favored_side == "HOME" else -rating_delta
        favored_spread = spread_favored_team
        rating_diff_favored_team = round(favored_diff, 2)
        rating_vs_odds_favored_team = round(favored_diff - favored_spread, 2)
    elif spread_home_relative is not None:
        rating_vs_odds_value = round(rating_delta - spread_home_relative, 2)

    return rating_delta, rating_vs_odds_value, rating_diff_favored_team, rating_vs_odds_favored_team


def build_game_records(
    season: int,
    week: int,
    relax_odds: bool = False,
    relax_sagarin: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_dir = _cfb_week_dir(season, week)
    schedule_df = _read_schedule(season, week, base_dir)
    metrics_path = base_dir / f"league_metrics_{season}_{week}.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"CFB league metrics CSV missing: {metrics_path}")
    metrics_map = _load_metrics_map(metrics_path)
    odds_path = base_dir / f"odds_{season}_wk{week}.jsonl"
    odds_map = _load_odds_map(odds_path)
    sagarin_path = base_dir / f"sagarin_cfb_{season}_wk{week}.jsonl"
    sagarin_map = _load_sagarin_map(sagarin_path)

    if schedule_df.empty:
        raise RuntimeError("CFB schedule empty; cannot build Game View outputs.")

    schedule_df["home_token"] = schedule_df["home_team_norm"].map(team_merge_key_cfb)
    schedule_df["away_token"] = schedule_df["away_team_norm"].map(team_merge_key_cfb)

    fbs_mask = schedule_df.apply(
        lambda row: (row["home_token"] in metrics_map) and (row["away_token"] in metrics_map), axis=1
    )
    fbs_df = schedule_df[fbs_mask].copy().reset_index(drop=True)

    fbs_rows = len(fbs_df)

    records: List[Dict[str, Any]] = []
    joined_metrics_rows = 0
    joined_odds_rows = 0
    joined_sagarin_rows = 0
    metrics_capped = 0
    odds_games = 0
    sagarin_games = 0
    missing_odds: List[Dict[str, Any]] = []
    missing_sagarin: List[Dict[str, Any]] = []
    metrics_gaps: List[Dict[str, Any]] = []

    for row_dict in fbs_df.to_dict(orient="records"):
        record = {column: None for column in OUTPUT_COLUMNS}
        record.update(
            {
                "season": _int_or_none(row_dict.get("season")),
                "week": _int_or_none(row_dict.get("week")),
                "kickoff_iso_utc": row_dict.get("kickoff_iso_utc"),
                "game_key": row_dict.get("game_key"),
                "source_uid": _int_or_none(row_dict.get("id")),
                "home_team_raw": row_dict.get("home_team"),
                "home_team_norm": row_dict.get("home_team_norm"),
                "away_team_raw": row_dict.get("away_team"),
                "away_team_norm": row_dict.get("away_team_norm"),
            }
        )

        home_token = row_dict["home_token"]
        away_token = row_dict["away_token"]
        home_metrics = metrics_map.get(home_token)
        away_metrics = metrics_map.get(away_token)
        if home_metrics and away_metrics:
            joined_metrics_rows += 1
            home_vals = home_metrics["metrics"]
            away_vals = away_metrics["metrics"]
            record.update(
                {
                    "home_pf_pg": home_vals["pf_pg"],
                    "home_pa_pg": home_vals["pa_pg"],
                    "home_ry_pg": home_vals["ry_pg"],
                    "home_py_pg": home_vals["py_pg"],
                    "home_ty_pg": home_vals["ty_pg"],
                    "home_ry_allowed_pg": home_vals["ry_allowed_pg"],
                    "home_py_allowed_pg": home_vals["py_allowed_pg"],
                    "home_ty_allowed_pg": home_vals["ty_allowed_pg"],
                    "home_to_margin_pg": home_vals["to_margin_pg"],
                    "home_su": home_vals["su"],
                    "home_ats": home_vals["ats"],
                    "home_rush_rank": home_vals["rush_rank"],
                    "home_pass_rank": home_vals["pass_rank"],
                    "home_tot_off_rank": home_vals["tot_off_rank"],
                    "home_rush_def_rank": home_vals["rush_def_rank"],
                    "home_pass_def_rank": home_vals["pass_def_rank"],
                    "home_tot_def_rank": home_vals["tot_def_rank"],
                    "away_pf_pg": away_vals["pf_pg"],
                    "away_pa_pg": away_vals["pa_pg"],
                    "away_ry_pg": away_vals["ry_pg"],
                    "away_py_pg": away_vals["py_pg"],
                    "away_ty_pg": away_vals["ty_pg"],
                    "away_ry_allowed_pg": away_vals["ry_allowed_pg"],
                    "away_py_allowed_pg": away_vals["py_allowed_pg"],
                    "away_ty_allowed_pg": away_vals["ty_allowed_pg"],
                    "away_to_margin_pg": away_vals["to_margin_pg"],
                    "away_su": away_vals["su"],
                    "away_ats": away_vals["ats"],
                    "away_rush_rank": away_vals["rush_rank"],
                    "away_pass_rank": away_vals["pass_rank"],
                    "away_tot_off_rank": away_vals["tot_off_rank"],
                    "away_rush_def_rank": away_vals["rush_def_rank"],
                    "away_pass_def_rank": away_vals["pass_def_rank"],
                    "away_tot_def_rank": away_vals["tot_def_rank"],
                }
            )
            if home_vals["pf_pg"] is not None and away_vals["pf_pg"] is not None:
                metrics_capped += 1
        else:
            metrics_gaps.append(_game_sample(record))

        odds_payload = odds_map.get(record["game_key"])
        spread_home_relative = None
        favored_side = None
        spread_favored_team = None
        if odds_payload:
            joined_odds_rows += 1
            spread_home_relative = _round_float(odds_payload.get("spread_home_relative"), 2)
            spread_favored_team = _round_float(odds_payload.get("spread_favored_team"), 2)
            record.update(
                {
                    "spread_home_relative": spread_home_relative,
                    "total": _round_float(odds_payload.get("total"), 1),
                    "moneyline_home": _int_or_none(odds_payload.get("moneyline_home")),
                    "moneyline_away": _int_or_none(odds_payload.get("moneyline_away")),
                    "odds_source": odds_payload.get("odds_source"),
                    "is_closing": bool(odds_payload.get("is_closing")),
                    "snapshot_at": odds_payload.get("snapshot_at"),
                    "favored_side": (odds_payload.get("favored_side") or "").upper() or None,
                    "spread_favored_team": spread_favored_team,
                }
            )
            favored_side = (odds_payload.get("favored_side") or "").upper() or None
            record["favored_side"] = favored_side
            odds_games += 1
        else:
            missing_odds.append(_game_sample(record))

        home_sagarin = sagarin_map.get(home_token)
        away_sagarin = sagarin_map.get(away_token)
        if home_sagarin and away_sagarin:
            joined_sagarin_rows += 1
            hfa_value: Optional[float]
            if home_sagarin["hfa"] is not None:
                hfa_value = home_sagarin["hfa"]
            elif away_sagarin["hfa"] is not None:
                hfa_value = away_sagarin["hfa"]
            else:
                hfa_value = 0.0
            record.update(
                {
                    "home_pr": home_sagarin["pr"],
                    "home_pr_rank": home_sagarin["pr_rank"],
                    "home_sos": home_sagarin["sos"],
                    "home_sos_rank": home_sagarin["sos_rank"],
                    "away_pr": away_sagarin["pr"],
                    "away_pr_rank": away_sagarin["pr_rank"],
                    "away_sos": away_sagarin["sos"],
                    "away_sos_rank": away_sagarin["sos_rank"],
                    "hfa": hfa_value,
                }
            )
            rating_delta, rating_vs_odds_val, rating_diff_fav, rating_vs_odds_fav = _compute_rating_vectors(
                home_sagarin["pr"],
                away_sagarin["pr"],
                favored_side,
                spread_favored_team,
                spread_home_relative,
            )
            record["rating_diff"] = rating_delta
            record["rating_vs_odds"] = rating_vs_odds_val
            record["rating_diff_favored_team"] = rating_diff_fav
            record["rating_vs_odds_favored_team"] = rating_vs_odds_fav
            sagarin_games += 1
        else:
            missing_sagarin.append(_game_sample(record))

        # Ratings vs odds (favored perspective)
        hfa_used = record.get("hfa")
        rvo_value, edge_value = _compute_rating_vs_odds(record, hfa_used)
        record["rating_vs_odds"] = rvo_value
        record["rating_diff_favored_team"] = edge_value

        raw_sources = {
            "schedule_row": _schedule_payload(row_dict),
        }
        if home_metrics:
            raw_sources["league_metrics_home"] = home_metrics["raw"]
        if away_metrics:
            raw_sources["league_metrics_away"] = away_metrics["raw"]
        if odds_payload:
            raw_sources["odds_row"] = _clean_structure(odds_payload)
        if home_sagarin:
            raw_sources["sagarin_home"] = home_sagarin["raw"]
        if away_sagarin:
            raw_sources["sagarin_away"] = away_sagarin["raw"]
        record["raw_sources"] = raw_sources

        records.append(record)

    records.sort(key=lambda rec: (rec.get("kickoff_iso_utc") or "", rec.get("game_key") or ""))

    eligible = 0
    edges_count = 0
    rvo_count = 0
    mismatch_samples: List[Dict[str, Any]] = []
    for rec in records:
        favored = (rec.get("favored_side") or "").upper()
        has_spread = rec.get("spread_favored_team") is not None
        has_prs = rec.get("home_pr") is not None and rec.get("away_pr") is not None
        if favored in ("HOME", "AWAY") and has_spread and has_prs:
            eligible += 1
            if rec.get("rating_diff_favored_team") is not None:
                edges_count += 1
            if rec.get("rating_vs_odds") is not None:
                rvo_count += 1
            if rec.get("rating_vs_odds") is None or rec.get("rating_diff_favored_team") is None:
                mismatch_samples.append(
                    {
                        "game_key": rec.get("game_key"),
                        "favored_side": rec.get("favored_side"),
                        "home_pr": rec.get("home_pr"),
                        "away_pr": rec.get("away_pr"),
                        "hfa": rec.get("hfa"),
                        "spread_favored_team": rec.get("spread_favored_team"),
                        "edge": rec.get("rating_diff_favored_team"),
                        "rvo": rec.get("rating_vs_odds"),
                    }
                )

    spot_key = "20251019_0000_cincinnati_oklahoma_state"
    for rec in records:
        if rec.get("game_key") == spot_key:
            print(
                f"Ratings check {spot_key}: favored={rec.get('favored_side')} "
                f"spread={rec.get('spread_favored_team')} edge={rec.get('rating_diff_favored_team')} "
                f"rvo={rec.get('rating_vs_odds')}"
            )
            break

    coverage_metrics = (metrics_capped / fbs_rows) if fbs_rows else 0.0

    receipt = {
        "fbs_rows": fbs_rows,
        "joined_metrics_rows": joined_metrics_rows,
        "joined_odds_rows": joined_odds_rows,
        "joined_sagarin_rows": joined_sagarin_rows,
        "output_rows": len(records),
        "odds_games": odds_games,
        "sagarin_games": sagarin_games,
        "coverage_metrics": coverage_metrics,
        "notes": [
            f"FBS games: {fbs_rows}",
            f"Metrics coverage: {metrics_capped}/{fbs_rows}",
            f"Odds coverage: {odds_games}/{fbs_rows}",
            f"Sagarin coverage: {sagarin_games}/{fbs_rows}",
        ],
        "samples": {
            "missing_odds": missing_odds[:10],
            "missing_sagarin": missing_sagarin[:10],
            "metrics_gaps": metrics_gaps[:10],
        },
        "ratings": {
            "eligible": eligible,
            "edges_count": edges_count,
            "rvo_count": rvo_count,
        },
    }

    odds_file_present = odds_path.exists() and odds_path.stat().st_size > 0
    sagarin_file_present = sagarin_path.exists() and sagarin_path.stat().st_size > 0

    failures: List[str] = []
    if len(records) != fbs_rows:
        failures.append("output row count mismatch")
    if coverage_metrics < 0.90:
        failures.append(f"metrics coverage below threshold ({coverage_metrics:.2%})")
    if odds_file_present and not relax_odds:
        required_odds = 0.60 * fbs_rows
        if odds_games < required_odds:
            failures.append("odds coverage below threshold")
    if sagarin_file_present and not relax_sagarin:
        required_sagarin = 0.90 * fbs_rows
        if sagarin_games < required_sagarin:
            failures.append("sagarin coverage below threshold")
    if eligible > 0 and rvo_count != eligible:
        debug_path = base_dir / "ratings_vs_odds_debug.json"
        debug_payload = mismatch_samples[:20]
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        failures.append(f"ratings vs odds coverage {rvo_count}/{eligible} (see {debug_path})")

    return records, receipt | {"failures": failures}


def write_outputs(
    season: int,
    week: int,
    records: List[Dict[str, Any]],
    receipt: Dict[str, Any],
) -> Tuple[Path, Path, Path]:
    base_dir = _cfb_week_dir(season, week)
    jsonl_path = base_dir / f"games_week_{season}_{week}.jsonl"
    csv_path = base_dir / f"games_week_{season}_{week}.csv"
    receipt_path = base_dir / RECEIPT_NAME

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    write_jsonl(records, jsonl_path)
    write_csv(df, csv_path)
    receipt_clean = receipt.copy()
    failures = receipt_clean.pop("failures", [])
    payload = receipt_clean
    payload["failures"] = failures
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonl_path, csv_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weekly College Football Game View outputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--relax-odds", action="store_true", default=False)
    parser.add_argument("--relax-sagarin", action="store_true", default=False)
    args = parser.parse_args()

    try:
        records, receipt = build_game_records(
            args.season,
            args.week,
            relax_odds=args.relax_odds,
            relax_sagarin=args.relax_sagarin,
        )
    except Exception as exc:
        print(f"FAIL: CFB game view build aborted -> {exc}")
        return 1

    jsonl_path, csv_path, receipt_path = write_outputs(args.season, args.week, records, receipt)

    failures: Iterable[str] = receipt.get("failures") or []
    failures = [entry for entry in failures if entry]
    if failures:
        detail = "; ".join(failures)
        print(f"FAIL: {detail}")
        print(f"Receipt: {receipt_path}")
        return 1

    print(
        f"PASS: CFB Game View rows={len(records)} "
        f"(jsonl={jsonl_path}, csv={csv_path}, receipt={receipt_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

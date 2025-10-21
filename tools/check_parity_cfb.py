"""Compare NFL and CFB artifact schemas for a given week."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

RELEVANT_JSON = ["games_week", "odds", "sagarin"]


def out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "out"


def csv_columns(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    df = pd.read_csv(path, nrows=0)
    return df.columns.tolist()


def json_keys(path: Path) -> Optional[Set[str]]:
    if not path.exists():
        return None
    keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                keys.update(record.keys())
            if keys:
                break
    return keys


def compare_sets(label: str, nfl: Optional[Iterable[str]], cfb: Optional[Iterable[str]]) -> bool:
    nfl_set = set(nfl or [])
    cfb_set = set(cfb or [])
    missing = nfl_set - cfb_set
    extra = cfb_set - nfl_set
    status = "PASS" if not missing and not extra else "FAIL"
    print(f"{status}: {label}")
    if missing:
        print(f"  missing in CFB: {sorted(missing)}")
    if extra:
        print(f"  extra in CFB:   {sorted(extra)}")
    return status == "PASS"


def pick_sidecar(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = sorted(p for p in dir_path.glob("*.json") if p.name != "")
    return candidates[0] if candidates else None


def sidecar_keys(path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    top_keys = set(data.keys())
    child: Dict[str, Set[str]] = {}
    for bucket in ("home_ytd", "away_ytd", "home_prev", "away_prev"):
        rows = data.get(bucket) or []
        if rows:
            child[bucket] = set(rows[0].keys())
        else:
            child[bucket] = set()
    return top_keys, child


def main() -> int:
    parser = argparse.ArgumentParser(description="Check schema parity between NFL and CFB outputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    base_nfl = out_dir() / f"{args.season}_week{args.week-1}"
    base_cfb = out_dir() / "cfb" / f"{args.season}_week{args.week}"

    all_pass = True

    # Games (CSV + JSONL)
    nfl_games_csv = base_nfl / f"games_week_{args.season}_{args.week-1}.csv"
    cfb_games_csv = base_cfb / f"games_week_{args.season}_{args.week}.csv"
    nfl_cols = csv_columns(nfl_games_csv)
    cfb_cols = csv_columns(cfb_games_csv)
    all_pass &= compare_sets("games_week CSV columns", nfl_cols, cfb_cols)

    nfl_games_json = base_nfl / f"games_week_{args.season}_{args.week-1}.jsonl"
    cfb_games_json = base_cfb / f"games_week_{args.season}_{args.week}.jsonl"
    nfl_json_keys = json_keys(nfl_games_json) or set(nfl_cols or [])
    cfb_json_keys = json_keys(cfb_games_json) or set(cfb_cols or [])
    all_pass &= compare_sets("games_week JSON keys", nfl_json_keys, cfb_json_keys)

    # League metrics CSV
    nfl_league = base_nfl / f"league_metrics_{args.season}_{args.week-1}.csv"
    cfb_league = base_cfb / f"league_metrics_{args.season}_{args.week}.csv"
    all_pass &= compare_sets("league_metrics CSV columns", csv_columns(nfl_league), csv_columns(cfb_league))

    # Odds JSONL
    nfl_odds = base_nfl / f"odds_{args.season}_wk{args.week-1}.jsonl"
    cfb_odds = base_cfb / f"odds_{args.season}_wk{args.week}.jsonl"
    all_pass &= compare_sets("odds JSON keys", json_keys(nfl_odds), json_keys(cfb_odds))

    # Sagarin weekly CSV/JSON
    nfl_sagarin_csv = base_nfl / f"sagarin_nfl_{args.season}_wk{args.week-1}.csv"
    cfb_sagarin_csv = base_cfb / f"sagarin_cfb_{args.season}_wk{args.week}.csv"
    all_pass &= compare_sets("sagarin CSV columns", csv_columns(nfl_sagarin_csv), csv_columns(cfb_sagarin_csv))

    nfl_sagarin_json = base_nfl / f"sagarin_nfl_{args.season}_wk{args.week-1}.jsonl"
    cfb_sagarin_json = base_cfb / f"sagarin_cfb_{args.season}_wk{args.week}.jsonl"
    nfl_sagarin_keys = json_keys(nfl_sagarin_json) or set(csv_columns(nfl_sagarin_csv) or [])
    cfb_sagarin_keys = json_keys(cfb_sagarin_json) or set(csv_columns(cfb_sagarin_csv) or [])
    all_pass &= compare_sets("sagarin JSON keys", nfl_sagarin_keys, cfb_sagarin_keys)

    # Sidecar structure
    nfl_sidecar = pick_sidecar(base_nfl / "game_schedules")
    cfb_sidecar = pick_sidecar(base_cfb / "game_schedules")
    if nfl_sidecar and cfb_sidecar:
        nfl_top, nfl_child = sidecar_keys(nfl_sidecar)
        cfb_top, cfb_child = sidecar_keys(cfb_sidecar)
        all_pass &= compare_sets("sidecar top-level keys", nfl_top, cfb_top)
        for bucket in ("home_ytd", "away_ytd", "home_prev", "away_prev"):
            all_pass &= compare_sets(f"sidecar {bucket} entry keys", nfl_child.get(bucket), cfb_child.get(bucket))
    else:
        if not nfl_sidecar:
            print("FAIL: NFL sidecar sample not found")
            all_pass = False
        if not cfb_sidecar:
            print("FAIL: CFB sidecar sample not found")
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

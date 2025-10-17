"""Stub builder for CFB game timelines sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from src.common.io_utils import ensure_out_dir


def cfb_week_dir(season: int, week: int) -> Path:
    base = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def read_game_keys(games_path: Path) -> List[str]:
    keys: List[str] = []
    if not games_path.exists():
        return keys
    with games_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            game_key = record.get("game_key")
            if isinstance(game_key, str):
                keys.append(game_key)
    return keys


TIMELINE_KEYS: List[str] = [
    "season",
    "week",
    "date",
    "opp",
    "site",
    "pf",
    "pa",
    "result",
    "pr",
    "pr_rank",
    "sos",
    "sos_rank",
    "opp_pr",
    "opp_pr_rank",
    "opp_sos",
    "opp_sos_rank",
]


def timeline_stub() -> dict:
    return {key: None for key in TIMELINE_KEYS}


def write_sidecar(path: Path, game_key: str, include_template: bool = False) -> None:
    payload = {
        "game_key": game_key,
        "home_ytd": [timeline_stub()] if include_template else [],
        "away_ytd": [timeline_stub()] if include_template else [],
        "home_prev": [timeline_stub()] if include_template else [],
        "away_prev": [timeline_stub()] if include_template else [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub CFB timelines builder.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    print(f"STUB: build_team_timelines_cfb season={args.season} week={args.week}")
    out_dir = cfb_week_dir(args.season, args.week)
    games_path = out_dir / f"games_week_{args.season}_{args.week}.jsonl"
    sidecar_dir = out_dir / "game_schedules"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    keys = read_game_keys(games_path)
    for key in keys:
        side_path = sidecar_dir / f"{key}.json"
        write_sidecar(side_path, key)

    if not keys:
        stub_path = sidecar_dir / "_schema_stub.json"
        write_sidecar(stub_path, "_schema_stub", include_template=True)

    print(f"PASS: processed {len(keys)} game keys -> {sidecar_dir}")
    if not keys:
        print("Info: no CFB games available yet; sidecars directory is ready for future data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

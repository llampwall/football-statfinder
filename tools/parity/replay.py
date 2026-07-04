"""Offline replay harness (Phase 2 WP-A, Part 1).

Rebuilds one season-1 week with the **new** ``football_statfinder`` pipeline,
entirely offline, from archived season-1 inputs, into a scratch output root.
Never writes into the real ``out/`` tree, never touches the network.

Injection model (mirrors ``tests/test_refresh_integration.py``): point
``paths.OUT_ROOT`` / ``run_summary.OUT_ROOT`` at a scratch root, copy the
season-1 master tables + the windowed raw-odds ledgers into that root, then
assign the fetch seams to offline fakes and call ``refresh_league`` so the real
stage sequencing (schedule -> sagarin -> stats -> odds staging -> gameview ->
promotion -> recompute -> sidecars) runs unchanged.

Seams faked (source of each stage's input):

* schedule  -> season-1 ``out/master/{league}_schedule_master.csv`` (copied into
  the scratch master; ``fetch_schedule`` returns empty so the copied master is
  the schedule of record and prior-season rows survive for sidecar timelines).
* sagarin   -> season-1 weekly snapshot ``sagarin_{league}_{S}_wk{W}.jsonl``.
* stats     -> season-1 ``league_metrics_{S}_{W}.csv`` (the *output* of the
  legacy stats stage; stats *computation* parity is out of scope — the source
  pages are gone).
* odds      -> season-1 raw ledgers ``out/staging/odds_raw/{league}/*.jsonl``
  whose fetch timestamp falls in [first_kickoff - 14d, last_kickoff]; the new
  pin + promote run against them.

Run:  python -m tools.parity.replay --league nfl --season 2025 --week 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from football_statfinder import paths as paths_mod
from football_statfinder import refresh as refresh_mod
from football_statfinder import run_summary as run_summary_mod
from football_statfinder.common.io_atomic import write_atomic_jsonl
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.config import BackfillSettings, OddsSettings, Settings, StorageSettings
from football_statfinder.leagues import get_league
from football_statfinder.sources import sagarin as sagarin_mod
from football_statfinder.sources import schedule as schedule_mod
from football_statfinder.sources import stats as stats_mod

REPO_ROOT = paths_mod.REPO_ROOT
DEFAULT_SCRATCH = Path(
    r"C:\Users\Jordan\AppData\Local\Temp\claude"
    r"\C--Users-Jordan-Documents-Football-Project-George-NEW-PROJECT-football-statfinder"
    r"\6accb4e8-a09e-41a6-842c-beb095472d52\scratchpad\parity"
)

ODDS_WINDOW_LEAD_DAYS = 14


# ---------------------------------------------------------------------------
# baseline artifact locators (real out/, READ-ONLY)
# ---------------------------------------------------------------------------


def baseline_week_dir(baseline_out: Path, league_code: str, season: int, week: int) -> Path:
    if league_code == "nfl":
        return baseline_out / f"{season}_week{week}"
    return baseline_out / league_code / f"{season}_week{week}"


def _read_metrics_rows(csv_path: Path) -> List[Dict[str, Any]]:
    # na_filter=False keeps season-1 blank sentinels as "" (the stats parser
    # turns "" into None); dtype=str avoids pandas re-formatting the numbers.
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    return frame.to_dict(orient="records")


def _read_sagarin_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(jsonl_path).rows


# ---------------------------------------------------------------------------
# odds window
# ---------------------------------------------------------------------------


def _parse_ledger_stamp(name: str) -> Optional[datetime]:
    stem = name.split(".")[0]
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute_odds_window(week_df: pd.DataFrame) -> Dict[str, Any]:
    kicks = pd.to_datetime(week_df["kickoff_iso_utc"], utc=True, errors="coerce").dropna()
    if kicks.empty:
        raise RuntimeError("cannot compute odds window: no parseable kickoffs in week rows")
    first = kicks.min().to_pydatetime()
    last = kicks.max().to_pydatetime()
    return {
        "first_kickoff": first,
        "last_kickoff": last,
        "start": first - timedelta(days=ODDS_WINDOW_LEAD_DAYS),
        "end": last,
    }


def copy_windowed_odds(
    baseline_out: Path, scratch_out: Path, league_code: str, window: Dict[str, Any]
) -> List[str]:
    src_dir = baseline_out / "staging" / "odds_raw" / league_code
    dst_dir = scratch_out / "staging" / "odds_raw" / league_code
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    if not src_dir.exists():
        return copied
    for path in sorted(src_dir.glob("*.jsonl")):
        stamp = _parse_ledger_stamp(path.name)
        if stamp is None:
            continue
        if window["start"] <= stamp <= window["end"]:
            shutil.copy2(path, dst_dir / path.name)
            copied.append(path.name)
    return copied


def _load_scratch_odds_records(scratch_out: Path, league_code: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    dst_dir = scratch_out / "staging" / "odds_raw" / league_code
    for path in sorted(dst_dir.glob("*.jsonl")):
        records.extend(read_jsonl(path).rows)
    return records


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeStatsProvider:
    def __init__(self, league, rows: List[Dict[str, Any]]):
        self._league = league
        self._rows = rows

    def league_metrics_rows(self, season, week, *, as_of_week=None):
        return self._rows

    def team_stats(self, season, week, *, as_of_week=None):
        return stats_mod.team_stats_from_metrics_rows(self._league, self._rows)


def _make_sagarin_fake(scratch_out: Path, league, season: int, week: int, rows: List[Dict[str, Any]]):
    weekly_jsonl = (
        paths_mod.week_dir(league.code, season, week, out_root=scratch_out, create=True)
        / f"sagarin_{league.code}_{season}_wk{week}.jsonl"
    )
    write_atomic_jsonl(weekly_jsonl, rows)
    hfa_values = [r.get("hfa") for r in rows if r.get("hfa") is not None]
    hfa = float(hfa_values[0]) if hfa_values else None

    def _fake(_league, _season, _week, _settings, **_kw):
        return sagarin_mod.SagarinStagingResult(
            league=league.display,
            season=season,
            week=week,
            page_season=season,
            page_week=week,
            teams_parsed=len(rows),
            teams_selected=len(rows),
            master_before=0,
            master_after=len(rows),
            latest_fetch_ts=(rows[0].get("fetched_at") if rows else None),
            hfa=hfa,
            page_stamp=(rows[0].get("page_stamp") if rows else "replay fixture"),
            source_url="replay://sagarin",
            raw_html_path=None,
            staging_path=weekly_jsonl.parent,
            weekly_csv=weekly_jsonl.with_suffix(".csv"),
            weekly_jsonl=weekly_jsonl,
        )

    return _fake


def _make_ingest_fake(scratch_out: Path):
    def _fake(league, settings, **_kw):
        records = _load_scratch_odds_records(scratch_out, league.code)
        return {
            "records": records,
            "fetch_ts": None,
            "path": None,
            "counts": {"raw": len(records)},
            "skipped_reason": None,
        }

    return _fake


def _empty_schedule_df(*_args, **_kw) -> pd.DataFrame:
    # Return an empty frame in the schedule schema: the copied master is the
    # schedule of record, so the upsert is a no-op and prior-season rows (which
    # sidecar prev-season timelines need) survive untouched.
    return pd.DataFrame(columns=schedule_mod.SCHEDULE_COLUMNS)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _replay_settings(select_policy: str) -> Settings:
    return Settings(
        the_odds_api_key="replay",  # ingest is faked; never used
        cfbd_api_key="replay",
        cfbd_refresh=False,  # CFB schedule stage must not fetch
        odds=OddsSettings(
            staging_enable=True,
            promotion_enable=True,
            select_policy=select_policy,
        ),
        backfill=BackfillSettings(scores_enable=False, ats_enable=False),
        # Replay must never dual-write into the real data/statfinder.sqlite3
        # (WP-C storage defaults to enabled and to the repo-anchored DB path).
        storage=StorageSettings(enable=False),
    )


def run_replay(
    league_code: str,
    season: int,
    week: int,
    *,
    scratch_root: Path = DEFAULT_SCRATCH,
    baseline_out: Path = paths_mod.OUT_ROOT,
    select_policy: str = "closing_pre_kickoff",
) -> Dict[str, Any]:
    """Rebuild one week offline into ``scratch_root/{league}_{S}_wk{W}/out``.

    Returns a manifest dict (paths, odds window, stage summary, coverage).
    """
    league = get_league(league_code)
    scratch_out = scratch_root / f"{league.code}_{season}_wk{week}" / "out"
    if scratch_out.exists():
        shutil.rmtree(scratch_out)
    scratch_out.mkdir(parents=True, exist_ok=True)

    baseline_out = Path(baseline_out)
    bl_week = baseline_week_dir(baseline_out, league.code, season, week)

    # --- copy season-1 masters (all seasons; sidecar prev timelines need them).
    # NOTE the season-1 sagarin master is named ``sagarin_{league}_master.csv``
    # but the NEW pipeline's path helper expects ``{league}_sagarin_master.csv``
    # (paths.sagarin_master_csv). Copy to the name the new pipeline reads.
    master_src = baseline_out / "master"
    master_dst = scratch_out / "master"
    master_dst.mkdir(parents=True, exist_ok=True)
    master_copies = {
        paths_mod.schedule_master_csv(league.code, out_root=scratch_out).name: (
            master_src / f"{league.code}_schedule_master.csv"
        ),
        paths_mod.sagarin_master_csv(league.code, out_root=scratch_out).name: (
            master_src / f"sagarin_{league.code}_master.csv"
        ),
    }
    for dst_name, src in master_copies.items():
        if not src.exists():
            raise FileNotFoundError(f"required season-1 master missing: {src}")
        shutil.copy2(src, master_dst / dst_name)

    # --- season-1 stage inputs
    metrics_rows = _read_metrics_rows(bl_week / f"league_metrics_{season}_{week}.csv")
    sagarin_rows = _read_sagarin_rows(bl_week / f"sagarin_{league.code}_{season}_wk{week}.jsonl")

    # --- odds window + windowed ledger copy
    schedule_master = pd.read_csv(master_dst / f"{league.code}_schedule_master.csv")
    week_df = schedule_master[
        (pd.to_numeric(schedule_master["season"], errors="coerce") == season)
        & (pd.to_numeric(schedule_master["week"], errors="coerce") == week)
    ].copy()
    window = compute_odds_window(week_df)
    copied = copy_windowed_odds(baseline_out, scratch_out, league.code, window)

    settings = _replay_settings(select_policy)

    # --- install offline seams + scratch out root ------------------------------
    saved = {
        "paths_out": paths_mod.OUT_ROOT,
        "summary_out": run_summary_mod.OUT_ROOT,
        "fetch_schedule": schedule_mod.fetch_schedule,
        "run_sagarin": refresh_mod.sagarin_mod.run_sagarin_staging,
        "get_provider": stats_mod.get_stats_provider,
        "ingest": refresh_mod.odds_ingest.ingest_raw,
    }
    try:
        paths_mod.OUT_ROOT = scratch_out
        run_summary_mod.OUT_ROOT = scratch_out
        # keep the aliased module globals (refresh imported some by name) in sync
        refresh_mod.paths.OUT_ROOT = scratch_out

        schedule_mod.fetch_schedule = _empty_schedule_df
        refresh_mod.sagarin_mod.run_sagarin_staging = _make_sagarin_fake(
            scratch_out, league, season, week, sagarin_rows
        )
        stats_mod.get_stats_provider = lambda _league, _settings: _FakeStatsProvider(
            league, metrics_rows
        )
        refresh_mod.odds_ingest.ingest_raw = _make_ingest_fake(scratch_out)

        summary = refresh_mod.refresh_league(league, settings, season=season, week=week)
    finally:
        paths_mod.OUT_ROOT = saved["paths_out"]
        run_summary_mod.OUT_ROOT = saved["summary_out"]
        refresh_mod.paths.OUT_ROOT = saved["paths_out"]
        schedule_mod.fetch_schedule = saved["fetch_schedule"]
        refresh_mod.sagarin_mod.run_sagarin_staging = saved["run_sagarin"]
        stats_mod.get_stats_provider = saved["get_provider"]
        refresh_mod.odds_ingest.ingest_raw = saved["ingest"]

    replay_jsonl = paths_mod.games_week_jsonl(league.code, season, week, out_root=scratch_out)
    replay_rows = read_jsonl(replay_jsonl).rows if replay_jsonl.exists() else []

    manifest = {
        "league": league.code,
        "season": season,
        "week": week,
        "select_policy": select_policy,
        "scratch_out": str(scratch_out),
        "baseline_week_dir": str(bl_week),
        "replay_games_week_jsonl": str(replay_jsonl),
        "schedule_master_week_rows": int(len(week_df)),
        "replay_output_rows": len(replay_rows),
        "odds_window": {
            "first_kickoff": window["first_kickoff"].isoformat(),
            "last_kickoff": window["last_kickoff"].isoformat(),
            "start": window["start"].isoformat(),
            "end": window["end"].isoformat(),
            "ledger_files_copied": len(copied),
        },
        "stages": [
            {"name": s.name, "ok": s.ok, "counts": s.counts, "notes": s.notes, "error": s.error}
            for s in summary.stages
        ],
        "ok": summary.ok,
    }
    manifest_path = scratch_out.parent / "replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline season-1 week replay with the new pipeline.")
    p.add_argument("--league", required=True, choices=["nfl", "cfb"])
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--week", required=True, type=int)
    p.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    p.add_argument("--baseline-out", type=Path, default=paths_mod.OUT_ROOT)
    p.add_argument(
        "--select-policy",
        default="closing_pre_kickoff",
        choices=["closing_pre_kickoff", "latest_by_fetch_ts"],
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = run_replay(
        args.league,
        args.season,
        args.week,
        scratch_root=args.scratch,
        baseline_out=args.baseline_out,
        select_policy=args.select_policy,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

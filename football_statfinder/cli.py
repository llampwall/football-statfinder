"""Console entry point (``statfinder`` script, or ``python -m football_statfinder.cli``).

Subcommands:

* ``refresh --league nfl|cfb|all [--season S --week W]`` — the weekly refresh.
  Emits the frozen one-line NOTIFY contract per league on success and exits
  non-zero if any league failed.
* ``current-week --league nfl|cfb`` — print the resolved season/week.
* ``seed-schedule --league nfl|cfb --season S`` — bootstrap/refresh the
  schedule master for a season (needed once before the first refresh of a new
  season, since current-week resolution reads the master).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .common.current_week import get_current_week
from .config import get_settings
from .leagues import CFB, NFL, League, get_league
from .run_summary import setup_logging

logger = logging.getLogger(__name__)


def _leagues_arg(value: str) -> List[League]:
    if value.strip().lower() == "all":
        return [CFB, NFL]  # CFB first, matching the season-1 runner order
    return [get_league(value)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="statfinder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser("refresh", help="run the weekly refresh")
    p_refresh.add_argument("--league", default="all", help="nfl, cfb, or all (default)")
    p_refresh.add_argument("--season", type=int, default=None)
    p_refresh.add_argument("--week", type=int, default=None)

    p_week = sub.add_parser("current-week", help="print the resolved current week")
    p_week.add_argument("--league", required=True)

    p_seed = sub.add_parser("seed-schedule", help="bootstrap the schedule master for a season")
    p_seed.add_argument("--league", required=True)
    p_seed.add_argument("--season", type=int, required=True)

    args = parser.parse_args(argv)
    setup_logging()
    settings = get_settings()

    if args.command == "current-week":
        league = get_league(args.league)
        season, week, computed_at = get_current_week(league, settings=settings, persist=False)
        print(f"{league.display} season={season} week={week} computed_at={computed_at}")
        return 0

    if args.command == "seed-schedule":
        from .sources.schedule_master import ensure_seasons_present

        league = get_league(args.league)
        inserted, updated = ensure_seasons_present(league, [args.season], settings)
        print(f"{league.display} schedule master: inserted={inserted} updated={updated}")
        return 0

    # refresh
    from .refresh import refresh_all

    logger.info(settings.banner())
    summaries = refresh_all(_leagues_arg(args.league), settings, season=args.season, week=args.week)
    exit_code = 0
    for display, summary in summaries.items():
        if summary.ok:
            gameview = next((s for s in summary.stages if s.name == "gameview"), None)
            rows = (gameview.counts.get("records") or gameview.counts.get("rows") or 0) if gameview else 0
            print(summary.notify_line(rows))
        else:
            failed = [s.name for s in summary.stages if not s.ok]
            print(f"FAILED: {display} refresh (stages: {', '.join(failed) or 'unknown'})")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

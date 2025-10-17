## Overview
Football Statfinder orchestrates weekly NFL schedule, rating, metric, and odds pulls into normalized JSONL/CSV outputs that drive the Week and Game static web views; the browser converts kickoff timestamps to Pacific time when rendering, while the stored data remains UTC.

## Deliverables -> Web pages -> Files they read
- **Week View** (maps to the Week deliverable in `SPEC_DELIVERABLES.md`)
  - Reads: `out/{season}_week{week}/games_week_{season}_{week}.jsonl`
  - Navigates to Game View.
- **Game View** (maps to the Game deliverable)
  - Reads: same games JSONL row (by `game_key`) + sidecar `out/{season}_week{week}/game_schedules/{game_key}.json`
  - Also reads prior-season EOY: `out/final_league_metrics_{priorSeason}.csv`
- **Masters used by generators**
  - `out/master/sagarin_nfl_master.csv`
  - `out/master/nfl_schedule_master.csv`

## Orchestration flow (weekly)
- `python -m src.refresh_week_data --season <YYYY> --week <W>` runs:
  - `src/fetch_games.py` -> downloads the nflverse schedules CSV, filters the target regular-season week, parses kickoff UTC stamps, normalizes teams, and captures the expected game count.
  - `src/fetch_year_to_date_stats.py` -> builds per-team year-to-date offense/defense metrics, PF/PA, SU/ATS, and writes `league_metrics_{season}_{week}.csv` while logging TeamRankings turnover merge coverage.
  - `src/fetch_sagarin_week_nfl.py` -> scrapes Sagarin's NFL page for PR/SoS/HFA, emits `sagarin_nfl_{season}_wk{week}.csv|.jsonl`, and reports record counts plus precision checks.
  - `src/sagarin_master.py` -> appends and dedupes the weekly Sagarin export into `out/master/sagarin_nfl_master.csv`, returning the before/after row totals.
  - `src/gameview_build.py` -> joins schedules, league metrics, Sagarin, and odds (schedule fallback) into `games_week_{season}_{week}.jsonl`/`.csv`, producing matchup-level derived fields and acceptance diagnostics.
  - `src/fetch_week_odds_nfl.py` -> when `THE_ODDS_API_KEY` is available, pulls Pinnacle (preferred) odds from The Odds API into `odds_{season}_wk{week}.jsonl`; otherwise logs the skip.
  - `src/schedule_master.py` -> ensures current and prior seasons populate `out/master/nfl_schedule_master.csv`, validating keys before sidecar construction.
  - `src/build_team_timelines.py` -> generates per-game sidecars (`home_ytd`, `away_ytd`, `home_prev`, `away_prev`) enriched with Sagarin joins and verifies PR coverage.
- Acceptance: orchestrator logs schedule vs gameview counts, league rows, Sagarin master delta, odds coverage, sidecar counts, league-metric join coverage, S2D completeness, rank validity, favored-metric coverage, and NaN/Inf diagnostics.

## Shared utilities
- `src/common/io_utils.py` centralizes output directory helpers (`week_out_dir`), env loading, and JSONL/CSV writers.
- `src/common/metrics.py` provides pure helpers for rating differentials, favored-side spreads, rating-vs-odds, dense ranks, and SU/ATS tallies.
- `src/common/team_names.py` normalizes raw labels to canonical display names and supplies `team_merge_key` for cross-source joins.

## Key data contracts (names only, not full schemas)
- **Games JSONL**: identity (`season`, `week`, `game_key`, `kickoff_iso_utc`, normalized team names), favorite-perspective odds (`favored_side`, `spread_favored_team`, `rating_diff_favored_team`, `rating_vs_odds_favored_team`), Sagarin fields (`home_pr`, `away_pr`, `home_sos`, `away_sos`, `hfa`), and S2D metrics (`*_pf_pg`, `*_pa_pg`, `*_ry_pg`, `*_py_pg`, `*_ty_pg`, offense/defense ranks).
- **Sidecar JSON**: `home_ytd`, `away_ytd`, `home_prev`, `away_prev` arrays of compact rows (`season`, `week`, `date`, `opp`, `site`, `pf`, `pa`, `result`, `pr`, `pr_rank`, `sos`, `sos_rank`, `opp_pr`, `opp_pr_rank`, `opp_sos`, `opp_sos_rank`).
- **Masters**: Sagarin master (`league`, `season`, `week`, `team_norm`, `pr`, `pr_rank`, `sos`, `sos_rank`) and schedule master (`league`, `season`, `week`, `game_type`, `kickoff_iso_utc`, `home_team_norm`, `away_team_norm`, merge keys, scores, `spread_line`, `total_line`, `source`).

## External sources (from code; keep as bullet list)
- `src.fetch_games` -> nflverse `games.csv` release for schedules/kickoff metadata.
- `src.fetch_year_to_date_stats` -> nflverse `stats_team_week_{season}.csv` plus TeamRankings turnover margin tables.
- `src.fetch_week_odds_nfl` -> The Odds API v4 (and Don Best XML when configured).
- `src.fetch_sagarin_week_nfl` -> `http://sagarin.com/sports/nflsend.htm` HTML scrape.
- `src.fetch_last_year_stats` -> nflverse schedules CSV and multiple TeamRankings season-average pages for prior-season finals.

## Runbook
- Weekly refresh: `python -m src.refresh_week_data --season <YYYY> --week <W>` (ensure `.env` carries `THE_ODDS_API_KEY` if odds are required).
- Serve web: `python -m http.server 8000` from repo root and open `web/week_view.html` (links route into Game View).
- Paths in `out/` are contracts per `SPEC_DELIVERABLES.md`; add new fields only as optional extensions; never rename keys or relocate files.

## Future: League awareness & CFB
- Upcoming work introduces a `league={nfl|cfb}` web param, with College Football outputs landing under `out/cfb/{season}_week{week}/...` while the existing NFL pipeline and paths stay unchanged.

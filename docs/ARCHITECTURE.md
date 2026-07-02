# football-statfinder Architecture

This document describes how the system works today, as built. It is descriptive, not prescriptive; known defects and rebuild recommendations live in a separate REBUILD document. Behavior that is surprising but load-bearing is flagged inline as "(known quirk)".

## Table of contents

- [1. System overview](#1-system-overview)
- [2. Production execution path](#2-production-execution-path)
  - [2.1 Call tree](#21-call-tree)
  - [2.2 The NOTIFY contract and Discord](#22-the-notify-contract-and-discord)
  - [2.3 Auto-commit: the repo is the database](#23-auto-commit-the-repo-is-the-database)
- [3. Subsystem guide](#3-subsystem-guide)
  - [3.1 NFL pipeline](#31-nfl-pipeline)
  - [3.2 CFB pipeline](#32-cfb-pipeline)
  - [3.3 Odds](#33-odds)
  - [3.4 Scores and ATS](#34-scores-and-ats)
  - [3.5 Common utilities and Sagarin ratings](#35-common-utilities-and-sagarin-ratings)
  - [3.6 Ops (workflow, tools, tests)](#36-ops-workflow-tools-tests)
  - [3.7 Docs and specs](#37-docs-and-specs)
- [4. Data contracts (the out/ tree)](#4-data-contracts-the-out-tree)
- [5. Current-week resolution](#5-current-week-resolution)
- [6. External data sources](#6-external-data-sources)
- [7. Configuration reference](#7-configuration-reference)
- [8. Frontend](#8-frontend)
- [9. Error-handling map](#9-error-handling-map)

## 1. System overview

football-statfinder produces weekly betting data packs for NFL and college football (CFB, FBS only). The product replaces hand-made Excel sheets that George built in 2019/2020 (scans in `context/DELIVERABLE 1 - WEEK VIEW.JPG` and the two `DELIVERABLE 2 - GAME VIEW` images); `context/SPEC_DELIVERABLES.md` is the written contract. Each week the pipeline joins five kinds of data per game:

- Sagarin power ratings (PR), strength of schedule (SoS), and home-field advantage (HFA), scraped from sagarin.com
- Schedules and final scores (nflverse for NFL, the CollegeFootballData API for CFB)
- Season-to-date team stats and ranks (nflverse team-week stats plus a TeamRankings scrape for NFL; CFBD for CFB)
- Betting odds: spreads, totals, moneylines from The Odds API
- Derived metrics: rating_diff (PR difference plus HFA), rating_vs_odds (model edge against the market), SU and ATS records

Output is a per-week "game view" pack of JSONL/CSV files plus per-game sidecar JSONs under `out/`, consumed directly by a static, build-free HTML/JS frontend with three pages: **Week View** (`web/week_view.html`, two rows per game across the whole slate), **Game View** (`web/game_view.html`, one game drill-down with schedule timelines and prior-season stats), and a **Printable** one-page game sheet (`web/game_view_printable.html`). There is no backend server and no database; the frontend fetches files from `out/` over plain HTTP.

Everything runs from GitHub Actions on a twice-daily cron, and the regenerated artifacts are committed back to the repo.

## 2. Production execution path

### 2.1 Call tree

The only production entry is `.github/workflows/refresh.yml` (cron `0 10,22 * * *` UTC, plus `workflow_dispatch`). It exports `THE_ODDS_API_KEY`, `CFBD_API_KEY`, `DISCORD_WEBHOOK_URL`, `BACKFILL_PROMOTE_PREV`, `BACKFILL_WEEKS`, and `ODDS_PROMOTION_ENABLE` into the job env (refresh.yml:14-20), then runs the orchestrator. Indented tree, with reads/writes per step:

```text
.github/workflows/refresh.yml                       cron 10:00/22:00 UTC
├─ pip install -r requirements.txt                  (unpinned deps)
├─ python tools/run_refresh_all_and_notify.py
│  │
│  ├─ SUBPROCESS python -m src.refresh_week_data_cfb          (no args -> auto week; runs first)
│  │  ├─ get_current_week("CFB")
│  │  │    reads  out/master/cfb_schedule_master.csv
│  │  │    writes out/state/current_week.json
│  │  ├─ ensure_weeks_present (schedule_master_cfb)           [CFBD_REFRESH gate; errors swallowed -> WARNING]
│  │  │    reads  CFBD /games (season, season-1)
│  │  │    writes out/master/cfb_schedule_master.csv (upsert)
│  │  ├─ SUBPROCESS src.fetch_games_cfb --season --week
│  │  │    permanent no-op: main() hardcodes abort=1 and returns 0 (src/fetch_games_cfb.py:183-186) (known quirk)
│  │  ├─ filter_week_reg(load_games(season))                  in-process CFBD /games fetch; 0 rows -> StageError abort
│  │  ├─ ingest_cfb_odds_raw                                  [ODDS_STAGING_ENABLE; fetch errors swallowed]
│  │  │    reads  The Odds API NCAAF odds
│  │  │    writes out/staging/odds_raw/cfb/{ts}.jsonl
│  │  ├─ pin_cfb_odds
│  │  │    reads  out/master/cfb_schedule_master.csv
│  │  │    appends out/staging/odds_pinned/cfb/{season}.jsonl; writes out/staging/odds_unmatched/cfb/{ts}.jsonl
│  │  ├─ SUBPROCESS src.fetch_year_to_date_stats_cfb          rc!=0 or CSV missing -> abort
│  │  │    reads  CFBD /teams/fbs, /games/teams weeks 1..W-1 (one call per week)
│  │  │    writes out/cfb/{S}_week{W}/league_metrics_{S}_{W}.csv + league_metrics_debug.json
│  │  ├─ SUBPROCESS src.fetch_last_year_stats_cfb             only with --include-eoy; production never passes it
│  │  ├─ SUBPROCESS src.fetch_week_odds_cfb --odds-days-before 6 --odds-days-after 6
│  │  │    reads  The Odds API NCAAF; CFBD schedule (again)
│  │  │    writes out/cfb/{S}_week{W}/odds_{S}_wk{W}.jsonl + odds_match_debug.json + odds_alias_debt.json
│  │  │    exits 1 if odds-to-schedule match coverage is below thresholds -> abort
│  │  ├─ enrich_from_local_odds                               [errors swallowed -> WARNING]
│  │  │    writes out/master/cfb_schedule_master.csv (fill spread/total) + master_enrichment_receipt.json
│  │  ├─ run_cfb_sagarin_staging                              [SAGARIN_STAGING_ENABLE default on; exceptions -> abort]
│  │  │    reads  http://sagarin.com/sports/cfsend.htm
│  │  │    writes data/sagarin/raw/cfb/cfsend_*.html (best effort),
│  │  │           out/staging/sagarin_latest/cfb/{season}.jsonl (append),
│  │  │           out/cfb/{S}_week{W}/sagarin_cfb_{S}_wk{W}.csv/.jsonl (atomic),
│  │  │           out/master/sagarin_cfb_master.csv (atomic upsert)
│  │  │    [flag off: SUBPROCESS src.fetch_sagarin_week_cfb then src.sagarin_master_cfb — unreachable with default env]
│  │  ├─ backfill_cfb_scores                                  [BACKFILL_WEEKS=2 default, week>1]
│  │  │    reads  cfb_schedule_master.csv, prior weeks' games_week + sidecars; closing spreads via The Odds API history
│  │  │    writes rewrites prior weeks' games_week jsonl/csv (atomic) + sidecar JSONs; may re-run gameview_build_cfb (check=False)
│  │  ├─ SUBPROCESS src.gameview_build_cfb (pass 1)           rc!=0 / empty / missing favorite fields -> abort
│  │  │    reads  CFBD /games (live, again), league_metrics CSV, sagarin JSONL, odds JSONL
│  │  │    writes out/cfb/{S}_week{W}/games_week_{S}_{W}.jsonl/.csv + gameview_build_receipt.json
│  │  ├─ SUBPROCESS src.build_team_timelines_cfb              rc!=0 or no sidecar dir -> abort
│  │  │    reads  re-runs ensure_weeks_present (CFBD again), both master CSVs
│  │  │    writes out/cfb/{S}_week{W}/game_schedules/{game_key}.json per game + sidecars_receipt.json + schedules_sagarin_receipt.json
│  │  ├─ SUBPROCESS src.gameview_build_cfb (pass 2)           verbatim duplicate; comment "(again - might be necessary)"
│  │  │    (refresh_week_data_cfb.py:638-660) (known quirk)
│  │  ├─ promote_week_odds (cfb_promote_week)                 [ODDS_PROMOTION_ENABLE default on]
│  │  │    reads  out/staging/odds_pinned/cfb/{season}.jsonl
│  │  │    writes games_week jsonl/csv rewritten if promoted_games > 0 (refresh_week_data_cfb.py:668-673)
│  │  ├─ build_team_ats + apply_ats_to_week (src/cfb_ats)
│  │  │    reads  out/cfb/{S}_week{1..W-1}/games_week_*.jsonl
│  │  │    writes current-week games_week JSONL rewritten (CSV not updated)
│  │  ├─ write_gaps_report -> out/cfb/{S}_week{W}/orchestrator_gaps.json
│  │  │    compares CFB game keys against the NFL week file out/{S}_week{W}/games_week_{S}_{W}.jsonl (known quirk: namespaces never overlap)
│  │  └─ prints "NOTIFY: CFB refresh complete week={S}-{W} rows={n} odds_promoted={p}"
│  │
│  ├─ SUBPROCESS python -m src.refresh_week_data_nfl          (no args -> auto week; runs even if CFB failed)
│  │  ├─ get_current_week("NFL")                              (refresh_week_data_nfl.py:91)
│  │  │    reads  out/master/nfl_schedule_master.csv; writes out/state/current_week.json
│  │  ├─ src.refresh_week_data.main (legacy core, IN-PROCESS via a sys.argv shim, refresh_week_data_nfl.py:112-120)
│  │  │  ├─ load_games / filter_week_reg                      reads nflverse games.csv (fresh download)
│  │  │  ├─ generate_league_metrics                           reads nflverse stats_team_week_{S}.csv + scrapes teamrankings.com turnover page
│  │  │  │    writes out/{S}_week{W}/league_metrics_{S}_{W}.csv
│  │  │  ├─ fetch_sagarin_week                                reads http://sagarin.com/sports/nflsend.htm; must parse exactly 32 teams or RuntimeError
│  │  │  │    writes out/{S}_week{W}/sagarin_nfl_{S}_wk{W}.csv/.jsonl
│  │  │  ├─ append_week (sagarin_master)                      writes out/master/sagarin_nfl_master.csv
│  │  │  ├─ build_gameview                                    re-downloads schedule + stats; joins sagarin CSV + league_metrics CSV
│  │  │  │    writes out/{S}_week{W}/games_week_{S}_{W}.jsonl/.csv
│  │  │  ├─ fetch_odds_theoddsapi                             [only if THE_ODDS_API_KEY set]
│  │  │  │    writes out/{S}_week{W}/odds_{S}_wk{W}.jsonl (legacy snapshot; never read back by any build step)
│  │  │  └─ timelines: ensure_weeks_present([S, S-1]) upserts out/master/nfl_schedule_master.csv,
│  │  │       then build_timelines writes out/{S}_week{W}/game_schedules/{game_key}.json per game;
│  │  │       hard-fails on missing schedule rows or PR join coverage < 80% (refresh_week_data.py:153, 172-175)
│  │  ├─ _run_odds_staging: ingest_nfl_odds_raw + pin_nfl_odds   (refresh_week_data_nfl.py:124, 41-76)
│  │  │    writes out/staging/odds_raw/nfl/{ts}.jsonl; appends out/staging/odds_pinned/nfl/{season}.jsonl
│  │  ├─ run_nfl_sagarin_staging                              (refresh_week_data_nfl.py:125; second sagarin.com fetch of the run)
│  │  │    writes data/sagarin/raw/nfl/*.html, out/staging/sagarin_latest/nfl/{season}.jsonl,
│  │  │           out/nfl/{S}_week{W}/sagarin_nfl_*.csv/.jsonl (note: out/nfl/ prefix), out/master/sagarin_nfl_master.csv
│  │  ├─ backfill_nfl_scores                                  (refresh_week_data_nfl.py:127; weeks W-2..W-1)
│  │  │    reads nfl_schedule_master.csv + prior weeks' games_week; rewrites them atomically
│  │  ├─ build_team_ats + apply_ats_to_week (src/ats/nfl_ats) (refresh_week_data_nfl.py:134-135)
│  │  │    reads out/nfl/{S}_week{w}/games_week_*.jsonl, a path where no games_week files exist -> permanent no-op (known quirk; see 3.4)
│  │  ├─ promote_week_odds (nfl_promote_week)                 (refresh_week_data_nfl.py:147-154)
│  │  │    reads out/staging/odds_pinned/nfl/{season}.jsonl; if promoted > 0, write_week_outputs atomically rewrites games_week jsonl/csv
│  │  └─ prints "NOTIFY: NFL refresh complete week={S}-{W} rows={n} odds_promoted={p}."
│  │
│  ├─ writes out/logs/refresh_{ISO-ts}.json (atomic) + appends out/logs/refresh_index.tsv
│  ├─ post_discord                                            [gated DISCORD_WEBHOOK_URL; failures swallowed]
│  └─ sys.exit(1) if either league had rc!=0 or its NOTIFY line was missing
│
└─ stefanzweifel/git-auto-commit-action@v7
     commits out/** and data/sagarin/raw/** with message "chore: auto-refresh football data" (refresh.yml:38-44)
     runs only if the refresh step exited 0; there is no `if: always()`, so a failed run commits nothing
```

### 2.2 The NOTIFY contract and Discord

Each league orchestrator must print exactly one machine-readable line to stdout:

```
NOTIFY: <LEAGUE> refresh complete week=<season>-<week> rows=<n> odds_promoted=<p>
```

(CFB at refresh_week_data_cfb.py:747-750; NFL at refresh_week_data_nfl.py:178-181; documented in AGENTS.md:28-40, though AGENTS.md's format string omits the `odds_promoted=<p>` suffix the code emits.) `tools/run_refresh_all_and_notify.py` captures each subprocess's stdout+stderr, regex-extracts the NOTIFY line, and marks a league "ok" only when the return code is 0 **and** the NOTIFY line appeared (run_refresh_all_and_notify.py:258-259). It then writes `out/logs/refresh_{ts}.json` and appends a row to `out/logs/refresh_index.tsv` (`ts_utc, ok_cfb, ok_nfl, sec_cfb, sec_nfl`), and POSTs a summary (trimmed to 1900 chars) to the Discord webhook when `DISCORD_WEBHOOK_URL` is set. Discord failures are logged and swallowed; they never fail the run.

### 2.3 Auto-commit: the repo is the database

`out/` is git-tracked (about 1,450 files), and CI commits `out/**` and `data/sagarin/raw/**` after every successful run. There is no external datastore: the masters, staging ledgers, weekly packs, state file, and run logs are all versioned files. This is load-bearing in two ways: the frontend serves data straight from `out/` on GitHub Pages or any static host, and the current-week service bootstraps from the committed schedule masters (a fresh clone can resolve the week without ever fetching). It also means every scheduled run adds a commit and the append-only staging files grow inside git history.

## 3. Subsystem guide

### 3.1 NFL pipeline

Assembles the NFL week pack under `out/{season}_week{week}/`.

**Entry points**
- `src/refresh_week_data_nfl.py`: the live entry (`python -m src.refresh_week_data_nfl [--season S --week W]`). A thin wrapper (~185 lines): resolves the current week, then runs the legacy core in-process, then bolts on odds staging, Sagarin staging, score backfill, ATS, and odds promotion.
- `src/refresh_week_data.py`: the legacy core builder. Not dead; it does most of the work. `refresh_week()` writes league metrics, fetches Sagarin, builds the gameview, fetches the legacy odds snapshot, maintains the schedule master, and builds timeline sidecars.

**Key modules**
- `src/gameview_build.py`: builds one record per REG game and writes `games_week_{S}_{W}.jsonl/.csv`. Computes `game_key` (kickoff `YYYYMMDD_HHMM` plus away/home slugs), spread/total from nflverse schedule columns, Sagarin joins, `rating_diff = home_pr - away_pr + HFA`, favored-team metrics, season-to-date per-game stats, and six league-wide dense rank tables.
- `src/fetch_year_to_date_stats.py`: writes `league_metrics_{S}_{W}.csv` from nflverse team-week stats plus a live TeamRankings turnover-margin scrape.
- `src/fetch_sagarin_week_nfl.py`: legacy Sagarin scraper and shared parser library (see 3.5).
- `src/fetch_games.py`: downloads nflverse `games.csv` fresh on every call (no caching), filters season/REG/week.
- `src/schedule_master.py` and `src/sagarin_master.py`: upsert the two NFL master CSVs under `out/master/`.
- `src/build_team_timelines.py`: writes one `game_schedules/{game_key}.json` sidecar per game (year-to-date and prior-season timelines joined to Sagarin PR/SoS by week).
- `src/fetch_last_year_stats.py`: manual/offseason only; no orchestrator calls it. Writes `out/final_league_metrics_{season}.csv`, which the frontend reads directly.

**Generational layers.** The NFL side has two generations running on every refresh. The legacy generation (`refresh_week_data.py` core, `fetch_sagarin_week_nfl.py`, the `odds_{S}_wk{W}.jsonl` snapshot) writes to `out/{S}_week{W}/`. The newer generation (odds staging/pin/promote under `src/odds/`, Sagarin staging under `src/ratings/`, scores backfill under `src/scores/`, ATS under `src/ats/`) is layered on by the wrapper. The two generations do not fully agree on paths: Sagarin staging writes weekly snapshots to `out/nfl/{S}_week{W}/` while everything else NFL lives at `out/{S}_week{W}/` (known quirk), and the ATS module reads the `out/nfl/` path where no games_week files exist.

### 3.2 CFB pipeline

Assembles the CFB week pack under `out/cfb/{season}_week{week}/`. Structurally a fork of the NFL tree, but the orchestrator diverged: `src/refresh_week_data_cfb.py` (~756 lines) is a subprocess-per-stage runner with a `StageError` type and per-stage coverage gates, rather than a thin wrapper.

**Entry points and key modules**
- `src/refresh_week_data_cfb.py`: end-to-end orchestrator (`python -m src.refresh_week_data_cfb [--season S --week W] [--include-eoy]`); auto mode via `get_current_week("CFB")` (line 285).
- `src/fetch_games_cfb.py`: as a library, fetches CFBD `/games` and normalizes the schedule; as a CLI it is a deliberate no-op (`abort = 1`, prints "CFB parity check aborting", returns 0, lines 183-186), so the orchestrator's "schedule ingest" stage validates nothing and every later stage that needs the schedule refetches CFBD live (known quirk).
- `src/fetch_year_to_date_stats_cfb.py`: league metrics from CFBD `/teams/fbs` + `/games/teams`; enforces coverage gates (at least max(100, 70%) of teams with 10 of 14 fields, ranks_ok >= 0.75).
- `src/fetch_week_odds_cfb.py`: The Odds API NCAAF fetch with mascot-stripped name matching to the CFBD schedule; writes the odds JSONL plus `odds_match_debug.json` and `odds_alias_debt.json`; exits 1 below match-coverage thresholds (MIN_ABS_MATCHED=10, MIN_MATCH_FRAC=0.50, MAX_UNMATCH_FRAC=0.60).
- `src/gameview_build_cfb.py`: joins schedule + metrics + odds + Sagarin into the games_week pair; FBS filtering is implicit (both teams must appear in the metrics map, lines 372-375); coverage gates metrics >= 0.90, odds >= 0.60, sagarin >= 0.90. Run twice per refresh (passes 1 and 2, refresh_week_data_cfb.py:581-605 and 638-660).
- `src/build_team_timelines_cfb.py`: per-game sidecars with nearest-week Sagarin fallback; fails if Sagarin coverage < 0.85; re-runs `ensure_weeks_present` internally (second full CFBD schedule refetch per run).
- `src/schedule_master_cfb.py`: maintains `out/master/cfb_schedule_master.csv`; `enrich_from_local_odds` backfills missing spread/total lines from the week's local odds JSONL.
- `src/fetch_last_year_stats_cfb.py`: prior-season EOY metrics; only runs with `--include-eoy`, which production never passes.

**Generational layers.** Same two Sagarin generations as NFL, but CFB gates them cleanly: with `SAGARIN_STAGING_ENABLE` on (the default), only `src/ratings/sagarin_cfb_fetch.py` runs; the legacy `src/fetch_sagarin_week_cfb.py` CLI and `src/sagarin_master_cfb.py` execute only when the flag is off. The CFB week dirs also carry a receipt/debug layer the NFL dirs lack entirely (gameview_build_receipt, orchestrator_gaps, odds_match_debug, odds_alias_debt, league_metrics_debug, master_enrichment_receipt, schedules_sagarin_receipt, sidecars_receipt).

### 3.3 Odds

Two parallel paths per league, both live.

**Legacy snapshot path.** `src/fetch_week_odds_nfl.py` (called from refresh_week_data.py:86) and `src/fetch_week_odds_cfb.py` (subprocess from the CFB orchestrator) fetch The Odds API and write `odds_{S}_wk{W}.jsonl` into the week dir. The NFL snapshot stamps the requested season/week on every event the API returns (unfiltered) and is never read back by any build step; the CFB snapshot does real schedule matching with coverage gates and is consumed by `enrich_from_local_odds`. The NFL module also contains a Don Best XML client (`--source donbest`, `DONBEST_TOKEN`), which is never provisioned and presumed unused.

**Staging path (current design).** Three stages per league under `src/odds/`:
1. **Ingest** (`nfl_ingest.py` / `cfb_ingest.py`): fetch every event x bookmaker x market unfiltered and atomically write `out/staging/odds_raw/{league}/{YYYYMMDDTHHMMSSZ}.jsonl`. No-op if `ODDS_STAGING_ENABLE` is off or the API key is missing; all fetch exceptions are swallowed.
2. **Pin** (`nfl_pin_to_schedule.py` / `cfb_pin_to_schedule.py`): match raw records to schedule-master games by team-token pair plus kickoff tolerance (env-tunable day window, kickoff delta, role-swap fallback). Matches append to `out/staging/odds_pinned/{league}/{season}.jsonl` (append-only, duplicates included every run); misses are quarantined to `out/staging/odds_unmatched/{league}/{ts}.jsonl` with a reason.
3. **Promote** (`nfl_promote_week.py` / `cfb_promote_week.py`): dedupe the pinned ledger to latest fetch_ts per (game_key, market, book), select per market by `ODDS_SELECT_POLICY` (default `latest_by_fetch_ts`), and overwrite spread/total/moneyline/odds fields in the week's games_week rows. The `game_key` is recomputed in the pin modules from kickoff plus name slugs and must byte-match the gameview builders' keys for promotion to land.

**Closing spreads.** `src/odds/odds_history.py::get_closing_spread` resolves an event_id from the pinned ledger and calls The Odds API's paid `odds-history` endpoint, one HTTP call per game, cached only in-process. `src/odds/ats_compute.py::resolve_closing_spread` nominally tries pinned -> snapshot -> history tiers, but the pinned and snapshot caches are never populated on main, so only the history tier ever fires (known quirk). A more complete ATS-from-API implementation exists only on the unmerged `feature/ats-api-backfill` branch (27 commits ahead; adds `odds_api_client.py`, `ats_backfill_api.py`, `historical_events.py`, `participants_cache.py`).

### 3.4 Scores and ATS

Late stages of both refreshes.

**Score backfill.** `src/scores/nfl_backfill.py::backfill_nfl_scores` and `src/scores/cfb_backfill.py::backfill_cfb_scores` fill final scores into the previous `BACKFILL_WEEKS` (default 2) weeks' games_week files from the schedule masters, run a per-game ATS repair against the sidecars, and atomically rewrite the week files. Both use `src/common/backfill_merge.py::merge_games_week`: incoming score-updated rows win, but existing non-blank odds fields, rating fields, and `raw_sources.odds*` keys are preserved (this merge-not-replace behavior was added after a backfill once wiped promoted odds; see `ACCOUNTING-2026-06-13.md`). With `BACKFILL_PROMOTE_PREV` set, staged odds are re-promoted into the backfilled weeks. After changes, the builder is re-run as a subprocess with `check=False` (exit code ignored).

**Season-to-date ATS.** `src/cfb_ats.py` tallies each team's ATS W-L-P from prior weeks' games_week rows and stamps cumulative strings plus `*_ats_{w,l,p}` counts onto the current week's rows (JSONL only; the CSV is not updated). The NFL twin, `src/ats/nfl_ats.py`, reads `out/nfl/{S}_week{w}/games_week_*.jsonl` (nfl_ats.py:36,41), a directory where games_week files are never written, so it is a permanent no-op in production; the `ATS(NFL): teams=0 rows_updated=0` log line is the visible symptom (known quirk). NFL games therefore carry only the ATS values computed by the gameview builder from schedule spread lines.

### 3.5 Common utilities and Sagarin ratings

**`src/common/`**
- `io_utils.py`: `.env` loading and `getenv` (loads `.env` with `override=True`, so a local `.env` wins over real environment variables; the older `read_env` in the same file uses `override=False`) (io_utils.py:51 vs 102); `download_csv`; `ensure_out_dir`/`week_out_dir` anchored to the repo root; non-atomic jsonl/csv writers.
- `io_atomic.py`: tmp + `os.replace` atomic writers used by the newer modules.
- `metrics.py`: pure math (rating_diff, dense_rank, SU/ATS record strings).
- `team_names.py`: exhaustive NFL normalizer (abbr/city/synonym to canonical "City Nickname" plus an alphanumeric merge key); closed-world, returns empty on unknown.
- `team_names_cfb.py`: minimal CFB normalizer (8 aliases plus a Title Case fallback that never fails) and a mascot-stripping variant for odds text; joins rely on spellings agreeing after cleanup.
- `cfb_source.py`: three thin CFBD REST wrappers (`/games`, `/teams/fbs`, `/games/teams`, Bearer `CFBD_API_KEY`).
- `current_week_service.py`: the Global Week service (section 5).
- `backfill_merge.py`: the preserve-fields merge described in 3.4.

**Sagarin ingestion: two generations, both live on NFL.**
- Legacy scrapers `src/fetch_sagarin_week_nfl.py` and `src/fetch_sagarin_week_cfb.py`: regex parsing over de-tagged page text. NFL requires exactly 32 teams (RuntimeError otherwise, which kills the whole NFL refresh); CFB keeps classification "A" (FBS) rows and expects 120-140 of them, writing a raw dump plus receipt on failure. These modules are also the parser **library** for the staging generation, which imports their functions.
- Staging wrappers `src/ratings/sagarin_nfl_fetch.py` and `src/ratings/sagarin_cfb_fetch.py`: fetch the same pages, archive raw HTML to `data/sagarin/raw/{league}/` (best effort, CWD-relative path), append parsed rows to the per-season ledger `out/staging/sagarin_latest/{league}/{season}.jsonl`, select the latest row per team by fetch_ts, atomically rewrite the weekly snapshot CSV/JSONL, and upsert `out/master/sagarin_{league}_master.csv`.

Asymmetries worth knowing: the NFL refresh runs **both** generations every time (legacy at refresh_week_data.py:72, staging at refresh_week_data_nfl.py:125), fetching sagarin.com twice per run and writing two weekly snapshot trees (`out/{S}_week{W}/` and `out/nfl/{S}_week{W}/`) plus two writers into one master; CFB runs only one generation per the flag. CFB staging validates and raises on bad parses; NFL staging skips validation and soft-fails with a WARNING. NFL staging also stamps the ledger season as `max(page_season, current_calendar_year)` (sagarin_nfl_fetch.py:249-250), which mislabels late-season fetches made in January/February (known quirk).

### 3.6 Ops (workflow, tools, tests)

- `.github/workflows/refresh.yml`: the only CI workflow; described in section 2. Secrets fall back to repository variables (`secrets.X || vars.X`). No pytest step, no concurrency group.
- `tools/run_refresh_all_and_notify.py`: the production runner (section 2.2).
- Manual one-off tools, not referenced by the workflow: `tools/run_refresh_cfb_and_notify.py` (superseded CFB-only ancestor; has an import-before-sys.path bug at line 14), `tools/recompute_current_week.py` and `tools/recompute_current_week_nfl.py` (duplicates differing only in default league), `tools/seed_schedule_master.py` (seeds from `data/NFL_SCHEDULE_SEED.csv`), `tools/replace_sagarin_master.py` (full master replace from `data/SAGARIN_WEEKLY_HISTORICAL_NFL.csv`), `tools/check_parity_cfb.py`, `tools/diff_games_week.py`.
- `webhooks/discord_notify.ps1`: a PowerShell Discord helper with a hardcoded webhook URL committed in plaintext (webhooks/discord_notify.ps1:20).
- `tests/`: 12 pytest tests across 3 modules (`test_metrics.py`, `test_sagarin_parser.py`, `test_team_names.py`), NFL-only coverage. CI never runs them.
- `requirements.txt`: pandas, requests, lxml, html5lib, python-dotenv, pytest, all unpinned.

### 3.7 Docs and specs

- `context/SPEC_DELIVERABLES.md` plus the three JPG scans: the product contract (Week View and Game View layouts).
- `context/global_week_and_provider_decoupling.md` and `context/merge_summary_global_week.md`: the canon for the Global Week service and the staging/promotion design; the closest docs to current truth.
- `AGENTS.md` and `context/CODEX_RULES.md`: agent working rules and the NOTIFY contract.
- `README.md`: rewritten in July 2026 to match the current system; earlier revisions described only the legacy NFL-only pipeline. `context/implementation.md` predates the CFB build-out and still calls CFB "future work".
- `ACCOUNTING-2026-06-13.md` (untracked): post-mortem of the November 2025 failures.
- Archived data drops: `data/archive/2025_w7_nfl/` (an early NFL week snapshot) and `data/archive/sagarin_wayback/cfb_2025_w1{0,1}.csv` (wayback-recovered Sagarin CFB exports with extra columns, unread by code). A larger batch of tracked debris (an `out-backup/` snapshot, `tmp/` scratch parsers, pasted run logs, an empty patch file) was removed in commit `84fff5c`.

## 4. Data contracts (the out/ tree)

All paths are relative to the repo root. `out/` is git-tracked and CI-committed; treat every file below as both build output and versioned data.

| Artifact | League/Scope | Written by |
|---|---|---|
| `out/{S}_week{W}/games_week_{S}_{W}.jsonl` + `.csv` | NFL week pack | gameview_build; rewritten by backfill and odds promotion |
| `out/cfb/{S}_week{W}/games_week_{S}_{W}.jsonl` + `.csv` | CFB week pack | gameview_build_cfb; rewritten by backfill, promotion, cfb_ats (JSONL only) |
| `out/{S}_week{W}/league_metrics_{S}_{W}.csv` (and `out/cfb/...`) | per-week team stats | fetch_year_to_date_stats(_cfb) |
| `out/{S}_week{W}/sagarin_nfl_{S}_wk{W}.csv/.jsonl` | NFL Sagarin (legacy path) | fetch_sagarin_week_nfl |
| `out/nfl/{S}_week{W}/sagarin_nfl_{S}_wk{W}.csv/.jsonl` | NFL Sagarin (staging path; separate tree, known quirk) | ratings/sagarin_nfl_fetch |
| `out/cfb/{S}_week{W}/sagarin_cfb_{S}_wk{W}.csv/.jsonl` | CFB Sagarin | ratings/sagarin_cfb_fetch |
| `out/{S}_week{W}/odds_{S}_wk{W}.jsonl` (and `out/cfb/...`) | legacy odds snapshot | fetch_week_odds_nfl / _cfb (NFL variant never read back) |
| `out/{S}_week{W}/game_schedules/{game_key}.json` (and `out/cfb/...`) | per-game sidecar | build_team_timelines(_cfb); updated by backfill ATS repair |
| `out/master/nfl_schedule_master.csv`, `cfb_schedule_master.csv` | schedule masters | schedule_master(_cfb).ensure_weeks_present (upsert) |
| `out/master/sagarin_nfl_master.csv`, `sagarin_cfb_master.csv` | Sagarin masters | sagarin_master append_week and staging upsert (two NFL writers) |
| `out/state/current_week.json` | per-league {season, week, computed_at} | current_week_service |
| `out/state/last_refresh_summary_cfb.json` | legacy CFB runner state | tools/run_refresh_cfb_and_notify (superseded) |
| `out/staging/odds_raw/{league}/{ts}.jsonl` | raw odds snapshots (append-only, one file per fetch) | odds ingest |
| `out/staging/odds_pinned/{league}/{season}.jsonl` | schedule-pinned odds ledger (append-only, duplicates per run) | odds pin |
| `out/staging/odds_unmatched/{league}/{ts}.jsonl` | pin misses with reasons | odds pin |
| `out/staging/sagarin_latest/{league}/{season}.jsonl` | Sagarin staging ledger (append-only) | ratings staging |
| `out/cfb/{S}_week{W}/*_receipt.json`, `*_debug.json`, `orchestrator_gaps.json` | CFB-only receipts/debug | CFB stages |
| `out/final_league_metrics_{season}.csv` (and `out/cfb/...`) | prior-season EOY stats for the frontend | fetch_last_year_stats(_cfb), manual |
| `out/logs/refresh_{ts}.json`, `out/logs/refresh_index.tsv` | run logs | run_refresh_all_and_notify |
| `data/sagarin/raw/{league}/*.html` | raw Sagarin page archive | ratings staging (best effort) |

**games_week row (the frontend contract).** One JSON object per game: `season, week, game_key, kickoff_iso_utc`, team fields (`home/away_team_norm`, `_raw`), market fields (`favored_side, spread_favored_team, spread_home_relative, total`, moneylines, `odds_source, is_closing, snapshot_at`), ratings (`home/away_pr, pr_rank, sos, sos_rank, hfa, rating_diff, rating_vs_odds, rating_diff_favored_team, rating_vs_odds_favored_team`), season-to-date stats (`home/away_pf_pg, pa_pg, ry_pg, py_pg, ty_pg, *_allowed_pg` plus offense/defense ranks), records (`home/away_su, home/away_ats, to_margin_pg`), scores when backfilled (`home_score, away_score`), and a `raw_sources` blob embedding the source schedule row, sagarin rows, league_metrics rows, and odds row. `game_key` format: `YYYYMMDD_HHMM_{away_slug}_{home_slug}` where NFL slugs are abbreviations (`20251121_0115_hou_buf`) and CFB slugs are name tokens (`20251112_0030_kent_state_akron`).

**league_metrics CSV header:** `Team, RY(O), R(O)_RY, PY(O), R(O)_PY, TY(O), R(O)_TY, RY(D), R(D)_RY, PY(D), R(D)_PY, TY(D), R(D)_TY, TO, PF, PA, SU, ATS`. The `final_league_metrics_{season}.csv` files use the same header.

**Sidecar JSON:** top-level `game_key, home_ytd, away_ytd, home_prev, away_prev`; each entry has `season, week, date, opp, site, pf, pa, result, ats, to_margin, pr, pr_rank, sos, sos_rank, opp_pr, opp_pr_rank, opp_sos, opp_sos_rank`.

**Sagarin master CSV header:** `league, season, week, team_norm, team_raw, pr, rank, sos, sos_rank`, keyed on (league, season, week, team_norm), keep-last upsert.

**Pinned odds row:** `fetch_ts, source, season, week, game_key, market, book, line{spread_home_relative, favored_side, spread_favored_team, prices, raw_outcomes}, home_norm, away_norm, kickoff_utc, role_swapped, raw_event`.

## 5. Current-week resolution

`src/common/current_week_service.py::get_current_week(league)` is the system clock. It reads the league's schedule master (`out/master/{nfl|cfb}_schedule_master.csv`), takes the **max season** present, groups that season's games by week, and builds one window per week: anchored at the Tuesday 00:00 UTC on or before the week's earliest kickoff, extending 7 days (current_week_service.py:48-105). The week whose window contains now wins; before the first window it clamps to the earliest week, after the last it stays pinned to the latest week (so in the offseason the cron keeps rebuilding the final played week). `WEEK_FORCE_LEAGUE` + `WEEK_FORCE` (`"season-week"`) override everything (lines 135-146). Unless `persist=False`, every call also writes `out/state/current_week.json` (lines 108-122, 175-176), even a pure read.

The hard dependency: if the schedule master is missing or empty, `get_current_week` raises (lines 149-150) and both orchestrators SystemExit in auto mode. The master is populated by `ensure_weeks_present`, which runs *inside* the refresh, so a fresh environment works only because the masters are committed to git. Note also that `sagarin_nfl_fetch.py` calls `get_current_week("NFL")` a second time internally (line 247) to pin its weekly snapshot.

## 6. External data sources

| Source | Provides | Access | Fragility |
|---|---|---|---|
| sagarin.com (`/sports/nflsend.htm`, `/sports/cfsend.htm`) | Power ratings, SoS, HFA | Plain HTTP GET with a spoofed Chrome UA; no key | Highest-risk input. Regex parsing over de-tagged fixed-width text; NFL rows must contain `(NFC`/`(AFC` and exactly 32 teams must parse or the NFL refresh dies; CFB depends on `COLLEGE FOOTBALL ... WEEK` / `CONFERENCE AVERAGES` markers, classification letter "A", and a 120-140 row window. Any cosmetic page change breaks ingestion. Fetched twice per NFL run. Raw HTML archived to `data/sagarin/raw/` as insurance. |
| nflverse (GitHub releases) | NFL schedules + scores (`games.csv`), team-week stats (`stats_team_week_{S}.csv`) | Public CSV downloads, no key | Stable format, but re-downloaded 4+ times per refresh with no caching (io_utils.py:58-62); each download is a fresh network-failure surface. |
| CollegeFootballData API | CFB schedules, scores, FBS team list, per-game team stats | `https://api.collegefootballdata.com` `/games`, `/teams/fbs`, `/games/teams`; Bearer `CFBD_API_KEY` | Rate/quota sensitive: one refresh makes dozens of calls (per-week stats loop, double schedule fetch, double gameview build). Fetch errors in `load_games` are swallowed into an empty DataFrame, surfacing later as "0 normalized rows". |
| The Odds API v4 | Spreads, totals, moneylines (`/sports/{sport}/odds`); closing lines (`/events/{id}/odds-history`, paid tier) | `THE_ODDS_API_KEY` query param | Quota-metered. Four independent client implementations exist. The legacy NFL snapshot plus staging ingest means double spend per run; every ATS closing-spread lookup is one paid history call with no on-disk cache. Event names must match schedule names via token matching; naming drift lands records in the unmatched quarantine (and can abort CFB via the coverage gate). |
| TeamRankings | NFL turnover margin (weekly) and season-final stat tables | `pd.read_html` scrapes of `teamrankings.com/nfl/stat/*` | HTML-table scrape; a markup change breaks `generate_league_metrics` and aborts the NFL refresh. Scraped live, so re-running past weeks injects today's turnover values (known quirk). |
| Discord | Run notifications | Webhook POST, `DISCORD_WEBHOOK_URL` | Best-effort; failures never affect the run. A live webhook URL is also hardcoded in `webhooks/discord_notify.ps1`. |
| Don Best XML v2 | Alternate NFL odds | `DONBEST_TOKEN` (never provisioned) | Unused in practice; CLI-only code path. |

## 7. Configuration reference

Env is loaded from `.env` at the repo root via `io_utils.getenv` with `override=True`, so `.env` values beat real environment variables in local runs (known quirk; CI has no `.env`). Names only; never commit values.

**The 16 keys present in `.env`:**

| Key | Read at | Effect |
|---|---|---|
| `THE_ODDS_API_KEY` | all odds fetchers (fetch_week_odds_*, odds/{nfl,cfb}_ingest, odds_history) | Enables The Odds API; absent means legacy snapshot skipped, ingest no-op, closing spreads unresolvable |
| `THE_ODDS_API_KEY_TEMP` | nothing on main | Belongs to the unmerged feature/ats-api-backfill branch |
| `CFBD_API_KEY` | src/common/cfb_source.py | Bearer auth for all CFBD calls; absent yields empty schedules |
| `DISCORD_WEBHOOK_URL` | tools/run_refresh_all_and_notify.py:291 | Where the run summary is posted |
| `ODDS_STAGING_ENABLE` | odds ingest (default on) | Gates raw odds ingest + pinning |
| `ODDS_PROMOTION_ENABLE` | refresh_week_data_nfl.py:147, refresh_week_data_cfb.py:668 (default on) | Gates promotion of staged odds into games_week rows |
| `ODDS_LEGACY_JOIN_ENABLE` | both promote blocks (default off) | Enables `diff_game_rows` legacy-vs-promoted comparison logging |
| `ODDS_SELECT_POLICY` | both promote blocks (default `latest_by_fetch_ts`) | Promotion selection: `latest_by_fetch_ts` or `closing_pre_kickoff` |
| `ODDS_CACHE_ONLY` | refresh_week_data_cfb.py:55 (informational only) | Echoed in a config line; dispatches nothing on main |
| `SAGARIN_STAGING_ENABLE` | sagarin_nfl_fetch.py:212 (inside fetcher); refresh_week_data_cfb.py:520-524 (in orchestrator) | Default on. Off switches CFB to the legacy scraper + sagarin_master_cfb path and no-ops NFL staging |
| `BACKFILL_WEEKS` | nfl_backfill.py:426, cfb_backfill.py:438 (default 2) | How many prior weeks get score backfill |
| `BACKFILL_PROMOTE_PREV` | nfl_backfill.py:443, cfb_backfill.py:454 (default off) | Re-promote staged odds into backfilled weeks |
| `ATS_BACKFILL_ENABLED` | only the never-called `_config_banner` in refresh_week_data_cfb | No effect on main |
| `ATS_BACKFILL_SOURCE` | refresh_week_data_cfb.py:58,62 (log line only) | No dispatch on main; the API implementation lives on feature/ats-api-backfill |
| `ATS_DEBUG` | nothing on main | Branch-only |
| `CFBD_REFRESH` | schedule_master_cfb.py:198, refresh_week_data_cfb.py:56 | Off skips the CFBD schedule-master update (dry-run knob) |

**Keys read by code but absent from `.env` (code defaults apply):**

| Key | Read at | Effect |
|---|---|---|
| `SCORES_BACKFILL_ENABLE` | src/scores/nfl_backfill.py:423 | Gates NFL score backfill (default on) |
| `ATS_ENABLE` | src/ats/nfl_ats.py:114,171 | Gates the (no-op) NFL ATS aggregate |
| `CFB_ATS_DRYRUN` | src/cfb_ats.py:182 | 1 computes CFB ATS but skips the write |
| `WEEK_FORCE` / `WEEK_FORCE_LEAGUE` | src/common/current_week_service.py:135-136 | Force `(season, week)` for a league (or `ALL`/`*`), bypassing the schedule windows |
| `ODDS_PIN_DAY_WINDOW` | refresh_week_data_nfl.py:45, refresh_week_data_cfb.py:359 | Pin match day window (default 3) |
| `ODDS_PIN_MAX_KICKOFF_DELTA_HOURS` | same call sites | Pin kickoff tolerance (default 36) |
| `ODDS_ROLE_SWAP_TOLERANCE` | same call sites | Allow home/away-swapped matches (default 1) |
| `CFB_WRITE_DEBUG_SCHEDULE` | src/fetch_games_cfb.py:202 | Would write `_schedule_norm.csv`; unreachable behind the CLI abort |

CI-only: the workflow maps `THE_ODDS_API_KEY`, `CFBD_API_KEY`, `DISCORD_WEBHOOK_URL` from secrets (falling back to repo variables) and `BACKFILL_PROMOTE_PREV`, `BACKFILL_WEEKS`, `ODDS_PROMOTION_ENABLE` from variables (refresh.yml:14-20).

## 8. Frontend

Static ES modules under `web/`, no build step, no runtime external services; `index.html` redirects to `web/week_view.html`. League is a `?league=` URL param persisted in localStorage; it flips the path prefix between `out/` (NFL) and `out/cfb/` (CFB) plus a handful of formatting branches. HFA display defaults live in the one genuinely shared module, `web/js/game_metrics.js` (nfl=2.2, cfb=2.1).

**Week discovery.** `week_view.js::listAvailableWeeks` (week_view.js:497-530) fetches the HTML directory listing of `out/` or `out/cfb/` and regex-matches `href="{season}_week{week}/"` anchors. This only works on servers that emit auto-index pages (for example `python -m http.server`); on GitHub Pages it silently returns nothing and the page falls back to the localStorage last-selection or manual season/week inputs (known quirk). There is no manifest file listing available weeks.

**Data loading.**
- Week View fetches `games_week_{S}_{W}.jsonl`, parses it line by line with a NaN/Infinity-to-null regex sanitizer (the Python writers can emit bare `NaN`), renders two `<tr>` per game, and caches the parsed week in localStorage under `week-view:games:v4:{league}:{season}:{week}` with staleness heuristics (missing rating keys, or under 60% CFB metric coverage, evict).
- Game View re-uses the same week loading (duplicated code), then fetches the sidecar `game_schedules/{game_key}.json` for the four schedule tables, and `final_league_metrics_{season-1}.csv` for the prior-season EOY block. Deep links without `?league=` probe the NFL then CFB week files for the game_key.
- Printable refetches the week JSONL, the sidecar, the current week's `league_metrics_{S}_{W}.csv` (for offense/defense ranks), and the prior-season EOY CSV, then fills fixed DOM ids. It resolves paths as `../out/...` relative to the page rather than using `web/js/base_path.js`, so the two pages have different hosting assumptions (known quirk).

**Notable hardcoded assumptions:** all CSV parsing is naive `split(",")` with no quote handling; the printable page's EOY heading is the literal string "2024 End of Year Statistics" (game_view_printable.html:235); a CFB "soft block" tri-state (localStorage `cfb_soft_block`, `?hard`/`?soft` params) controls whether a week with zero odds coverage renders or shows an error row; NFL team alias tables are duplicated with different shapes across the three JS files.

## 9. Error-handling map

What kills a run vs what is silently absorbed, condensed from the production trace.

**Aborts the CFB refresh** (uncaught `StageError` traceback, exit 1; StageError is raised at 15+ gates and never caught anywhere):
- Schedule produced 0 normalized rows (including the swallowed-CFBD-error path that returns an empty DataFrame)
- League metrics subprocess rc != 0 or output CSV missing; coverage gates below threshold
- Odds subprocess rc != 0, debug receipt missing, or odds match coverage below thresholds (a provider naming drift can fail the whole refresh here)
- Sagarin staging: any exception, validation failure, or empty snapshot
- Gameview build (either pass): rc != 0, empty output, or missing favorite fields
- Timelines subprocess rc != 0, missing sidecar dir, or Sagarin sidecar coverage < 0.85

**Aborts the NFL refresh** (the legacy core runs in-process, so any uncaught exception is fatal):
- nflverse `games.csv` or `stats_team_week` download failure
- TeamRankings `pd.read_html` scrape failure
- Sagarin page parsing anything other than exactly 32 teams (fetch_sagarin_week_nfl.py:334-338)
- Schedule-master duplicate keys (schedule_master.py:150)
- Timelines: missing schedule rows for any game, or Sagarin PR join coverage < 80% (refresh_week_data.py:153, 172-175)
- Latent, currently masked: the NFL backfill's sidecar ATS repair calls `_update_sidecar_entry` with an incompatible signature (nfl_backfill.py:345-350 vs the def at 224) and would TypeError, unguarded, the moment its preconditions (sidecar present, scores present, closing spread resolved) are met; and `nfl_ats.py:203` references `pd` without importing pandas, masked only because the module is a no-op

**Swallowed (WARNING or silent, run continues):**
- CFB schedule-master CFBD update failure; master odds enrichment failure
- All odds staging ingest fetch exceptions, both leagues (an API outage looks identical to an off week: raw=0)
- Missing `THE_ODDS_API_KEY` / `CFBD_API_KEY` (empty data instead of failure, except where downstream 0-row gates fire)
- NFL legacy odds fetch: HTTP 401/403/429 print `ODDS_FETCH_ERROR` and return empty; other HTTP errors abort
- NFL Sagarin staging fetch/parse failures (WARNING plus zero-count summary; stale ratings stay in place)
- Sagarin raw-HTML archiving failures (best effort)
- Malformed JSONL lines in every reader in the repo (skipped without logging or counting)
- Backfill's subprocess gameview rebuilds (`check=False`; a failed rebuild leaves stale files with no signal)
- Discord posting failures (stderr WARN only)

**Isolation and publication:** the two leagues run as separate subprocesses, so one failing never blocks the other, but if either fails the runner exits 1 and the auto-commit step is skipped entirely, so a failed run publishes nothing (including that run's Sagarin HTML archives and logs).

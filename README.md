# football-statfinder

Weekly betting data packs for NFL and college football (FBS). Twice a day, an automated pipeline scrapes Sagarin power ratings, pulls schedules, scores, season-to-date stats, and betting odds, joins them per game, and publishes per-week "game view" packs that a static HTML/JS frontend renders as three printable views (Week View, Game View, Printable game sheet). The system replaces a hand-maintained set of Excel sheets; the scanned originals in `context/` are the product spec.

**Start here:**

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the system works today: production call tree, subsystems, data contracts, config reference, error-handling map.
- [`docs/REBUILD.md`](docs/REBUILD.md) — known bugs, duplication inventory, dead/stranded code (including unmerged branch work), and the phased rebuild plan for next season.

**Two trees coexist during the rebuild:** `src/` is the season-1 system and still what production runs; `football_statfinder/` is the season-2 package (REBUILD.md Phase 1, built 2026-07-02) — one league-parameterized pipeline, typed config, unified `out/{league}/{season}_week{week}/` layout, and a `statfinder` CLI. The new package does not import from `src/` and does not run in production yet.

## Season-2 package (rebuild)

```bash
pip install -e .            # installs the statfinder console script

statfinder refresh --league all              # weekly refresh, both leagues
statfinder refresh --league nfl --season 2026 --week 3
statfinder current-week --league nfl
statfinder seed-schedule --league nfl --season 2026   # bootstrap a new season's master
statfinder export --league nfl --season 2026 --week 3 # rebuild a week's files from the DB

python -m pytest tests/     # 203 tests (season-1 + season-2)
```

Config precedence in the new package is the reverse of season 1: real environment variables beat `.env`. Each refresh writes a machine-readable run summary to `out/state/run_summary_{league}.json` alongside the legacy NOTIFY line, and dual-writes every stage's results to a SQLite mirror at `data/statfinder.sqlite3` (gitignored; flat files stay canonical — `statfinder export` reproduces them byte-identically from the DB). Output parity with season 1 is proven by the offline replay harness in `tools/parity/` (reports in `docs/parity/`; method and delta whitelist in `docs/PHASE2_SPEC.md`).

## How it runs in production

`.github/workflows/refresh.yml` (cron 10:00 and 22:00 UTC) runs `tools/run_refresh_all_and_notify.py`, which refreshes CFB then NFL as isolated subprocesses, posts a summary to Discord, and auto-commits the regenerated `out/**` and `data/sagarin/raw/**` back to this repo. There is no external database: `out/` is git-tracked and *is* the datastore, and the frontend fetches files from it directly.

Each league orchestrator ends with exactly one machine-readable line the runner greps for:

```
NOTIFY: <LEAGUE> refresh complete week=<season>-<week> rows=<n> odds_promoted=<p>
```

## Manual runs

```bash
pip install -r requirements.txt

# Full production run (CFB + NFL + Discord + logs)
python tools/run_refresh_all_and_notify.py

# One league, current week auto-detected from the schedule masters
python -m src.refresh_week_data_nfl
python -m src.refresh_week_data_cfb

# One league, explicit week
python -m src.refresh_week_data_nfl --season 2025 --week 7
python -m src.refresh_week_data_cfb --season 2025 --week 7

# Tests (203 tests, pure logic only; CI does not run them)
python -m pytest tests/
```

Credentials come from a `.env` at the repo root (never committed). Minimum useful set: `THE_ODDS_API_KEY` (The Odds API), `CFBD_API_KEY` (CollegeFootballData, CFB only), `DISCORD_WEBHOOK_URL` (optional notifications). The full flag reference is in `docs/ARCHITECTURE.md` section 7. Note that `.env` values override real environment variables in local runs.

## Frontend

Static ES modules under `web/`, no build step. Serve the repo root and open `web/week_view.html`:

```bash
python -m http.server 8000
# http://localhost:8000/web/week_view.html?league=nfl
```

Week auto-discovery relies on server directory listings, so it works under `python -m http.server` but not on GitHub Pages (manual season/week entry works everywhere). League switching (`?league=nfl|cfb`) flips the data root between `out/` and `out/cfb/`.

## Key outputs

| Artifact | What it is |
| --- | --- |
| `out/{S}_week{W}/games_week_{S}_{W}.jsonl` + `.csv` | NFL week pack: one record per game with ratings, odds, stats, ranks |
| `out/cfb/{S}_week{W}/games_week_{S}_{W}.jsonl` + `.csv` | CFB week pack (same schema family) |
| `out/{...}/game_schedules/{game_key}.json` | Per-game sidecar: home/away year-to-date and prior-season timelines |
| `out/{...}/league_metrics_{S}_{W}.csv` | Season-to-date team stats and ranks |
| `out/{...}/sagarin_*_{S}_wk{W}.csv/.jsonl` | Weekly Sagarin ratings snapshot |
| `out/master/*.csv` | Schedule and Sagarin masters (also drive current-week detection) |
| `out/staging/` | Append-only odds and Sagarin staging ledgers |
| `data/sagarin/raw/` | Archived raw Sagarin HTML pages (parser insurance and test fixtures) |

The full artifact catalog and record schemas are in `docs/ARCHITECTURE.md` section 4.

## Repo layout

```
football_statfinder/  Season-2 package: league-parameterized pipeline (config, leagues, paths,
                      common/, sources/, pipeline/, refresh.py orchestrator, cli.py)
src/                  Season-1 pipeline (per-league orchestrators + subsystem modules; production)
src/common/           Shared utils: env/config, atomic writes, team names, current-week service
src/odds/             Odds staging pipeline: ingest -> pin to schedule -> promote
src/ratings/          Sagarin staging fetchers (current generation)
src/scores/, src/ats/ Score backfill and against-the-spread computation
web/                  Static frontend (Week View, Game View, Printable)
tools/                Production runner + one-off utilities
context/              Product spec, scanned deliverables, design canon docs
tests/                Pure-logic tests (metrics, Sagarin parser, team names)
out/                  Generated data (git-tracked; committed by CI)
```

Working conventions for agentic sessions are in `AGENTS.md`.

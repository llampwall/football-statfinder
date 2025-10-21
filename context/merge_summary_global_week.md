# Global Week & Provider Decoupling Summary

## 1. Overview
Provider week drift and one-off refresh arguments had us babysitting both leagues, leading to stale odds, mismatched weeks, and manual recoveries. The Global Week service replaces that guesswork with schedule-driven season/week selection and removes provider labels from the control loop.

We now ingest odds and ratings into append-only staging layers, pin them to schedule master rows, and promote the latest snapshots only when the schedule week is active. Refreshes run the same way at any time: staged data accumulates safely, odds promotion and ATS backfill operate deterministically, and the Game/Week views stay stable.

Twice-daily refreshes (or manual runs) now read the same centralized schedule, stage odds/ratings, backfill scores, compute ATS, and emit one concise summary with a Notification hook. No UI schema changes were required to deliver week-independent stability.

## 2. Branches & Scope
- Source branches: `feature/cfb-global-week`, `feature/nfl-global-week`
- Target branch: `integration/global-week` (to be merged into `main`)
- Out of scope: UI redesign, schema changes (none introduced)

## 3. Major Changes by Area
- **Global Week service:** `src/common/current_week_service.py`, state persisted in `out/state/current_week.json`, helper script `tools/recompute_current_week.py`
- **Odds staging & pinning (CFB):** `src/odds/cfb_ingest.py`, `src/odds/cfb_pin_to_schedule.py`, promotion via `src/odds/cfb_promote_week.py`
- **Odds staging & pinning (NFL):** `src/odds/nfl_ingest.py`, `src/odds/nfl_pin_to_schedule.py`, promotion via `src/odds/nfl_promote_week.py`
- **Ratings staging (Sagarin):** `src/ratings/sagarin_cfb_fetch.py`, `src/ratings/sagarin_nfl_fetch.py`
- **Scores backfill:** `src/scores/cfb_backfill.py`, `src/scores/nfl_backfill.py`
- **ATS aggregation:** `src/ats/cfb_ats.py`, `src/ats/nfl_ats.py`
- **Orchestrators:** `src/refresh_week_data_cfb.py`, `src/refresh_week_data_nfl.py`
- **Diff helper:** `tools/diff_games_week.py` (league-agnostic)
- **Env access:** `src/common/io_utils.py` (expanded `read_env()` / `getenv()` usage)
- **Discord notify wrappers:** `tools/run_refresh_cfb_and_notify.py` and (if configured) the NFL counterpart for schedule automation

## 4. File Outputs & Paths
- **Staging (append-only):**
  - `out/staging/odds_raw/{league}/<YYYYMMDDTHHMMSSZ>.jsonl`
  - `out/staging/odds_pinned/{league}/{season}.jsonl`
  - `out/staging/odds_unmatched/{league}/<YYYYMMDDTHHMMSSZ>.jsonl`
  - `out/staging/sagarin_latest/{league}/{season}.jsonl`
- **Public week artifacts (unchanged schemas):**
  - CFB: `out/cfb/{season}_week{week}/games_week_{season}_{week}.{jsonl,csv}`
  - NFL: `out/{season}_week{week}/games_week_{season}_{week}.{jsonl,csv}`
- **Master tables:** `out/master/sagarin_{league}_master.csv`
- No unintended `out/nfl/...` duplicates remain (legacy parity scripts removed); public file naming stayed intact.

## 5. Runtime Knobs (via `.env` / `getenv()`)
- `ODDS_STAGING_ENABLE` (default `1`)
- `ODDS_PROMOTION_ENABLE` (default `1`)
- `ODDS_LEGACY_JOIN_ENABLE` (default `0`)
- `SAGARIN_STAGING_ENABLE` (default `1`)
- `BACKFILL_WEEKS` (default `2`)
- `DISCORD_WEBHOOK_URL` (used by `tools/run_refresh_*_and_notify.py`)
- Optional debug/testing toggles: `WEEK_FORCE`, `WEEK_FORCE_LEAGUE`, `ATS_DRY_RUN` (all default to safe/disabled).

## 6. Logging Contract (per refresh run)
- Start: `CurrentWeek(<LEAGUE>)=<SEASON> W<WK> computed_at=<ISOZ>; args=<auto|manual>`
- Odds staging: `<LEAGUE> ODDS STAGING: raw=<n> pinned=<n> unmatched=<n> candidate_sets_zero=<n> candidate_sets_multi=<n> markets={...} books={...}`
- Ratings staging: `Sagarin(<LEAGUE>): latest_fetch_ts=<ISOZ> teams=<n> wrote_master_rows=<n> selected_for_build=<n>`
- Scores backfill: `Scores(<LEAGUE>): weeks=[Wk,...] updated=<u> skipped=<s>`
- ATS: `ATS(<LEAGUE>): teams=<t> rows_updated=<r>`
- Odds promotion: `<LEAGUE> ODDS PROMOTION: week=<season-week> promoted=<m> by_market={...} by_book={...} legacy_mismatch=<k>` (+ `hint="provider ahead; staged"` when applicable)
- Final notify: `NOTIFY: <LEAGUE> refresh complete week=<season-week> rows=<n> odds_promoted=<m>`

## 7. How to Run (Smoke Tests)
- CFB auto: `python -m src.refresh_week_data_cfb`
- CFB simulation: `python -m src.refresh_week_data_cfb --season 2025 --week 10`
- NFL auto: `python -m src.refresh_week_data_nfl`
- NFL simulation: `python -m src.refresh_week_data_nfl --season 2025 --week 9`
- Each command should emit the final NOTIFY line above and leave the corresponding `games_week_{season}_{week}` JSONL/CSV in place.

## 8. Ops / Scheduling
- Recommended cadence: twice daily (morning/evening, UTC anchored) to align with provider data and schedule updates.
- Entry points (no args):
  - `python -m tools.run_refresh_cfb_and_notify`
  - `python -m tools.run_refresh_nfl_and_notify` (if present in ops tooling)
- Ensure the scheduler/task runner sets `WorkingDirectory` to the repo root so `.env` is detected and staging paths resolve correctly.

## 9. UI Stabilization Notes
- **Week View:** odds display signed spreads, PR/SOS appear on both sides, SOS diff only on the higher-SOS row, and teams are sorted alphabetically; week navigation no longer depends on provider labels.
- **Game View:** CFB reads `home_ats` / `away_ats` when populated (falls back to em dash otherwise); NFL ATS now fills from finalized lined games; ratings columns still align with legacy expectations.

## 10. Risk & Rollback
- Risks: schedule boundary heuristics per league, staging path drift, or legacy scripts writing unexpected files.
- Rollback mitigation:
  - Disable odds staging via `ODDS_STAGING_ENABLE=0` (legacy odds join can still be toggled via `ODDS_LEGACY_JOIN_ENABLE=1`).
  - Disable ratings staging with `SAGARIN_STAGING_ENABLE=0` to fall back to existing weekly fetch output.
  - Reduce `BACKFILL_WEEKS` to limit prior-week rewrites if churn becomes an issue.

## 11. PR Checklist Template
```
### Verification
- [ ] CFB current (auto) and simulated run produced NOTIFY lines
- [ ] NFL current (auto) and simulated run produced NOTIFY lines
- [ ] UI spot-check: CFB & NFL Week 8/9 Game View + Week View render correctly
- [ ] No public schema changes; new files only under out/staging/*
- [ ] Refresh schedules configured (or rollout plan documented)
```

## 12. Appendix: Verification Commands
```
python -m src.refresh_week_data_cfb
python -m src.refresh_week_data_cfb --season 2025 --week 10
python -m src.refresh_week_data_nfl
python -m src.refresh_week_data_nfl --season 2025 --week 9
```
Expect each command to conclude with `NOTIFY: <LEAGUE> refresh complete week=<season-week> rows=<n> odds_promoted=<m>`.

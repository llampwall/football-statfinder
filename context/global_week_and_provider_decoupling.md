# 🧭 GLOBAL WEEK + PROVIDER DECOUPLING SPEC

**Goal:**
Make all available, relevant, and correct data appear on the Week and Game Views at any time—without human week input or provider “current week” drift.

---

## A. Global Week Service (Source of Truth)

### A1) Persistent global state

Store once, read everywhere:

```json
{
  "NFL": {"season": 2025, "week": 7, "computed_at": "2025-10-19T18:45Z"},
  "CFB": {"season": 2025, "week": 8, "computed_at": "2025-10-19T18:45Z"}
}
```

Path: `out/state/current_week.json`
Optional mirror: `public.state_current_week` (DB).
Access function: `get_current_week(league) -> (season, week)`.

### A2) Computation (schedule-driven)

* Determine **current week solely from the schedule master**.
* For each league, find the week whose kickoff window (from schedule data) contains “now” (UTC).
* Default boundaries:

  * NFL: Tue 00:00 UTC → Tue 00:00 UTC.
  * CFB: Tue 00:00 UTC → Tue 00:00 UTC.
* Postseason & bowls: always prefer the **week identifiers in schedule master** over generic UTC rules.

### A3) Update cadence

* Auto-recompute twice daily (03:00 UTC & 15:00 UTC), UTC-anchored to avoid DST skew.
* Allow overrides (`WEEK_FORCE_LEAGUE`, `WEEK_FORCE`) for manual testing.

---

## B. Ingestion Philosophy — “Never Throw Data Away”

### B1) Odds API ingestion

* Fetch **everything** from the Odds API; do not pre-filter by time.
* Keep every raw snapshot (append-only):

  * `staging/odds_raw/<league>/<iso-date>.jsonl`
  * `staging/odds_pinned/<league>/<season>.jsonl`
* Normalize team names via `team_names_<league>.py`.

### B2) Pinning odds to schedule

Match each provider event to a scheduled game:

1. Build schedule keys `(season, home_norm, away_norm, kickoff_date)` for *all* season games.
2. For each provider event:

   * Match on `(home_norm, away_norm)` within ±3 days of event time.
   * If no match, try swapped `(away, home)` (role-swap tolerance).
   * Pick the candidate with smallest kickoff-time delta ≤ 36 h.
   * Prefer neutral-site schedule keys where applicable.
   * If still ambiguous → quarantine to `staging/odds_unmatched` and log.
3. Record the mapping: provider → `game_key` (+ book, market, timestamp).
4. Keep all snapshots; the **promotion** step selects the *latest* per `game_key/market/book`.

### B3) Week-promotion

During `refresh_week_data_<league>`:

* Load odds for that league/season from `staging/odds_pinned`.
* Promote only those whose `game_key` belongs to the current week’s schedule.
* Overwrite week file’s odds fields with the latest snapshot per game.

Result: future odds stay quietly staged; current odds appear automatically when their week arrives.

---

## C. Ratings (Sagarin) Decoupling

* Scrape the latest page each run, ignore the page’s “Week #”.
* Write:

  * `staging/sagarin_latest/<season>.jsonl` (append with `fetch_ts`).
  * `out/master/sagarin_*_master.csv` (dedup `league,season,week,team_norm`).
* Builders pick the **latest** values at build time.
* This preserves reproducibility and ignores mis-labeled provider weeks.

---

## D. Score Backfill + ATS Timing

* Before ATS, update scores for the **previous two weeks** (default; tunable via `BACKFILL_WEEKS`).
* Fetch finals from official stats source.
* Atomic update of each prior week's `games_week` file.
* If the staged snapshot matches disk content (after canonical ordering), skip the rewrite and log `already up to date`, preserving prior-week artifacts on reruns while still running odds/ATS repair when needed.
* Then run ATS builder:

  * Uses only finished + lined games.
  * Writes to current week's `games_week` rows.

---

## E. Unified Refresh Builders (Argument-Free)

### Command

```
python -m src.refresh_week_data_<league>
```

### Flow

1. `(season, week) = get_current_week(league)`
2. Update schedule master.
3. Ingest odds → normalize → pin → promote.
4. Fetch & stage ratings.
5. Backfill scores (2 weeks back).
6. Build `games_week_{season}_{week}.jsonl` / CSV.
7. Compute ATS.
8. Emit concise log summary.

All writes are atomic and idempotent.
Manual `--season`/`--week` args remain optional for replays only.

---

## F. Observability and Logging

Example one-block summary:

```
Current week: 2025 W8 (computed 05:41Z)
Schedule: weeks 1-15, games 59
Odds: raw=382 pinned=220 promoted_to_week=41 (new=7)
Markets: spreads=38 totals=35 h2h=22
Books: draftkings=41 fanduel=39 betmgm=37
Scores: prior_week_updates=23
Ratings: sagarin_latest_ts=2025-10-19T05:00Z
Games JSONL: rows=59; coverage {spread:41,total:39,rvo:41,sos:59,pr:59,ats:59}
Provider ahead; odds queued in staging (raw>0 but promoted=0)
```

Extra diagnostics:

* Count & top 3 entries from `odds_unmatched`.
* `alias_hits` count.
* Time since last `current_week` recompute.

---

## G. UI Contract (Unchanged)

* Week View / Game View read only from `out/<league_path>/{season}_week{week}/games_week_{season}_{week}.jsonl`.
* No recompute in JS. Missing → em-dash.
* Cache bump only for schema changes.
* Banner:
  *“No lines found for this week yet (provider may be ahead). Last refresh <timestamp>.”*

---

## H. Edge-Case Policy

| Case                                 | Behavior                                             |
| ------------------------------------ | ---------------------------------------------------- |
| Provider posts future odds           | Ingest → stage → wait until week arrives.            |
| Provider late                        | Show — until data present; populate on next refresh. |
| Postponed/canceled                   | Remain in schedule; no odds/scores; ATS skips.       |
| Team rename                          | Fix in `team_names_<league>.py`; alias_hits logged.  |
| Dual-week overlap (postseason/bowls) | Use schedule’s explicit week IDs.                    |

---

## I. Rollout & Safety

1. **Phase 1:** Add global-week service (read-only), log current week.
2. **Phase 2:** Add staging layers (odds, ratings) with dual-write; week builders still read old path.
3. **Phase 3:** Flip builders to staging inputs → promote outputs.
4. **Phase 4:** Parallel run (dual-write old/new) for 1 week; diff `games_week` files ignoring timestamps; PASS = flip default.
5. **Phase 5:** Mirror logic to NFL.

---

## J. Acceptance & Anti-Drift Tests

* **Unit-style:**

  * `get_current_week()` returns correct week for known timestamps.
  * Odds pinning matches correct `game_key` even with side swap/neutral site.
  * Ratings selector picks latest regardless of provider “week.”
* **Pipeline:**

  * Future-only odds → promoted=0, build succeeds.
  * Mixed odds → only current-week promoted.
  * Prior-week finals present → ATS > 0 for finished teams.
* **Idempotence:** back-to-back refresh runs identical (except timestamps).
* **Schema:** no new public fields without Spec update.

---

## K. Implementation Notes

* UTC everywhere; all timestamps ISO 8601.
* Append-only staging; atomic replacement for outputs.
* Week artifacts are written in a stable `(season, week, kickoff_iso_utc, game_key)` order, and JSON payloads use sorted keys so cosmetic diffs disappear between runs.
* No changes to filenames or schemas consumed by UI until explicitly versioned.
* Daily cron: `refresh_week_data_NFL` + `refresh_week_data_CFB` (auto).
* Logs routed to Discord via existing webhook.
* `CFBD_REFRESH=0` suppresses schedule re-fetches, enabling dry runs against a frozen master without triggering placeholder kickoffs.

---

### ✅ Summary

This unified plan:

* Removes manual week handling.
* Decouples all sources from their own week semantics.
* Ensures every run ingests *all* data, pins it deterministically to schedule, and only promotes what belongs.
* Guarantees stable Week/Game views even when providers are early, late, or mislabeled.




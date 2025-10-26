# ATS API Backfill — Historical Flow (NFL + CFB)

## Goal

Fill missing ATS for **closed** games using The Odds API’s **historical** endpoints. Idempotent + merge-only.

## High-level pipeline

1. **Participants cache**

   * Fetch Odds API canonical teams (use `full_name` + `id`) per league.
   * Cache on disk; if file exists but is **empty**, treat as a miss and refetch.
   * Canonicalize our labels using: (a) league normalizers, (b) alias tokens (CFB), (c) mascot-tolerant compact compare as a fallback.

2. **Historical events (per week)**

   * One snapshot **inside the week** (pre-kick) → `date=<ISO8601>Z` (e.g., Tue 12:00Z).
   * Endpoint: `/v4/historical/sports/{sport}/events?apiKey=<key>&date=<Z>`
   * Locally filter to the week window.
   * Match our games → events by:

     * canonical **home/away names** (participants `full_name`), and
     * kickoff within a guard (±60–90 min).
   * Persist/return `eventId` for matched games (use pinned map path if present).

3. **Historical event odds (per game)**

   * Endpoint: `/v4/historical/sports/{sport}/events/{eventId}/odds?apiKey=<key>&date=<kickoff Z>&regions=us&markets=spreads&oddsFormat=american`
   * Select **latest pre-kick** snapshot, prefer **Pinnacle**, else newest pre-kick.
   * Parse outcomes → favored (`HOME|AWAY|PICK`) + absolute spread.

4. **ATS compute + writes**

   * Compute from final scores + (favored, spread).
   * **Merge-only** into week rows: `home_ats`, `away_ats`, `to_margin_home`, `to_margin_away`.
   * **Merge-only** into sidecar YTD for that (season, week).
   * Re-run ⇒ no changes (idempotent).

## Never do

* ❌ Do **not** call **current** odds endpoints for past games.
* ❌ Do **not** overwrite non-blank ATS fields or sidecar values.

## Timestamps

* Always send ISO8601 **with `Z`** (UTC) for `date`.
* Let the HTTP client encode params (don’t hand-encode `+00:00`).

## Env / flags

* `THE_ODDS_API_KEY`
* `ATS_BACKFILL_ENABLED=1`
* `ATS_BACKFILL_SOURCE=api`
* (optional) timeouts/retries/region knobs

## Logging (single-line, low noise)

* On first participants load per league:

  * `ATSDBG(PARTICIPANTS): league=<L> count=<N> sample=[...]`
* Per week fetch:

  * `ATSDBG(HIST-EVENTS): league=<L> snapshot=<ISO>Z fetched=<n>`
  * `ATSDBG(HIST-EVENTS): league=<L> filtered=<m>`
* Per game during resolve:

  * `ATSDBG(RESOLVE): league=<L> week=<W> game=<key> h='<ours>'->'<provider|->' a='<ours>'->'<provider|->'`
* Summary line:

  * `ATS_BACKFILL(API <LEAGUE>): week=<YYYY-W> games_fixed=<n> sources={history:<a>,current:<b>} resolve={pinned:...,events:...,failed:...} usage=used:<X>,remaining:<Y>`

## Matching rules (deterministic)

* Try canonical token/alias; if miss, fall back to mascot-tolerant compact compare (school-only labels OK).
* Time guard default: ±60–90 min (tighten after we confirm provider timing).

## Error handling

* Historical events/odds: surface a single line with status:

  * `ODDS_API_ERROR(get_historical_events): league=<L> status=<code>`
* On API failure, skip gracefully and continue; don’t spam logs.

## Acceptance checklist

* Participants cache for league contains **`full_name` + `id`**, `count>0`.
* Historical events: `fetched>0`, `filtered>0`, multiple games resolve `eventId`.
* Odds calls use **historical event odds** at **kickoff Z** (not current odds).
* First run: `games_fixed>0`; immediate second run: `games_fixed=0`.
* Week rows + sidecars show ATS/to-margin (merge-only).

## Nice-to-have

* Persist newly discovered `eventId`s to a pinned map for future runs.
* Shrink guard to ±15–30 min after a few green weeks.
# ATS API BACKFILL SPEC

## Goal

Backfill **Against-the-Spread (ATS)** results for any week where they’re missing by pulling the **closing spread before kickoff** from The Odds API, computing ATS for each finished game, and **merge-only** writing:

* `home_ats`, `away_ats`, `to_margin_home`, `to_margin_away` onto week rows, and
* `ats`, `to_margin` onto the current season’s sidecar YTD rows (home/away tables).

No schema changes. Deterministic, idempotent. One line of provenance logging per run.

---

## External source (verified)

We use The Odds API **V4**:

* **Find upcoming/in-season events (for event IDs, if needed):** `GET /v4/sports/{sport}/events` (free; returns id, teams, commence_time) ([The Odds API][1])
  Sport keys include `americanfootball_nfl` and `americanfootball_ncaaf`. ([The Odds API][1])

* **Current odds for a single event:** `GET /v4/sports/{sport}/events/{eventId}/odds?regions=us&markets=spreads...` (bookmakers → markets → outcomes with `point`) ([The Odds API][1])

* **Historical event odds (our primary “closing” source):**
  `GET /v4/historical/sports/{sport}/events/{eventId}/odds?date={iso_ts}&regions=us&markets=spreads...`
  Returns odds snapshot(s) at/around the specified timestamp; schema mirrors “event odds”. The earlier /v4/sports/{sport}/odds-history is deprecated.

---

## Resolution order (API-first, but pinned map assists)

1. **Event ID resolution**

   * Look up `event_id` using our pinned map if present (we already write `out/staging/odds_pinned/{league}/{season}.jsonl`; map `game_key → event_id`).
   * If missing, derive event_id by fuzzy match: compare normalized participants (`team_merge_key`) and kickoff window via `GET /v4/sports/{sport}/events` (filter by `commenceTimeFrom`/`commenceTimeTo`). ([The Odds API][1])
   * If still no ID, **skip** the game (log once).

2. **Closing spread fetch (strict pre-kick)**

   * Try **Historical Event Odds** at **`date = kickoff_ts`**; pick the **latest bookmaker snapshot with `timestamp/last_update <= kickoff`** for market `spreads`.
   * If history returns nothing, call **Event Odds** and use the **most recent bookmaker snapshot with `last_update <= kickoff`** for `spreads`. ([The Odds API][1])
   * Bookmaker priority: `"pinnacle"` first if present; else the bookmaker with the **latest valid pre-kick update**; tie-break by most outcomes covered.

3. **Normalize a single spread line**

   * Outcomes include each team with a `point`; normalize to `(favored_side, spread)`:

     * If `home_point` is numeric and `home_point < 0`, favored = `HOME`, spread = `abs(home_point)`.
     * If `away_point < 0`, favored = `AWAY`, spread = `abs(away_point)`.
     * If both are `0` → **pick’em** (treat as push when margin = 0).
     * If data inconsistent (only one side with `point` and it equals 0), **skip**.

---

## ATS computation

Given `home_score`, `away_score`, `favored ∈ {HOME, AWAY}`, `spread ≥ 0`:

```text
home_line = -(spread) if favored == HOME else +(spread)
to_margin_home  = (home_score - away_score) + home_line     # positive means cover
to_margin_away  = -(to_margin_home)
home_ats = 'W' if to_margin_home  > 0 else ('L' if to_margin_home  < 0 else 'P')
away_ats = 'W' if to_margin_away  > 0 else ('L' if to_margin_away  < 0 else 'P')
```

* If a score is missing, **do not compute** (skip).
* Treat non-finite values as `None`.

---

## Files & touch-points (minimal surface)

### New helpers

* `src/odds/odds_api_client.py`

  * `get_historical_spread(league, event_id, kickoff_iso) -> Optional[{favored: 'HOME'|'AWAY', spread: float, book: str, fetched_ts: str}]`
  * `get_current_spread(league, event_id, kickoff_iso) -> Optional[...]`
  * Uses sport keys: `NFL → americanfootball_nfl`, `CFB → americanfootball_ncaaf`. ([The Odds API][1])

* `src/odds/ats_backfill_api.py`

  * `resolve_event_id(league, season, game_row) -> Optional[str]` (pinned map → events fallback)
  * `resolve_closing_spread(league, season, game_row) -> Optional[{favored, spread, source: 'history'|'current', book, fetched_ts}]`
  * `compute_ats(home_score, away_score, favored, spread) -> {home_ats, away_ats, to_margin_home, to_margin_away}`

### Wire-in (existing refreshers only)

* `src/refresh_week_data_nfl.py`
* `src/refresh_week_data_cfb.py`

Right **after scores are available** (same spot you update SU/metrics, before writing week and sidecars):

```python
if getenv("ATS_BACKFILL_ENABLED") == "1" and getenv("ATS_BACKFILL_SOURCE","api") == "api":
    games_fixed, sidecar_rows = backfill_ats_api(LEAGUE, season, week, games_list, sidecar_map)
    print(f"ATS_BACKFILL({LEAGUE}): week={season}-{week} "
          f"games_fixed={games_fixed} rows_updated={sidecar_rows}")
```

Where `backfill_ats_api(...)`:

1. **Skip if already present** — For each game:

   * If any of `home_ats`, `away_ats`, `to_margin_home`, `to_margin_away` is populated (non-blank), leave the row **unchanged**.

2. **Get the closing spread** — Call `resolve_closing_spread(...)`. If `None`, skip.

3. **Compute ATS** — `compute_ats(...)`.

4. **Merge-only patch: week rows**

   * Write those four fields **only** if blank (`None`, `""`, `"—"`, or NaN).

5. **Merge-only patch: sidecars (current season YTD only)**

   * Select the team’s YTD array (`home_ytd` for home, `away_ytd` for away).
   * Find entry with `season == season && week == week`; if **absent**, append a **skeleton** `{season, week, ats: None, to_margin: None}` (do **not** touch other columns).
   * Set `ats` and `to_margin` **only if blank**.
   * Never touch prior-season arrays here.

6. **Log provenance (no schema change):** Print one summary line per run (example above). If you want richer audit, add a short breadcrumb into your existing `raw_sources.odds_row` map (nested), but it’s optional.

7. **Idempotence:** Second run with the same data should result in `games_fixed=0`, `rows_updated=0` and no file diffs.

---

## Normalization & matching rules

* Team names: always run through the existing `team_merge_key` / `team_merge_key_cfb` before comparing odds API team names with our schedule/sidecar names.
* Pre-kick guard: when scanning a bookmaker’s odds, select the **latest snapshot at or before kickoff** (never after).
* Cancelled/postponed/no-score games: skip entirely.
* Bookmaker choice: `pinnacle` > “most recent pre-kick update”.
* Timezone handling: normalize to UTC; keep kickoff in UTC too (Odds API uses ISO8601; you’re already converting).

---

## Environment switches

* `ATS_BACKFILL_ENABLED=1`
* `ATS_BACKFILL_SOURCE=api`  (hard-code in code if you prefer; flag retained for quick off switch)
* `THE_ODDS_API_KEY=...`

---

## Error handling & rate strategy

* Timeouts: 10–20s with a single retry on 5xx.
* Rate awareness: read `x-requests-remaining/used/last` headers for observability (Odds API returns them) ([The Odds API][1])
* Cache results in-run by `(league, event_id)`.

---

## Minimal implementation outline (so Codex stays on-rails)

```python
# src/odds/ats_backfill_api.py
def backfill_ats_api(league, season, week, games, sidecars):
    games_fixed = 0
    rows_updated = 0
    for g in games:
        if _has_ats(g):
            continue
        event_id = resolve_event_id(league, season, g)
        if not event_id:
            continue
        res = resolve_closing_spread(league, season, g, event_id)
        if not res:
            continue
        out = compute_ats(int_or_none(g.get("home_score")),
                          int_or_none(g.get("away_score")),
                          res["favored"], float(res["spread"]))
        if not out:
            continue
        games_fixed += _merge_week(g, out)  # returns 1 if any field updated
        rows_updated += _merge_sidecars(league, season, week, sidecars, g, out)  # returns count of ytd rows updated
    return games_fixed, rows_updated
```

Notes:

* `_merge_sidecars` must **create** a `{season, week}` skeleton row when not present (this was a prior blocker).
* `_has_ats` = any of the four week-row ATS fields populated → skip.

---

## Acceptance

1. **Guardrails:** with `ATS_BACKFILL_ENABLED=0` or missing API key, refresh week runs cleanly and prints `games_fixed=0`.
2. **NFL week with holes (e.g., 2025 W7/W8):** run refresher → expect `games_fixed>0`, `rows_updated>0`, Game View shows ATS in **Team Snapshot** and **Schedule tables**. Re-run → zeros.
3. **Parity:** Repeat for a CFB week with known blanks.
4. **Spot checks:** Confirm at least one game’s ATS matches a public closing line & final score.

---

## Commit message template

```
feat(ats): API-gated ATS backfill via The Odds API (history → current pre-kick).
- Resolve event_id (pinned map; fallback to events)
- Pull closing spreads from odds-history; fallback to event-odds pre-kick
- Compute ATS; merge-only onto week rows + sidecar YTD (create skeleton if missing)
- Strict idempotence & provenance logging
```

---

## Why API-first (and how this differs from earlier work)

* We **always** compute from the provider’s **closing** line (history endpoint) and only fall back to current odds when a **pre-kick snapshot** exists. This avoids the fuzzy “rebuild rank/sidecar” machinery that caused churn yesterday. The endpoint paths and behavior are spelled out in the Odds API docs (see “GET event odds”, “GET historical event odds”, and sport keys) ([The Odds API][1]).
* We **reuse** your pinned map only to resolve event IDs (cutting calls and ambiguity) instead of trusting staged odds values as “closing.”
* All writes are **merge-only** with a **skeleton insert** for the current week’s YTD row—this was the missing piece that previously left sidecars unchanged.



[1]: https://the-odds-api.com/liveapi/guides/v4/ "Odds API Documentation V4 | The Odds API"

# Phase 2 execution spec: parity gate + SQLite storage + export

Status: spec of record for REBUILD.md Phase 2, written 2026-07-03. Execution is delegated
to subagents; this document is the contract. The plan-level rationale lives in
`docs/REBUILD.md` (section 6 and the Phases list); read that first for context.

**Ground rules for every work package (WP):**

- Do NOT modify anything under `src/`, `web/`, `.github/`, or `out/` (except: harness runs
  write to a temp/scratch output root, never the real `out/`). `src/` is still production.
- Do NOT add new third-party dependencies. `sqlite3` is stdlib.
- The legacy test suite and all `football_statfinder` tests must stay green:
  `python -m pytest tests/` (baseline: 177 passing, 0 failing).
- New code goes in `football_statfinder/` (package) or `tools/parity/` (one-shot harness).
- Atomic writes only, via `football_statfinder/common/io_atomic.py`.
- Windows machine, PowerShell 7. No bash-isms in any commands you run.
- If this spec is ambiguous or wrong about a fact you can check, check the code and say so
  in your report; do not silently guess.

---

## Part 1 — Parity gate (WP-A, WP-B)

### Goal

Rebuild one season-1 week per league with the **new** pipeline, entirely offline, from
archived season-1 inputs, and diff the result against the season-1 artifacts. The gate
passes when every delta is explained by a numbered whitelist rule below. Unexplained
deltas are either new-pipeline bugs (fix them) or new whitelist rules (require main-loop
approval — an agent may PROPOSE a rule in its report but must NOT add it to this spec).

### Replay targets

| League | Week | Baseline artifacts (all exist, verified 2026-07-03) |
|---|---|---|
| NFL | 2025 week 16 | `out/2025_week16/` — games_week jsonl (16 rows) + csv, league_metrics csv (32 teams), game_schedules/ (16 sidecars), sagarin_nfl_2025_wk16.jsonl (32 rows), odds_2025_wk16.jsonl |
| CFB | 2025 week 13 | `out/cfb/2025_week13/` — games_week jsonl (60 rows) + csv, league_metrics csv (136 teams), game_schedules/ (60 sidecars), sagarin_cfb_2025_wk13.jsonl (134 rows), odds file is empty (CFB odds live only in staging ledgers) |

Secondary spot-check targets (run after primaries pass, same harness): NFL 2025 week 17,
CFB 2025 week 14.

### Input sourcing per stage (replay is hermetic — zero network)

The canonical injection pattern is `tests/test_refresh_integration.py`: set
`paths_mod.OUT_ROOT` and `run_summary_mod.OUT_ROOT` to a scratch root, monkeypatch-style
inject the fetch seams, call `refresh_mod.refresh_league(...)` or the stage functions
directly. The harness is a plain script (no pytest), so assign the module attributes
directly and restore after.

1. **Schedule** — build the new 20-column `SCHEDULE_COLUMNS` DataFrame
   (`football_statfinder/sources/schedule.py`) from the legacy schedule master:
   `out/master/nfl_schedule_master.csv` / `out/master/cfb_schedule_master.csv`, filtered
   to the replay season+week. Derive the column mapping by reading the legacy master
   header plus the legacy master builders (`src/` — find them via grep for the master
   filename). Map every column you can; `game_key` comes from the NEW builder
   (`common/game_key.build_game_key`), not from any legacy key column. Preserve legacy
   `spread_line` / `total_line` columns if the master has them — the NFL gameview
   schedule-odds fallback needs them (see W7/known-check below).
2. **Sagarin** — construct a `SagarinStagingResult` (see the integration test for the
   shape) whose `weekly_jsonl` you write from the legacy weekly snapshot
   `sagarin_{league}_2025_wk{W}.jsonl` in the baseline week dir. Map legacy row fields to
   what `pipeline/gameview.sagarin_map_from_rows` consumes (`team_norm`, `pr`, `pr_rank`,
   `sos`, `sos_rank`, `hfa`). Read the legacy snapshot first and report its exact field
   set. `hfa` on the result = the per-page hfa in those rows.
3. **Stats** — a fake provider (again per the integration test) whose
   `league_metrics_rows` returns the rows of the legacy `league_metrics_2025_{W}.csv`
   parsed with the same null/blank conventions as
   `football_statfinder/sources/stats.py::team_stats_from_metrics_rows` expects. CFB ATS
   column may be empty (trailing comma) — that is the blank sentinel, becomes None.
   This injects the *output* of the legacy stats stage as input; stats *computation*
   parity is out of scope (the source pages are gone — document this in the report).
4. **Odds** — replay the staging pipeline from the real raw ledgers:
   copy (or point the new pipeline's readers at) `out/staging/odds_raw/{league}/*.jsonl`
   files with fetch timestamps from 14 days before the week's first kickoff through the
   week's last kickoff, into the scratch root's odds_raw dir. Run the NEW pin
   (`pipeline/odds_pin.pin_to_schedule`) against the replayed schedule master, then the
   NEW promote (`pipeline/odds_promote.promote_week`). Do NOT reuse the legacy pinned
   ledger `out/staging/odds_pinned/{league}/2025.jsonl` as input (its NFL keys are
   away-first and its role-swap rows carry the bug-20 sign flip); it is a *comparison*
   aid only.
5. **Gameview, sidecars, recompute** — run the new stages as `refresh_league` sequences
   them (schedule → sagarin → stats → gameview → promotion → recompute → sidecars).
   Sidecars need the sagarin master too: source it from `out/master/sagarin_{league}_master.csv`
   (read `pipeline/sidecars.build_sidecars`'s signature for what it takes).

### Diff and classification (the differ, `tools/parity/diff.py`)

`tools/diff_games_week.py` is prior art but insufficient: it joins on `game_key`
(changed for NFL) and only ignores two volatile fields. Build a new differ:

- **Row join**: match old row ↔ new row on (kickoff instant, home merge-key, away
  merge-key). Merge-keys via `football_statfinder.leagues.{NFL,CFB}.merge_key` applied
  to each side's `home_team_norm`/`away_team_norm` (this absorbs legacy NFL
  abbreviations like `"tb"` vs new full names). Kickoffs parsed to aware datetimes and
  compared as instants (legacy uses both `+00:00` and `Z` suffixes).
- **Field comparison**: every top-level field of the old record vs the new record.
  Floats: equal within 1e-9. Strings: exact after normalization rules below.
  Fields present in only one side are deltas (classified, not ignored).
- **`raw_sources`**: excluded from field comparison ONLY IF the frontend does not read
  it — verify by grepping `web/` for `raw_sources` and record the result in the report.
  If the frontend reads any of it, compare those subfields.
- **Classification**: every delta gets exactly one label:
  `W<n>` (whitelisted, cite the rule), `BUGFIX-<n>` (known legacy bug corrected, cite
  REBUILD.md bug number, and prove it on at least one sampled game by tracing the raw
  input), or `UNEXPLAINED`. The gate requires zero UNEXPLAINED.
- **Report**: write `docs/parity/parity_{league}_{season}_wk{week}.md` (human summary:
  join coverage, per-field delta counts by class, sampled evidence for each BUGFIX
  class, proposed-new-whitelist section) plus a `.json` with full per-game detail.

### Delta whitelist (frozen — agents cite, never extend)

- **W1 game_key format.** NFL keys changed from abbreviation slugs home-first to
  full-name slugs; the join must not use game_key, and key-value deltas are W1.
- **W2 team representation.** `home/away_team_norm` and `_team_raw` may differ in
  representation (legacy NFL: abbreviations); identity must still match via merge_key.
- **W3 ATS record format.** New pipeline always emits `W-L-P`; legacy may emit `W-L`
  or blank/em-dash. Normalize before comparing: pad 2-part records with `-0`; em-dash,
  empty string, and None all mean blank. After normalization the *values* must match;
  a padded-format-only delta is W3.
- **W4 blank sentinel.** Legacy em-dash / empty-string sentinels vs new JSON null.
  Normalize both to null; value-equal-after-normalization deltas are W4.
- **W5 volatile provenance.** `snapshot_at`, any `fetch_ts`, `fetched_at`,
  `computed_at`, `page_stamp`-adjacent timestamps, receipt files, and `source_uid`
  *format* (legacy NFL uses nflverse game_id, CFB uses a numeric id; new may carry the
  same value or a normalized one — value differences here are W5 only if the underlying
  game identity matches; otherwise UNEXPLAINED).
- **W6 kickoff string format.** `+00:00` vs `Z` suffix and second-precision padding;
  compare as instants (a format-only difference is W6).
- **W7 known legacy bug corrections.** Classified `BUGFIX-<n>` per REBUILD.md bugs
  (expected to appear: 7 — one rating-vs-odds formula, so CFB rating_vs_odds sign/def
  may shift; 20 — role-swap pin sign flip, so a minority of NFL spreads/moneylines may
  correct; 4/19 — ATS dead tiers). Each BUGFIX class needs sampled proof from raw
  inputs, not just an assertion.

### Known-check list (things WP-A must explicitly test and report, pass or fail)

- **K1 schedule-odds fallback**: legacy NFL rows show `odds_source: "schedule"` with
  spread/total sourced from schedule columns when no odds were promoted. Read
  `football_statfinder/pipeline/gameview.py` and determine whether the new build has
  this tier. If it does not, that is a real gap: report it as UNEXPLAINED with evidence;
  do not fix it without main-loop sign-off (the fix belongs to WP-B triage).
- **K2 row coverage**: every legacy game must join to exactly one new game and vice
  versa (16/16 NFL, 60/60 CFB). Any orphan is UNEXPLAINED.
- **K3 sidecar parity**: for each joined game, diff the legacy
  `game_schedules/{legacy_key}.json` against the new `game_schedules/{new_key}.json`
  with the same normalization rules. Report per-field delta counts.
- **K4 promoted odds sanity**: for at least 3 NFL games with promoted odds, trace the
  chosen (book, market, fetch_ts) in the raw ledger and confirm the new promote picked
  the latest pre-kickoff record per the select policy.
- **K5 FBS filtering (CFB)**: 60/60 games and no FBS-classification drift.

### WP-B (triage — main loop + spot agents)

Runs after WP-A's first report. Every UNEXPLAINED delta gets a root cause: new-pipeline
bug (fix + test in `football_statfinder/`, rerun harness) or a proposed whitelist rule
(main loop decides). Iterate until zero UNEXPLAINED, then run the secondary weeks.
The final reports land in `docs/parity/` and REBUILD.md gets a Phase 2 parity note.

---

## Part 2 — SQLite storage + export (WP-C, WP-D)

### Design decisions (made; do not relitigate)

- **Document-style schema.** Payload-JSON columns with indexed key columns. The flat-file
  contract is the product; relational decomposition of record internals adds risk with no
  consumer. SQLite here replaces "855 git-tracked CSVs" as the system of record, it is not
  an analytics schema.
- **DB location**: `data/statfinder.sqlite3` (repo-relative, add to `.gitignore` — it must
  never be committed). Path helper in `football_statfinder/paths.py` (`db_path(out_root=...)`
  style keyword for tests, consistent with the existing helpers).
- **Single writer.** The orchestrator is the only writer; WAL mode on; `PRAGMA foreign_keys`
  off (no FKs — cross-table integrity is the orchestrator's job, same as the flat files).
- **Dual-write, files stay canonical in Phase 2.** Stages keep writing flat files exactly as
  today; the orchestrator additionally records results to the DB after each stage. The
  export step proves the DB can reproduce the files; the *flip* (DB primary, files
  export-only) is a later phase-gate, not part of WP-C/D.
- **Byte-parity is the export gate.** Records are stored as the exact dicts the pipeline
  wrote (JSON round-trip preserves key order), and the export step reuses the pipeline's
  own writer functions (same `json.dumps` args, same CSV builder, same row ordering
  function). Export output must be byte-identical to pipeline output.

### Schema v1 (`football_statfinder/storage/schema.sql` or inline DDL)

```sql
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);          -- schema_version=1
CREATE TABLE schedule_games(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, kickoff_iso_utc TEXT,
  home_team_key TEXT, away_team_key TEXT,
  payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, game_key));
CREATE INDEX idx_schedule_week ON schedule_games(league, season, week);
CREATE TABLE sagarin_ratings(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  team_norm TEXT NOT NULL, fetch_ts TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY(league, season, week, team_norm, fetch_ts));
CREATE TABLE team_metrics(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  team TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, team));
CREATE TABLE odds_pinned(
  league TEXT NOT NULL, fetch_ts TEXT NOT NULL, game_key TEXT NOT NULL,
  market TEXT NOT NULL, book TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(league, fetch_ts, game_key, market, book));  -- mirrors ledger dedupe key
CREATE TABLE games(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, game_key));
CREATE TABLE sidecars(
  league TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
  game_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(league, season, week, game_key));
```

`odds_raw` stays on disk as the append-only JSONL ledger (it is already the right shape
and mirroring megabyte raw fetches into SQLite buys nothing in Phase 2). Note this in the
module docstring.

### WP-C — storage layer

`football_statfinder/storage/` package:

- `db.py`: `connect(db_path=None)` (creates file + schema on first open, WAL,
  `schema_version` check that raises on mismatch), context-manager transaction helper.
- `store.py`: `record_schedule(conn, league, df)`, `record_sagarin(conn, league, season,
  week, rows)`, `record_metrics(conn, league, season, week, rows)`,
  `record_pinned_odds(conn, league, records)`, `record_games(conn, league, season, week,
  rows)`, `record_sidecars(conn, league, season, week, sidecar_payloads)` — all upserts
  (`INSERT ... ON CONFLICT ... DO UPDATE`), all taking the in-memory objects the
  orchestrator already has (no re-reading files). `updated_at` = timezone-aware UTC now.
- Orchestrator wiring in `refresh.py`: a `storage` recording call after each relevant
  stage, and after promotion/recompute/backfill re-record the affected `games` rows and
  sidecars. Controlled by `StorageSettings(enable: bool = True, db_path: Optional[Path])`
  added to `config.py` (env: `STORAGE_ENABLE`, `STORAGE_DB_PATH`). Recording failures are
  stage failures (fail loud), not warnings.
- Tests: schema creation, upsert-idempotence, orchestrator integration (extend the
  pattern of `tests/test_refresh_integration.py`; assert row counts and payload
  round-trip equality after a fake refresh), version-mismatch raise.

### WP-D — export step (depends on WP-C)

- `storage/export.py`: `export_week(conn, league, season, week, out_root=None)` writes
  `games_week_{S}_{W}.jsonl` + `.csv`, `league_metrics_{S}_{W}.csv`, and
  `game_schedules/*.json` from DB payloads by CALLING the same writer/ordering functions
  the pipeline uses (import them; if a writer is not importable as a pure function,
  refactor the pipeline module to expose it rather than duplicating logic).
- CLI: `statfinder export --league X --season S --week W [--out DIR]`.
- Test gate: run a fake refresh (integration-test pattern) into root A with storage on,
  `export_week` into root B, assert **byte equality** of every exported file pair.
  Also: export of a week absent from the DB fails loudly with a clear message.

---

## Part 3 — Retire the data auto-commit (WP-F, plan only in Phase 2)

Production (`.github/workflows/refresh.yml`, `src/`) still auto-commits `out/**` twice
daily and the frontend reads those files. No workflow change ships in Phase 2. WP-F is a
written proposal (a section appended to this file) covering: what publishes the exported
artifacts (Pages artifact deploy vs object storage), how the frontend's data root moves,
the cutover order relative to switching production to the new package, and rollback.
Requires Jordan's sign-off before any of it is implemented.

---

## Work-package handoff summary

| WP | What | Depends on | Executor |
|---|---|---|---|
| A | Replay harness + differ + first parity reports (NFL 2025w16, CFB 2025w13) | — | agent (high effort) |
| B | Triage UNEXPLAINED → fixes or approved whitelist rules; secondary weeks; REBUILD.md note | A | main loop + spot agents |
| C | storage/ package + orchestrator dual-write + tests | — | agent |
| D | export step + byte-parity gate + CLI | C | agent |
| E | README/REBUILD.md doc updates for storage + export | C, D | agent (cheap) |
| F | Auto-commit retirement proposal (text only) | B, D | main loop, then Jordan |

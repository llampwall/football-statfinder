# tools/parity — Phase 2 parity harness (WP-A)

Offline replay + classified differ for the Phase 2 parity gate. Rebuilds one
season-1 week with the **new** `football_statfinder` pipeline, entirely offline
from archived season-1 inputs, and diffs the result against the season-1
baseline artifacts. See `docs/PHASE2_SPEC.md` Part 1 for the contract.

## Hard invariants

- **100% offline.** No network. Every fetch seam is replaced by a fake that
  reads a season-1 archive file. If a code path tries the network, the run
  fails rather than reaching out.
- **Never writes the real `out/` tree.** All replay output goes to a scratch
  root (default: the session scratchpad `.../scratchpad/parity/`). The only
  repo writes are `docs/parity/*.md` + `*.json` (the reports) and this
  `tools/parity/` code.
- **Never mutates `football_statfinder/`.** The differ classifies deltas; it
  does not fix bugs. It cites the frozen whitelist and never extends it —
  proposals live in each report's "Proposed whitelist rules" section for
  main-loop (WP-B) approval.

## Usage

Run from the repo root (repo root on `sys.path`):

```powershell
# 1. rebuild a week offline into the scratch root
python -m tools.parity.replay --league nfl --season 2025 --week 16
python -m tools.parity.replay --league cfb --season 2025 --week 13

# 2. diff the rebuilt week against the season-1 baseline and write the reports
python -m tools.parity.diff --league nfl --season 2025 --week 16
python -m tools.parity.diff --league cfb --season 2025 --week 13
```

`diff` reads the manifest that `replay` wrote, so run `replay` first for the
same league/season/week. Reports land in `docs/parity/parity_{league}_{season}_wk{week}.{md,json}`.

Optional flags: `--scratch <dir>` (replay output root), `--baseline-out <dir>`
(season-1 `out/`, default the repo `out/`), and `replay --select-policy`
(`closing_pre_kickoff` default, or `latest_by_fetch_ts`).

## How the replay sources each stage (hermetic)

| stage | season-1 input | note |
|---|---|---|
| schedule | `out/master/{league}_schedule_master.csv` | copied into the scratch master (all seasons, so sidecar prev-season timelines survive); `fetch_schedule` returns empty so the copied master is the schedule of record |
| sagarin | `out/{week}/sagarin_{league}_{S}_wk{W}.jsonl` | injected as a `SagarinStagingResult`; the season-1 `sagarin_{league}_master.csv` is copied to the name the new pipeline reads (`{league}_sagarin_master.csv`) |
| stats | `out/{week}/league_metrics_{S}_{W}.csv` | injected as the *output* of the legacy stats stage — stats *computation* parity is out of scope (the source pages are gone) |
| odds | `out/staging/odds_raw/{league}/*.jsonl` in `[first_kickoff - 14d, last_kickoff]` | copied into the scratch root; the new pin + promote run against them |

The replay then calls the real `refresh_league` so stage sequencing (schedule →
sagarin → stats → odds staging → gameview → promotion → recompute → sidecars)
is production-faithful. Score/ATS backfill is disabled (single-week parity; the
season-1 baseline sidecars have `ats`/`to_margin` = null anyway).

## The differ

- **Join** on `(kickoff instant, home merge-key, away merge-key)` — never on
  `game_key` (NFL keys changed). Legacy NFL abbreviations (`"tb"`) and new full
  names both collapse through `leagues.{NFL,CFB}.merge_key`.
- **Classify** every field delta as exactly one of: a frozen whitelist rule
  `W1`–`W7`, a known-bug correction `BUGFIX-<n>` (proven arithmetically), or
  `UNEXPLAINED`. `raw_sources` is compared only for the subfields the frontend
  reads (grep of `web/`: `sagarin_row_{home,away}.{team,hfa}`, `odds_row`,
  `schedule_row.{game_no,rotation,gsis}`).
- **Report** the join coverage, per-field delta counts by class, K1–K5 findings,
  a K4 promoted-odds trace, BUGFIX evidence, the full UNEXPLAINED list grouped
  by reason, sidecar (K3) deltas, and proposed whitelist rules.

The gate is expected to **fail** at WP-A (non-zero UNEXPLAINED). Driving it to
zero — via fixes in `football_statfinder/` or approved whitelist rules — is
WP-B's job.

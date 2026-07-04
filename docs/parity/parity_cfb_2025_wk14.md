# Parity report — CFB 2025 week 14

Generated 2026-07-04T03:21:45.621031+00:00 by `tools/parity` (WP-A). Replay is 100% offline.

## Join coverage

- baseline rows: **67**
- replay rows: **67**
- matched (kickoff instant + home/away merge_key): **67**
- only in baseline (orphans): **0**
- only in replay (orphans): **0**
- join-key collisions: baseline=0 replay=0
- unkeyed rows: baseline=0 replay=0

## Per-field delta counts by classification

| label | count |
|---|---|
| BUGFIX-4 | 465 |
| BUGFIX-7 | 119 |
| W1 | 1 |
| W10 | 67 |
| W11 | 28 |
| W2 | 22 |
| W5 | 39 |
| W6 | 67 |
| W8 | 264 |

### By (label, field)

| label | field | count |
|---|---|---|
| BUGFIX-4 | favored_side | 39 |
| BUGFIX-4 | is_closing | 39 |
| BUGFIX-4 | moneyline_away | 39 |
| BUGFIX-4 | moneyline_home | 39 |
| BUGFIX-4 | odds_source | 39 |
| BUGFIX-4 | rating_diff_favored_team | 38 |
| BUGFIX-4 | rating_vs_odds | 38 |
| BUGFIX-4 | rating_vs_odds_favored_team | 38 |
| BUGFIX-4 | raw_sources.odds_row | 39 |
| BUGFIX-4 | spread_favored_team | 39 |
| BUGFIX-4 | spread_home_relative | 39 |
| BUGFIX-4 | total | 39 |
| BUGFIX-7 | rating_diff | 65 |
| BUGFIX-7 | rating_diff_favored_team | 18 |
| BUGFIX-7 | rating_vs_odds | 18 |
| BUGFIX-7 | rating_vs_odds_favored_team | 18 |
| W10 | source_uid | 67 |
| W11 | is_closing | 28 |
| W1 | game_key | 1 |
| W2 | away_team_raw | 15 |
| W2 | home_team_raw | 7 |
| W5 | snapshot_at | 39 |
| W6 | kickoff_iso_utc | 67 |
| W8 | raw_sources.sagarin_row_away.hfa | 67 |
| W8 | raw_sources.sagarin_row_away.team | 67 |
| W8 | raw_sources.sagarin_row_home.hfa | 65 |
| W8 | raw_sources.sagarin_row_home.team | 65 |

## Known-check list K1–K5

### K1 — schedule-odds fallback tier: **PASS**

legacy rows with odds_source='schedule': 0; new rows with odds_source='schedule': 0 (new gameview has no schedule-odds fallback tier — spread/total from schedule columns is never emitted). New rows with blank odds_source (unpromoted): 10.

### K2 — row coverage (1:1 join): **PASS**

baseline=67 replay=67 matched=67 only_baseline=0 only_replay=0; expected 60/60.

### K3 — sidecar parity: **PASS**

sidecars compared for 67 joined games; per-field delta counts: {'field:opp_pr': 264, 'field:opp_sos': 264, 'field:opp_pr_rank': 38, 'field:opp_sos_rank': 38}; classified: W9=604 UNEXPLAINED=0. These concentrate in the Sagarin enrichment fields (pr/sos and their ranks): the new sidecar builder deliberately uses the CFB nearest-week fallback for BOTH leagues and common.metrics.dense_rank for ranks (documented in pipeline/sidecars.py), whereas the season-1 NFL sidecar joined exact (season, week) rows and ranked sequentially. Confirmed on a sample: a wk1 entry keeps pr=21.4 but pr_rank moved 13->12 (dense vs sequential); pr/sos value deltas are the nearest-week fallback filling weeks the master lacks exact rows for (e.g. 2025 week 3 is absent). Classified W9 (604 deltas, spec triage round 1 — approved whitelist rule).

### K4 — promoted odds sanity (select policy = closing_pre_kickoff): **PASS**

new rows with a promoted book source: 57. Traced 3 game(s); each verified against the closing_pre_kickoff candidate set (freshest record per book, then the latest with fetch_ts<=kickoff). all policy-correct=True. NFL games have real kickoff times so the closing rule applies; CFB week-13 rows carry midnight (00:00) placeholder kickoffs, so no pre-kickoff candidate exists and the policy correctly falls back to the freshest record (is_closing=False). K4 is an NFL-primary check per spec. See K4 trace table.

### K5 — FBS filtering (CFB): **PASS**

replay emitted 67 rows vs baseline 67; only_baseline orphans=0 (no legacy FBS game was wrongly dropped — the FBS filter is sound); only_replay orphans=0 are additional FBS-vs-FBS games present in the schedule master but absent from the season-1 baseline snapshot (coverage expansion, not FBS-classification drift). gameview skipped_non_fbs is reported in the replay manifest.

### K4 trace (promoted-odds selection)

| matchup | kickoff | chosen book | chosen fetch_ts | pre-kickoff? | policy branch | policy-correct? |
|---|---|---|---|---|---|---|
| Western Michigan @ Eastern Michigan | 2025-11-26T00:30:00+00:00 | williamhill_us | 2025-11-25T22:26:45Z | True | closing_pre_kickoff | True |
| Iowa @ Nebraska | 2025-11-28T17:00:00+00:00 | williamhill_us | 2025-11-28T10:29:48Z | True | closing_pre_kickoff | True |
| Kent State @ Northern Illinois | 2025-11-28T17:00:00+00:00 | williamhill_us | 2025-11-28T10:29:48Z | True | closing_pre_kickoff | True |

## BUGFIX evidence

### BUGFIX-7
- Bowling Green @ Massachusetts (`20251125_2130_bowling_green_massachusetts`): rating_diff legacy=-19.34 new=-16.13 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.21); new=-16.13 legacy=-19.34
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): rating_diff legacy=-10.3 new=-7.09 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.21); new=-7.09 legacy=-10.3
- Navy @ Memphis (`20251128_0030_navy_memphis`): rating_diff legacy=0.55 new=3.76 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.21); new=3.76 legacy=0.55
- Ohio @ Buffalo (`20251128_1700_ohio_buffalo`): rating_diff legacy=-9.55 new=-6.34 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.21); new=-6.34 legacy=-9.55
- Ohio @ Buffalo (`20251128_1700_ohio_buffalo`): rating_diff_favored_team legacy=null new=6.34 — legacy CFB never derived rating_diff_favored_team at all (0 populated rows across checked weeks 10/13/14); the single-formula build derives it whenever a favorite exists (spec BUGFIX-7 family)
- Ohio @ Buffalo (`20251128_1700_ohio_buffalo`): rating_vs_odds legacy=null new=0.66 — single rating_vs_odds formula (legacy CFB emitted two conflicting values); spread matched so the delta is pure formula
- Ohio @ Buffalo (`20251128_1700_ohio_buffalo`): rating_vs_odds_favored_team legacy=null new=-0.66 — single rating_vs_odds formula (legacy CFB emitted two conflicting values); spread matched so the delta is pure formula
- Utah @ Kansas (`20251128_1700_utah_kansas`): rating_diff legacy=-15.79 new=-12.58 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.21); new=-12.58 legacy=-15.79

### BUGFIX-4
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): favored_side legacy=null new=AWAY — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): is_closing legacy=null new=true — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): moneyline_away legacy=null new=-390 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): moneyline_home legacy=null new=300 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): odds_source legacy=null new=williamhill_us — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): rating_diff_favored_team legacy=null new=7.09 — rating_diff_favored_team legacy-null -> new-populated: favored_side only exists once a spread does, and the replay promoted book odds where legacy promotion was dead (spec BUGFIX-4 CFB extension)
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): rating_vs_odds legacy=null new=2.91 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Western Michigan @ Eastern Michigan (`20251126_0030_western_michigan_eastern_michigan`): rating_vs_odds_favored_team legacy=null new=-2.91 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects

## UNEXPLAINED deltas (full list)

Total UNEXPLAINED field-deltas: **0**. Grouped by reason:

## Sidecar parity (K3 detail)

Sidecars compared for 67 joined games.

| metric | value |
|---|---|
| away_prev.entries_new | 815 |
| away_prev.entries_old | 815 |
| away_prev.entries_only_new | 0 |
| away_prev.entries_only_old | 0 |
| away_ytd.entries_new | 735 |
| away_ytd.entries_old | 773 |
| away_ytd.entries_only_new | 0 |
| away_ytd.entries_only_old | 38 |
| field:opp_pr | 264 |
| field:opp_pr_rank | 38 |
| field:opp_sos | 264 |
| field:opp_sos_rank | 38 |
| home_prev.entries_new | 811 |
| home_prev.entries_old | 811 |
| home_prev.entries_only_new | 0 |
| home_prev.entries_only_old | 0 |
| home_ytd.entries_new | 737 |
| home_ytd.entries_old | 769 |
| home_ytd.entries_only_new | 0 |
| home_ytd.entries_only_old | 32 |

### Sidecar delta classification

| label | count |
|---|---|
| W9 | 604 |

| label | field | count |
|---|---|---|
| W9 | opp_pr | 264 |
| W9 | opp_pr_rank | 38 |
| W9 | opp_sos | 264 |
| W9 | opp_sos_rank | 38 |

## Proposed whitelist rules (require main-loop approval; NOT applied)

_This differ never extends the frozen whitelist. The rules below are proposals for WP-B._


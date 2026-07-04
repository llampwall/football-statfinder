# Parity report — CFB 2025 week 13

Generated 2026-07-04T03:21:22.005350+00:00 by `tools/parity` (WP-A). Replay is 100% offline.

## Join coverage

- baseline rows: **60**
- replay rows: **60**
- matched (kickoff instant + home/away merge_key): **60**
- only in baseline (orphans): **0**
- only in replay (orphans): **0**
- join-key collisions: baseline=0 replay=0
- unkeyed rows: baseline=0 replay=0

## Per-field delta counts by classification

| label | count |
|---|---|
| BUGFIX-4 | 96 |
| BUGFIX-7 | 73 |
| W1 | 1 |
| W10 | 60 |
| W11 | 48 |
| W2 | 20 |
| W5 | 8 |
| W6 | 60 |
| W8 | 236 |

### By (label, field)

| label | field | count |
|---|---|---|
| BUGFIX-4 | favored_side | 8 |
| BUGFIX-4 | is_closing | 8 |
| BUGFIX-4 | moneyline_away | 8 |
| BUGFIX-4 | moneyline_home | 8 |
| BUGFIX-4 | odds_source | 8 |
| BUGFIX-4 | rating_diff_favored_team | 8 |
| BUGFIX-4 | rating_vs_odds | 8 |
| BUGFIX-4 | rating_vs_odds_favored_team | 8 |
| BUGFIX-4 | raw_sources.odds_row | 8 |
| BUGFIX-4 | spread_favored_team | 8 |
| BUGFIX-4 | spread_home_relative | 8 |
| BUGFIX-4 | total | 8 |
| BUGFIX-7 | rating_diff | 58 |
| BUGFIX-7 | rating_diff_favored_team | 5 |
| BUGFIX-7 | rating_vs_odds | 5 |
| BUGFIX-7 | rating_vs_odds_favored_team | 5 |
| W10 | source_uid | 60 |
| W11 | is_closing | 48 |
| W1 | game_key | 1 |
| W2 | away_team_raw | 6 |
| W2 | home_team_raw | 14 |
| W5 | snapshot_at | 8 |
| W6 | kickoff_iso_utc | 60 |
| W8 | raw_sources.sagarin_row_away.hfa | 58 |
| W8 | raw_sources.sagarin_row_away.team | 58 |
| W8 | raw_sources.sagarin_row_home.hfa | 60 |
| W8 | raw_sources.sagarin_row_home.team | 60 |

## Known-check list K1–K5

### K1 — schedule-odds fallback tier: **PASS**

legacy rows with odds_source='schedule': 0; new rows with odds_source='schedule': 0 (new gameview has no schedule-odds fallback tier — spread/total from schedule columns is never emitted). New rows with blank odds_source (unpromoted): 47.

### K2 — row coverage (1:1 join): **PASS**

baseline=60 replay=60 matched=60 only_baseline=0 only_replay=0; expected 60/60.

### K3 — sidecar parity: **PASS**

sidecars compared for 60 joined games; per-field delta counts: {'field:opp_pr': 252, 'field:opp_sos': 252, 'field:opp_pr_rank': 38, 'field:opp_sos_rank': 38}; classified: W9=580 UNEXPLAINED=0. These concentrate in the Sagarin enrichment fields (pr/sos and their ranks): the new sidecar builder deliberately uses the CFB nearest-week fallback for BOTH leagues and common.metrics.dense_rank for ranks (documented in pipeline/sidecars.py), whereas the season-1 NFL sidecar joined exact (season, week) rows and ranked sequentially. Confirmed on a sample: a wk1 entry keeps pr=21.4 but pr_rank moved 13->12 (dense vs sequential); pr/sos value deltas are the nearest-week fallback filling weeks the master lacks exact rows for (e.g. 2025 week 3 is absent). Classified W9 (580 deltas, spec triage round 1 — approved whitelist rule).

### K4 — promoted odds sanity (select policy = closing_pre_kickoff): **PASS**

new rows with a promoted book source: 13. Traced 3 game(s); each verified against the closing_pre_kickoff candidate set (freshest record per book, then the latest with fetch_ts<=kickoff). all policy-correct=True. NFL games have real kickoff times so the closing rule applies; CFB week-13 rows carry midnight (00:00) placeholder kickoffs, so no pre-kickoff candidate exists and the policy correctly falls back to the freshest record (is_closing=False). K4 is an NFL-primary check per spec. See K4 trace table.

### K5 — FBS filtering (CFB): **PASS**

replay emitted 60 rows vs baseline 60; only_baseline orphans=0 (no legacy FBS game was wrongly dropped — the FBS filter is sound); only_replay orphans=0 are additional FBS-vs-FBS games present in the schedule master but absent from the season-1 baseline snapshot (coverage expansion, not FBS-classification drift). gameview skipped_non_fbs is reported in the replay manifest.

### K4 trace (promoted-odds selection)

| matchup | kickoff | chosen book | chosen fetch_ts | pre-kickoff? | policy branch | policy-correct? |
|---|---|---|---|---|---|---|
| Colorado State @ Boise State | 2025-11-23T00:00:00+00:00 | williamhill_us | 2025-11-23T01:56:48Z | False | fallback_to_freshest (no pre-kickoff candidate) | True |
| Nebraska @ Penn State | 2025-11-23T00:00:00+00:00 | williamhill_us | 2025-11-23T01:56:48Z | False | fallback_to_freshest (no pre-kickoff candidate) | True |
| New Mexico @ Air Force | 2025-11-23T00:00:00+00:00 | williamhill_us | 2025-11-23T01:56:48Z | False | fallback_to_freshest (no pre-kickoff candidate) | True |

## BUGFIX evidence

### BUGFIX-7
- Akron @ Bowling Green (`20251119_0000_akron_bowling_green`): rating_diff legacy=0.62 new=4.05 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=4.05 legacy=0.62
- Western Michigan @ Northern Illinois (`20251119_0000_western_michigan_northern_illinois`): rating_diff legacy=-11.47 new=-8.04 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=-8.04 legacy=-11.47
- Massachusetts @ Ohio (`20251119_0000_massachusetts_ohio`): rating_diff legacy=28.37 new=31.8 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=31.8 legacy=28.37
- Miami (oh) @ Buffalo (`20251120_0000_miami_oh_buffalo`): rating_diff legacy=-10.17 new=-6.74 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=-6.74 legacy=-10.17
- Central Michigan @ Kent State (`20251120_0000_central_michigan_kent_state`): rating_diff legacy=-13.4 new=-9.97 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=-9.97 legacy=-13.4
- Louisiana @ Arkansas State (`20251121_0030_louisiana_arkansas_state`): rating_diff legacy=-2.23 new=1.2 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=1.2 legacy=-2.23
- Florida State @ Nc State (`20251122_0100_florida_state_nc_state`): rating_diff legacy=-2.87 new=0.56 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=0.56 legacy=-2.87
- Hawai'i @ Unlv (`20251122_0330_hawai_i_unlv`): rating_diff legacy=4.94 new=8.37 — rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, new = legacy + hfa (3.43); new=8.37 legacy=4.94

### BUGFIX-4
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): favored_side legacy=null new=AWAY — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): is_closing legacy=null new=false — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): moneyline_away legacy=null new=-10000 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): moneyline_home legacy=null new=1625 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): odds_source legacy=null new=williamhill_us — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): rating_diff_favored_team legacy=null new=2.51 — rating_diff_favored_team legacy-null -> new-populated: favored_side only exists once a spread does, and the replay promoted book odds where legacy promotion was dead (spec BUGFIX-4 CFB extension)
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): rating_vs_odds legacy=null new=16.99 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- New Mexico @ Air Force (`20251123_0000_new_mexico_air_force`): rating_vs_odds_favored_team legacy=null new=-16.99 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects

## UNEXPLAINED deltas (full list)

Total UNEXPLAINED field-deltas: **0**. Grouped by reason:

## Sidecar parity (K3 detail)

Sidecars compared for 60 joined games.

| metric | value |
|---|---|
| away_prev.entries_new | 727 |
| away_prev.entries_old | 727 |
| away_prev.entries_only_new | 0 |
| away_prev.entries_only_old | 0 |
| away_ytd.entries_new | 602 |
| away_ytd.entries_old | 618 |
| away_ytd.entries_only_new | 0 |
| away_ytd.entries_only_old | 16 |
| field:opp_pr | 252 |
| field:opp_pr_rank | 38 |
| field:opp_sos | 252 |
| field:opp_sos_rank | 38 |
| home_prev.entries_new | 730 |
| home_prev.entries_old | 730 |
| home_prev.entries_only_new | 0 |
| home_prev.entries_only_old | 0 |
| home_ytd.entries_new | 599 |
| home_ytd.entries_old | 623 |
| home_ytd.entries_only_new | 0 |
| home_ytd.entries_only_old | 24 |

### Sidecar delta classification

| label | count |
|---|---|
| W9 | 580 |

| label | field | count |
|---|---|---|
| W9 | opp_pr | 252 |
| W9 | opp_pr_rank | 38 |
| W9 | opp_sos | 252 |
| W9 | opp_sos_rank | 38 |

## Proposed whitelist rules (require main-loop approval; NOT applied)

_This differ never extends the frozen whitelist. The rules below are proposals for WP-B._


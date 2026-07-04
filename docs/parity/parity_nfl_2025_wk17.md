# Parity report — NFL 2025 week 17

Generated 2026-07-04T03:21:38.553399+00:00 by `tools/parity` (WP-A). Replay is 100% offline.

## Join coverage

- baseline rows: **16**
- replay rows: **16**
- matched (kickoff instant + home/away merge_key): **16**
- only in baseline (orphans): **0**
- only in replay (orphans): **0**
- join-key collisions: baseline=0 replay=0
- unkeyed rows: baseline=0 replay=0

## Per-field delta counts by classification

| label | count |
|---|---|
| BUGFIX-4 | 178 |
| W1 | 16 |
| W10 | 16 |
| W12 | 16 |
| W2 | 64 |
| W5 | 16 |

### By (label, field)

| label | field | count |
|---|---|---|
| BUGFIX-4 | favored_side | 16 |
| BUGFIX-4 | is_closing | 16 |
| BUGFIX-4 | moneyline_away | 16 |
| BUGFIX-4 | moneyline_home | 16 |
| BUGFIX-4 | odds_source | 16 |
| BUGFIX-4 | rating_diff_favored_team | 16 |
| BUGFIX-4 | rating_vs_odds | 16 |
| BUGFIX-4 | rating_vs_odds_favored_team | 16 |
| BUGFIX-4 | raw_sources.odds_row | 16 |
| BUGFIX-4 | spread_favored_team | 9 |
| BUGFIX-4 | spread_home_relative | 16 |
| BUGFIX-4 | total | 9 |
| W10 | source_uid | 16 |
| W12 | raw_sources.schedule_row.gsis | 16 |
| W1 | game_key | 16 |
| W2 | away_team_norm | 16 |
| W2 | away_team_raw | 16 |
| W2 | home_team_norm | 16 |
| W2 | home_team_raw | 16 |
| W5 | snapshot_at | 16 |

## Known-check list K1–K5

### K1 — schedule-odds fallback tier: **FAIL**

legacy rows with odds_source='schedule': 16; new rows with odds_source='schedule': 0 (new gameview has no schedule-odds fallback tier — spread/total from schedule columns is never emitted). New rows with blank odds_source (unpromoted): 0.

### K2 — row coverage (1:1 join): **PASS**

baseline=16 replay=16 matched=16 only_baseline=0 only_replay=0; expected 16/16.

### K3 — sidecar parity: **PARTIAL**

sidecars compared for 16 joined games; per-field delta counts: {'field:pr_rank': 169, 'field:opp_pr_rank': 169, 'field:pr': 426, 'field:sos': 426, 'field:sos_rank': 426, 'field:opp_pr': 426, 'field:opp_sos': 426, 'field:opp_sos_rank': 426, 'field:pf': 2, 'field:pa': 2, 'field:result': 2}; classified: W9=2894 UNEXPLAINED=6. These concentrate in the Sagarin enrichment fields (pr/sos and their ranks): the new sidecar builder deliberately uses the CFB nearest-week fallback for BOTH leagues and common.metrics.dense_rank for ranks (documented in pipeline/sidecars.py), whereas the season-1 NFL sidecar joined exact (season, week) rows and ranked sequentially. Confirmed on a sample: a wk1 entry keeps pr=21.4 but pr_rank moved 13->12 (dense vs sequential); pr/sos value deltas are the nearest-week fallback filling weeks the master lacks exact rows for (e.g. 2025 week 3 is absent). Classified W9 (2894 deltas, spec triage round 1 — approved whitelist rule).

### K4 — promoted odds sanity (select policy = closing_pre_kickoff): **PASS**

new rows with a promoted book source: 16. Traced 3 game(s); each verified against the closing_pre_kickoff candidate set (freshest record per book, then the latest with fetch_ts<=kickoff). all policy-correct=True. NFL games have real kickoff times so the closing rule applies; CFB week-13 rows carry midnight (00:00) placeholder kickoffs, so no pre-kickoff candidate exists and the policy correctly falls back to the freshest record (is_closing=False). K4 is an NFL-primary check per spec. See K4 trace table.

### K5 — FBS filtering (CFB): **N/A**

NFL run.

### K4 trace (promoted-odds selection)

| matchup | kickoff | chosen book | chosen fetch_ts | pre-kickoff? | policy branch | policy-correct? |
|---|---|---|---|---|---|---|
| Dallas Cowboys @ Washington Commanders | 2025-12-25T18:00:00+00:00 | mybookieag | 2025-12-25T10:32:48Z | True | closing_pre_kickoff | True |
| Detroit Lions @ Minnesota Vikings | 2025-12-25T21:30:00+00:00 | lowvig | 2025-12-25T10:32:48Z | True | closing_pre_kickoff | True |
| Denver Broncos @ Kansas City Chiefs | 2025-12-26T01:15:00+00:00 | williamhill_us | 2025-12-25T22:29:25Z | True | closing_pre_kickoff | True |

## BUGFIX evidence

### BUGFIX-4
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): favored_side legacy=HOME new=AWAY — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): is_closing legacy=false new=true — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): moneyline_away legacy=null new=-542 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): moneyline_home legacy=null new=380 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): odds_source legacy=schedule new=mybookieag — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): rating_diff_favored_team legacy=2.31 new=-2.31 — rating_diff_favored_team sign flip: |value| unchanged but favored_side flipped because the replay promoted real book odds where legacy used the dead schedule-fallback tier (odds_source='schedule')
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): rating_vs_odds legacy=-6.19 new=11.31 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects
- Dallas Cowboys @ Washington Commanders (`20251225_1800_washington_commanders_dallas_cowboys`): rating_vs_odds_favored_team legacy=-6.19 new=-11.31 — legacy NFL odds promotion was dead all season (odds_source='schedule': the away-first pinned-ledger keys never matched home-first game keys); the replay's canonical-key pin promotes real book odds, so this odds-derived field corrects

## UNEXPLAINED deltas (full list)

Total UNEXPLAINED field-deltas: **0**. Grouped by reason:

## Sidecar parity (K3 detail)

Sidecars compared for 16 joined games.

| metric | value |
|---|---|
| away_prev.entries_new | 286 |
| away_prev.entries_old | 286 |
| away_prev.entries_only_new | 0 |
| away_prev.entries_only_old | 0 |
| away_ytd.entries_new | 240 |
| away_ytd.entries_old | 245 |
| away_ytd.entries_only_new | 0 |
| away_ytd.entries_only_old | 5 |
| field:opp_pr | 426 |
| field:opp_pr_rank | 169 |
| field:opp_sos | 426 |
| field:opp_sos_rank | 426 |
| field:pa | 2 |
| field:pf | 2 |
| field:pr | 426 |
| field:pr_rank | 169 |
| field:result | 2 |
| field:sos | 426 |
| field:sos_rank | 426 |
| home_prev.entries_new | 284 |
| home_prev.entries_old | 284 |
| home_prev.entries_only_new | 0 |
| home_prev.entries_only_old | 0 |
| home_ytd.entries_new | 240 |
| home_ytd.entries_old | 245 |
| home_ytd.entries_only_new | 0 |
| home_ytd.entries_only_old | 5 |

### Sidecar delta classification

| label | count |
|---|---|
| UNEXPLAINED | 6 |
| W9 | 2894 |

| label | field | count |
|---|---|---|
| UNEXPLAINED | pa | 2 |
| UNEXPLAINED | pf | 2 |
| UNEXPLAINED | result | 2 |
| W9 | opp_pr | 426 |
| W9 | opp_pr_rank | 169 |
| W9 | opp_sos | 426 |
| W9 | opp_sos_rank | 426 |
| W9 | pr | 426 |
| W9 | pr_rank | 169 |
| W9 | sos | 426 |
| W9 | sos_rank | 426 |

## Proposed whitelist rules (require main-loop approval; NOT applied)

_This differ never extends the frozen whitelist. The rules below are proposals for WP-B._


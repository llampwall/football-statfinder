# Parity report — NFL 2025 week 16

Generated 2026-07-04T02:44:45.835793+00:00 by `tools/parity` (WP-A). Replay is 100% offline.

## Join coverage

- baseline rows: **16**
- replay rows: **20**
- matched (kickoff instant + home/away merge_key): **16**
- only in baseline (orphans): **0**
- only in replay (orphans): **4**
- join-key collisions: baseline=0 replay=0
- unkeyed rows: baseline=0 replay=0

### Orphans only in replay
- `20251219_1800_chicago_bears_green_bay_packers` — Green Bay Packers @ Chicago Bears — key=['2025-12-19T18:00:00+00:00', 'chicagobears', 'greenbaypackers']
- `20251219_1800_washington_commanders_philadelphia_eagles` — Philadelphia Eagles @ Washington Commanders — key=['2025-12-19T18:00:00+00:00', 'washingtoncommanders', 'philadelphiaeagles']
- `20251221_1800_baltimore_ravens_new_england_patriots` — New England Patriots @ Baltimore Ravens — key=['2025-12-21T18:00:00+00:00', 'baltimoreravens', 'newenglandpatriots']
- `20251222_0120_miami_dolphins_cincinnati_bengals` — Cincinnati Bengals @ Miami Dolphins — key=['2025-12-22T01:20:00+00:00', 'miamidolphins', 'cincinnatibengals']

## Per-field delta counts by classification

| label | count |
|---|---|
| UNEXPLAINED | 203 |
| W1 | 16 |
| W2 | 64 |
| W5 | 16 |

### By (label, field)

| label | field | count |
|---|---|---|
| UNEXPLAINED | favored_side | 16 |
| UNEXPLAINED | is_closing | 16 |
| UNEXPLAINED | moneyline_away | 16 |
| UNEXPLAINED | moneyline_home | 16 |
| UNEXPLAINED | odds_source | 16 |
| UNEXPLAINED | rating_diff_favored_team | 16 |
| UNEXPLAINED | rating_vs_odds | 16 |
| UNEXPLAINED | rating_vs_odds_favored_team | 16 |
| UNEXPLAINED | raw_sources.odds_row | 16 |
| UNEXPLAINED | raw_sources.schedule_row.gsis | 16 |
| UNEXPLAINED | source_uid | 16 |
| UNEXPLAINED | spread_favored_team | 4 |
| UNEXPLAINED | spread_home_relative | 16 |
| UNEXPLAINED | total | 7 |
| W1 | game_key | 16 |
| W2 | away_team_norm | 16 |
| W2 | away_team_raw | 16 |
| W2 | home_team_norm | 16 |
| W2 | home_team_raw | 16 |
| W5 | snapshot_at | 16 |

## Known-check list K1–K5

### K1 — schedule-odds fallback tier: **FAIL**

legacy rows with odds_source='schedule': 16; new rows with odds_source='schedule': 0 (new gameview has no schedule-odds fallback tier — spread/total from schedule columns is never emitted). New rows with blank odds_source (unpromoted): 4.

### K2 — row coverage (1:1 join): **FAIL**

baseline=16 replay=20 matched=16 only_baseline=0 only_replay=4; expected 16/16. Replay orphans are schedule-master rows absent from the season-1 baseline snapshot: for NFL these are stale duplicate-kickoff master rows (the same matchup appears at two kickoff times because the master upsert KEY includes kickoff_iso_utc, so a corrected kickoff adds a row instead of replacing the stale one); for CFB these are additional FBS-vs-FBS games the season-1 games_week never captured. The new pipeline faithfully emits one row per master row; no legacy game is missing (only_baseline=0).

### K3 — sidecar parity: **PARTIAL**

sidecars compared for 16 joined games; per-field delta counts: {'field:pr_rank': 169, 'field:opp_pr_rank': 169, 'field:pr': 426, 'field:sos': 426, 'field:sos_rank': 426, 'field:opp_pr': 426, 'field:opp_sos': 426, 'field:opp_sos_rank': 426}. These concentrate in the Sagarin enrichment fields (pr/sos and their ranks): the new sidecar builder deliberately uses the CFB nearest-week fallback for BOTH leagues and common.metrics.dense_rank for ranks (documented in pipeline/sidecars.py), whereas the season-1 NFL sidecar joined exact (season, week) rows and ranked sequentially. Confirmed on a sample: a wk1 entry keeps pr=21.4 but pr_rank moved 13->12 (dense vs sequential); pr/sos value deltas are the nearest-week fallback filling weeks the master lacks exact rows for (e.g. 2025 week 3 is absent). WP-B should confirm this is the intended change.

### K4 — promoted odds sanity (select policy = closing_pre_kickoff): **PASS**

new rows with a promoted book source: 16. Traced 3 game(s); each verified against the closing_pre_kickoff candidate set (freshest record per book, then the latest with fetch_ts<=kickoff). all policy-correct=True. NFL games have real kickoff times so the closing rule applies; CFB week-13 rows carry midnight (00:00) placeholder kickoffs, so no pre-kickoff candidate exists and the policy correctly falls back to the freshest record (is_closing=False). K4 is an NFL-primary check per spec. See K4 trace table.

### K5 — FBS filtering (CFB): **N/A**

NFL run.

### K4 trace (promoted-odds selection)

| matchup | kickoff | chosen book | chosen fetch_ts | pre-kickoff? | policy branch | policy-correct? |
|---|---|---|---|---|---|---|
| Los Angeles Rams @ Seattle Seahawks | 2025-12-19T01:15:00+00:00 | draftkings | 2025-12-18T22:30:11Z | True | closing_pre_kickoff | True |
| Philadelphia Eagles @ Washington Commanders | 2025-12-20T22:00:00+00:00 | betus | 2025-12-20T10:30:11Z | True | closing_pre_kickoff | True |
| Green Bay Packers @ Chicago Bears | 2025-12-21T01:20:00+00:00 | draftkings | 2025-12-20T22:28:11Z | True | closing_pre_kickoff | True |

## BUGFIX evidence

No BUGFIX-class deltas were observed in this run (see report body for why — e.g. legacy NFL week 16 promoted no odds, so no both-promoted sign-flip could occur).

## UNEXPLAINED deltas (full list)

Total UNEXPLAINED field-deltas: **203**. Grouped by reason:

### 10× — odds_sourcing_divergence: legacy used schedule-fallback/none (odds_source='schedule'), new promotes book odds (odds_source='betus'); new pipeline has no schedule-odds fallback tier (K1)
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `favored_side`: legacy=HOME new=AWAY
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `is_closing`: legacy=false new=true
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `moneyline_away`: legacy=null new=-335
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `moneyline_home`: legacy=null new=270
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `odds_source`: legacy=schedule new=betus
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `rating_vs_odds`: legacy=-12.26 new=1.74
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `rating_vs_odds_favored_team`: legacy=-12.26 new=-1.74
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `spread_home_relative`: legacy=-7.0 new=7.0
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `total`: legacy=43.5 new=44.5
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `raw_sources.odds_row`: legacy=null new=<present>

### 104× — odds_sourcing_divergence: legacy used schedule-fallback/none (odds_source='schedule'), new promotes book odds (odds_source='draftkings'); new pipeline has no schedule-odds fallback tier (K1)
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `favored_side`: legacy=AWAY new=HOME
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `is_closing`: legacy=false new=true
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `moneyline_away`: legacy=null new=100
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `moneyline_home`: legacy=null new=-120
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `odds_source`: legacy=schedule new=draftkings
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `rating_vs_odds`: legacy=2.27 new=-0.73
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `rating_vs_odds_favored_team`: legacy=-2.27 new=-0.73
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `spread_home_relative`: legacy=1.5 new=-1.5
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `raw_sources.odds_row`: legacy=null new=<present>
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `favored_side`: legacy=AWAY new=HOME
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `is_closing`: legacy=false new=true
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `moneyline_away`: legacy=null new=-102
- … 92 more (see .json)

### 30× — odds_sourcing_divergence: legacy used schedule-fallback/none (odds_source='schedule'), new promotes book odds (odds_source='lowvig'); new pipeline has no schedule-odds fallback tier (K1)
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `favored_side`: legacy=HOME new=AWAY
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `is_closing`: legacy=false new=true
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `moneyline_away`: legacy=null new=-157
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `moneyline_home`: legacy=null new=127
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `odds_source`: legacy=schedule new=lowvig
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `rating_vs_odds`: legacy=-1.24 new=3.76
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `rating_vs_odds_favored_team`: legacy=-1.24 new=-3.76
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `spread_home_relative`: legacy=-2.5 new=2.5
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `raw_sources.odds_row`: legacy=null new=<present>
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `favored_side`: legacy=AWAY new=HOME
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `is_closing`: legacy=false new=true
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `moneyline_away`: legacy=null new=161
- … 18 more (see .json)

### 11× — odds_sourcing_divergence: legacy used schedule-fallback/none (odds_source='schedule'), new promotes book odds (odds_source='mybookieag'); new pipeline has no schedule-odds fallback tier (K1)
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `favored_side`: legacy=AWAY new=HOME
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `is_closing`: legacy=false new=true
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `moneyline_away`: legacy=null new=263
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `moneyline_home`: legacy=null new=-325
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `odds_source`: legacy=schedule new=mybookieag
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `rating_vs_odds`: legacy=14.02 new=-0.48
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `rating_vs_odds_favored_team`: legacy=-14.02 new=-0.48
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `spread_favored_team`: legacy=-7.5 new=-7.0
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `spread_home_relative`: legacy=7.5 new=-7.0
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `total`: legacy=52.5 new=52.0
- Pittsburgh Steelers @ Detroit Lions (`20251221_2125_detroit_lions_pittsburgh_steelers`) `raw_sources.odds_row`: legacy=null new=<present>

### 16× — rating_diff_favored_team sign flip: |value| unchanged but favored_side flipped because the promoted spread put the other team as favorite (downstream of the odds-sourcing divergence, K1/bug-4)
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `rating_diff_favored_team`: legacy=-0.77 new=0.77
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `rating_diff_favored_team`: legacy=-5.26 new=5.26
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `rating_diff_favored_team`: legacy=0.59 new=-0.59
- Tampa Bay Buccaneers @ Carolina Panthers (`20251221_1800_carolina_panthers_tampa_bay_buccaneers`) `rating_diff_favored_team`: legacy=-0.82 new=0.82
- Buffalo Bills @ Cleveland Browns (`20251221_1800_cleveland_browns_buffalo_bills`) `rating_diff_favored_team`: legacy=-10.11 new=10.11
- Los Angeles Chargers @ Dallas Cowboys (`20251221_1800_dallas_cowboys_los_angeles_chargers`) `rating_diff_favored_team`: legacy=1.76 new=-1.76
- Cincinnati Bengals @ Miami Dolphins (`20251221_1800_miami_dolphins_cincinnati_bengals`) `rating_diff_favored_team`: legacy=1.94 new=-1.94
- New York Jets @ New Orleans Saints (`20251221_1800_new_orleans_saints_new_york_jets`) `rating_diff_favored_team`: legacy=-2.75 new=2.75
- Minnesota Vikings @ New York Giants (`20251221_1800_new_york_giants_minnesota_vikings`) `rating_diff_favored_team`: legacy=-3.01 new=3.01
- Kansas City Chiefs @ Tennessee Titans (`20251221_1800_tennessee_titans_kansas_city_chiefs`) `rating_diff_favored_team`: legacy=-10.4 new=10.4
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `rating_diff_favored_team`: legacy=1.26 new=-1.26
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `rating_diff_favored_team`: legacy=-2.87 new=2.87
- … 4 more (see .json)

### 16× — schedule provenance dropped: frontend reads schedule_row.game_no/rotation/gsis; SCHEDULE_COLUMNS omits them so new pipeline emits None
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `raw_sources.schedule_row.gsis`: legacy=60067.0 new=null
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `raw_sources.schedule_row.gsis`: legacy=60069.0 new=null
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `raw_sources.schedule_row.gsis`: legacy=60068.0 new=null
- Tampa Bay Buccaneers @ Carolina Panthers (`20251221_1800_carolina_panthers_tampa_bay_buccaneers`) `raw_sources.schedule_row.gsis`: legacy=60071.0 new=null
- Buffalo Bills @ Cleveland Browns (`20251221_1800_cleveland_browns_buffalo_bills`) `raw_sources.schedule_row.gsis`: legacy=60072.0 new=null
- Los Angeles Chargers @ Dallas Cowboys (`20251221_1800_dallas_cowboys_los_angeles_chargers`) `raw_sources.schedule_row.gsis`: legacy=60073.0 new=null
- Cincinnati Bengals @ Miami Dolphins (`20251221_1800_miami_dolphins_cincinnati_bengals`) `raw_sources.schedule_row.gsis`: legacy=60081.0 new=null
- New York Jets @ New Orleans Saints (`20251221_1800_new_orleans_saints_new_york_jets`) `raw_sources.schedule_row.gsis`: legacy=60074.0 new=null
- Minnesota Vikings @ New York Giants (`20251221_1800_new_york_giants_minnesota_vikings`) `raw_sources.schedule_row.gsis`: legacy=60075.0 new=null
- Kansas City Chiefs @ Tennessee Titans (`20251221_1800_tennessee_titans_kansas_city_chiefs`) `raw_sources.schedule_row.gsis`: legacy=60076.0 new=null
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `raw_sources.schedule_row.gsis`: legacy=60077.0 new=null
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `raw_sources.schedule_row.gsis`: legacy=60078.0 new=null
- … 4 more (see .json)

### 16× — source_uid dropped: SCHEDULE_COLUMNS has no source_uid, new pipeline emits None
- Los Angeles Rams @ Seattle Seahawks (`20251219_0115_seattle_seahawks_los_angeles_rams`) `source_uid`: legacy=2025_16_LA_SEA new=null
- Philadelphia Eagles @ Washington Commanders (`20251220_2200_washington_commanders_philadelphia_eagles`) `source_uid`: legacy=2025_16_PHI_WAS new=null
- Green Bay Packers @ Chicago Bears (`20251221_0120_chicago_bears_green_bay_packers`) `source_uid`: legacy=2025_16_GB_CHI new=null
- Tampa Bay Buccaneers @ Carolina Panthers (`20251221_1800_carolina_panthers_tampa_bay_buccaneers`) `source_uid`: legacy=2025_16_TB_CAR new=null
- Buffalo Bills @ Cleveland Browns (`20251221_1800_cleveland_browns_buffalo_bills`) `source_uid`: legacy=2025_16_BUF_CLE new=null
- Los Angeles Chargers @ Dallas Cowboys (`20251221_1800_dallas_cowboys_los_angeles_chargers`) `source_uid`: legacy=2025_16_LAC_DAL new=null
- Cincinnati Bengals @ Miami Dolphins (`20251221_1800_miami_dolphins_cincinnati_bengals`) `source_uid`: legacy=2025_16_CIN_MIA new=null
- New York Jets @ New Orleans Saints (`20251221_1800_new_orleans_saints_new_york_jets`) `source_uid`: legacy=2025_16_NYJ_NO new=null
- Minnesota Vikings @ New York Giants (`20251221_1800_new_york_giants_minnesota_vikings`) `source_uid`: legacy=2025_16_MIN_NYG new=null
- Kansas City Chiefs @ Tennessee Titans (`20251221_1800_tennessee_titans_kansas_city_chiefs`) `source_uid`: legacy=2025_16_KC_TEN new=null
- Atlanta Falcons @ Arizona Cardinals (`20251221_2105_arizona_cardinals_atlanta_falcons`) `source_uid`: legacy=2025_16_ATL_ARI new=null
- Jacksonville Jaguars @ Denver Broncos (`20251221_2105_denver_broncos_jacksonville_jaguars`) `source_uid`: legacy=2025_16_JAX_DEN new=null
- … 4 more (see .json)

## Sidecar parity (K3 detail)

Sidecars compared for 16 joined games.

| metric | value |
|---|---|
| away_prev.entries_new | 289 |
| away_prev.entries_old | 289 |
| away_prev.entries_only_new | 0 |
| away_prev.entries_only_old | 0 |
| away_ytd.entries_new | 224 |
| away_ytd.entries_old | 227 |
| away_ytd.entries_only_new | 0 |
| away_ytd.entries_only_old | 3 |
| field:opp_pr | 426 |
| field:opp_pr_rank | 169 |
| field:opp_sos | 426 |
| field:opp_sos_rank | 426 |
| field:pr | 426 |
| field:pr_rank | 169 |
| field:sos | 426 |
| field:sos_rank | 426 |
| home_prev.entries_new | 281 |
| home_prev.entries_old | 281 |
| home_prev.entries_only_new | 0 |
| home_prev.entries_only_old | 0 |
| home_ytd.entries_new | 224 |
| home_ytd.entries_old | 227 |
| home_ytd.entries_only_new | 0 |
| home_ytd.entries_only_old | 3 |

## Proposed whitelist rules (require main-loop approval; NOT applied)

_This differ never extends the frozen whitelist. The rules below are proposals for WP-B._

- **PROP-source_uid** (pipeline-gap or whitelist): new pipeline emits source_uid=None for every game (SCHEDULE_COLUMNS carries no source_uid column, so gameview reads None). Either add source_uid to the schedule schema (populate from nflverse game_id / CFBD id) or whitelist None-vs-legacy-id when the joined game identity matches.
- **PROP-schedule_provenance** (pipeline-gap or whitelist): the frontend reads raw_sources.schedule_row.game_no/rotation/gsis, but SCHEDULE_COLUMNS omits them so the new pipeline emits None. Either carry these columns through the schedule schema or whitelist their loss (frontend already has a fallback).
- **PROP-BUGFIX-4-odds-promotion** (BUGFIX candidate (needs WP-B sign-off)): NFL: legacy week rows fell back to odds_source='schedule' because the legacy pin wrote AWAY-first game_keys ({ts}_{away}_{home}) that never matched the home-first games_week keys (bug 4). The new canonical-key pin promotes real book odds. Evidence: legacy out/staging/odds_pinned/nfl/2025.jsonl keys are away-first; the replayed pinned ledger keys are home-first. Propose classifying the resulting spread/total/moneyline/odds_source/rating_vs_odds deltas as BUGFIX-4 rather than UNEXPLAINED. Also decide K1: whether the new pipeline should keep a schedule-odds fallback for genuinely-unpromoted games.

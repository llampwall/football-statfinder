# Football Stats System Spec

This spec defines the schema, API surface, and rules for the Football Stats System.

## Deliverables
- **Week View**: Two rows per game (away, home). Columns: Date, Time, Team #, Game #, Team, Odds, O/U, Current Power Rating, Rating Diff, Rating vs Odds, Current Sched Strength, Power Rating (Last 5), Trend, Last Year PR, (optional SU W-L).
- **Game View**: One game. Sections:
  - Header: Teams, kickoff, odds, PR, Rating Diff, Rating vs Odds, HFA.
  - Team tables: PF, PA, SU, ATS, offense/defense per game + ranks, TO.
  - Schedules: per-week opponent, score, result, PR/Rank, Opp PR/Rank, SoS, Opp SoS.
  - Prior-season summary: PF, PA, SU, ATS, O/D totals + ranks, TO.

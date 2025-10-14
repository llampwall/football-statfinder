# Football Statfinder

Automated tooling for producing weekly NFL data packs: league metrics, game view records, and odds snapshots. The refactored pipeline keeps calculations identical to the original scripts while splitting responsibilities into focused modules for easier maintenance and reuse.

## Quickstart

```
pip install -r requirements.txt
python -m src.refresh_week_data --season 2025 --week 7
python -m src.fetch_last_year_stats              # defaults to last year
```

## Outputs

| File | Description |
| ---- | ----------- |
| `out/final_league_metrics_{season}.csv` | Prior-season finals with offense/defense per game, dense ranks, TO, PF/PA/SU/ATS. |
| `out/league_metrics_{season}_{week}.csv` | In-season league metrics aggregated through the selected week. |
| `out/games_week_{season}_{week}.jsonl` & `.csv` | Game View records with matchup derivations, ranks, and S2D context. |
| `out/odds_{season}_wk{week}.jsonl` | Weekly odds snapshot (The Odds API, when credentials are available). |

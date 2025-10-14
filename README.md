# Football Statfinder

Automated tooling for producing weekly NFL data packs: league metrics, game view records, and odds snapshots. The refactored pipeline keeps calculations identical to the original scripts while splitting responsibilities into focused modules for easier maintenance and reuse.

## Quickstart

```
pip install -r requirements.txt
python -m src.refresh_week_data --season 2025 --week 7
python -m src.fetch_last_year_stats              # defaults to last year
python -m src.fetch_sagarin_week_nfl --season 2025 --week 7
```

## Outputs

| File | Description |
| ---- | ----------- |
| `out/final_league_metrics_{season}.csv` | Prior-season finals with offense/defense per game, dense ranks, TO, PF/PA/SU/ATS. |
| `out/league_metrics_{season}_{week}.csv` | In-season league metrics aggregated through the selected week. |
| `out/games_week_{season}_{week}.jsonl` & `.csv` | Game View records with matchup derivations, ranks, and S2D context. |
| `out/odds_{season}_wk{week}.jsonl` | Weekly odds snapshot (The Odds API, when credentials are available). |
| `out/sagarin_nfl_{season}_wk{week}.jsonl` & `.csv` | Sagarin power ratings with PR/SOS ranks, hfa, and audit metadata. |

### Sagarin Scraper

The Sagarin fetcher scrapes the fixed-width table at `http://sagarin.com/sports/nflsend.htm`, normalizes every team label, and writes both CSV and JSONL records sorted by power-rating rank. CLI flags:

```
python -m src.fetch_sagarin_week_nfl --season 2025 --week 6 \
    --out out/sagarin_nfl_2025_wk6               # optional basename
python -m src.fetch_sagarin_week_nfl --season 2025 --week 6 --local-html fixtures/nflsend.html
```

The script enforces 32-team coverage, rank validation, decimal precision, and optional SoS completeness, exiting non-zero on failure and emitting a debug text file when parsing issues arise.

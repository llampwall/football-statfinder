"""Produce prior-season final league metrics from nflverse and TeamRankings.

Purpose:
    Generate the final_league_metrics_{season}.csv file with offense/defense
    per-game values, dense ranks, turnover margin, and PF/PA/SU/ATS.
Inputs:
    Season label (defaults to last completed season).
Outputs:
    /out/final_league_metrics_{season}.csv
Source(s) of truth:
    nflverse schedules plus TeamRankings season-average tables.
Example:
    python -m src.fetch_last_year_stats --season 2024
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import requests

from src.common.io_utils import ensure_out_dir, write_csv
from src.common.metrics import dense_rank
from src.common.team_names import normalize_team_display, team_merge_key

NFLVERSE_GAMES_CSV = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TR_PAGES = {
    "RY(O)": "https://www.teamrankings.com/nfl/stat/rushing-yards-per-game",
    "PY(O)": "https://www.teamrankings.com/nfl/stat/passing-yards-per-game",
    "TY(O)": "https://www.teamrankings.com/nfl/stat/yards-per-game",
    "RY(D)": "https://www.teamrankings.com/nfl/stat/opponent-rushing-yards-per-game",
    "PY(D)": "https://www.teamrankings.com/nfl/stat/opponent-passing-yards-per-game",
    "TY(D)": "https://www.teamrankings.com/nfl/stat/opponent-yards-per-game",
    "TO": "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game",
}

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


def resolve_default_season() -> int:
    return datetime.now(timezone.utc).year - 1


def _safe_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(url, headers=UA_HEADERS, timeout=25)
    html = response.text
    try:
        return pd.read_html(html)
    except ValueError:
        return []


def _read_tr(url: str, season: int) -> pd.DataFrame:
    tables = _safe_tables(url)
    if not tables:
        raise RuntimeError(f"TeamRankings: no tables at {url}")
    want = str(season)
    for table in tables:
        if "Team" not in table.columns:
            continue
        columns = list(table.columns)
        if want in columns:
            col = want
        else:
            numeric_cols = [c for c in columns if c != "Team" and pd.api.types.is_numeric_dtype(table[c])]
            if not numeric_cols:
                continue
            year_cols = [c for c in numeric_cols if re.fullmatch(r"\d{4}", str(c))]
            col = sorted(year_cols, reverse=True)[0] if year_cols else numeric_cols[0]
        df = table[["Team", col]].rename(columns={col: "value"})
        return df
    raise RuntimeError(f"TeamRankings: suitable table not found at {url}")


def fetch_tr_metric(name: str, season: int) -> pd.Series:
    df = _read_tr(TR_PAGES[name], season)
    df["Team"] = df["Team"].astype(str).str.replace(r"[\*\+]+$", "", regex=True).str.strip()
    df["Team"] = df["Team"].map(normalize_team_display)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("Team")["value"]



def _rank_column_name(metric: str) -> str:
    match = re.fullmatch(r"([A-Z]+)\(([OD])\)", metric)
    if not match:
        raise ValueError(f"Unexpected metric label: {metric}")
    stat, side = match.groups()
    return f"R({side})_{stat}"


def fetch_pf_pa_su_ats_from_nflverse(season: int) -> pd.DataFrame:
    games = pd.read_csv(NFLVERSE_GAMES_CSV)
    subset = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    subset = subset.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    pf: Dict[str, int] = {}
    pa: Dict[str, int] = {}
    su_w: Dict[str, int] = {}
    su_l: Dict[str, int] = {}
    su_t: Dict[str, int] = {}
    ats_w: Dict[str, int] = {}
    ats_l: Dict[str, int] = {}
    ats_p: Dict[str, int] = {}

    def bump(store: Dict[str, int], key: str, value: int) -> None:
        store[key] = store.get(key, 0) + value

    for _, row in subset.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        bump(pf, home, home_score)
        bump(pa, home, away_score)
        bump(pf, away, away_score)
        bump(pa, away, home_score)

        if home_score > away_score:
            bump(su_w, home, 1)
            bump(su_l, away, 1)
        elif home_score < away_score:
            bump(su_w, away, 1)
            bump(su_l, home, 1)
        else:
            bump(su_t, home, 1)
            bump(su_t, away, 1)

        spread = row.get("spread_line")
        try:
            spread_val = float(spread) if pd.notna(spread) else None
        except Exception:
            spread_val = None

        if spread_val is not None:
            diff = (home_score + spread_val) - away_score
            if abs(diff) < 1e-9:
                bump(ats_p, home, 1)
                bump(ats_p, away, 1)
            elif diff > 0:
                bump(ats_w, home, 1)
                bump(ats_l, away, 1)
            else:
                bump(ats_w, away, 1)
                bump(ats_l, home, 1)

    rows = []
    teams = sorted(set(list(pf.keys()) + list(pa.keys())))
    for abbr in teams:
        display = normalize_team_display(abbr)
        merge = team_merge_key(display)
        wins = su_w.get(abbr, 0)
        losses = su_l.get(abbr, 0)
        ties = su_t.get(abbr, 0)
        su = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
        aw = ats_w.get(abbr, 0)
        al = ats_l.get(abbr, 0)
        ap = ats_p.get(abbr, 0)
        ats = f"{aw}-{al}-{ap}"
        rows.append(
            {
                "Team": display,
                "merge_key": merge,
                "PF": pf.get(abbr, 0),
                "PA": pa.get(abbr, 0),
                "SU": su,
                "ATS": ats,
            }
        )
    return pd.DataFrame(rows)


def build_final_league_metrics(season: int) -> pd.DataFrame:
    metrics = {}
    for name in ["RY(O)", "PY(O)", "TY(O)", "RY(D)", "PY(D)", "TY(D)", "TO"]:
        metrics[name] = fetch_tr_metric(name, season)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = "Team"
    metrics_df = metrics_df.reset_index()
    metrics_df["Team"] = metrics_df["Team"].map(normalize_team_display)
    metrics_df["merge_key"] = metrics_df["Team"].map(team_merge_key)

    pfpa = fetch_pf_pa_su_ats_from_nflverse(season)
    merged = metrics_df.merge(
        pfpa[["merge_key", "PF", "PA", "SU", "ATS"]],
        on="merge_key",
        how="left",
        validate="one_to_one",
    )

    for column, higher in [
        ("RY(O)", True),
        ("PY(O)", True),
        ("TY(O)", True),
        ("RY(D)", False),
        ("PY(D)", False),
        ("TY(D)", False),
    ]:
        series = pd.Series(
            merged[column].values,
            index=merged["merge_key"],
        )
        rank = dense_rank(series, higher_is_better=higher)
        rank_col = _rank_column_name(column)
        merged[rank_col] = rank.reindex(merged["merge_key"]).astype("Int64")

    for col in ["RY(O)", "PY(O)", "TY(O)", "RY(D)", "PY(D)", "TY(D)", "TO"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").round(1)

    final = merged[
        [
            "Team",
            "RY(O)",
            "R(O)_RY",
            "PY(O)",
            "R(O)_PY",
            "TY(O)",
            "R(O)_TY",
            "RY(D)",
            "R(D)_RY",
            "PY(D)",
            "R(D)_PY",
            "TY(D)",
            "R(D)_TY",
            "TO",
            "PF",
            "PA",
            "SU",
            "ATS",
        ]
    ].copy()

    return final.sort_values("Team").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate prior-season final league metrics.")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season label (defaults to last completed season).",
    )
    args = parser.parse_args()
    season = args.season if args.season is not None else resolve_default_season()
    df = build_final_league_metrics(season)
    out_path = ensure_out_dir() / f"final_league_metrics_{season}.csv"
    write_csv(df, out_path)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

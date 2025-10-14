#!/usr/bin/env python3
import re
import requests
import pandas as pd
import argparse
from datetime import datetime, timezone


# ---------- CONSTANTS ----------
NFLVERSE_GAMES_CSV = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

# TeamRankings metric pages (season-final per-game values)
TR_PAGES = {
    # Offense per game
    "RY(O)": "https://www.teamrankings.com/nfl/stat/rushing-yards-per-game",
    "PY(O)": "https://www.teamrankings.com/nfl/stat/passing-yards-per-game",
    "TY(O)": "https://www.teamrankings.com/nfl/stat/yards-per-game",
    # Defense allowed per game
    "RY(D)": "https://www.teamrankings.com/nfl/stat/opponent-rushing-yards-per-game",
    "PY(D)": "https://www.teamrankings.com/nfl/stat/opponent-passing-yards-per-game",
    "TY(D)": "https://www.teamrankings.com/nfl/stat/opponent-yards-per-game",
    # Turnover margin per game
    "TO":    "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game",
}

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}

# Pretty full names from nflverse team abbreviations
ABBR_TO_FULL = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","MIA":"Miami Dolphins",
    "MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
    "SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders",
}

CITY_TO_FULL = {
    "arizona": "arizona cardinals",
    "atlanta": "atlanta falcons",
    "baltimore": "baltimore ravens",
    "buffalo": "buffalo bills",
    "carolina": "carolina panthers",
    "chicago": "chicago bears",
    "cincinnati": "cincinnati bengals",
    "cleveland": "cleveland browns",
    "dallas": "dallas cowboys",
    "denver": "denver broncos",
    "detroit": "detroit lions",
    "green bay": "green bay packers",
    "houston": "houston texans",
    "indianapolis": "indianapolis colts",
    "jacksonville": "jacksonville jaguars",
    "kansas city": "kansas city chiefs",
    "la": "los angeles rams",
    "la rams": "los angeles rams",
    "la chargers": "los angeles chargers",
    "las vegas": "las vegas raiders",
    "miami": "miami dolphins",
    "minnesota": "minnesota vikings",
    "new england": "new england patriots",
    "new orleans": "new orleans saints",
    "new york giants": "new york giants",
    "new york jets": "new york jets",
    "philadelphia": "philadelphia eagles",
    "pittsburgh": "pittsburgh steelers",
    "san francisco": "san francisco 49ers",
    "seattle": "seattle seahawks",
    "tampa bay": "tampa bay buccaneers",
    "tennessee": "tennessee titans",
    "washington": "washington commanders",
}


# ---------- UTILS ----------
def _safe_tables(url: str) -> list[pd.DataFrame]:
    """Fetch URL with desktop UA and return any HTML tables; empty list if none."""
    html = requests.get(url, headers=UA, timeout=25).text
    try:
        return pd.read_html(html)
    except ValueError:
        return []

# Canonical merge key: robust across abbreviations / city shorthands / punctuation
_SYNONYMS = {
    "la rams":"los angeles rams",
    "la chargers":"los angeles chargers",
    "st louis rams":"los angeles rams",
    "oakland raiders":"las vegas raiders",
    "washington":"washington commanders",
    "was":"washington commanders","wsh":"washington commanders",
    "ny giants":"new york giants","ny jets":"new york jets",
}
def team_key(name: str) -> str:
    if not isinstance(name, str):
        name = "" if name is None else str(name)

    s = name.strip()

    # nflverse abbr -> full first (LAR/LAC/etc.)
    if s in ABBR_TO_FULL:
        s = ABBR_TO_FULL[s]

    # normalize weird forms so "L.A. Rams", "LA Rams" (NBSP), "L-A Rams" all become "la rams"
    s = s.replace("\u00a0", " ").replace(".", "").replace("-", " ")
    s = s.lower()
    s = re.sub(r"[\*\+]+$", "", s)
    s = s.replace("&","and").replace("st.","saint").replace("st ","saint ")
    s = re.sub(r"\s+", " ", s).strip()

    # expand city-only names (handles "la" and "la rams")
    s = CITY_TO_FULL.get(s, s)

    # historical/other synonyms
    s = _SYNONYMS.get(s, s)

    return re.sub(r"[^a-z0-9]", "", s)





def dense_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = series.sort_values(ascending=not higher_is_better)
    ranks, last_val, r = {}, None, 0
    for team, val in s.items():
        if pd.isna(val): continue
        if last_val is None or val != last_val:
            r += 1
            last_val = val
        ranks[team] = r
    return pd.Series(ranks)

# ---------- CORE: PF/PA/SU/ATS from nflverse games.csv ----------
def fetch_pf_pa_su_ats_from_nflverse(season: int) -> pd.DataFrame:
    """
    Build PF, PA, SU, ATS for the given season (REG only) using nflverse games.csv.
    - PF/PA from final scores
    - SU from score outcomes
    - ATS from home-relative closing line (spread_line), push when (home_score + line) == away_score
    Returns: DataFrame[Team (full name), PF, PA, SU, ATS]
    """
    games = pd.read_csv(NFLVERSE_GAMES_CSV)
    g = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    g = g.dropna(subset=["home_team","away_team"])
    g = g[(g["home_score"].notna()) & (g["away_score"].notna())].copy()

    pf, pa, w, l, t = {}, {}, {}, {}, {}
    ats_w, ats_l, ats_p = {}, {}, {}

    def bump(d, k, v): d[k] = d.get(k, 0) + v

    for _, r in g.iterrows():
        ht, at = str(r["home_team"]), str(r["away_team"])
        hs, as_ = int(r["home_score"]), int(r["away_score"])

        # PF/PA (both teams)
        bump(pf, ht, hs); bump(pa, ht, as_)
        bump(pf, at, as_); bump(pa, at, hs)

        # SU (straight up)
        if   hs > as_: bump(w, ht, 1); bump(l, at, 1)
        elif hs < as_: bump(w, at, 1); bump(l, ht, 1)
        else:          bump(t, ht, 1); bump(t, at, 1)

        # ATS (closing, home-relative spread_line). Rule: home covers if (home_score + line) > away_score; push if equal
        line = r.get("spread_line")
        try:
            line = float(line) if pd.notna(line) else None
        except Exception:
            line = None

        if line is not None:
            adj = (hs + line) - as_
            if abs(adj) < 1e-9:
                bump(ats_p, ht, 1); bump(ats_p, at, 1)
            elif adj > 0:
                bump(ats_w, ht, 1); bump(ats_l, at, 1)
            else:
                bump(ats_w, at, 1); bump(ats_l, ht, 1)

    # Build output rows (Team as full name)
    teams = sorted(set(list(pf.keys()) + list(pa.keys())))
    rows = []
    for abbr in teams:
        if abbr == "LA":                     # <— add this special case
            team = "Los Angeles Rams"
        else:
            team = ABBR_TO_FULL.get(abbr, abbr)
        win = w.get(abbr, 0); loss = l.get(abbr, 0); tie = t.get(abbr, 0)
        su = f"{win}-{loss}" + (f"-{tie}" if tie else "")
        aw, al, apu = ats_w.get(abbr, 0), ats_l.get(abbr, 0), ats_p.get(abbr, 0)
        ats = f"{aw}-{al}-{apu}"
        rows.append({"Team": team, "PF": pf.get(abbr, 0), "PA": pa.get(abbr, 0), "SU": su, "ATS": ats})

    return pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)

# ---------- TeamRankings metrics: RY/PY/TY (O/D) and TO ----------
def _read_tr(url: str, season: int) -> pd.DataFrame:
    """
    Read TeamRankings table and return 'Team' + the exact season column (e.g., '2024').
    If the exact season column isn't present, fall back to the first numeric column.
    """
    tables = _safe_tables(url)
    if not tables:
        raise RuntimeError(f"TeamRankings: no tables at {url}")
    want = str(season)
    for t in tables:
        if "Team" in t.columns:
            cols = list(t.columns)
            if want in cols:
                return t[["Team", want]].rename(columns={want: "val"})
            # fallback: first numeric column if exact season is missing
            num_cols = [c for c in cols if c != "Team" and pd.api.types.is_numeric_dtype(t[c])]
            if num_cols:
                return t[["Team", num_cols[0]]].rename(columns={num_cols[0]: "val"})
    raise RuntimeError(f"TeamRankings: suitable table not found at {url}")


def fetch_tr_metric(name: str, season: int) -> pd.Series:
    df = _read_tr(TR_PAGES[name], season)
    df["Team"] = df["Team"].astype(str).str.replace(r"[\*\+]+$","", regex=True).str.strip()
    df["val"]  = pd.to_numeric(df["val"], errors="coerce")
    return df.set_index("Team")["val"]

# ---------- Finals emitter (Caroline-style metrics + ranks + TO + PF/PA/SU/ATS) ----------
def emit_2024_league_metrics(pfpa_su_ats: pd.DataFrame, season: int, out_csv="final_league_metrics.csv"):
    # Pull TeamRankings metrics for that season
    ry_o  = fetch_tr_metric("RY(O)", season)
    py_o  = fetch_tr_metric("PY(O)", season)
    ty_o  = fetch_tr_metric("TY(O)", season)
    ry_d  = fetch_tr_metric("RY(D)", season)
    py_d  = fetch_tr_metric("PY(D)", season)
    ty_d  = fetch_tr_metric("TY(D)", season)
    to_pg = fetch_tr_metric("TO",    season)

    metrics = pd.DataFrame({
        "RY(O)": ry_o, "PY(O)": py_o, "TY(O)": ty_o,
        "RY(D)": ry_d, "PY(D)": py_d, "TY(D)": ty_d,
        "TO": to_pg
    })

    # Dense ranks: offense higher is better; defense allowed lower is better
    metrics["R(O)_RY"] = dense_rank(metrics["RY(O)"], True)
    metrics["R(O)_PY"] = dense_rank(metrics["PY(O)"], True)
    metrics["R(O)_TY"] = dense_rank(metrics["TY(O)"], True)
    metrics["R(D)_RY"] = dense_rank(metrics["RY(D)"], False)
    metrics["R(D)_PY"] = dense_rank(metrics["PY(D)"], False)
    metrics["R(D)_TY"] = dense_rank(metrics["TY(D)"], False)

    # --- Canonical join with PF/PA/SU/ATS ---
    left = metrics.reset_index().rename(columns={"index":"Team"})
    left["merge_key"] = left["Team"].map(team_key)

    right = pfpa_su_ats.copy()
    right["Team"]      = right["Team"].astype(str).str.replace(r"[\*\+]+$","", regex=True).str.strip()
    right["merge_key"] = right["Team"].map(team_key)

    # keep the authoritative full-name from the nflverse side
    right_name_by_key = dict(zip(right["merge_key"], right["Team"]))

    joined = left.merge(
        right[["merge_key","PF","PA","SU","ATS"]],
        on="merge_key",
        how="left",
        validate="many_to_one"
    )

    # 👉 force the display name to your nflverse full name (city + nickname)
    joined["Team"] = joined["merge_key"].map(right_name_by_key)

    # Warn if any team failed to merge (should be empty)
    miss = joined[joined[["PF","PA","SU","ATS"]].isna().all(axis=1)]
    if not miss.empty:
        print("WARN (finals merge unmatched):", miss["Team"].tolist())

    # Arrange columns & round floats
    out = joined[[
        "Team",
        "RY(O)","R(O)_RY","PY(O)","R(O)_PY","TY(O)","R(O)_TY",
        "RY(D)","R(D)_RY","PY(D)","R(D)_PY","TY(D)","R(D)_TY",
        "TO","PF","PA","SU","ATS"
    ]].sort_values("Team")

    for c in ["RY(O)","PY(O)","TY(O)","RY(D)","PY(D)","TY(D)","TO"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

def resolve_default_season() -> int:
    # “last year” by NFL season label
    return datetime.now(timezone.utc).year - 1


# ---------- MAIN ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None,
                   help="NFL season label (e.g., 2024). Defaults to last year.")
    # single-word flags (no hyphens, no underscores)
    p.add_argument("--emitstandings", action="store_true",
                   help="Also write the PF/PA/SU/ATS CSV.")
    p.add_argument("--outstandings", default="final_pf_pa_su_ats.csv",
                   help="Filename for PF/PA/SU/ATS CSV.")
    p.add_argument("--outmetrics", default="final_league_metrics_prev_season.csv",
                   help="Filename for the league metrics CSV.")
    args = p.parse_args()

    season = args.season if args.season is not None else resolve_default_season()
    print(f"Season selected: {season}")

    # 1) Build PF, PA, SU, ATS from nflverse for the chosen season
    pfpa_su_ats = fetch_pf_pa_su_ats_from_nflverse(season=season)
    if args.emitstandings:
        pfpa_su_ats.to_csv(args.outstandings, index=False)
        print(f"Wrote {args.outstandings}")

    # 2) Build full Caroline-style league metrics CSV for the chosen season
    emit_2024_league_metrics(pfpa_su_ats, season=season, out_csv=args.outmetrics)

if __name__ == "__main__":
    main()

"""Season-to-date team stats providers feeding the gameview build.

Ports the essentials of two season-1 twins:

* ``src/fetch_year_to_date_stats.py``      (NFL: nflverse team-week stats +
  schedule-derived PF/PA/SU/ATS + a TeamRankings turnover-margin scrape)
* ``src/fetch_year_to_date_stats_cfb.py``  (CFB: CFBD ``/teams/fbs`` +
  ``/games/teams`` aggregation)

Both providers emit the same frozen ``league_metrics`` table (header
``Team, RY(O), R(O)_RY, ..., TO, PF, PA, SU, ATS``) and the same typed
:class:`TeamStats` mapping the gameview builder joins.

Deliberate changes from legacy behavior:

* ``common.metrics.dense_rank`` is the ONLY ranking implementation. Season 1
  had three (this module's ancestor, an inline ``Series.rank(method="dense")``
  in ``gameview_build.py``, and ``build_team_timelines.py``); the inline copies
  are gone.
* Every HTTP fetch is injectable (constructor/keyword callables); nothing here
  reads env vars — CFB takes an explicit :class:`~football_statfinder.config.Settings`
  and fails loud via ``settings.require("cfbd_api_key")`` instead of silently
  producing an empty table (REBUILD.md bug 8).
* ``as_of_week`` is surfaced on the provider API so an orchestrator can pin
  historical rebuilds. nflverse and CFBD stats are inherently pinned by the
  week window; the TeamRankings turnover scrape is NOT — it serves live
  values only (REBUILD.md section 2, unverified lead: past-week rebuilds
  inject current turnover values). The current implementation documents that
  limitation by logging a warning when ``as_of_week`` differs from the build
  week; TeamRankings stat pages appear to accept a ``?date=YYYY-MM-DD`` query
  that a future revision can use to honor the pin.
* Season-1 NFL quirk preserved on purpose: the ``PF``/``PA`` columns are
  season TOTALS for NFL but PER-GAME averages for CFB, and downstream the
  gameview builder maps them into ``home_pf_pg``/``away_pf_pg`` unchanged.
  Flagged in the port notes as a contract decision to review.
* Legacy printed ``TO merge: found ...``; this module logs instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import pandas as pd
import requests

from ..common import metrics
from ..common.io_atomic import write_atomic_csv
from ..common.team_names import normalize_team_display, team_merge_key
from ..common.team_names_cfb import normalize_team_name_cfb_stats, team_merge_key_cfb
from ..config import Settings
from ..leagues import League
from .. import paths

try:  # pragma: no cover - fallback for hosts without tz database
    from zoneinfo import ZoneInfo

    _TZ_NY: Optional[Any] = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _TZ_NY = None

logger = logging.getLogger(__name__)

# Frozen league_metrics CSV header (frontend reads these files directly).
LEAGUE_METRICS_COLUMNS: List[str] = [
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

NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
NFLVERSE_TEAM_WEEK_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
)
TEAMRANKINGS_TURNOVER_URL = "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game"
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# CFB has no per-game closing spreads in the stats source; season 1 left ATS
# blank at this stage (a later ATS stage fills the gameview rows).
CFB_ATS_BLANK = ""


# ---------------------------------------------------------------------------
# Typed view of a league_metrics row
# ---------------------------------------------------------------------------


def _num(value: Any) -> Optional[float]:
    """Coerce CSV-ish values ('', '123.4', NaN, 7) to float or None."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _num_int(value: Any) -> Optional[int]:
    result = _num(value)
    if result is None:
        return None
    return int(round(result))


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


@dataclass(frozen=True)
class TeamStats:
    """Parsed league_metrics row for one team (the gameview join payload)."""

    team: str
    pf_pg: Optional[float] = None
    pa_pg: Optional[float] = None
    ry_pg: Optional[float] = None
    py_pg: Optional[float] = None
    ty_pg: Optional[float] = None
    ry_allowed_pg: Optional[float] = None
    py_allowed_pg: Optional[float] = None
    ty_allowed_pg: Optional[float] = None
    to_margin_pg: Optional[float] = None
    rush_rank: Optional[int] = None
    pass_rank: Optional[int] = None
    tot_off_rank: Optional[int] = None
    rush_def_rank: Optional[int] = None
    pass_def_rank: Optional[int] = None
    tot_def_rank: Optional[int] = None
    su: Optional[str] = None
    ats: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)


def team_stats_from_metrics_rows(
    league: League, rows: Iterable[Mapping[str, Any]]
) -> Dict[str, TeamStats]:
    """Key league_metrics rows by the league merge key (keep-last on dupes)."""
    out: Dict[str, TeamStats] = {}
    for row in rows:
        team = _text(row.get("Team"))
        if not team:
            continue
        token = league.merge_key(team)
        if not token:
            continue
        out[token] = TeamStats(
            team=team,
            pf_pg=_num(row.get("PF")),
            pa_pg=_num(row.get("PA")),
            ry_pg=_num(row.get("RY(O)")),
            py_pg=_num(row.get("PY(O)")),
            ty_pg=_num(row.get("TY(O)")),
            ry_allowed_pg=_num(row.get("RY(D)")),
            py_allowed_pg=_num(row.get("PY(D)")),
            ty_allowed_pg=_num(row.get("TY(D)")),
            to_margin_pg=_num(row.get("TO")),
            rush_rank=_num_int(row.get("R(O)_RY")),
            pass_rank=_num_int(row.get("R(O)_PY")),
            tot_off_rank=_num_int(row.get("R(O)_TY")),
            rush_def_rank=_num_int(row.get("R(D)_RY")),
            pass_def_rank=_num_int(row.get("R(D)_PY")),
            tot_def_rank=_num_int(row.get("R(D)_TY")),
            su=_text(row.get("SU")),
            ats=_text(row.get("ATS")),
            raw={key: (None if isinstance(val, float) and pd.isna(val) else val) for key, val in dict(row).items()},
        )
    return out


def league_metrics_csv_path(league: League, season: int, week: int) -> Path:
    return paths.week_dir(league.code, season, week) / f"league_metrics_{int(season)}_{int(week)}.csv"


def write_league_metrics_csv(
    league: League,
    season: int,
    week: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Optional[Path] = None,
) -> Path:
    """Atomically write the frozen-format league metrics CSV for a week."""
    target = path if path is not None else league_metrics_csv_path(league, season, week)
    frame = pd.DataFrame(list(rows), columns=LEAGUE_METRICS_COLUMNS)
    write_atomic_csv(target, frame)
    logger.info("wrote league metrics: %s (%d teams)", target, len(frame))
    return target


def load_league_metrics_csv(
    league: League,
    season: int,
    week: int,
    *,
    path: Optional[Path] = None,
) -> Dict[str, TeamStats]:
    """Read a week's league metrics CSV back into the gameview join mapping."""
    target = path if path is not None else league_metrics_csv_path(league, season, week)
    frame = pd.read_csv(target)
    return team_stats_from_metrics_rows(league, frame.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class StatsProvider(Protocol):
    """Season-to-date stats for one league, as of the start of a week."""

    league: League

    def league_metrics_rows(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Frozen-format league_metrics rows (strings formatted like season 1)."""

    def team_stats(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> Dict[str, TeamStats]:
        """The same data keyed by league merge key for the gameview join."""


def _effective_week(week: int, as_of_week: Optional[int]) -> int:
    return int(as_of_week) if as_of_week is not None else int(week)


# ---------------------------------------------------------------------------
# NFL: nflverse + TeamRankings
# ---------------------------------------------------------------------------


def fetch_nflverse_games(season: int, *, read_csv: Callable[[str], pd.DataFrame] = pd.read_csv) -> pd.DataFrame:
    """Download the nflverse games table filtered to one season (injectable)."""
    frame = read_csv(NFLVERSE_GAMES_URL)
    return frame[frame["season"] == season].copy()


def fetch_nflverse_team_week_stats(
    season: int, *, read_csv: Callable[[str], pd.DataFrame] = pd.read_csv
) -> pd.DataFrame:
    """Download the nflverse team-week stats release for a season (injectable)."""
    return read_csv(NFLVERSE_TEAM_WEEK_STATS_URL.format(season=season))


def fetch_teamrankings_turnover_margin(
    *,
    read_html: Callable[[str], List[pd.DataFrame]] = pd.read_html,
    as_of_week: Optional[int] = None,
) -> Dict[str, float]:
    """Scrape TeamRankings turnover margin per game, keyed by display name.

    KNOWN LIMITATION (documented, not yet fixed): the page serves live,
    current-day values; ``as_of_week`` cannot be honored yet, so rebuilding a
    past week injects today's turnover numbers (REBUILD.md section 2 lead).
    The parameter exists so orchestrators already declare their intent; a
    future revision can translate it to the page's ``?date=`` query.
    """
    if as_of_week is not None:
        logger.warning(
            "TeamRankings turnover scrape serves live values only; as_of_week=%s "
            "is recorded but cannot be honored yet (historical rebuilds get "
            "current turnover margins).",
            as_of_week,
        )
    tables = read_html(TEAMRANKINGS_TURNOVER_URL)
    frame: Optional[pd.DataFrame] = None
    for table in tables:
        if "Team" not in table.columns:
            continue
        numeric_cols = [c for c in table.columns if c != "Team" and pd.api.types.is_numeric_dtype(table[c])]
        if not numeric_cols:
            continue
        year_cols = [c for c in numeric_cols if re.fullmatch(r"\d{4}", str(c))]
        if year_cols:
            keep_col = sorted(year_cols, reverse=True)[0]
        else:
            filtered = [c for c in numeric_cols if str(c).strip().lower() not in {"rank"}]
            keep_col = filtered[0] if filtered else numeric_cols[0]
        frame = table[["Team", keep_col]].rename(columns={keep_col: "TO_pg"})
        break
    if frame is None:
        logger.warning("TeamRankings turnover page yielded no usable table")
        return {}
    frame["Team"] = frame["Team"].astype(str).str.strip()
    frame["TO_pg"] = pd.to_numeric(frame["TO_pg"], errors="coerce")
    return {
        normalize_team_display(team): float(value)
        for team, value in frame[["Team", "TO_pg"]].itertuples(index=False, name=None)
        if pd.notna(value) and normalize_team_display(team)
    }


def _parse_kickoff_utc(row: pd.Series) -> Optional[datetime]:
    """Kickoff datetime in UTC from nflverse schedule columns (legacy port)."""
    if "start_time_utc" in row and pd.notna(row["start_time_utc"]):
        try:
            return pd.to_datetime(row["start_time_utc"], utc=True).to_pydatetime()
        except (ValueError, TypeError):
            pass
    gameday = next(
        (str(row[c]) for c in ("gameday", "gamedate", "game_date") if c in row and pd.notna(row[c])), None
    )
    gametime = next(
        (str(row[c]) for c in ("gametime", "game_time_eastern", "start_time") if c in row and pd.notna(row[c])),
        None,
    )
    if not gameday or not gametime:
        return None
    try:
        dt_naive = pd.to_datetime(f"{gameday} {gametime}").to_pydatetime()
    except (ValueError, TypeError):
        return None
    if _TZ_NY is None:
        return dt_naive.replace(tzinfo=timezone.utc)
    return dt_naive.replace(tzinfo=_TZ_NY).astimezone(timezone.utc)


def _filter_week_reg(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if "game_type" in games.columns:
        mask_type = games["game_type"] == "REG"
    else:
        mask_type = pd.Series(True, index=games.index)
    subset = games[(games["season"] == season) & (games["week"] == week) & mask_type].copy()
    subset["kickoff_dt_utc"] = subset.apply(_parse_kickoff_utc, axis=1)
    return subset


def _find_col(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand in frame.columns:
            return cand
        key = cand.lower()
        if key in cols:
            return cols[key]
    norm = {c.lower().replace("_", ""): c for c in frame.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    return None


def _build_opponent_map(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = []
    sched_season = schedule[schedule["season"] == season]
    for _, row in sched_season.iterrows():
        if pd.isna(row.get("week")) or pd.isna(row.get("home_team")) or pd.isna(row.get("away_team")):
            continue
        week = int(row["week"])
        rows.append({"team": str(row["home_team"]), "week": week, "opponent": str(row["away_team"])})
        rows.append({"team": str(row["away_team"]), "week": week, "opponent": str(row["home_team"])})
    return pd.DataFrame(rows)


def _team_games(prior: pd.DataFrame, team_abbr: str) -> List[dict]:
    """Team-centric finished games with team lines from ``spread_line``."""
    records: List[dict] = []
    mask = (prior["home_team"] == team_abbr) | (prior["away_team"] == team_abbr)
    for _, row in prior.loc[mask].iterrows():
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        side = "HOME" if row["home_team"] == team_abbr else "AWAY"
        team_points = int(row["home_score"] if side == "HOME" else row["away_score"])
        opp_points = int(row["away_score"] if side == "HOME" else row["home_score"])
        spread = row.get("spread_line")
        if spread is None or pd.isna(spread):
            team_line = None
        else:
            # nflverse spread_line is the expected home margin; the home
            # team-centric line is its negation (legacy convention).
            team_line = -float(spread) if side == "HOME" else float(spread)
        records.append(
            {
                "team_points": team_points,
                "opp_points": opp_points,
                "team_margin": team_points - opp_points,
                "team_line": team_line,
            }
        )
    return records


def build_nfl_league_metrics_rows(
    season: int,
    week: int,
    *,
    schedule: pd.DataFrame,
    team_week_stats: pd.DataFrame,
    turnover_by_display: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Pure computation of the NFL league metrics rows from injected frames.

    Faithful port of ``generate_league_metrics`` minus network and prints.
    ``PF``/``PA`` stay season totals (season-1 NFL contract quirk, see module
    docstring); per-game yardage keeps the legacy one-decimal formatting.
    """
    if "kickoff_dt_utc" not in schedule.columns:
        schedule = schedule.copy()
        schedule["kickoff_dt_utc"] = schedule.apply(_parse_kickoff_utc, axis=1)
    week_games = _filter_week_reg(schedule, season, week)
    if week_games.empty:
        raise RuntimeError(f"No regular season games found for season {season} week {week}.")

    stats_df = team_week_stats
    col_team = _find_col(stats_df, ["team", "recent_team", "team_abbr"])
    col_week = _find_col(stats_df, ["week"])
    if not col_team or not col_week:
        raise RuntimeError("stats_team_week missing required team/week columns.")

    col_opp = _find_col(stats_df, ["opponent", "opp"])
    col_off_ry = _find_col(stats_df, ["rushing_yards", "rush_yards", "offense_rushing_yards"])
    col_off_py = _find_col(stats_df, ["net_passing_yards", "passing_yards", "pass_yards"])
    if col_off_ry is None or col_off_py is None:
        raise RuntimeError("stats_team_week missing offensive rushing/passing columns.")

    col_def_ry = _find_col(
        stats_df,
        [
            "opponent_rushing_yards",
            "rushing_yards_against",
            "rushing_yards_allowed",
            "defense_rushing_yards_allowed",
            "defense_rushing_yards",
        ],
    )
    col_def_py = _find_col(
        stats_df,
        [
            "opponent_net_passing_yards",
            "opponent_passing_yards",
            "net_passing_yards_against",
            "passing_yards_against",
            "passing_yards_allowed",
            "defense_passing_yards_allowed",
            "defense_passing_yards",
        ],
    )
    col_turnovers = _find_col(stats_df, ["turnovers", "giveaways"])
    col_takeaways = _find_col(stats_df, ["takeaways", "defensive_takeaways", "opponent_turnovers"])

    if col_opp is None:
        opp_df = _build_opponent_map(schedule, season)
        stats_df = stats_df.merge(
            opp_df, left_on=[col_team, col_week], right_on=["team", "week"], how="left"
        )
        col_opp = "opponent"

    base_df = stats_df.copy()

    need_mirror_def = (col_def_ry is None) or (col_def_py is None) or (col_takeaways is None)
    if need_mirror_def:
        opp_cols = [c for c in [col_off_ry, col_off_py, col_turnovers] if c]
        opp_side = base_df[[col_opp, col_week] + opp_cols].copy()
        rename_map = {
            col_opp: "opp_team",
            col_off_ry: "opp_off_ry",
            col_off_py: "opp_off_py",
            (col_turnovers if col_turnovers else "turnovers"): "opp_turnovers",
            col_week: "week",
        }
        opp_side = opp_side.rename(columns=rename_map)
        left = base_df[[col_team, col_week, col_opp]].rename(
            columns={col_team: "team", col_week: "week", col_opp: "opponent"}
        )
        mirror = left.merge(
            opp_side, left_on=["opponent", "week"], right_on=["opp_team", "week"], how="left"
        )
        ry_allowed_series = (
            base_df[col_def_ry]
            if col_def_ry is not None
            else mirror.get("opp_off_ry", pd.Series(pd.NA, index=base_df.index))
        )
        py_allowed_series = (
            base_df[col_def_py]
            if col_def_py is not None
            else mirror.get("opp_off_py", pd.Series(pd.NA, index=base_df.index))
        )
        takeaways_series = (
            base_df[col_takeaways]
            if col_takeaways is not None
            else mirror.get("opp_turnovers", pd.Series(pd.NA, index=base_df.index))
        )
    else:
        ry_allowed_series = base_df[col_def_ry]
        py_allowed_series = base_df[col_def_py]
        takeaways_series = (
            base_df[col_takeaways] if col_takeaways else pd.Series(pd.NA, index=base_df.index)
        )

    work = pd.DataFrame(
        {
            "team": base_df[col_team].astype(str),
            "week": base_df[col_week],
            "ry_off": base_df[col_off_ry],
            "py_off": base_df[col_off_py],
            "ry_allowed": ry_allowed_series,
            "py_allowed": py_allowed_series,
            "turnovers": base_df[col_turnovers] if col_turnovers else pd.NA,
            "takeaways": takeaways_series,
        }
    )

    teams = sorted(work["team"].dropna().astype(str).unique().tolist())

    def _s2d(team_label: str) -> Dict[str, Any]:
        subset = work[(work["team"] == team_label) & (work["week"] < week)]
        gp = int(subset.shape[0])
        if gp == 0:
            return dict(gp=0, ry=None, py=None, ty=None, ry_a=None, py_a=None, ty_a=None, to_pg=None)
        ry = float(subset["ry_off"].fillna(0).sum())
        py = float(subset["py_off"].fillna(0).sum())
        ty = ry + py
        ry_a = float(subset["ry_allowed"].fillna(0).sum())
        py_a = float(subset["py_allowed"].fillna(0).sum())
        ty_a = ry_a + py_a
        give = float(subset["turnovers"].fillna(0).sum())
        take = float(subset["takeaways"].fillna(0).sum())
        return dict(
            gp=gp,
            ry=ry / gp,
            py=py / gp,
            ty=ty / gp,
            ry_a=ry_a / gp,
            py_a=py_a / gp,
            ty_a=ty_a / gp,
            to_pg=(take - give) / gp,
        )

    per_team = {team: _s2d(team) for team in teams}

    def build_rank(metric_key: str, higher_is_better: bool) -> Dict[str, int]:
        data = {
            team_merge_key(team): stats[metric_key]
            for team, stats in per_team.items()
            if stats["gp"] > 0 and team_merge_key(team)
        }
        if not data:
            return {}
        ranked = metrics.dense_rank(pd.Series(data), higher_is_better=higher_is_better)
        return ranked.astype(int).to_dict()

    off_ry_rank = build_rank("ry", True)
    off_py_rank = build_rank("py", True)
    off_ty_rank = build_rank("ty", True)
    def_ry_rank = build_rank("ry_a", False)
    def_py_rank = build_rank("py_a", False)
    def_ty_rank = build_rank("ty_a", False)

    week_min_kick = week_games["kickoff_dt_utc"].min()
    prior_games = schedule[
        pd.notna(schedule["kickoff_dt_utc"]) & (schedule["kickoff_dt_utc"] < week_min_kick)
    ].copy()

    pfpa_su_ats: Dict[str, tuple] = {}
    for team in teams:
        games = _team_games(prior_games, team)
        pf = sum(g["team_points"] for g in games)
        pa = sum(g["opp_points"] for g in games)
        pfpa_su_ats[team] = (pf, pa, metrics.compute_su(games), metrics.compute_ats(games))

    to_pg_map: Dict[str, float] = {}
    for team in teams:
        display = normalize_team_display(team)
        val = turnover_by_display.get(display)
        if val is None and display:
            val = turnover_by_display.get(display.title())
        if val is not None:
            to_pg_map[team] = float(val)
    logger.info("TeamRankings turnover values matched for %d/%d teams", len(to_pg_map), len(teams))

    rows: List[Dict[str, Any]] = []
    for team in teams:
        display = normalize_team_display(team) or team
        stats = per_team[team]
        pf, pa, su, ats = pfpa_su_ats[team]
        key = team_merge_key(team)
        to_value: Optional[float] = to_pg_map.get(team, stats["to_pg"])
        rows.append(
            {
                "Team": display,
                "RY(O)": "" if stats["ry"] is None else f"{stats['ry']:.1f}",
                "R(O)_RY": "" if stats["gp"] == 0 else str(off_ry_rank.get(key, "")),
                "PY(O)": "" if stats["py"] is None else f"{stats['py']:.1f}",
                "R(O)_PY": "" if stats["gp"] == 0 else str(off_py_rank.get(key, "")),
                "TY(O)": "" if stats["ty"] is None else f"{stats['ty']:.1f}",
                "R(O)_TY": "" if stats["gp"] == 0 else str(off_ty_rank.get(key, "")),
                "RY(D)": "" if stats["ry_a"] is None else f"{stats['ry_a']:.1f}",
                "R(D)_RY": "" if stats["gp"] == 0 else str(def_ry_rank.get(key, "")),
                "PY(D)": "" if stats["py_a"] is None else f"{stats['py_a']:.1f}",
                "R(D)_PY": "" if stats["gp"] == 0 else str(def_py_rank.get(key, "")),
                "TY(D)": "" if stats["ty_a"] is None else f"{stats['ty_a']:.1f}",
                "R(D)_TY": "" if stats["gp"] == 0 else str(def_ty_rank.get(key, "")),
                "TO": "" if to_value is None else f"{to_value:.1f}",
                "PF": pf,
                "PA": pa,
                "SU": su,
                "ATS": ats,
            }
        )
    return rows


@dataclass(frozen=True)
class NflStatsProvider:
    """NFL season-to-date stats from nflverse + TeamRankings.

    All fetchers are injectable; tests pass frame/dict factories and never
    touch the network.
    """

    fetch_schedule: Callable[[int], pd.DataFrame] = fetch_nflverse_games
    fetch_team_week_stats: Callable[[int], pd.DataFrame] = fetch_nflverse_team_week_stats
    fetch_turnover_margin: Callable[..., Dict[str, float]] = fetch_teamrankings_turnover_margin

    @property
    def league(self) -> League:
        from ..leagues import NFL

        return NFL

    def league_metrics_rows(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        effective = _effective_week(week, as_of_week)
        schedule = self.fetch_schedule(season)
        team_week_stats = self.fetch_team_week_stats(season)
        turnover = self.fetch_turnover_margin(as_of_week=as_of_week)
        return build_nfl_league_metrics_rows(
            season,
            effective,
            schedule=schedule,
            team_week_stats=team_week_stats,
            turnover_by_display=turnover,
        )

    def team_stats(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> Dict[str, TeamStats]:
        rows = self.league_metrics_rows(season, week, as_of_week=as_of_week)
        return team_stats_from_metrics_rows(self.league, rows)


# ---------------------------------------------------------------------------
# CFB: CollegeFootballData API
# ---------------------------------------------------------------------------


def _cfbd_get_json(path: str, params: Mapping[str, Any], api_key: str) -> Any:
    response = requests.get(
        f"{CFBD_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=dict(params),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_cfbd_fbs_teams(
    season: int,
    api_key: str,
    *,
    get_json: Callable[[str, Mapping[str, Any], str], Any] = _cfbd_get_json,
) -> List[dict]:
    data = get_json("/teams/fbs", {"year": season}, api_key)
    return data if isinstance(data, list) else []


def fetch_cfbd_team_game_stats(
    season: int,
    week: int,
    api_key: str,
    *,
    get_json: Callable[[str, Mapping[str, Any], str], Any] = _cfbd_get_json,
) -> List[dict]:
    data = get_json("/games/teams", {"year": season, "seasonType": "regular", "week": week}, api_key)
    return data if isinstance(data, list) else []


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "-" in text:
            head = text.split("-", 1)[0]
            try:
                return float(head)
            except ValueError:
                return None
        return None


def _safe_int(value: Any) -> Optional[int]:
    result = _safe_float(value)
    return None if result is None else int(round(result))


def _extract_stat(stats: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in stats:
            return _safe_float(stats.get(key))
    return None


def _cfb_team_baseline(fbs_teams: List[dict]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    for entry in fbs_teams:
        raw_name = entry.get("school") or entry.get("team")
        if not raw_name:
            continue
        display = normalize_team_name_cfb_stats(raw_name)
        token = team_merge_key_cfb(display)
        if not token:
            continue
        teams.setdefault(
            token,
            {
                "team_display": display,
                "games": 0,
                "ry": 0.0,
                "py": 0.0,
                "ty": 0.0,
                "ry_allowed": 0.0,
                "py_allowed": 0.0,
                "ty_allowed": 0.0,
                "pf": 0.0,
                "pa": 0.0,
                "giveaways": 0.0,
                "takeaways": 0.0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
            },
        )
    return teams


def _cfb_parse_team_entry(team_entry: dict) -> tuple:
    team_name = team_entry.get("team") or ""
    display = normalize_team_name_cfb_stats(team_name)
    token = team_merge_key_cfb(display)
    stats_map = {
        item.get("category"): item.get("stat")
        for item in team_entry.get("stats") or []
        if isinstance(item, dict) and item.get("category")
    }
    return token, {
        "points": _safe_int(team_entry.get("points")),
        "rushing": _extract_stat(stats_map, ("rushingYards",)) or 0.0,
        "passing": _extract_stat(stats_map, ("netPassingYards", "passingYards")) or 0.0,
        "total": _extract_stat(stats_map, ("totalYards",)),
        "turnovers": _extract_stat(stats_map, ("turnovers",)) or 0.0,
    }


def _cfb_aggregate_games(games_payload: List[dict], teams: Dict[str, dict]) -> None:
    for game in games_payload:
        team_entries = game.get("teams") or []
        if len(team_entries) < 2:
            continue
        parsed = []
        for entry in team_entries:
            token, parsed_entry = _cfb_parse_team_entry(entry)
            parsed_entry["token"] = token
            parsed.append(parsed_entry)
        for idx, team_info in enumerate(parsed):
            token = team_info["token"]
            if token not in teams:
                continue
            opponent_info = parsed[1 - idx] if len(parsed) == 2 else next(
                (p for i, p in enumerate(parsed) if i != idx), None
            )
            if opponent_info is None:
                continue
            record = teams[token]
            record["games"] += 1
            team_points = team_info["points"]
            opp_points = opponent_info["points"]
            record["pf"] += float(team_points or 0)
            record["pa"] += float(opp_points or 0)
            record["ry"] += float(team_info["rushing"])
            record["py"] += float(team_info["passing"])
            team_total = (
                float(team_info["total"])
                if team_info["total"] is not None
                else float(team_info["rushing"]) + float(team_info["passing"])
            )
            opp_total = (
                float(opponent_info["total"])
                if opponent_info["total"] is not None
                else float(opponent_info["rushing"]) + float(opponent_info["passing"])
            )
            record["ty"] += team_total
            record["ry_allowed"] += float(opponent_info["rushing"])
            record["py_allowed"] += float(opponent_info["passing"])
            record["ty_allowed"] += opp_total
            record["giveaways"] += float(team_info["turnovers"])
            record["takeaways"] += float(opponent_info["turnovers"])
            if team_points is not None and opp_points is not None:
                if team_points > opp_points:
                    record["wins"] += 1
                elif team_points < opp_points:
                    record["losses"] += 1
                else:
                    record["ties"] += 1


def _cfb_ranks(teams: Dict[str, dict]) -> Dict[str, Dict[str, Optional[int]]]:
    valid = {token: data for token, data in teams.items() if data["games"] > 0}
    metric_columns = {
        "ry": ("R(O)_RY", True),
        "py": ("R(O)_PY", True),
        "ty": ("R(O)_TY", True),
        "ry_allowed": ("R(D)_RY", False),
        "py_allowed": ("R(D)_PY", False),
        "ty_allowed": ("R(D)_TY", False),
    }
    ranks: Dict[str, Dict[str, Optional[int]]] = {token: {} for token in teams}
    for metric, (column, higher_is_better) in metric_columns.items():
        series = pd.Series(
            {token: data[metric] / data["games"] for token, data in valid.items()}, dtype=float
        )
        if series.empty:
            continue
        rank_series = metrics.dense_rank(series, higher_is_better=higher_is_better).astype(int)
        for token, value in rank_series.to_dict().items():
            ranks[token][column] = int(value)
    return ranks


def build_cfb_league_metrics_rows(
    season: int,
    week: int,
    *,
    fbs_teams: List[dict],
    team_game_stats: List[dict],
) -> List[Dict[str, Any]]:
    """Pure computation of CFB league metrics rows from injected payloads.

    ``team_game_stats`` is the flat list of CFBD ``/games/teams`` game payloads
    for weeks 1..week-1 (the provider collects them). ATS is intentionally
    blank at this stage (season-1 behavior; a later stage fills it). One
    legacy divergence kept: PF/PA here are per-game averages while the NFL
    table carries totals (see module docstring).
    """
    del season, week  # the window is decided by the caller's payload
    teams = _cfb_team_baseline(fbs_teams)
    _cfb_aggregate_games(team_game_stats, teams)
    ranks = _cfb_ranks(teams)

    def fmt(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.1f}"

    def rank_str(token: str, column: str) -> str:
        value = ranks.get(token, {}).get(column)
        return "" if value is None else str(int(value))

    rows: List[Dict[str, Any]] = []
    for token, data in sorted(teams.items(), key=lambda item: item[1]["team_display"]):
        games = data["games"]

        def per_game(key: str) -> Optional[float]:
            return (data[key] / games) if games else None

        wins, losses, ties = data["wins"], data["losses"], data["ties"]
        su = f"{wins}-{losses}-{ties}" if ties else f"{wins}-{losses}"
        rows.append(
            {
                "Team": data["team_display"],
                "RY(O)": fmt(per_game("ry")),
                "R(O)_RY": rank_str(token, "R(O)_RY"),
                "PY(O)": fmt(per_game("py")),
                "R(O)_PY": rank_str(token, "R(O)_PY"),
                "TY(O)": fmt(per_game("ty")),
                "R(O)_TY": rank_str(token, "R(O)_TY"),
                "RY(D)": fmt(per_game("ry_allowed")),
                "R(D)_RY": rank_str(token, "R(D)_RY"),
                "PY(D)": fmt(per_game("py_allowed")),
                "R(D)_PY": rank_str(token, "R(D)_PY"),
                "TY(D)": fmt(per_game("ty_allowed")),
                "R(D)_TY": rank_str(token, "R(D)_TY"),
                "TO": fmt(
                    ((data["takeaways"] - data["giveaways"]) / games) if games else None
                ),
                "PF": fmt(per_game("pf")),
                "PA": fmt(per_game("pa")),
                "SU": su,
                "ATS": CFB_ATS_BLANK,
            }
        )
    return rows


@dataclass(frozen=True)
class CfbStatsProvider:
    """CFB season-to-date stats from the CFBD API.

    Requires ``settings.cfbd_api_key`` and fails loud when it is missing
    (legacy returned an empty table and let downstream stages guess why).
    """

    settings: Settings
    fetch_fbs_teams: Callable[..., List[dict]] = fetch_cfbd_fbs_teams
    fetch_team_game_stats: Callable[..., List[dict]] = fetch_cfbd_team_game_stats

    @property
    def league(self) -> League:
        from ..leagues import CFB

        return CFB

    def league_metrics_rows(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self.settings.require("cfbd_api_key")
        api_key = self.settings.cfbd_api_key or ""
        effective = _effective_week(week, as_of_week)
        fbs_teams = self.fetch_fbs_teams(season, api_key)
        if not fbs_teams:
            raise RuntimeError(f"CFBD returned no FBS teams for season {season}")
        game_stats: List[dict] = []
        for stats_week in range(1, effective):
            game_stats.extend(self.fetch_team_game_stats(season, stats_week, api_key) or [])
        return build_cfb_league_metrics_rows(
            season, effective, fbs_teams=fbs_teams, team_game_stats=game_stats
        )

    def team_stats(
        self, season: int, week: int, *, as_of_week: Optional[int] = None
    ) -> Dict[str, TeamStats]:
        rows = self.league_metrics_rows(season, week, as_of_week=as_of_week)
        return team_stats_from_metrics_rows(self.league, rows)


def get_stats_provider(league: League, settings: Settings) -> StatsProvider:
    """Resolve the stats provider for a league."""
    if league.code == "nfl":
        return NflStatsProvider()
    if league.code == "cfb":
        return CfbStatsProvider(settings=settings)
    raise ValueError(f"no stats provider for league {league.code!r}")


__all__ = [
    "CFB_ATS_BLANK",
    "CfbStatsProvider",
    "LEAGUE_METRICS_COLUMNS",
    "NflStatsProvider",
    "StatsProvider",
    "TeamStats",
    "build_cfb_league_metrics_rows",
    "build_nfl_league_metrics_rows",
    "fetch_cfbd_fbs_teams",
    "fetch_cfbd_team_game_stats",
    "fetch_nflverse_games",
    "fetch_nflverse_team_week_stats",
    "fetch_teamrankings_turnover_margin",
    "get_stats_provider",
    "league_metrics_csv_path",
    "load_league_metrics_csv",
    "team_stats_from_metrics_rows",
    "write_league_metrics_csv",
]

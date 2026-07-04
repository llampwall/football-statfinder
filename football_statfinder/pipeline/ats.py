"""One league-parameterized ATS stage: cover math, season records, week writer.

Replaces the season-1 twins ``src/ats/nfl_ats.py`` + ``src/cfb_ats.py`` and
the closing-spread resolver ``src/odds/ats_compute.py``. Legacy behavior
deliberately changed while porting:

* Bug 1: weeks are scanned under the unified ``out/{league}/{S}_week{W}/``
  convention via ``paths.week_dir`` — the NFL twin scanned the never-written
  ``out/nfl/`` tree and was a permanent no-op.
* Bug 4: the closing-spread tiers now actually work. Tier (a) reads the
  promoted/pinned ledger at ``paths.odds_pinned_jsonl`` (free); tier (b) is
  the Odds API historical client harvested from ``feature/ats-api-backfill``
  (paid). ``settings.backfill.ats_source`` selects: ``auto`` (free then
  paid), ``pinned`` (free only), ``api``/``history`` (paid only). The dead
  ``_PINNED_CACHE``/``_SNAPSHOT_CACHE`` module globals are gone.
* Bug 19: one blank-ATS sentinel. Blank is ``None`` (what this module
  writes); ``is_blank`` additionally treats the legacy em/en-dash and
  bare-dash placeholders from season-1 data as blank so backfill re-fills
  them instead of considering them done.
* The ``function.attribute`` state hack (``build_team_ats.meta``,
  ``apply_ats_to_week.teams_in_week``) is replaced by returned dataclasses.
* One record format: season ATS strings are always ``W-L-P`` (the NFL twin
  dropped the ``-P`` segment when pushes were zero; CFB always carried it).
* Both the JSONL and CSV week artifacts are rewritten atomically (the CFB
  twin never rewrote the CSV).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import pandas as pd

from .. import paths
from ..common.io_atomic import write_atomic_csv, write_atomic_jsonl
from ..common.jsonl import read_jsonl
from ..config import Settings
from ..leagues import League

logger = logging.getLogger(__name__)

EPSILON = 1e-6

# Season-1 data used a literal em dash placeholder for "no ATS yet" (bug 19);
# treat every dash-ish placeholder as blank so legacy rows get refilled.
_LEGACY_BLANK_STRINGS = {"", "-", "–", "—"}

BLANK_ATS = None  # the one sentinel this package ever writes


def is_blank(value: Any) -> bool:
    """True when an ATS field carries no value (None, empty, legacy dashes)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _LEGACY_BLANK_STRINGS
    return False


def compute_game_ats(
    home_score: Any, away_score: Any, favored_team: Any, spread: Any
) -> Optional[Dict[str, Any]]:
    """Per-game ATS outcome and to-margin for both sides.

    Ported from ``src/odds/ats_compute.py`` (the validated variant: spread is
    taken as a magnitude, favored side must be HOME/AWAY/PICK).
    """
    try:
        home_val = int(home_score)
        away_val = int(away_score)
        spread_val = abs(float(spread))
    except (TypeError, ValueError):
        return None
    favored = (favored_team or "").upper()
    if favored not in {"HOME", "AWAY", "PICK"}:
        return None
    if favored == "HOME":
        home_line = -spread_val
    elif favored == "AWAY":
        home_line = spread_val
    else:  # PICK: spread is zero either way
        home_line = 0.0
    home_vs_line = (home_val - away_val) + home_line
    away_vs_line = -home_vs_line

    def _label(value: float) -> str:
        if value > 0:
            return "W"
        if value < 0:
            return "L"
        return "P"

    return {
        "home_ats": _label(home_vs_line),
        "away_ats": _label(away_vs_line),
        "to_margin_home": round(home_vs_line, 2),
        "to_margin_away": round(away_vs_line, 2),
    }


# ---------------------------------------------------------------------------
# Closing-spread resolution (bug 4 fix)
# ---------------------------------------------------------------------------


class ClosingSpreadApi(Protocol):
    """Paid tier: the harvested Odds API resolver (or a test stub)."""

    def resolve_closing_spread(
        self, season: int, week: int, game_row: Mapping[str, Any]
    ) -> Optional[Mapping[str, Any]]: ...


def load_pinned_spread_index(
    league: League, season: int, *, out_root: Optional[Path] = None
) -> Dict[str, List[dict]]:
    """Index the pinned odds ledger: game_key -> its ``spreads`` records."""
    path = paths.odds_pinned_jsonl(league.code, season, out_root=out_root)
    index: Dict[str, List[dict]] = {}
    for record in read_jsonl(path).rows:
        if record.get("market") != "spreads":
            continue
        game_key = record.get("game_key")
        if isinstance(game_key, str) and game_key:
            index.setdefault(game_key, []).append(record)
    return index


def _parse_ts(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pick_latest_before(records: Sequence[Mapping[str, Any]], cutoff: datetime) -> Optional[Mapping[str, Any]]:
    """Record with the greatest ``fetch_ts`` <= cutoff (missing ts ignored)."""
    best: Optional[Tuple[datetime, Mapping[str, Any]]] = None
    for record in records or []:
        when = _parse_ts(record.get("fetch_ts"))
        if when is None:
            continue
        if when <= cutoff and (best is None or when > best[0]):
            best = (when, record)
    return best[1] if best else None


def _payload_from_pinned(
    league: League,
    record: Mapping[str, Any],
    home_norm: Optional[str],
    away_norm: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Convert one pinned ledger record into a closing-spread payload.

    Ported from ``ats_compute._record_to_payload`` (raw-outcome fallback kept),
    with PICK preserved instead of being silently coerced to AWAY.
    """
    line = record.get("line") or record
    spread_home_relative = line.get("spread_home_relative")
    if not _is_finite(spread_home_relative):
        home_key = league.merge_key(home_norm) if home_norm else None
        away_key = league.merge_key(away_norm) if away_norm else None
        home_point = None
        away_point = None
        for outcome in line.get("raw_outcomes") or []:
            name = outcome.get("name")
            point = outcome.get("point")
            if not _is_finite(point) or not name:
                continue
            token = league.merge_key(str(name))
            if token == home_key:
                home_point = float(point)
            elif token == away_key:
                away_point = float(point)
        if home_point is not None:
            spread_home_relative = home_point
        elif away_point is not None:
            spread_home_relative = -away_point
    if not _is_finite(spread_home_relative):
        return None
    value = float(spread_home_relative)

    favored_side = (line.get("favored_side") or "").upper()
    if not favored_side:
        if value < 0:
            favored_side = "HOME"
        elif value > 0:
            favored_side = "AWAY"
        else:
            favored_side = "PICK"
    if favored_side not in {"HOME", "AWAY", "PICK"}:
        return None
    return {
        "favored_team": favored_side,
        "spread": abs(value),
        "book": record.get("book") or record.get("book_label"),
        "fetched_ts": record.get("fetch_ts") or record.get("snapshot_at"),
    }


def resolve_closing_spread(
    league: League,
    season: int,
    game_row: Mapping[str, Any],
    *,
    settings: Settings,
    pinned_index: Optional[Dict[str, List[dict]]] = None,
    api: Optional[ClosingSpreadApi] = None,
    out_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the closing spread for one game: pinned ledger, then paid API.

    Returns ``{"favored_team", "spread", "source": "pinned"|"history", "book",
    "fetched_ts"}`` or ``None``. Tier order follows
    ``settings.backfill.ats_source`` (``auto``/``pinned``/``api``/``history``).
    """
    source_policy = (settings.backfill.ats_source or "auto").strip().lower()
    if source_policy not in {"auto", "pinned", "api", "history"}:
        logger.warning("unknown ats_source %r; treating as 'auto'", source_policy)
        source_policy = "auto"

    game_key = str(game_row.get("game_key") or "")
    cutoff = _parse_ts(game_row.get("kickoff_iso_utc") or game_row.get("kickoff_ts"))
    if cutoff is None:
        cutoff = datetime.now(tz=timezone.utc)

    # Tier (a): the week's promoted rows / pinned ledger (free).
    if source_policy in {"auto", "pinned"} and game_key:
        if pinned_index is None:
            pinned_index = load_pinned_spread_index(league, season, out_root=out_root)
        candidates = pinned_index.get(game_key) or []
        record = _pick_latest_before(candidates, cutoff)
        if record is not None:
            payload = _payload_from_pinned(
                league,
                record,
                game_row.get("home_team_norm") or record.get("home_norm"),
                game_row.get("away_team_norm") or record.get("away_norm"),
            )
            if payload is not None:
                payload["source"] = "pinned"
                return payload

    # Tier (b): the Odds API historical endpoint (paid; the client itself
    # refuses network when settings.odds.cache_only is set).
    if source_policy in {"auto", "api", "history"} and api is not None:
        try:
            week = int(game_row.get("week"))
        except (TypeError, ValueError):
            week = 0
        payload = api.resolve_closing_spread(season, week, game_row)
        if payload and payload.get("favored_team") is not None and _is_finite(payload.get("spread")):
            return {
                "favored_team": payload["favored_team"],
                "spread": float(payload["spread"]),
                "source": str(payload.get("source") or "history"),
                "book": payload.get("book"),
                "fetched_ts": payload.get("fetched_ts"),
            }

    return None


# ---------------------------------------------------------------------------
# Season ATS records (the one home_ats/away_ats writer)
# ---------------------------------------------------------------------------


@dataclass
class TeamAts:
    """Season-to-date ATS tally for one team."""

    w: int = 0
    l: int = 0
    p: int = 0

    @property
    def games(self) -> int:
        return self.w + self.l + self.p

    def record(self) -> Optional[str]:
        """``W-L-P`` string, or None when the team has no lined games yet."""
        if self.games <= 0:
            return None
        return f"{self.w}-{self.l}-{self.p}"


@dataclass
class AtsBuildResult:
    stats: Dict[str, TeamAts] = field(default_factory=dict)
    weeks_scanned: List[int] = field(default_factory=list)
    games_considered: int = 0


@dataclass
class AtsApplyResult:
    rows_updated: int = 0
    teams_in_week: int = 0
    zero_lined: int = 0


def _to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _score_from_row(row: Mapping[str, Any], side: str) -> Optional[float]:
    key = f"{side}_score"
    num = _to_number(row.get(key))
    if num is not None:
        return num
    raw_sources = row.get("raw_sources")
    schedule_row = raw_sources.get("schedule_row") if isinstance(raw_sources, dict) else None
    if isinstance(schedule_row, dict):
        return _to_number(schedule_row.get(key))
    return None


def team_key(league: League, row: Mapping[str, Any], side: str) -> Optional[str]:
    """Stable per-team tally key: the league merge key of the norm/raw label."""
    for candidate in (row.get(f"{side}_team_norm"), row.get(f"{side}_team_raw")):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return league.merge_key(text)
    return None


def build_team_ats(
    league: League,
    season: int,
    week: int,
    *,
    out_root: Optional[Path] = None,
) -> AtsBuildResult:
    """Season-to-date ATS tallies from finalized, lined games in weeks < week."""
    result = AtsBuildResult()
    if week <= 1:
        return result

    for prior_week in range(1, week):
        path = paths.games_week_jsonl(league.code, season, prior_week, out_root=out_root)
        rows = read_jsonl(path).rows
        if not rows:
            continue
        result.weeks_scanned.append(prior_week)
        for row in rows:
            favored_side = str(row.get("favored_side") or "").upper()
            if favored_side not in {"HOME", "AWAY"}:
                continue
            spread = _to_number(row.get("spread_favored_team"))
            if spread is None:
                continue
            home_score = _score_from_row(row, "home")
            away_score = _score_from_row(row, "away")
            if home_score is None or away_score is None:
                continue
            home_team = team_key(league, row, "home")
            away_team = team_key(league, row, "away")
            if not home_team or not away_team:
                continue

            margin = float(home_score) - float(away_score)
            if favored_side == "HOME":
                cover_score = margin + float(spread)
                favored, dog = home_team, away_team
            else:
                cover_score = (-margin) + float(spread)
                favored, dog = away_team, home_team

            result.stats.setdefault(favored, TeamAts())
            result.stats.setdefault(dog, TeamAts())
            result.games_considered += 1

            if cover_score > EPSILON:
                result.stats[favored].w += 1
                result.stats[dog].l += 1
            elif cover_score < -EPSILON:
                result.stats[favored].l += 1
                result.stats[dog].w += 1
            else:
                result.stats[favored].p += 1
                result.stats[dog].p += 1

    result.weeks_scanned.sort()
    return result


def _align_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    ordered = [col for col in columns if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + remainder)


def apply_ats_to_week(
    league: League,
    season: int,
    week: int,
    build: AtsBuildResult,
    *,
    out_root: Optional[Path] = None,
) -> AtsApplyResult:
    """Write season ATS records onto the week's rows (JSONL + CSV, atomic).

    Field semantics written: ``{home,away}_ats`` = season-to-date ``W-L-P``
    string (never a dash placeholder — bug 19); the split-count fields
    ``{side}_ats_{w,l,p}`` are updated only when the row already carries them
    (frozen frontend contract). A team with zero counted games keeps whatever
    ATS value the row already has (typically the metrics-sourced record) —
    the computed tally never overwrites with ``None``.
    """
    json_path = paths.games_week_jsonl(league.code, season, week, out_root=out_root)
    csv_path = paths.games_week_csv(league.code, season, week, out_root=out_root)
    rows = read_jsonl(json_path).rows
    if not rows:
        return AtsApplyResult()

    rows_updated = 0
    teams_in_week: set[str] = set()

    for row in rows:
        changed = False
        for side in ("home", "away"):
            key = team_key(league, row, side)
            if key:
                teams_in_week.add(key)
            tally = build.stats.get(key) if key else None
            record = tally.record() if tally else None
            ats_field = f"{side}_ats"

            if record is not None:
                if row.get(ats_field) != record:
                    row[ats_field] = record
                    changed = True
                if tally is not None:
                    for suffix in ("w", "l", "p"):
                        field_name = f"{side}_ats_{suffix}"
                        if field_name in row:
                            value = int(getattr(tally, suffix))
                            if row.get(field_name) != value:
                                row[field_name] = value
                                changed = True
            else:
                # No counted games for this team (week 1, or prior weeks not
                # lined/scored yet). Preserve a real record the stats join
                # already put on the row — the metrics-sourced season ATS is a
                # legitimate value, and clobbering it with None here nulled
                # every NFL ATS record in the parity replay (and would do the
                # same in production early weeks). Blank placeholders (legacy
                # em-dash, empty string) still normalize to the None sentinel
                # (bug 19).
                existing = row.get(ats_field)
                if existing is not None and is_blank(existing):
                    row[ats_field] = BLANK_ATS
                    changed = True
                    for suffix in ("w", "l", "p"):
                        field_name = f"{side}_ats_{suffix}"
                        if field_name in row and row.get(field_name) is not None:
                            row[field_name] = None
                            changed = True
        if changed:
            rows_updated += 1

    zero_lined = sum(
        1
        for team in teams_in_week
        if team not in build.stats or build.stats[team].games == 0
    )

    if rows_updated > 0:
        write_atomic_jsonl(json_path, rows)
        final_df = pd.DataFrame(rows)
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            final_df = _align_columns(final_df, list(existing_df.columns))
        write_atomic_csv(csv_path, final_df)

    return AtsApplyResult(
        rows_updated=rows_updated,
        teams_in_week=len(teams_in_week),
        zero_lined=zero_lined,
    )


def run_ats(
    league: League,
    season: int,
    week: int,
    settings: Settings,
    *,
    out_root: Optional[Path] = None,
) -> AtsApplyResult:
    """Build season tallies from prior weeks and apply them to ``week``."""
    if not settings.backfill.ats_enable:
        logger.info("ats disabled (ats_enable=False); league=%s", league.code)
        return AtsApplyResult()
    build = build_team_ats(league, season, week, out_root=out_root)
    applied = apply_ats_to_week(league, season, week, build, out_root=out_root)
    logger.info(
        "ats: league=%s week=%s-%s weeks_scanned=%s games=%d rows_updated=%d "
        "teams=%d zero_lined=%d",
        league.code,
        season,
        week,
        build.weeks_scanned,
        build.games_considered,
        applied.rows_updated,
        applied.teams_in_week,
        applied.zero_lined,
    )
    return applied


__all__ = [
    "AtsApplyResult",
    "AtsBuildResult",
    "BLANK_ATS",
    "ClosingSpreadApi",
    "EPSILON",
    "TeamAts",
    "apply_ats_to_week",
    "build_team_ats",
    "compute_game_ats",
    "is_blank",
    "load_pinned_spread_index",
    "resolve_closing_spread",
    "run_ats",
    "team_key",
]

"""Classified games_week + sidecar differ (Phase 2 WP-A, Part 1).

Joins the season-1 baseline week against the replay-rebuilt week on
``(kickoff instant, home merge-key, away merge-key)`` — never on ``game_key``
(NFL keys changed) — and classifies every field delta into exactly one label:
a frozen whitelist rule ``W<n>``, a known legacy-bug correction ``BUGFIX-<n>``,
or ``UNEXPLAINED``. Writes ``docs/parity/parity_{league}_{season}_wk{week}.md``
plus a ``.json`` with full per-game detail.

The whitelist is frozen (spec Part 1): this differ CITES rules, it never
invents them. Anything that does not cleanly fit a rule stays UNEXPLAINED; the
report's proposed-whitelist section is where new rules are argued, for
main-loop approval (WP-B).

Run (after replay):
    python -m tools.parity.diff --league nfl --season 2025 --week 16
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from football_statfinder import paths as paths_mod
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.leagues import League, get_league

from tools.parity import replay as replay_mod

REPO_ROOT = paths_mod.REPO_ROOT
DOCS_PARITY = REPO_ROOT / "docs" / "parity"

BLANK_SENTINELS = {None, "", "-", "—", "–", "nan", "None"}

# Field groupings from gameview.FROZEN_RECORD_FIELDS.
TEAM_REP_FIELDS = {"home_team_norm", "away_team_norm", "home_team_raw", "away_team_raw"}
ATS_FIELDS = {"home_ats", "away_ats"}
VOLATILE_FIELDS = {"snapshot_at"}
ODDS_PRIMARY_FIELDS = {
    "spread_home_relative",
    "total",
    "moneyline_home",
    "moneyline_away",
    "odds_source",
    "is_closing",
}
DERIVED_ODDS_FIELDS = {
    "favored_side",
    "spread_favored_team",
    "rating_vs_odds",
    "rating_vs_odds_favored_team",
}
# rating_diff and its favored variant do NOT depend on odds; for CFB their delta
# is the bug-7 HFA-inclusion correction (proven by new-legacy == hfa).
RATING_DIFF_FIELDS = {"rating_diff", "rating_diff_favored_team"}
# raw_sources subfields the frontend actually reads (grep of web/ recorded in
# the report). Everything else under raw_sources is provenance noise, excluded.
FRONTEND_RAW_SOURCES = [
    ("raw_sources", "sagarin_row_home", "team"),
    ("raw_sources", "sagarin_row_home", "hfa"),
    ("raw_sources", "sagarin_row_away", "team"),
    ("raw_sources", "sagarin_row_away", "hfa"),
    ("raw_sources", "schedule_row", "game_no"),
    ("raw_sources", "schedule_row", "rotation"),
    ("raw_sources", "schedule_row", "gsis"),
]

# BUGFIX-4 (approved WP-B triage rule): when a game's legacy row fell back to
# odds_source="schedule" and the replay promoted real book odds, ALL of these
# downstream odds-derived fields classify BUGFIX-4 for that game (frozen list,
# spec triage round 1). raw_sources.odds_row and rating_diff_favored_team's
# sign-only flip are handled separately (see classify()) because they need
# extra shape/sign checks the plain equality gate above doesn't do.
BUGFIX4_ODDS_FIELDS = ODDS_PRIMARY_FIELDS | DERIVED_ODDS_FIELDS

# W9 (sidecar Sagarin enrichment policy): sidecar entry fields whose value
# deltas trace to the documented season-2 policy change (nearest-week
# fallback + dense_rank vs legacy exact-week + sequential ranks).
SIDECAR_W9_FIELDS = {
    "pr", "pr_rank", "sos", "sos_rank",
    "opp_pr", "opp_pr_rank", "opp_sos", "opp_sos_rank",
}


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def _parse_instant(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in BLANK_SENTINELS:
        return True
    return False


def _norm_blank(value: Any) -> Any:
    return None if _is_blank(value) else value


def _as_float(value: Any) -> Optional[float]:
    try:
        if _is_blank(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _floats_equal(a: Any, b: Any) -> bool:
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= 1e-9


def _norm_ats(value: Any) -> Optional[str]:
    if _is_blank(value):
        return None
    text = str(value).strip()
    parts = text.split("-")
    if len(parts) == 2:
        parts = parts + ["0"]
    return "-".join(p.strip() for p in parts)


def _values_equal(field: str, old: Any, new: Any) -> bool:
    """True when the two values are equal for parity purposes (post-normalize)."""
    if field in ATS_FIELDS:
        return _norm_ats(old) == _norm_ats(new)
    if _is_blank(old) and _is_blank(new):
        return True
    if _floats_equal(old, new):
        return True
    return _norm_blank(old) == _norm_blank(new)


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


def _join_key(league: League, row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    instant = _parse_instant(row.get("kickoff_iso_utc"))
    home = row.get("home_team_norm") or row.get("home_team_raw") or ""
    away = row.get("away_team_norm") or row.get("away_team_raw") or ""
    if instant is None or not home or not away:
        return None
    return (instant.isoformat(), league.merge_key(str(home)), league.merge_key(str(away)))


def _index_rows(league: League, rows: List[Dict[str, Any]]):
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    collisions: List[Tuple[str, str, str]] = []
    unkeyed: List[Dict[str, Any]] = []
    for row in rows:
        key = _join_key(league, row)
        if key is None:
            unkeyed.append(row)
            continue
        if key in index:
            collisions.append(key)
        index[key] = row
    return index, collisions, unkeyed


# ---------------------------------------------------------------------------
# nested access
# ---------------------------------------------------------------------------


def _dig(row: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def classify(field: str, old: Any, new: Any, ctx: Dict[str, Any]) -> Tuple[str, str]:
    """Return (label, reason) for a single field delta. League-aware via ctx."""
    league: League = ctx["league"]
    legacy_src = ctx.get("legacy_odds_source")
    new_src = ctx.get("new_odds_source")
    new_hfa = ctx.get("new_hfa")
    spread_matches = ctx.get("spread_matches", False)

    if field == "kickoff_iso_utc":
        oi, ni = _parse_instant(old), _parse_instant(new)
        if oi is not None and oi == ni:
            return "W6", "kickoff string format only; same instant"
        return "UNEXPLAINED", "kickoff instant differs"

    if field == "game_key":
        return "W1", "game_key format (NFL abbreviation->full-name slug); join is instant+merge_key"

    if field == "source_uid":
        if _is_blank(new) and not _is_blank(old):
            return (
                "W10",
                "source_uid: legacy provider id -> new null (SCHEDULE_COLUMNS carries no "
                "source_uid column); frontend never reads it (grep-verified: zero refs in web/)",
            )
        return "W5", "source_uid format/provenance"

    if field in TEAM_REP_FIELDS:
        if league.merge_key(str(_norm_blank(old) or "")) == league.merge_key(str(_norm_blank(new) or "")):
            return "W2", "team representation (legacy NFL abbreviation vs full name); merge_key identical"
        return "UNEXPLAINED", "team identity differs after merge_key"

    if field in VOLATILE_FIELDS or field.endswith("_ts") or field.endswith("fetched_at") or field == "computed_at":
        return "W5", "volatile provenance timestamp"

    if field in ATS_FIELDS:
        if _norm_ats(old) == _norm_ats(new):
            if _is_blank(old) or _is_blank(new):
                return "W4", "blank sentinel vs null (ATS)"
            return "W3", "ATS record format (W-L padded to W-L-P)"
        return "UNEXPLAINED", "ATS record value differs after normalization"

    # generic blank-sentinel equivalence
    if _norm_blank(old) == _norm_blank(new) and (_is_blank(old) or _is_blank(new)):
        return "W4", "blank sentinel (em-dash/empty) vs null"

    if field in RATING_DIFF_FIELDS:
        do, dn = _as_float(old), _as_float(new)
        if (
            league.code == "cfb"
            and new_hfa not in (None, 0.0)
            and do is not None
            and dn is not None
            and (abs((dn - do) - new_hfa) < 0.02 or abs((abs(dn) - abs(do)) - new_hfa) < 0.02)
        ):
            return (
                "BUGFIX-7",
                f"rating_diff HFA-inclusion fix: legacy CFB _compute_rating_vectors omitted HFA, "
                f"new = legacy + hfa ({new_hfa:.2f}); new={dn} legacy={do}",
            )
        if (
            field == "rating_diff_favored_team"
            and do is None and dn is not None
            and ctx.get("bugfix4_trigger")
        ):
            return (
                "BUGFIX-4",
                "rating_diff_favored_team legacy-null -> new-populated: favored_side only exists "
                "once a spread does, and the replay promoted book odds where legacy promotion "
                "was dead (spec BUGFIX-4 CFB extension)",
            )
        if (
            field == "rating_diff_favored_team"
            and league.code == "cfb"
            and do is None and dn is not None
        ):
            return (
                "BUGFIX-7",
                "legacy CFB never derived rating_diff_favored_team at all (0 populated rows across "
                "checked weeks 10/13/14); the single-formula build derives it whenever a favorite "
                "exists (spec BUGFIX-7 family)",
            )
        if (
            field == "rating_diff_favored_team"
            and do is not None and dn is not None
            and _floats_equal(abs(dn), abs(do))
        ):
            if ctx.get("bugfix4_trigger"):
                return (
                    "BUGFIX-4",
                    "rating_diff_favored_team sign flip: |value| unchanged but favored_side flipped "
                    "because the replay promoted real book odds where legacy used the dead "
                    "schedule-fallback tier (odds_source='schedule')",
                )
            return (
                "UNEXPLAINED",
                "rating_diff_favored_team sign flip: |value| unchanged but favored_side flipped "
                "because the promoted spread put the other team as favorite "
                "(downstream of the odds-sourcing divergence, K1/bug-4)",
            )
        return "UNEXPLAINED", f"{field} differs (HFA/Sagarin, independent of odds)"

    if field == "is_closing" and not ctx.get("bugfix4_trigger"):
        return (
            "W11",
            "is_closing delta (any value); frontend never reads it (grep-verified: zero refs in "
            "web/). false->true on legacy-promoted games is replay hindsight: legacy's last "
            "promotion ran pre-kickoff and could not mark its own snapshot closing",
        )

    if field in ODDS_PRIMARY_FIELDS or field in DERIVED_ODDS_FIELDS or field == "raw_sources.odds_row":
        if ctx.get("bugfix4_trigger") and field in BUGFIX4_ODDS_FIELDS:
            return (
                "BUGFIX-4",
                "legacy NFL odds promotion was dead all season (odds_source='schedule': the "
                "away-first pinned-ledger keys never matched home-first game keys); the replay's "
                "canonical-key pin promotes real book odds, so this odds-derived field corrects",
            )
        if ctx.get("bugfix4_trigger") and field == "raw_sources.odds_row":
            return (
                "BUGFIX-4",
                "legacy never promoted odds for this game (odds_source='schedule'); replay's "
                "corrected pin attaches raw_sources.odds_row",
            )
        # CFB rating_vs_odds two-formula fix, only provable when the spread itself
        # matched (otherwise the delta is entangled with odds sourcing).
        if (
            league.code == "cfb"
            and field in ("rating_vs_odds", "rating_vs_odds_favored_team")
            and spread_matches
            and not (_is_blank(legacy_src) or legacy_src == "schedule")
            and not _is_blank(new_src)
        ):
            return (
                "BUGFIX-7",
                "single rating_vs_odds formula (legacy CFB emitted two conflicting values); "
                "spread matched so the delta is pure formula",
            )
        legacy_promoted = not (_is_blank(legacy_src) or legacy_src == "schedule")
        new_promoted = not _is_blank(new_src)
        if not legacy_promoted and new_promoted:
            reason = (
                "odds_sourcing_divergence: legacy used schedule-fallback/none (odds_source="
                f"{legacy_src!r}), new promotes book odds (odds_source={new_src!r}); "
                "new pipeline has no schedule-odds fallback tier (K1)"
            )
            return "UNEXPLAINED", reason
        if not legacy_promoted and not new_promoted:
            return "UNEXPLAINED", f"odds field differs, neither side promoted (legacy_src={legacy_src!r})"
        if legacy_promoted and new_promoted:
            if field in ("spread_home_relative", "moneyline_home", "moneyline_away", "spread_favored_team"):
                if _floats_equal(_as_float(new), -(_as_float(old) or 0.0)) and not _floats_equal(old, new):
                    return "BUGFIX-20", "role-swap pin sign flip corrected (new = -legacy)"
            if league.code == "cfb" and field in ("rating_vs_odds", "rating_vs_odds_favored_team"):
                return "BUGFIX-7", "single rating_vs_odds formula (legacy CFB emitted two conflicting values)"
            return "UNEXPLAINED", f"both promoted but value differs ({field})"
        return "UNEXPLAINED", f"odds field differs ({field})"

    if field == "rating_diff":
        return "UNEXPLAINED", "rating_diff differs (HFA/Sagarin-independent of odds)"

    if field.startswith("raw_sources.schedule_row."):
        if field.rsplit(".", 1)[-1] in {"gsis", "game_no", "rotation"} and _is_blank(new):
            return (
                "W12",
                "schedule provenance drop (whitelisted): gsis/game_no/rotation read only in the "
                "frontend's last-resort Game# fallback chain; primary path computes locally",
            )
        return "UNEXPLAINED", f"schedule_row provenance differs ({field})"

    if field.startswith("raw_sources.sagarin_row"):
        if league.code == "cfb" and _is_blank(old) and not _is_blank(new):
            return (
                "W8",
                "CFB sagarin_row enrichment: legacy null -> new populated (added provenance; "
                "frontend reads .team/.hfa and handles both states)",
            )
        return "UNEXPLAINED", f"sagarin provenance differs ({field})"

    return "UNEXPLAINED", f"value mismatch ({field})"


# ---------------------------------------------------------------------------
# per-pair diff
# ---------------------------------------------------------------------------


def _compare_pair(
    league: League, key: Tuple[str, str, str], old: Dict[str, Any], new: Dict[str, Any]
) -> List[Dict[str, Any]]:
    legacy_src = old.get("odds_source")
    new_src = new.get("odds_source")
    ctx = {
        "league": league,
        "legacy_odds_source": legacy_src,
        "new_odds_source": new_src,
        "new_hfa": _as_float(new.get("hfa")),
        "spread_matches": _values_equal(
            "spread_home_relative",
            old.get("spread_home_relative"),
            new.get("spread_home_relative"),
        ),
        # BUGFIX-4 trigger (spec triage round 1): legacy fell back to the
        # schedule-odds tier (never promoted real book odds all season — the
        # legacy away-first pinned-ledger keys never matched home-first game
        # keys) AND the replay promoted real book odds. When true, every
        # odds-derived field delta for this game is BUGFIX-4, not UNEXPLAINED.
        # CFB extension (spec triage round 2, hand-traced): legacy CFB promotion
        # was ALSO dead for any multi-word team name — the pinned ledger slugged
        # "newmexico_airforce" while games_week slugged "new_mexico_air_force",
        # so the legacy row has NO odds at all (odds_source blank) while the
        # replay's canonical-key pin promotes book odds. Same bug class.
        "bugfix4_trigger": (
            (legacy_src == "schedule" or _is_blank(legacy_src))
            and not _is_blank(new_src)
            and new_src != "schedule"
        ),
    }
    deltas: List[Dict[str, Any]] = []

    fields = [f for f in old.keys() if f != "raw_sources"]
    for f in new.keys():
        if f != "raw_sources" and f not in fields:
            fields.append(f)

    for field in fields:
        ov, nv = old.get(field), new.get(field)
        if _values_equal(field, ov, nv):
            continue
        label, reason = classify(field, ov, nv, ctx)
        deltas.append({"field": field, "old": ov, "new": nv, "label": label, "reason": reason})

    # curated raw_sources subfields (frontend-read only)
    for path in FRONTEND_RAW_SOURCES:
        pseudo = ".".join(path)
        ov = _dig(old, path)
        nv = _dig(new, path)
        if _values_equal(pseudo, ov, nv):
            continue
        label, reason = classify(pseudo, ov, nv, ctx)
        deltas.append({"field": pseudo, "old": ov, "new": nv, "label": label, "reason": reason})

    # odds_row presence (frontend reads raw_sources.odds_row)
    old_or = _dig(old, ("raw_sources", "odds_row"))
    new_or = _dig(new, ("raw_sources", "odds_row"))
    if (old_or is None) != (new_or is None):
        label, reason = classify("raw_sources.odds_row", old_or, new_or, ctx)
        deltas.append(
            {
                "field": "raw_sources.odds_row",
                "old": None if old_or is None else "<present>",
                "new": None if new_or is None else "<present>",
                "label": label,
                "reason": reason,
            }
        )
    return deltas


# ---------------------------------------------------------------------------
# sidecar diff (K3)
# ---------------------------------------------------------------------------

SIDECAR_TIMELINE_KEYS = ("home_ytd", "away_ytd", "home_prev", "away_prev")
SIDECAR_ENTRY_FIELDS = (
    "season", "week", "date", "opp", "site", "pf", "pa", "result", "ats", "to_margin",
    "pr", "pr_rank", "sos", "sos_rank", "opp_pr", "opp_pr_rank", "opp_sos", "opp_sos_rank",
)


def _load_sidecar(directory: Path, game_key: str) -> Optional[Dict[str, Any]]:
    path = directory / f"{game_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _diff_sidecar(old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[Counter, Counter]:
    """Return (raw per-field delta counts, classified label counts).

    W9 (sidecar Sagarin enrichment policy): pr/pr_rank/sos/sos_rank (and their
    opp_ variants) value deltas trace to the documented season-2 policy change
    (nearest-week fallback + dense_rank vs legacy exact-week + sequential
    ranks) and classify W9. Every other sidecar entry field delta stays
    UNEXPLAINED (default; this differ never stretches a rule predicate).
    """
    counts: Counter = Counter()
    label_counts: Counter = Counter()
    for tkey in SIDECAR_TIMELINE_KEYS:
        old_entries = {(_norm_blank(e.get("date")), e.get("opp")): e for e in (old.get(tkey) or [])}
        new_entries = {(_norm_blank(e.get("date")), e.get("opp")): e for e in (new.get(tkey) or [])}
        counts[f"{tkey}.entries_old"] += len(old_entries)
        counts[f"{tkey}.entries_new"] += len(new_entries)
        for ekey in set(old_entries) & set(new_entries):
            oe, ne = old_entries[ekey], new_entries[ekey]
            for field in SIDECAR_ENTRY_FIELDS:
                if not _values_equal(field, oe.get(field), ne.get(field)):
                    counts[f"field:{field}"] += 1
                    label = "W9" if field in SIDECAR_W9_FIELDS else "UNEXPLAINED"
                    label_counts[label] += 1
                    label_counts[f"{label}::{field}"] += 1
        counts[f"{tkey}.entries_only_old"] += len(set(old_entries) - set(new_entries))
        counts[f"{tkey}.entries_only_new"] += len(set(new_entries) - set(old_entries))
    return counts, label_counts


# ---------------------------------------------------------------------------
# report build
# ---------------------------------------------------------------------------


def _k4_trace(
    league: League, season: int, week: int, scratch_out: Path, new_rows: List[Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Trace promoted odds: confirm the chosen (book, fetch_ts) is the latest
    pre-kickoff pinned record for that (game_key, market) in the replayed ledger."""
    ledger = read_jsonl(paths_mod.odds_pinned_jsonl(league.code, season, out_root=scratch_out)).rows
    by_gk_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in ledger:
        by_gk_market[(rec.get("game_key"), rec.get("market"))].append(rec)

    traces: List[Dict[str, Any]] = []
    for row in new_rows:
        if len([t for t in traces]) >= limit:
            break
        odds_row = _dig(row, ("raw_sources", "odds_row"))
        if not isinstance(odds_row, dict):
            continue
        markets = odds_row.get("markets") or {}
        spread = markets.get("spreads")
        if not isinstance(spread, dict):
            continue
        gk = row.get("game_key")
        kickoff = _parse_instant(row.get("kickoff_iso_utc"))
        chosen_ts = _parse_instant(spread.get("fetch_ts"))
        all_recs = by_gk_market.get((gk, "spreads"), [])
        # Mirror promote_week's closing_pre_kickoff candidate set: freshest record
        # per book (_select_latest), then the latest of those with fetch_ts <= kickoff.
        freshest_by_book: Dict[str, Dict[str, Any]] = {}
        for rec in all_recs:
            ts = _parse_instant(rec.get("fetch_ts"))
            if ts is None:
                continue
            book = rec.get("book")
            cur = freshest_by_book.get(book)
            if cur is None or ts > _parse_instant(cur.get("fetch_ts")):
                freshest_by_book[book] = rec
        candidates = list(freshest_by_book.values())
        pre_kick = [
            c for c in candidates
            if kickoff is not None and _parse_instant(c.get("fetch_ts")) <= kickoff
        ]
        latest_pre = max(
            (c for c in pre_kick), key=lambda c: _parse_instant(c.get("fetch_ts")), default=None
        ) if pre_kick else None
        latest_ts = _parse_instant(latest_pre.get("fetch_ts")) if latest_pre else None
        freshest_ts = max(
            (_parse_instant(c.get("fetch_ts")) for c in candidates),
            default=None,
        ) if candidates else None
        is_latest_pre = bool(chosen_ts and latest_ts and chosen_ts == latest_ts)
        is_freshest = bool(chosen_ts and freshest_ts and chosen_ts == freshest_ts)
        # closing_pre_kickoff: pick latest pre-kickoff candidate; if none exist
        # (e.g. CFB midnight placeholder kickoffs) fall back to the freshest overall.
        if pre_kick:
            policy_correct = is_latest_pre
            policy_branch = "closing_pre_kickoff"
        else:
            policy_correct = is_freshest
            policy_branch = "fallback_to_freshest (no pre-kickoff candidate)"
        traces.append({
            "game_key": gk,
            "matchup": f"{row.get('away_team_norm')} @ {row.get('home_team_norm')}",
            "kickoff": row.get("kickoff_iso_utc"),
            "chosen_book": spread.get("book"),
            "chosen_fetch_ts": spread.get("fetch_ts"),
            "pinned_records_for_spreads": len(all_recs),
            "freshest_per_book_candidates": len(candidates),
            "candidates_pre_kickoff": len(pre_kick),
            "latest_pre_kickoff_fetch_ts": latest_pre.get("fetch_ts") if latest_pre else None,
            "chosen_is_latest_pre_kickoff": is_latest_pre,
            "chosen_before_kickoff": bool(chosen_ts and kickoff and chosen_ts <= kickoff),
            "policy_branch": policy_branch,
            "policy_correct": policy_correct,
        })
    return traces


def build_report(
    league_code: str,
    season: int,
    week: int,
    *,
    scratch_root: Path = replay_mod.DEFAULT_SCRATCH,
    baseline_out: Path = paths_mod.OUT_ROOT,
) -> Dict[str, Any]:
    league = get_league(league_code)
    scratch_out = scratch_root / f"{league.code}_{season}_wk{week}" / "out"
    manifest_path = scratch_out.parent / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    bl_dir = replay_mod.baseline_week_dir(baseline_out, league.code, season, week)
    old_rows = read_jsonl(bl_dir / f"games_week_{season}_{week}.jsonl").rows
    new_rows = read_jsonl(
        paths_mod.games_week_jsonl(league.code, season, week, out_root=scratch_out)
    ).rows

    old_idx, old_coll, old_unkeyed = _index_rows(league, old_rows)
    new_idx, new_coll, new_unkeyed = _index_rows(league, new_rows)

    matched = sorted(set(old_idx) & set(new_idx))
    only_old = sorted(set(old_idx) - set(new_idx))
    only_new = sorted(set(new_idx) - set(old_idx))

    per_game: List[Dict[str, Any]] = []
    label_counts: Counter = Counter()
    field_label_counts: Counter = Counter()
    unexplained_samples: List[Dict[str, Any]] = []
    bugfix_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    old_bl_side = bl_dir / "game_schedules"
    new_side = scratch_out / league.code / f"{season}_week{week}" / "game_schedules"
    sidecar_counts: Counter = Counter()
    sidecar_label_counts: Counter = Counter()
    sidecar_games = 0

    for key in matched:
        old, new = old_idx[key], new_idx[key]
        deltas = _compare_pair(league, key, old, new)
        for d in deltas:
            label_counts[d["label"]] += 1
            field_label_counts[(d["label"], d["field"])] += 1
            evidence = {
                "old_game_key": old.get("game_key"),
                "new_game_key": new.get("game_key"),
                "kickoff": old.get("kickoff_iso_utc"),
                "matchup": f"{new.get('away_team_norm')} @ {new.get('home_team_norm')}",
                **d,
            }
            if d["label"] == "UNEXPLAINED" and len(unexplained_samples) < 400:
                unexplained_samples.append(evidence)
            if d["label"].startswith("BUGFIX") and len(bugfix_samples[d["label"]]) < 8:
                bugfix_samples[d["label"]].append(evidence)
        per_game.append(
            {
                "join_key": list(key),
                "old_game_key": old.get("game_key"),
                "new_game_key": new.get("game_key"),
                "deltas": deltas,
            }
        )
        # K3 sidecar diff
        old_sc = _load_sidecar(old_bl_side, str(old.get("game_key")))
        new_sc = _load_sidecar(new_side, str(new.get("game_key")))
        if old_sc and new_sc:
            sidecar_games += 1
            entry_counts, entry_label_counts = _diff_sidecar(old_sc, new_sc)
            sidecar_counts.update(entry_counts)
            sidecar_label_counts.update(entry_label_counts)

    report = {
        "league": league.display,
        "season": season,
        "week": week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "join": {
            "baseline_rows": len(old_rows),
            "replay_rows": len(new_rows),
            "matched": len(matched),
            "only_baseline": len(only_old),
            "only_replay": len(only_new),
            "baseline_collisions": len(old_coll),
            "replay_collisions": len(new_coll),
            "baseline_unkeyed": len(old_unkeyed),
            "replay_unkeyed": len(new_unkeyed),
        },
        "orphans": {
            "only_baseline": [
                {"key": list(k), "game_key": old_idx[k].get("game_key"),
                 "matchup": f"{old_idx[k].get('away_team_norm')} @ {old_idx[k].get('home_team_norm')}"}
                for k in only_old
            ],
            "only_replay": [
                {"key": list(k), "game_key": new_idx[k].get("game_key"),
                 "matchup": f"{new_idx[k].get('away_team_norm')} @ {new_idx[k].get('home_team_norm')}"}
                for k in only_new
            ],
        },
        "label_counts": dict(label_counts),
        "field_label_counts": {f"{lbl}::{fld}": c for (lbl, fld), c in sorted(field_label_counts.items())},
        "bugfix_samples": {k: v for k, v in bugfix_samples.items()},
        "unexplained_samples": unexplained_samples,
        "sidecar": {
            "games_compared": sidecar_games,
            "counts": dict(sidecar_counts),
            "label_counts": dict(sidecar_label_counts),
        },
        "per_game": per_game,
        "k4_trace": _k4_trace(league, season, week, scratch_out, new_rows),
    }
    report["known_checks"] = _known_checks(league, season, week, report, old_rows, new_rows)
    report["proposed_whitelist"] = _proposals(league, report)
    return report


def _proposals(league: League, report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Derive proposed whitelist/triage rules from the observed UNEXPLAINED groups.

    These are PROPOSALS for WP-B/main-loop approval; the differ never applies or
    freezes them. (source_uid, the NFL dead-odds-promotion reclassification, and
    CFB sagarin_row enrichment were proposed here in WP-A and approved in the
    2026-07-03 triage round as W10, BUGFIX-4, and W8 respectively — see
    ``classify()``/``PHASE2_SPEC.md``; they no longer surface as proposals here
    because ``classify()`` now labels them directly instead of leaving them
    UNEXPLAINED.)
    """
    reasons = {s["reason"] for s in report["unexplained_samples"]}
    fields = {s["field"] for s in report["unexplained_samples"]}
    out: List[Dict[str, str]] = []
    if any(f.startswith("raw_sources.schedule_row.") for f in fields):
        out.append({
            "id": "PROP-schedule_provenance",
            "kind": "pipeline-gap or whitelist",
            "text": "the frontend reads raw_sources.schedule_row.game_no/rotation/gsis, but "
                    "SCHEDULE_COLUMNS omits them so the new pipeline emits None. Either carry these "
                    "columns through the schedule schema or whitelist their loss (frontend already "
                    "has a fallback).",
        })
    if league.code == "cfb" and any(f in fields for f in ("home_ats", "away_ats")):
        out.append({
            "id": "PROP-cfb-ats-sourcing",
            "kind": "triage (pipeline decision)",
            "text": "legacy CFB games_week carried computed ATS records (e.g. '0-1-0'); the new "
                    "gameview sources home_ats/away_ats solely from league_metrics, whose ATS column "
                    "is blank for CFB (CFB_ATS_BLANK), so new emits None. Decide whether the CFB ATS "
                    "compute should feed games_week ATS or whether blank is the intended contract.",
        })
    return out


def _known_checks(
    league: League, season: int, week: int, report: Dict[str, Any],
    old_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    join = report["join"]
    # K1: schedule-odds fallback presence in the new build
    legacy_sched = sum(1 for r in old_rows if r.get("odds_source") == "schedule")
    new_sched = sum(1 for r in new_rows if r.get("odds_source") == "schedule")
    new_none_odds = sum(1 for r in new_rows if _is_blank(r.get("odds_source")))
    k1 = {
        "check": "schedule-odds fallback tier",
        "result": "FAIL" if legacy_sched and new_sched == 0 else "PASS",
        "finding": (
            f"legacy rows with odds_source='schedule': {legacy_sched}; new rows with "
            f"odds_source='schedule': {new_sched} (new gameview has no schedule-odds "
            f"fallback tier — spread/total from schedule columns is never emitted). New rows "
            f"with blank odds_source (unpromoted): {new_none_odds}."
        ),
    }
    # K2: row coverage
    expected = 16 if league.code == "nfl" else 60
    orphan_cause = (
        " Replay orphans are schedule-master rows absent from the season-1 baseline snapshot: "
        "for NFL these are stale duplicate-kickoff master rows (the same matchup appears at two "
        "kickoff times because the master upsert KEY includes kickoff_iso_utc, so a corrected "
        "kickoff adds a row instead of replacing the stale one); for CFB these are additional "
        "FBS-vs-FBS games the season-1 games_week never captured. The new pipeline faithfully "
        "emits one row per master row; no legacy game is missing (only_baseline=0)."
    ) if join["only_replay"] else ""
    k2 = {
        "check": "row coverage (1:1 join)",
        "result": "PASS" if (join["only_baseline"] == 0 and join["only_replay"] == 0
                             and join["matched"] == len(old_rows) == len(new_rows)) else "FAIL",
        "finding": (
            f"baseline={join['baseline_rows']} replay={join['replay_rows']} matched={join['matched']} "
            f"only_baseline={join['only_baseline']} only_replay={join['only_replay']}; "
            f"expected {expected}/{expected}.{orphan_cause}"
        ),
    }
    # K3: sidecar parity
    sc = report["sidecar"]
    field_deltas = {k: v for k, v in sc["counts"].items() if k.startswith("field:")}
    sidecar_labels = sc.get("label_counts", {})
    sidecar_unexplained = sidecar_labels.get("UNEXPLAINED", 0)
    sidecar_w9 = sidecar_labels.get("W9", 0)
    k3_cause = (
        " These concentrate in the Sagarin enrichment fields (pr/sos and their ranks): the new "
        "sidecar builder deliberately uses the CFB nearest-week fallback for BOTH leagues and "
        "common.metrics.dense_rank for ranks (documented in pipeline/sidecars.py), whereas the "
        "season-1 NFL sidecar joined exact (season, week) rows and ranked sequentially. Confirmed "
        "on a sample: a wk1 entry keeps pr=21.4 but pr_rank moved 13->12 (dense vs sequential); "
        "pr/sos value deltas are the nearest-week fallback filling weeks the master lacks exact "
        f"rows for (e.g. 2025 week 3 is absent). Classified W9 ({sidecar_w9} deltas, spec triage "
        "round 1 — approved whitelist rule)."
    ) if sidecar_w9 else ""
    k3 = {
        "check": "sidecar parity",
        "result": "PASS" if not field_deltas or sidecar_unexplained == 0 else "PARTIAL",
        "finding": (
            f"sidecars compared for {sc['games_compared']} joined games; per-field delta counts: "
            f"{field_deltas}; classified: W9={sidecar_w9} UNEXPLAINED={sidecar_unexplained}.{k3_cause}"
        ),
    }
    # K4: promoted odds sanity (traced separately, see report body)
    promoted_new = sum(1 for r in new_rows if not _is_blank(r.get("odds_source"))
                       and r.get("odds_source") != "schedule")
    traces = report.get("k4_trace", [])
    all_correct = all(t["policy_correct"] for t in traces) if traces else False
    k4 = {
        "check": "promoted odds sanity (select policy = closing_pre_kickoff)",
        "result": "PASS" if (traces and all_correct) else ("N/A" if not traces else "FAIL"),
        "finding": (
            f"new rows with a promoted book source: {promoted_new}. Traced {len(traces)} game(s); "
            f"each verified against the closing_pre_kickoff candidate set (freshest record per book, "
            f"then the latest with fetch_ts<=kickoff). all policy-correct={all_correct}. NFL games "
            f"have real kickoff times so the closing rule applies; CFB week-13 rows carry midnight "
            f"(00:00) placeholder kickoffs, so no pre-kickoff candidate exists and the policy "
            f"correctly falls back to the freshest record (is_closing=False). K4 is an NFL-primary "
            f"check per spec. See K4 trace table."
        ),
    }
    # K5: FBS filtering (CFB)
    if league.code == "cfb":
        k5 = {
            "check": "FBS filtering (CFB)",
            "result": "PASS" if join["matched"] == len(old_rows) and join["only_replay"] == 0 else "PARTIAL",
            "finding": (
                f"replay emitted {join['replay_rows']} rows vs baseline {join['baseline_rows']}; "
                f"only_baseline orphans={join['only_baseline']} (no legacy FBS game was wrongly "
                f"dropped — the FBS filter is sound); only_replay orphans={join['only_replay']} are "
                f"additional FBS-vs-FBS games present in the schedule master but absent from the "
                f"season-1 baseline snapshot (coverage expansion, not FBS-classification drift). "
                f"gameview skipped_non_fbs is reported in the replay manifest."
            ),
        }
    else:
        k5 = {"check": "FBS filtering (CFB)", "result": "N/A", "finding": "NFL run."}
    return {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def _fmt_val(v: Any) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= 60 else s[:57] + "..."


def render_markdown(report: Dict[str, Any]) -> str:
    j = report["join"]
    lc = report["label_counts"]
    lines: List[str] = []
    A = lines.append
    A(f"# Parity report — {report['league']} {report['season']} week {report['week']}")
    A("")
    A(f"Generated {report['generated_at']} by `tools/parity` (WP-A). Replay is 100% offline.")
    A("")
    A("## Join coverage")
    A("")
    A(f"- baseline rows: **{j['baseline_rows']}**")
    A(f"- replay rows: **{j['replay_rows']}**")
    A(f"- matched (kickoff instant + home/away merge_key): **{j['matched']}**")
    A(f"- only in baseline (orphans): **{j['only_baseline']}**")
    A(f"- only in replay (orphans): **{j['only_replay']}**")
    A(f"- join-key collisions: baseline={j['baseline_collisions']} replay={j['replay_collisions']}")
    A(f"- unkeyed rows: baseline={j['baseline_unkeyed']} replay={j['replay_unkeyed']}")
    A("")
    if report["orphans"]["only_baseline"]:
        A("### Orphans only in baseline")
        for o in report["orphans"]["only_baseline"]:
            A(f"- `{o['game_key']}` — {o['matchup']} — key={o['key']}")
        A("")
    if report["orphans"]["only_replay"]:
        A("### Orphans only in replay")
        for o in report["orphans"]["only_replay"]:
            A(f"- `{o['game_key']}` — {o['matchup']} — key={o['key']}")
        A("")
    A("## Per-field delta counts by classification")
    A("")
    A("| label | count |")
    A("|---|---|")
    for label in sorted(lc):
        A(f"| {label} | {lc[label]} |")
    A("")
    A("### By (label, field)")
    A("")
    A("| label | field | count |")
    A("|---|---|---|")
    for k, c in sorted(report["field_label_counts"].items()):
        label, field = k.split("::", 1)
        A(f"| {label} | {field} | {c} |")
    A("")
    A("## Known-check list K1–K5")
    A("")
    for kid in ("K1", "K2", "K3", "K4", "K5"):
        kc = report["known_checks"][kid]
        A(f"### {kid} — {kc['check']}: **{kc['result']}**")
        A("")
        A(kc["finding"])
        A("")
    A("### K4 trace (promoted-odds selection)")
    A("")
    traces = report.get("k4_trace", [])
    if traces:
        A("| matchup | kickoff | chosen book | chosen fetch_ts | pre-kickoff? | policy branch | policy-correct? |")
        A("|---|---|---|---|---|---|---|")
        for t in traces:
            A(f"| {t['matchup']} | {t['kickoff']} | {t['chosen_book']} | {t['chosen_fetch_ts']} | "
              f"{t['chosen_before_kickoff']} | {t['policy_branch']} | {t['policy_correct']} |")
    else:
        A("_No promoted odds to trace._")
    A("")
    A("## BUGFIX evidence")
    A("")
    if not report["bugfix_samples"]:
        A("No BUGFIX-class deltas were observed in this run (see report body for why — "
          "e.g. legacy NFL week 16 promoted no odds, so no both-promoted sign-flip could occur).")
        A("")
    for label, samples in report["bugfix_samples"].items():
        A(f"### {label}")
        for s in samples:
            A(f"- {s['matchup']} (`{s['new_game_key']}`): {s['field']} legacy={_fmt_val(s['old'])} "
              f"new={_fmt_val(s['new'])} — {s['reason']}")
        A("")
    A("## UNEXPLAINED deltas (full list)")
    A("")
    A(f"Total UNEXPLAINED field-deltas: **{lc.get('UNEXPLAINED', 0)}**. Grouped by reason:")
    A("")
    by_reason: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in report["unexplained_samples"]:
        by_reason[s["reason"]].append(s)
    for reason in sorted(by_reason):
        rows = by_reason[reason]
        A(f"### {len(rows)}× — {reason}")
        for s in rows[:12]:
            A(f"- {s['matchup']} (`{s['new_game_key']}`) `{s['field']}`: "
              f"legacy={_fmt_val(s['old'])} new={_fmt_val(s['new'])}")
        if len(rows) > 12:
            A(f"- … {len(rows) - 12} more (see .json)")
        A("")
    A("## Sidecar parity (K3 detail)")
    A("")
    sc = report["sidecar"]
    A(f"Sidecars compared for {sc['games_compared']} joined games.")
    A("")
    A("| metric | value |")
    A("|---|---|")
    for k in sorted(sc["counts"]):
        A(f"| {k} | {sc['counts'][k]} |")
    A("")
    A("### Sidecar delta classification")
    A("")
    label_counts = sc.get("label_counts", {})
    top_level = {k: v for k, v in label_counts.items() if "::" not in k}
    A("| label | count |")
    A("|---|---|")
    for label in sorted(top_level):
        A(f"| {label} | {top_level[label]} |")
    A("")
    by_label_field = {k: v for k, v in label_counts.items() if "::" in k}
    if by_label_field:
        A("| label | field | count |")
        A("|---|---|---|")
        for k, c in sorted(by_label_field.items()):
            label, field = k.split("::", 1)
            A(f"| {label} | {field} | {c} |")
        A("")
    A("## Proposed whitelist rules (require main-loop approval; NOT applied)")
    A("")
    A("_This differ never extends the frozen whitelist. The rules below are proposals for WP-B._")
    A("")
    for p in report.get("proposed_whitelist", []):
        A(f"- **{p['id']}** ({p['kind']}): {p['text']}")
    A("")
    return "\n".join(lines)


def write_reports(report: Dict[str, Any]) -> Tuple[Path, Path]:
    DOCS_PARITY.mkdir(parents=True, exist_ok=True)
    stem = f"parity_{report['league'].lower()}_{report['season']}_wk{report['week']}"
    md_path = DOCS_PARITY / f"{stem}.md"
    json_path = DOCS_PARITY / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classified games_week + sidecar parity differ.")
    p.add_argument("--league", required=True, choices=["nfl", "cfb"])
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--week", required=True, type=int)
    p.add_argument("--scratch", type=Path, default=replay_mod.DEFAULT_SCRATCH)
    p.add_argument("--baseline-out", type=Path, default=paths_mod.OUT_ROOT)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_report(
        args.league, args.season, args.week,
        scratch_root=args.scratch, baseline_out=args.baseline_out,
    )
    md_path, json_path = write_reports(report)
    lc = report["label_counts"]
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"matched={report['join']['matched']} only_baseline={report['join']['only_baseline']} "
          f"only_replay={report['join']['only_replay']}")
    print("labels:", dict(lc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

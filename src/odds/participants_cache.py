"""
Participants cache and provider mapping helpers for Odds API lookups.

Purpose:
    Persist the provider's canonical participant list per league and expose
    deterministic mappings from our team labels to those provider names.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - Cache is refreshed per run; optional disk snapshot avoids repeated calls.
    - Provider canonical records preserve full_name + id exactly as returned.
Side effects:
    - Writes snapshot under out/cache/participants/{league}.json when fetched.
Do not:
    - Invent aliases or fuzzy matches; mapping is deterministic per week.
Log contract:
    - Single `ATSDBG(PARTICIPANTS)` line the first time a league is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src.common.io_utils import ensure_out_dir
from src.common.team_names import team_merge_key
from src.common.team_names_cfb import normalize_team_name_cfb_odds, team_merge_key_cfb
from src.odds.odds_api_client import get_participants as _api_get_participants

_PARTICIPANT_LIST: Dict[str, List[Dict[str, str]]] = {}
_PROVIDER_INDEX: Dict[str, Dict[str, Set[str]]] = {}
_PROVIDER_MAP: Dict[str, Dict[str, str]] = {}
_PROVIDER_AMBIGUOUS: Dict[str, Set[str]] = {}
_PROVIDER_UNKNOWN: Dict[str, Set[str]] = {}
_EMPTY_NOTIFIED: set[str] = set()

_CACHE_DIR = ensure_out_dir() / "cache" / "participants"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(league: str) -> Path:
    return _CACHE_DIR / f"{league.lower()}.json"


def _reset_provider_state(league: str) -> None:
    key = league.lower()
    _PROVIDER_INDEX.pop(key, None)
    _PROVIDER_MAP.pop(key, None)
    _PROVIDER_AMBIGUOUS.pop(key, None)
    _PROVIDER_UNKNOWN.pop(key, None)


def _load_from_disk(league: str) -> Optional[List[Dict[str, str]]]:
    path = _cache_path(league)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list):
        return None
    records: List[Dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(
            entry.get("name")
            or entry.get("full_name")
            or entry.get("fullName")
            or entry.get("team")
            or ""
        ).strip()
        if not name:
            continue
        record: Dict[str, str] = {"name": name}
        participant_id = entry.get("id") or entry.get("participant_id") or entry.get("par_id")
        if isinstance(participant_id, str) and participant_id.strip():
            record["id"] = participant_id.strip()
        records.append(record)
    return records


def _save_to_disk(league: str, participants: List[Dict[str, str]]) -> None:
    path = _cache_path(league)
    try:
        path.write_text(json.dumps(participants, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def get_participants(league: str) -> Optional[List[Dict[str, str]]]:
    """Return the canonical participant records (name/id) for the league."""
    league_key = league.lower()
    cached = _PARTICIPANT_LIST.get(league_key)
    if cached and len(cached) >= 10:
        return cached

    participants = _load_from_disk(league_key)
    net_new = False
    if not participants or len(participants) < 10:
        participants = _api_get_participants(league_key)
        net_new = True
        _reset_provider_state(league_key)

    participants = participants or []
    _save_to_disk(league_key, participants)
    _PARTICIPANT_LIST[league_key] = participants

    if league_key not in _EMPTY_NOTIFIED:
        count = len(participants)
        sample = ", ".join(record.get("name", "") for record in participants[:3])
        freshness = "fresh" if net_new else "cache"
        print(
            f"ATSDBG(PARTICIPANTS): league={league_key} count={count} ({freshness}) sample=[{sample}]",
            flush=True,
        )
        _EMPTY_NOTIFIED.add(league_key)

    return participants


def _our_token(league: str, name: str | None) -> str:
    if not name:
        return ""
    if league.lower() == "cfb":
        return team_merge_key_cfb(name)
    return team_merge_key(name)


def _provider_token_internal(league: str, name: str | None) -> str:
    if not name:
        return ""
    if league.lower() == "cfb":
        return normalize_team_name_cfb_odds(name)
    return team_merge_key(name)


def _provider_index(league: str) -> Dict[str, Set[str]]:
    league_key = league.lower()
    if league_key in _PROVIDER_INDEX:
        return _PROVIDER_INDEX[league_key]
    records = _PARTICIPANT_LIST.get(league_key) or []
    index: Dict[str, Set[str]] = {}
    for record in records:
        token = _provider_token_internal(league_key, record.get("name"))
        if not token:
            continue
        index.setdefault(token, set()).add(record["name"])
    _PROVIDER_INDEX[league_key] = index
    return index


def build_provider_map(league: str, team_labels: Iterable[str]) -> Dict[str, int]:
    """
    Build a deterministic mapping from our team tokens to provider full names.

    Returns a summary dict with counts: total unique teams, mapped, ambiguous, unknown.
    """
    league_key = league.lower()
    participants = get_participants(league_key) or []
    if not participants:
        _PROVIDER_MAP[league_key] = {}
        _PROVIDER_AMBIGUOUS[league_key] = set()
        _PROVIDER_UNKNOWN[league_key] = set()
        return {"total": 0, "mapped": 0, "ambiguous": 0, "unknown": 0}

    index = _provider_index(league_key)
    mapped: Dict[str, str] = {}
    ambiguous: Set[str] = set()
    unknown: Set[str] = set()
    observed: Set[str] = set()

    for label in team_labels:
        token = _our_token(league_key, label)
        if not token:
            continue
        if token in observed:
            continue
        observed.add(token)
        options = index.get(token, set())
        if len(options) == 1:
            mapped[token] = next(iter(options))
        elif len(options) == 0:
            unknown.add(token)
        else:
            ambiguous.add(token)

    _PROVIDER_MAP[league_key] = mapped
    _PROVIDER_AMBIGUOUS[league_key] = ambiguous
    _PROVIDER_UNKNOWN[league_key] = unknown

    return {
        "total": len(observed),
        "mapped": len(mapped),
        "ambiguous": len(ambiguous),
        "unknown": len(unknown),
    }


def provider_name_for(league: str, team_label: str) -> tuple[Optional[str], str]:
    """Return the provider full_name for our team label, plus a status reason."""
    league_key = league.lower()
    mapping = _PROVIDER_MAP.get(league_key)
    if mapping is None:
        return None, "map_not_built"
    token = _our_token(league_key, team_label)
    if not token:
        return None, "no_token"
    if token in mapping:
        return mapping[token], "mapped"
    if token in _PROVIDER_AMBIGUOUS.get(league_key, set()):
        return None, "ambiguous_provider"
    if token in _PROVIDER_UNKNOWN.get(league_key, set()):
        return None, "no_provider_map"
    return None, "no_provider_map"


def provider_token(league: str, provider_name: str) -> str:
    """Return the provider token used for deterministic comparisons."""
    return _provider_token_internal(league, provider_name)


__all__ = ["get_participants", "build_provider_map", "provider_name_for", "provider_token"]

"""
Participants cache and canonical name matcher for Odds API lookups.

Purpose:
    Fetch and cache the participant list per league and map our internal team
    names to the provider's canonical labels.
Spec anchors:
    - /context/ats_api_backfill_spec.md
    - /context/global_week_and_provider_decoupling.md
Invariants:
    - Cache is refreshed per run; optional disk snapshot avoids repeated calls.
    - Matching relies on merge-key normalisation plus a small alias map.
Side effects:
    - Writes snapshot under out/cache/participants/{league}.json when fetched.
Do not:
    - Mutate caller-provided dictionaries.
Log contract:
    - API errors surface via odds_api_client logging; this module remains silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.common.io_utils import ensure_out_dir
from src.odds.odds_api_client import get_participants as _api_get_participants

_PARTICIPANT_LIST: Dict[str, List[Dict[str, str]]] = {}
_NORMALIZED_MAP: Dict[str, Dict[str, str]] = {}
_EMPTY_NOTIFIED: set[str] = set()

_CACHE_DIR = ensure_out_dir() / "cache" / "participants"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CFB_ALIAS_TOKENS: Dict[str, str] = {
    "utep": "texas_el_paso",
    "texaselpaso": "texas_el_paso",
    "texas_el_paso": "texas_el_paso",
    "louisiana_lafayette": "louisiana",
    "la_lafayette": "louisiana",
    "ole_miss": "mississippi",
    "uconn": "connecticut",
}


def _normalizer(league: str):
    from src.common.team_names import team_merge_key
    from src.common.team_names_cfb import team_merge_key_cfb

    return team_merge_key_cfb if league.lower() == "cfb" else team_merge_key


def _normalizer_for_league(league: str):
    league = (league or "").lower()
    if league == "cfb":
        from src.common.team_names_cfb import team_merge_key_cfb as norm
    else:
        from src.common.team_names import team_merge_key as norm
    return norm


def _cache_path(league: str) -> Path:
    return _CACHE_DIR / f"{league.lower()}.json"


def _load_from_disk(league: str) -> Optional[List[Dict[str, str]]]:
    path = _cache_path(league)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            normalized: List[Dict[str, str]] = []
            for entry in data:
                if isinstance(entry, dict):
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
                    normalized.append(record)
                elif isinstance(entry, str):
                    token = entry.strip()
                    if token:
                        normalized.append({"name": token})
            return normalized
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _save_to_disk(league: str, participants: List[Dict[str, str]]) -> None:
    path = _cache_path(league)
    try:
        path.write_text(json.dumps(participants, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def get_participants(league: str) -> Optional[List[Dict[str, str]]]:
    """Return the list of participant records (name/id) for the league (cached per run)."""
    league_key = league.lower()
    if league_key in _PARTICIPANT_LIST:
        return _PARTICIPANT_LIST[league_key]

    participants = _load_from_disk(league_key)
    net_new = False
    if participants is None:
        participants = _api_get_participants(league_key)
        net_new = True

    _save_to_disk(league_key, participants or [])
    _PARTICIPANT_LIST[league_key] = participants or []

    if league_key not in _EMPTY_NOTIFIED:
        count = len(_PARTICIPANT_LIST[league_key])
        sample_names = [entry.get("name", "") for entry in _PARTICIPANT_LIST[league_key][:3]]
        sample = ", ".join(sample_names)
        freshness = "fresh" if net_new else "cache"
        print(
            f"ATSDBG(PARTICIPANTS): league={league_key} count={count} ({freshness}) sample=[{sample}]",
            flush=True,
        )
        _EMPTY_NOTIFIED.add(league_key)

    if _PARTICIPANT_LIST[league_key]:
        _build_normalized_map(league_key, _PARTICIPANT_LIST[league_key])
    return _PARTICIPANT_LIST[league_key]


def _build_normalized_map(league: str, participants: List[Dict[str, str]]) -> None:
    normalizer = _normalizer(league)
    alias_map = _CFB_ALIAS_TOKENS if league == "cfb" else {}

    token_to_name: Dict[str, str] = {}
    for record in participants:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        token = normalizer(name)
        if not token:
            continue
        token_to_name.setdefault(token, name)
        for alias_token, canonical in alias_map.items():
            if canonical == token:
                token_to_name.setdefault(alias_token, name)

    _NORMALIZED_MAP[league] = token_to_name


def match_team_name(league: str, our_name: str) -> Optional[str]:
    """Return the provider participant name matching our internal team label."""
    league_key = league.lower()
    participants = get_participants(league_key)
    if not participants:
        norm = _normalizer_for_league(league_key)
        token = norm(our_name or "")
        return our_name if token else None

    normalizer = _normalizer(league_key)
    alias_map = _CFB_ALIAS_TOKENS if league_key == "cfb" else {}

    token = normalizer(our_name or "")
    if not token:
        return None
    token = alias_map.get(token, token)

    normalized_map = _NORMALIZED_MAP.get(league_key)
    if not normalized_map:
        _build_normalized_map(league_key, participants)
        normalized_map = _NORMALIZED_MAP.get(league_key, {})

    return normalized_map.get(token)


def canonical_equals(league: str, a: str, b: str) -> bool:
    """Return True when two labels refer to the same team (fallback aware)."""
    league_key = league.lower()
    norm = _normalizer_for_league(league_key)
    token_a = norm(a or "")
    token_b = norm(b or "")
    if not token_a or not token_b:
        return False

    participants = _PARTICIPANT_LIST.get(league_key)
    if participants:
        alias_map = _NORMALIZED_MAP.get(league_key)
        if not alias_map:
            _build_normalized_map(league_key, participants)
            alias_map = _NORMALIZED_MAP.get(league_key, {})
        return alias_map.get(token_a) is not None and alias_map.get(token_b) is not None and token_a == token_b

    return token_a == token_b


__all__ = ["get_participants", "match_team_name", "canonical_equals"]


__all__ = ["get_participants", "match_team_name"]

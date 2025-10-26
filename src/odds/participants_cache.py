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
_PARTICIPANT_CANONICALS: Dict[str, List[Dict[str, str]]] = {}
_ALIAS_TOKEN_CACHE: Dict[str, Dict[str, str]] = {}
_EMPTY_NOTIFIED: set[str] = set()

_CACHE_DIR = ensure_out_dir() / "cache" / "participants"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CFB_ALIAS_NAMES: Dict[str, str] = {
    "ole miss": "Mississippi Rebels",
    "olemiss": "Mississippi Rebels",
    "mississippi": "Mississippi Rebels",
    "utep": "UTEP Miners",
    "texas el paso": "UTEP Miners",
    "texaselpaso": "UTEP Miners",
    "louisiana lafayette": "Louisiana Ragin' Cajuns",
    "louisiana-lafayette": "Louisiana Ragin' Cajuns",
    "la lafayette": "Louisiana Ragin' Cajuns",
    "uconn": "Connecticut Huskies",
    "connecticut": "Connecticut Huskies",
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
    cached = _PARTICIPANT_LIST.get(league_key)
    if cached and len(cached) >= 10:
        return cached

    participants = _load_from_disk(league_key)
    net_new = False
    need_fetch = not participants or len(participants) < 10
    if need_fetch:
        participants = _api_get_participants(league_key)
        net_new = True
    else:
        participants = participants or []

    participants = participants or []
    _save_to_disk(league_key, participants)
    _PARTICIPANT_LIST[league_key] = participants

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
    alias_map = _alias_token_map(league)

    token_to_name: Dict[str, str] = {}
    canonical_records: List[Dict[str, str]] = []
    for record in participants or []:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        token = normalizer(name)
        if not token:
            continue
        token_to_name.setdefault(token, name)
        canonical_records.append(
            {
                "name": name,
                "token": token,
                "compact": _compact(name),
                "first": _first_token_compact(name),
            }
        )
    for alias_token, canonical_token in alias_map.items():
        target_name = token_to_name.get(canonical_token)
        if target_name:
            token_to_name.setdefault(alias_token, target_name)

    _NORMALIZED_MAP[league] = token_to_name
    _PARTICIPANT_CANONICALS[league] = canonical_records


def match_team_name(league: str, our_name: str) -> Optional[str]:
    """Return the provider participant name matching our internal team label."""
    league_key = league.lower()
    participants = get_participants(league_key)
    if not participants:
        norm = _normalizer_for_league(league_key)
        token = norm(our_name or "")
        return our_name if token else None

    normalizer = _normalizer(league_key)
    alias_map = _alias_token_map(league_key)

    token = normalizer(our_name or "")
    if not token:
        return None
    token = alias_map.get(token, token)

    normalized_map = _NORMALIZED_MAP.get(league_key)
    if not normalized_map:
        _build_normalized_map(league_key, participants)
        normalized_map = _NORMALIZED_MAP.get(league_key, {})

    provider = normalized_map.get(token)
    if provider:
        return provider

    records = _PARTICIPANT_CANONICALS.get(league_key) or []
    compact_ours = _compact(our_name or "")
    for item in records:
        prov_token = item["token"]
        if prov_token.startswith(token) or token.startswith(prov_token):
            return item["name"]
        if compact_ours:
            prov_compact = item["compact"]
            if prov_compact.startswith(compact_ours) or compact_ours.startswith(prov_compact):
                return item["name"]
            first = item.get("first")
            if first and (first == compact_ours or first == token or token.startswith(first)):
                return item["name"]
    return None


def canonical_equals(league: str, a: str, b: str) -> bool:
    """Return True when two labels refer to the same team (fallback aware)."""
    league_key = league.lower()
    norm = _normalizer_for_league(league_key)
    alias_tokens = _alias_token_map(league_key)

    token_a = norm(a or "")
    token_b = norm(b or "")
    if token_a:
        token_a = alias_tokens.get(token_a, token_a)
    if token_b:
        token_b = alias_tokens.get(token_b, token_b)

    if token_a and token_b and token_a == token_b:
        return True

    participants = _PARTICIPANT_LIST.get(league_key)
    if participants:
        if not _NORMALIZED_MAP.get(league_key):
            _build_normalized_map(league_key, participants)
        normalized_map = _NORMALIZED_MAP.get(league_key, {})
        if token_a in normalized_map and token_b in normalized_map:
            if normalized_map[token_a] == normalized_map[token_b]:
                return True
        if token_a and token_b and (token_a.startswith(token_b) or token_b.startswith(token_a)):
            return True
        compact_a = _compact(a or "")
        compact_b = _compact(b or "")
        if compact_a and compact_b and (
            compact_a == compact_b or compact_a.startswith(compact_b) or compact_b.startswith(compact_a)
        ):
            return True
        first_a = _first_token_compact(normalized_map.get(token_a, "") or a or "")
        first_b = _first_token_compact(normalized_map.get(token_b, "") or b or "")
        if first_a and first_b and first_a == first_b:
            return True
        return False

    # Participants unavailable: rely on tokens/compacts only.
    if token_a and token_b and (token_a.startswith(token_b) or token_b.startswith(token_a)):
        return True
    compact_a = _compact(a or "")
    compact_b = _compact(b or "")
    if compact_a and compact_b and (
        compact_a == compact_b or compact_a.startswith(compact_b) or compact_b.startswith(compact_a)
    ):
        return True
    return False


def _alias_token_map(league: str) -> Dict[str, str]:
    league_key = league.lower()
    if league_key in _ALIAS_TOKEN_CACHE:
        return _ALIAS_TOKEN_CACHE[league_key]
    normalizer = _normalizer(league_key)
    raw_map = _CFB_ALIAS_NAMES if league_key == "cfb" else {}
    token_map: Dict[str, str] = {}
    for raw_alias, target in raw_map.items():
        alias_token = normalizer(raw_alias)
        target_token = normalizer(target)
        if alias_token and target_token:
            token_map[alias_token] = target_token
    _ALIAS_TOKEN_CACHE[league_key] = token_map
    return token_map


def _compact(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _first_token_compact(value: str) -> str:
    if not value:
        return ""
    words = [
        word
        for word in "".join(ch if ch.isalnum() else " " for ch in value.lower()).split()
        if word
    ]
    for word in words:
        if word in {"the", "university", "college"}:
            continue
        return word
    return words[0] if words else ""


__all__ = ["get_participants", "match_team_name", "canonical_equals"]

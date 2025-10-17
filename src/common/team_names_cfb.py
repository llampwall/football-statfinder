"""Minimal College Football team name normalization utilities.

Purpose:
    Provide placeholder normalization helpers so the CFB stubs can emit
    consistent keys that mirror the NFL pipeline APIs.
"""

from __future__ import annotations

import re
from typing import Dict

_ALIAS_MAP: Dict[str, str] = {
    "alabama": "Alabama Crimson Tide",
    "clemson": "Clemson Tigers",
    "georgia": "Georgia Bulldogs",
    "notre dame": "Notre Dame Fighting Irish",
    "ohio state": "Ohio State Buckeyes",
    "usc": "USC Trojans",
    "texas": "Texas Longhorns",
    "oregon": "Oregon Ducks",
}


def _clean(label: str) -> str:
    text = label.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_team_name_cfb(name: str | None) -> str:
    """Return a display name for the provided label."""
    if not name:
        return ""
    cleaned = _clean(str(name))
    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]
    words = [word.capitalize() for word in cleaned.split(" ")]
    return " ".join(words)


def team_merge_key_cfb(name: str | None) -> str:
    """Return a simple merge key (lowercase, alphanumeric)."""
    display = normalize_team_name_cfb(name)
    basis = display or (name or "")
    basis = basis.lower()
    basis = basis.replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", basis)


__all__ = ["normalize_team_name_cfb", "team_merge_key_cfb"]

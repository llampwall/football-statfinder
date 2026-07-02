"""Football team name normalization utilities.

Purpose:
    Provide a single source of truth for team display names and merge keys
    so every module (games, odds, metrics) resolves franchises identically.
Inputs:
    Raw team labels from public feeds (abbr, city-only, nicknames).
Outputs:
    Canonical NFL team display names (City + Nickname) and merge-safe keys.
Source(s) of truth:
    nflverse team nomenclature for display; internal synonym tables for merges.
Example:
    >>> normalize_team_display("LA Rams")
    'Los Angeles Rams'
    >>> team_merge_key("ny jets")
    'newyorkjets'
"""

from __future__ import annotations

import re
from typing import Dict

# Canonical nflverse display names keyed by abbreviation.
TEAM_ABBR_TO_FULL: Dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LVR": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
}

CANONICAL_NAMES = set(TEAM_ABBR_TO_FULL.values())

# City-only labels that have a single unambiguous franchise.
_CITY_TO_FULL = {
    "arizona": "Arizona Cardinals",
    "atlanta": "Atlanta Falcons",
    "baltimore": "Baltimore Ravens",
    "buffalo": "Buffalo Bills",
    "carolina": "Carolina Panthers",
    "chicago": "Chicago Bears",
    "cincinnati": "Cincinnati Bengals",
    "cleveland": "Cleveland Browns",
    "dallas": "Dallas Cowboys",
    "denver": "Denver Broncos",
    "detroit": "Detroit Lions",
    "green bay": "Green Bay Packers",
    "houston": "Houston Texans",
    "indianapolis": "Indianapolis Colts",
    "jacksonville": "Jacksonville Jaguars",
    "kansas city": "Kansas City Chiefs",
    "miami": "Miami Dolphins",
    "minnesota": "Minnesota Vikings",
    "new england": "New England Patriots",
    "new orleans": "New Orleans Saints",
    "philadelphia": "Philadelphia Eagles",
    "pittsburgh": "Pittsburgh Steelers",
    "san francisco": "San Francisco 49ers",
    "seattle": "Seattle Seahawks",
    "tampa bay": "Tampa Bay Buccaneers",
    "tennessee": "Tennessee Titans",
    "washington": "Washington Commanders",
    # Los Angeles defaults to Rams for legacy partners; Chargers must be explicit.
    "la": "Los Angeles Rams",
    "los angeles": "Los Angeles Rams",
    "saint louis": "Los Angeles Rams",
    "st louis": "Los Angeles Rams",
    "st. louis": "Los Angeles Rams",
    "oakland": "Las Vegas Raiders",
    "las vegas": "Las Vegas Raiders",
    "ny": "New York Giants",
    "new york": "New York Giants",
}

# Synonym table seeded from prior debugging sessions.
_RAW_SYNONYMS = {
    "los angeles rams": "Los Angeles Rams",
    "l a rams": "Los Angeles Rams",
    "la rams": "Los Angeles Rams",
    "l.a. rams": "Los Angeles Rams",
    "l.a rams": "Los Angeles Rams",
    "la rams": "Los Angeles Rams",  # NBSP variant
    "rams": "Los Angeles Rams",
    "st louis rams": "Los Angeles Rams",
    "saint louis rams": "Los Angeles Rams",
    "los angeles chargers": "Los Angeles Chargers",
    "la chargers": "Los Angeles Chargers",
    "l.a. chargers": "Los Angeles Chargers",
    "chargers": "Los Angeles Chargers",
    "los angeles raiders": "Las Vegas Raiders",
    "oakland raiders": "Las Vegas Raiders",
    "raiders": "Las Vegas Raiders",
    "washington football team": "Washington Commanders",
    "washington redskins": "Washington Commanders",
    "redskins": "Washington Commanders",
    "football team": "Washington Commanders",
    "commanders": "Washington Commanders",
    "ny giants": "New York Giants",
    "ny jets": "New York Jets",
    "giants": "New York Giants",
    "jets": "New York Jets",
    "bucs": "Tampa Bay Buccaneers",
    "buccaneers": "Tampa Bay Buccaneers",
    "saints": "New Orleans Saints",
    "niners": "San Francisco 49ers",
    "sf 49ers": "San Francisco 49ers",
    "49ers": "San Francisco 49ers",
    "san fran 49ers": "San Francisco 49ers",
    "patriots": "New England Patriots",
    "steelers": "Pittsburgh Steelers",
    "bears": "Chicago Bears",
    "bengals": "Cincinnati Bengals",
    "chiefs": "Kansas City Chiefs",
    "bills": "Buffalo Bills",
    "lions": "Detroit Lions",
    "cowboys": "Dallas Cowboys",
    "broncos": "Denver Broncos",
    "packers": "Green Bay Packers",
    "vikings": "Minnesota Vikings",
    "colts": "Indianapolis Colts",
    "texans": "Houston Texans",
    "titans": "Tennessee Titans",
    "falcons": "Atlanta Falcons",
    "panthers": "Carolina Panthers",
    "browns": "Cleveland Browns",
    "ravens": "Baltimore Ravens",
    "dolphins": "Miami Dolphins",
    "jaguars": "Jacksonville Jaguars",
    "jets": "New York Jets",
    "cardinals": "Arizona Cardinals",
    "chargers": "Los Angeles Chargers",
    "seahawks": "Seattle Seahawks",
    "eagles": "Philadelphia Eagles",
    "bucaneers": "Tampa Bay Buccaneers",
    "buccs": "Tampa Bay Buccaneers",
}


def _clean_for_lookup(name: str) -> str:
    """Normalize a label for dictionary lookups."""
    cleaned = (
        name.replace("\u00a0", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace(",", " ")
        .lower()
    )
    cleaned = cleaned.replace("&", "and")
    cleaned = re.sub(r"saint ", "saint ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _canonicalize_display(name: str) -> str:
    """Resolve a raw name to canonical display or empty string."""
    if not name:
        return ""

    trimmed = name.strip()
    # Abbreviation direct hit
    upper = trimmed.upper()
    if upper in TEAM_ABBR_TO_FULL:
        return TEAM_ABBR_TO_FULL[upper]

    # Direct canonical match ignoring case.
    for canonical in CANONICAL_NAMES:
        if trimmed.lower() == canonical.lower():
            return canonical

    cleaned = _clean_for_lookup(trimmed)
    # Synonym table overrides.
    if cleaned in _RAW_SYNONYMS:
        return _RAW_SYNONYMS[cleaned]

    # City-only fallback.
    if cleaned in _CITY_TO_FULL:
        return _CITY_TO_FULL[cleaned]

    return ""


def normalize_team_display(name: str) -> str:
    """Return the canonical nflverse display name for a raw team label."""
    resolved = _canonicalize_display(name)
    return resolved or name.strip() if isinstance(name, str) else ""


def team_merge_key(name: str) -> str:
    """Return a stable lowercase merge key for team joins across sources."""
    display = normalize_team_display(name)
    basis = display if display and display in CANONICAL_NAMES else (name or "")
    basis = str(basis)
    basis = basis.replace("\u00a0", " ")
    basis = basis.lower()
    basis = basis.replace("&", "and")
    basis = re.sub(r"\s+", " ", basis).strip()
    basis = _RAW_SYNONYMS.get(basis, basis)
    if basis in _CITY_TO_FULL:
        basis = _CITY_TO_FULL[basis]
    if basis.upper() in TEAM_ABBR_TO_FULL:
        basis = TEAM_ABBR_TO_FULL[basis.upper()]
    # Remove non-alphanumeric to match legacy merges.
    return re.sub(r"[^a-z0-9]", "", basis.lower())


__all__ = ["TEAM_ABBR_TO_FULL", "normalize_team_display", "team_merge_key"]

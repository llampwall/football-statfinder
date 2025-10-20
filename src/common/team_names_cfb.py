"""Minimal College Football team name normalization utilities."""

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


def normalize_team_name_cfb_stats(name: str | None) -> str:
    """Normalize team labels from CFBD stats sources."""
    return normalize_team_name_cfb(name)


ODDS_TEXT_ALIASES = {
    # collapsed/book variants → canonical program label
    "umass": "massachusetts",
    "buffalo": "buffalo",
    "appalachianstate": "app state",
    "coastalcarolina": "coastal carolina",
    "indianahoosiers": "indiana",
    "michiganstate": "michigan state",
    "louisiana": "louisiana",
    "southernmississippi": "southern miss",
    "virginiacavaliers": "virginia",
    "washingtonstate": "washington state",
    "uclabruins": "ucla",
    "marylandterrapins": "maryland",
    "alabamacrimsontide": "alabama",
    "tennesseevolunteers": "tennessee",
    "syracuseorange": "syracuse",
    "pittsburgh": "pittsburgh",
    "byu": "byu",
    "utahutes": "utah",
    "newmexicolobos": "new mexico",
    "nevadawolfpack": "nevada",
    "oregonstatebeavers": "oregon state",
    "lafayetteleopards": "lafayette",
    "stanfordcardinal": "stanford",
    "floridastate": "florida state",

    "miami (oh) redhawks": "miami (oh)",
    "liberty flames": "liberty",
    "brigham young cougars": "byu",
    "byu cougars": "byu",
    "central florida golden knights": "ucf",
    "central florida knights": "ucf",
    "utsa roadrunners": "utsa",
    "texas san antonio roadrunners": "utsa",
    "texas a&m aggies": "texas a&m",
    "southern cal": "usc",
    "usc trojans": "usc",
    "unlv rebels": "unlv",
    "ole miss rebels": "ole miss",
    "southern miss golden eagles": "southern miss",
    "louisiana lafayette": "louisiana",
    "louisiana ragin cajuns": "louisiana",
    "vanderbilt commodores": "vanderbilt",
    "charlotte 49ers": "charlotte",
    "south carolina gamecocks": "south carolina",
    "south florida bulls": "south florida",
    "kent state golden flashes": "kent state",
    "texas tech red raiders": "texas tech",
    "pittsburgh panthers": "pittsburgh",
    "western kentucky hilltoppers": "western kentucky",
    "boise state broncos": "boise state",
    "louisiana monroe warhawks": "louisiana monroe",
    "michigan state spartans": "michigan state",
    "wisconsin badgers": "wisconsin",
    "penn state nittany lions": "penn state",
    "arizona state sun devils": "arizona state",
    "north texas mean green": "north texas",
    "rutgers scarlet knights": "rutgers",
    "colorado state rams": "colorado state",
    "hawaii rainbow warriors": "hawaii",
    "southern mississippi": "southern miss",
}

MASCOT_WORDS = {
    "aggies",
    "bluehens",
    "broncos",
    "bruins",
    "cardinal",
    "cavaliers",
    "hilltoppers",
    "hokies",
    "jaguars",
    "leopards",
    "badgers",
    "bears",
    "boilermakers",
    "buckeyes",
    "bulls",
    "chanticleers",
    "chippewas",
    "buffaloes",
    "crimson",
    "dukes",
    "hawks",
    "hawkeyes",
    "devils",
    "jackets",
    "gophers",
    "cornhuskers",
    "wave",
    "knights",
    "frogs",
    "rebels",
    "trojans",
    "bulldogs",
    "tigers",
    "gators",
    "cougars",
    "cardinals",
    "hurricanes",
    "eagles",
    "falcons",
    "warhawks",
    "rockets",
    "minutemen",
    "mountaineers",
    "cowboys",
    "sooners",
    "spartans",
    "wolverines",
    "huskies",
    "panthers",
    "wildcats",
    "razorbacks",
    "bearcats",
    "ducks",
    "lions",
    "herd",
    "bobcats",
    "zips",
    "seminoles",
    "longhorns",
    "blazers",
    "mustangs",
    "monarchs",
    "rams",
    "gamecocks",
    "cyclones",
    "jayhawks",
    "owls",
    "warriors",
    "tide",
    "orange",
    "lobos",
    "tarheels",
    "volunteers",
    "utes",
    "wolfpack",
    "terrapins",
}

MULTI_WORD_MASCOTS = {
    "blue devils",
    "yellow jackets",
    "horned frogs",
    "green wave",
    "black knights",
    "golden gophers",
    "golden eagles",
    "golden bears",
    "golden flashes",
    "thundering herd",
    "sun devils",
    "mean green",
    "rainbow warriors",
    "scarlet knights",
    "nittany lions",
    "crimson tide",
    "fighting illini",
    "blue hens",
    "blue raiders",
    "tar heels",
    "demon deacons",
}


def normalize_team_name_cfb_odds(name: str) -> str:
    base = normalize_team_name_cfb(name)
    lowered = _clean(base)
    alias = ODDS_TEXT_ALIASES.get(lowered)
    if alias is None:
        alias = ODDS_TEXT_ALIASES.get(lowered.replace(" ", ""))
    if alias:
        lowered = _clean(alias)
    for mascot in MULTI_WORD_MASCOTS:
        if lowered.endswith(mascot) and len(lowered) > len(mascot):
            lowered = lowered[: -len(mascot)].strip()
            break
    words = lowered.split()
    while len(words) > 1 and words[-1] in MASCOT_WORDS:
        words.pop()
    token = team_merge_key_cfb(" ".join(words))
    return token


__all__ = [
    "normalize_team_name_cfb",
    "team_merge_key_cfb",
    "normalize_team_name_cfb_odds",
    "normalize_team_name_cfb_stats",
]

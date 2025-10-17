"""Helper utilities to fetch CFB schedules and team data from external sources."""

from __future__ import annotations

from typing import List, Optional

import requests

CFBD_BASE_URL = "https://api.collegefootballdata.com"


def fetch_cfbd_games(season: int, week: Optional[int], api_key: str) -> Optional[List[dict]]:
    """Return the raw CFBD games payload for the requested season/week."""
    url = f"{CFBD_BASE_URL}/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "year": season,
        "seasonType": "regular",
    }
    if week is not None:
        params["week"] = week
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return None



def fetch_cfbd_fbs_teams(season: int, api_key: str) -> Optional[List[dict]]:
    """Return the list of FBS teams for the provided season."""
    url = f"{CFBD_BASE_URL}/teams/fbs"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"year": season}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return None


def fetch_cfbd_team_game_stats(
    season: int,
    week: Optional[int],
    api_key: str,
    season_type: str = "regular",
) -> Optional[List[dict]]:
    """Return the CFBD team game stats payload."""
    url = f"{CFBD_BASE_URL}/games/teams"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "year": season,
        "seasonType": season_type,
    }
    if week is not None:
        params["week"] = week
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return None


__all__ = ["fetch_cfbd_games", "fetch_cfbd_team_game_stats", "fetch_cfbd_fbs_teams"]

"""Helper utilities to fetch CFB schedules from external sources."""

from __future__ import annotations

from typing import List, Optional

import requests

CFBD_BASE_URL = "https://api.collegefootballdata.com"


def fetch_cfbd_week_games(season: int, week: int, api_key: str) -> Optional[List[dict]]:
    """Return the raw CFBD games payload for the requested week."""
    url = f"{CFBD_BASE_URL}/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "year": season,
        "week": week,
        "seasonType": "regular",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return None


__all__ = ["fetch_cfbd_week_games"]

"""The one game_key builder.

Season 1 recomputed game keys from slugs in three places with two different
team orders (REBUILD.md section 6). This module is now the only place a
game_key is ever constructed. The format is a frozen frontend contract —
sidecar filenames and the ``?game_key=`` URL parameter depend on it:

    NFL: {YYYYMMDD_HHMM}_{home_slug}_{away_slug}
    CFB: {YYYYMMDD_HHMM}_{away_slug}_{home_slug}

Timestamps are the kickoff in UTC. Slugs are lowercase alphanumeric tokens
joined by single underscores (``_slug`` here matches the season-1 pin/master
slug: ``&`` becomes ``and``, every other non-alphanumeric run collapses to one
underscore).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; keeps this module import-cycle-free
    from ..leagues import League


def slug(name: str) -> str:
    """Lowercase snake-case slug of a normalized team display name."""
    cleaned = (name or "").lower().replace("&", "and")
    out: list[str] = []
    for char in cleaned:
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def kickoff_stamp(kickoff_utc: datetime) -> str:
    """``YYYYMMDD_HHMM`` stamp of a kickoff datetime, coerced to UTC."""
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
    return kickoff_utc.astimezone(timezone.utc).strftime("%Y%m%d_%H%M")


def build_game_key(league: "League", kickoff_utc: datetime, home_norm: str, away_norm: str) -> str:
    """Canonical game_key for a game, honoring the league's frozen team order."""
    stamp = kickoff_stamp(kickoff_utc)
    home = slug(home_norm)
    away = slug(away_norm)
    if league.game_key_home_first:
        return f"{stamp}_{home}_{away}"
    return f"{stamp}_{away}_{home}"


__all__ = ["build_game_key", "kickoff_stamp", "slug"]

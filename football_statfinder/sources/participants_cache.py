"""Participants cache and provider-name mapping for Odds API lookups.

Port of the harvested ``src/odds/participants_cache.py``. Logic preserved;
adaptations:

* Module-level global dicts become instance state on :class:`ParticipantsCache`
  (one instance per league per run), so nothing leaks across runs or tests.
* The disk snapshot moves from ``out/cache/participants/{league}.json`` to
  ``paths.participants_cache_path`` (``out/staging/participants/``).
* The provider fetch goes through an injected :class:`OddsApiClient`, which
  already honors ``settings.odds.cache_only`` (no paid call when set).
* ``build_provider_map`` is now *additive*: repeated calls extend the mapping
  with unseen team labels instead of replacing it, so callers can feed labels
  incrementally (the legacy orchestrator always passed the full week at once).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .. import paths
from ..common import team_names_cfb
from ..leagues import League
from .odds_api import OddsApiClient

logger = logging.getLogger(__name__)

_MANUAL_PROVIDER_OVERRIDES: Dict[str, Dict[str, str]] = {
    "cfb": {
        "arkansasstate": "Arkansas State Red Wolves",
        "samhouston": "Sam Houston Bearkats",
        "samhoustonstate": "Sam Houston Bearkats",
        "tulsa": "Tulsa Golden Hurricane",
        "eastcarolina": "East Carolina Pirates",
        "sanjosestate": "San Jose State Spartans",
        "sanjosstate": "San Jose State Spartans",
        "massachusetts": "Massachusetts Minutemen",
        "umass": "Massachusetts Minutemen",
        "appalachianstate": "Appalachian State Mountaineers",
        "appstate": "Appalachian State Mountaineers",
        "southernmiss": "Southern Mississippi Golden Eagles",
        "southernmississippi": "Southern Mississippi Golden Eagles",
        "kennesawstate": "Kennesaw State Owls",
        "utep": "UTEP Miners",
        "texasstate": "Texas State Bobcats",
    }
}


def provider_token(league: League, name: Optional[str]) -> str:
    """Deterministic token for comparing provider participant names.

    Mirrors the legacy pairing exactly: CFB uses the odds-label normalizer
    (which already returns a merge token); NFL uses the merge key.
    """
    if not name:
        return ""
    if league.code == "cfb":
        return team_names_cfb.normalize_team_name_cfb_odds(name)
    return league.merge_key(name)


class ParticipantsCache:
    """Per-league provider participant list plus our-label -> provider mapping."""

    def __init__(
        self,
        league: League,
        client: OddsApiClient,
        *,
        out_root: Optional[Path] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self._league = league
        self._client = client
        self._cache_path = cache_path or paths.participants_cache_path(
            league.code, out_root=out_root
        )
        self._participants: Optional[List[Dict[str, str]]] = None
        self._provider_index: Optional[Dict[str, Set[str]]] = None
        self._mapped: Dict[str, str] = {}
        self._ambiguous: Set[str] = set()
        self._unknown: Set[str] = set()
        self._observed: Set[str] = set()
        self._map_built = False
        self._logged_load = False

    # -- participant list -----------------------------------------------------

    def _load_from_disk(self) -> Optional[List[Dict[str, str]]]:
        path = self._cache_path
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

    def _save_to_disk(self, participants: List[Dict[str, str]]) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(participants, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("could not persist participants cache: %s", self._cache_path)

    def get_participants(self) -> List[Dict[str, str]]:
        """Return the canonical participant records (name/id) for the league."""
        if self._participants is not None and len(self._participants) >= 10:
            return self._participants

        participants = self._load_from_disk()
        net_new = False
        if not participants or len(participants) < 10:
            participants = self._client.get_participants()
            net_new = True
            self._provider_index = None

        participants = participants or []
        self._save_to_disk(participants)
        self._participants = participants

        if not self._logged_load:
            sample = ", ".join(record.get("name", "") for record in participants[:3])
            logger.info(
                "participants: league=%s count=%d (%s) sample=[%s]",
                self._league.code,
                len(participants),
                "fresh" if net_new else "cache",
                sample,
            )
            self._logged_load = True
        return participants

    # -- provider mapping -------------------------------------------------------

    def _our_token(self, name: Optional[str]) -> str:
        if not name:
            return ""
        return self._league.merge_key(name)

    def _index(self) -> Dict[str, Set[str]]:
        if self._provider_index is not None:
            return self._provider_index
        index: Dict[str, Set[str]] = {}
        for record in self._participants or []:
            token = provider_token(self._league, record.get("name"))
            if not token:
                continue
            index.setdefault(token, set()).add(record["name"])
        self._provider_index = index
        return index

    def build_provider_map(self, team_labels: Iterable[str]) -> Dict[str, int]:
        """Extend the our-token -> provider-full-name mapping with new labels.

        Returns cumulative counts: total unique teams, mapped, ambiguous, unknown.
        """
        participants = self.get_participants()
        self._map_built = True
        if not participants:
            return {"total": len(self._observed), "mapped": 0, "ambiguous": 0, "unknown": 0}

        index = self._index()
        for label in team_labels:
            token = self._our_token(label)
            if not token or token in self._observed:
                continue
            self._observed.add(token)
            options = index.get(token, set())
            if len(options) == 1:
                self._mapped[token] = next(iter(options))
            elif len(options) == 0:
                self._unknown.add(token)
            else:
                self._ambiguous.add(token)

        overrides = _MANUAL_PROVIDER_OVERRIDES.get(self._league.code, {})
        for token, provider_name in overrides.items():
            if token in self._observed and token not in self._mapped:
                self._mapped[token] = provider_name
                self._ambiguous.discard(token)
                self._unknown.discard(token)

        return {
            "total": len(self._observed),
            "mapped": len(self._mapped),
            "ambiguous": len(self._ambiguous),
            "unknown": len(self._unknown),
        }

    def provider_name_for(self, team_label: str) -> tuple[Optional[str], str]:
        """Return the provider full_name for our team label, plus a status reason."""
        if not self._map_built:
            return None, "map_not_built"
        token = self._our_token(team_label)
        if not token:
            return None, "no_token"
        if token not in self._observed:
            # Additive behavior: map labels lazily on first sight.
            self.build_provider_map([team_label])
        if token in self._mapped:
            return self._mapped[token], "mapped"
        if token in self._ambiguous:
            return None, "ambiguous_provider"
        return None, "no_provider_map"


__all__ = ["ParticipantsCache", "provider_token"]

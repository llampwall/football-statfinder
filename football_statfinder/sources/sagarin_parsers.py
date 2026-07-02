"""Pure Sagarin page parsers for both leagues (no network, no IO).

Extracted from the generation-1 scrapers ``src/fetch_sagarin_week_nfl.py`` and
``src/fetch_sagarin_week_cfb.py`` so the staging engine
(:mod:`football_statfinder.sources.sagarin`) can use them as a library and the
legacy modules can eventually be deleted (REBUILD.md section 3, "Two Sagarin
ingestion generations").

The line-parsing regexes were derived empirically against real sagarin.com
pages and are ported VERBATIM — do not "improve" them:

* NFL: token scan over lines carrying an ``(AFC``/``(NFC`` division tag,
  bounded to ranks 1..32.
* CFB: ``CFB_LINE_PATTERN`` classification regex; only ``A`` (FBS) rows are
  kept and re-ranked contiguously after filtering.
* Shared: HFA extraction and per-league page season/week stamp extraction.

What changed from legacy: nothing semantic. Functions are renamed with
league prefixes because both leagues now live in one module, and the two
byte-decoding helpers are collapsed into the pure :func:`decode_bytes`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# Spoofed browser User-Agent the legacy fetchers used (identical in both
# generation-1 files); the staging engine sends it on every fetch.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

# --- shared patterns (identical in both legacy files) -----------------------

HFA_PATTERN = re.compile(r"HOME\s+ADVANTAGE=\[\s*([0-9]+\.\d+)\s*\]", re.IGNORECASE)

# --- NFL patterns (src/fetch_sagarin_week_nfl.py) ---------------------------

NFL_HEADER_PATTERN = re.compile(
    r"NFL\s+(\d{4})\s+through games of\s+.*?Week\s+(\d+)", re.IGNORECASE
)

# --- CFB patterns (src/fetch_sagarin_week_cfb.py) ---------------------------

CFB_HEADER_PATTERN = re.compile(r"COLLEGE\s+FOOTBALL\s+(\d{4}).*WEEK\s+(\d+)", re.IGNORECASE)
CFB_LINE_PATTERN = re.compile(r"^\s*(\d+)\s+(.+?)\s+([A-Z]{1,2})\s*=\s*(-?\d+\.\d+)", re.ASCII)
CFB_SCHEDULE_PATTERN = re.compile(r"(-?\d+\.\d+)\(\s*(\d+)\s*\)")

FBS_CLASSIFICATION = "A"


@dataclass(frozen=True)
class RatedTeam:
    """League-agnostic parsed rating row consumed by the staging engine."""

    team_raw: str
    pr: float
    pr_rank: int
    sos: Optional[float]
    sos_rank: Optional[int]


@dataclass
class NflSagarinRecord:
    rank: int
    team_raw: str
    pr: float
    pr_rank: int
    sos: Optional[float]
    sos_rank: Optional[int]


@dataclass
class CfbSagarinRecord:
    rank: int
    team_raw: str
    classification: str
    pr: float
    sos: Optional[float]
    sos_rank: Optional[int]


# --- pure text helpers -------------------------------------------------------


def decode_bytes(data: bytes, encodings: Sequence[Optional[str]]) -> str:
    """Decode raw page bytes trying ``encodings`` in order (legacy fallbacks)."""
    for enc in encodings:
        if not enc:
            continue
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    """Reduce the Sagarin page to plain text (verbatim legacy strip)."""
    text = unescape(html)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    return text


def parse_hfa(text: str) -> Optional[float]:
    """Extract the HOME ADVANTAGE value; None when absent/unparseable."""
    match = HFA_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# --- NFL parsing (verbatim port) ---------------------------------------------


def clean_team_raw_nfl(team_raw: str) -> str:
    return re.sub(r"[\*\+]+$", "", team_raw.strip())


def parse_nfl_line(line: str) -> Optional[NflSagarinRecord]:
    """Parse one NFL table line (legacy ``_parse_rank_line``, verbatim)."""
    sanitized = line.replace("=", " ").replace("\u00a0", " ")
    stripped = sanitized.strip()
    if not stripped or not stripped[0].isdigit():
        return None
    upper = sanitized.upper()
    if "(NFC" not in upper and "(AFC" not in upper:
        return None
    tokens = stripped.split()
    if len(tokens) < 2:
        return None
    rank = int(tokens[0])
    team_tokens: List[str] = []
    pr: Optional[float] = None
    for token in tokens[1:]:
        if re.fullmatch(r"-?\d+\.\d+", token):
            pr = float(token)
            break
        team_tokens.append(token)
    if pr is None:
        return None
    team_raw = clean_team_raw_nfl(" ".join(team_tokens))
    sos = sos_rank = None
    sos_match = re.search(r"(-?\d+\.\d+)\(\s*(\d+)\s*\)", sanitized)
    if sos_match:
        sos = float(sos_match.group(1))
        sos_rank = int(sos_match.group(2))
    return NflSagarinRecord(rank=rank, team_raw=team_raw, pr=pr, pr_rank=rank, sos=sos, sos_rank=sos_rank)


def parse_nfl_table(text: str) -> List[NflSagarinRecord]:
    """32-team NFC/AFC scan (legacy ``parse_table_lines``, verbatim)."""
    records: List[NflSagarinRecord] = []
    seen_ranks: set[int] = set()
    seen_teams: set[str] = set()
    for line in text.splitlines():
        parsed = parse_nfl_line(line)
        if not parsed:
            continue
        pr_rank = int(parsed.pr_rank)
        if not (1 <= pr_rank <= 32):
            continue
        if pr_rank in seen_ranks or parsed.team_raw in seen_teams:
            continue
        records.append(parsed)
        seen_ranks.add(pr_rank)
        seen_teams.add(parsed.team_raw)
        if len(records) == 32:
            break
    return records


def extract_nfl_table_week(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """NFL page season/week stamp extraction (verbatim)."""
    for line in text.splitlines():
        match = NFL_HEADER_PATTERN.search(line)
        if match:
            return int(match.group(1)), int(match.group(2)), line.strip()
    return None, None, None


def parse_nfl_page_stamp(lines: Iterable[str]) -> Optional[str]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("HOME ADVANTAGE"):
            break
        if "SAGARIN" in stripped.upper() or "NFL" in stripped.upper():
            return stripped
    return None


def nfl_rated_teams(records: Sequence[NflSagarinRecord]) -> List[RatedTeam]:
    """NFL records already carry page ranks; pass them through."""
    return [
        RatedTeam(team_raw=r.team_raw, pr=r.pr, pr_rank=int(r.pr_rank), sos=r.sos, sos_rank=r.sos_rank)
        for r in records
    ]


# --- CFB parsing (verbatim port) ---------------------------------------------


def clean_team_raw_cfb(team_raw: str) -> str:
    return re.sub(r"[\*\+\^\#]+$", "", team_raw.strip())


def parse_cfb_table(text: str) -> List[CfbSagarinRecord]:
    """FBS classification scan (legacy ``parse_table_lines``, verbatim)."""
    records: List[CfbSagarinRecord] = []
    seen: set[str] = set()
    started = False
    for line in text.splitlines():
        upper = line.upper()
        if not started and "COLLEGE FOOTBALL" in upper and "WEEK" in upper and re.search(r"\d{4}", upper):
            started = True
            continue
        if started and upper.strip().startswith("CONFERENCE AVERAGES"):
            break
        if not started:
            continue
        match = CFB_LINE_PATTERN.match(line)
        if not match:
            continue
        rank = int(match.group(1))
        team_raw = clean_team_raw_cfb(match.group(2))
        classification = match.group(3)
        if team_raw in seen:
            continue
        seen.add(team_raw)
        try:
            pr = float(match.group(4))
        except ValueError:
            continue
        sos = sos_rank = None
        sched_match = CFB_SCHEDULE_PATTERN.search(line)
        if sched_match:
            try:
                sos = float(sched_match.group(1))
                sos_rank = int(sched_match.group(2))
            except ValueError:
                sos = None
                sos_rank = None
        records.append(
            CfbSagarinRecord(
                rank=rank,
                team_raw=team_raw,
                classification=classification,
                pr=pr,
                sos=sos,
                sos_rank=sos_rank,
            )
        )
    return records


def extract_cfb_table_week(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """CFB page season/week stamp extraction (verbatim)."""
    for line in text.splitlines():
        match = CFB_HEADER_PATTERN.search(line)
        if match:
            return int(match.group(1)), int(match.group(2)), line.strip()
    return None, None, None


def parse_cfb_page_stamp(lines: Iterable[str]) -> Optional[str]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("HOME ADVANTAGE"):
            break
        if "COLLEGE FOOTBALL" in stripped.upper():
            return stripped
    return None


def select_fbs_and_rank(records: Sequence[CfbSagarinRecord]) -> List[RatedTeam]:
    """Keep FBS (classification ``A``) rows and re-rank contiguously.

    Verbatim port of the FBS filter + re-rank step from the legacy CFB
    ``records_to_dataframe`` (rank ties broken by team name).
    """
    fbs = [rec for rec in records if rec.classification == FBS_CLASSIFICATION]
    fbs.sort(key=lambda rec: (rec.rank, rec.team_raw))
    return [
        RatedTeam(team_raw=rec.team_raw, pr=rec.pr, pr_rank=idx, sos=rec.sos, sos_rank=rec.sos_rank)
        for idx, rec in enumerate(fbs, start=1)
    ]


# --- per-league dispatch ------------------------------------------------------


@dataclass(frozen=True)
class SagarinParserSpec:
    """The three parser entry points the staging engine needs per league."""

    rated_teams: Callable[[str], List[RatedTeam]]
    extract_table_week: Callable[[str], Tuple[Optional[int], Optional[int], Optional[str]]]
    parse_page_stamp: Callable[[Iterable[str]], Optional[str]]


_PARSERS = {
    "nfl": SagarinParserSpec(
        rated_teams=lambda text: nfl_rated_teams(parse_nfl_table(text)),
        extract_table_week=extract_nfl_table_week,
        parse_page_stamp=parse_nfl_page_stamp,
    ),
    "cfb": SagarinParserSpec(
        rated_teams=lambda text: select_fbs_and_rank(parse_cfb_table(text)),
        extract_table_week=extract_cfb_table_week,
        parse_page_stamp=parse_cfb_page_stamp,
    ),
}


def get_parser(league_code: str) -> SagarinParserSpec:
    """Resolve the parser spec for a league; raises on unknowns."""
    spec = _PARSERS.get((league_code or "").strip().lower())
    if spec is None:
        known = ", ".join(sorted(_PARSERS))
        raise ValueError(f"no Sagarin parser for league {league_code!r} (known: {known})")
    return spec


__all__ = [
    "CFB_HEADER_PATTERN",
    "CFB_LINE_PATTERN",
    "CFB_SCHEDULE_PATTERN",
    "CfbSagarinRecord",
    "FBS_CLASSIFICATION",
    "HFA_PATTERN",
    "NFL_HEADER_PATTERN",
    "NflSagarinRecord",
    "RatedTeam",
    "SagarinParserSpec",
    "USER_AGENT",
    "clean_team_raw_cfb",
    "clean_team_raw_nfl",
    "decode_bytes",
    "extract_cfb_table_week",
    "extract_nfl_table_week",
    "get_parser",
    "nfl_rated_teams",
    "parse_cfb_page_stamp",
    "parse_cfb_table",
    "parse_hfa",
    "parse_nfl_line",
    "parse_nfl_page_stamp",
    "parse_nfl_table",
    "select_fbs_and_rank",
    "strip_html",
]

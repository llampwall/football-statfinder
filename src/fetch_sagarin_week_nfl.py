"""Fetch weekly NFL Sagarin ratings and emit CSV/JSONL outputs.

Purpose:
    Scrape Sagarin's NFL ratings page, parse per-team power ratings, and store
    normalized records for downstream consumers.
Inputs:
    Season/week CLI flags, optional output basename, optional local HTML file.
Outputs:
    /out/sagarin_nfl_<season>_wk<week>.{csv,jsonl}
Source(s) of truth:
    http://sagarin.com/sports/nflsend.htm (plain-text HTML table).
Example:
    python -m src.fetch_sagarin_week_nfl --season 2025 --week 6
"""

from __future__ import annotations

import argparse
import warnings
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from pandas import DataFrame

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl
from src.common.team_names import normalize_team_display

SAGARIN_URL = "http://sagarin.com/sports/nflsend.htm"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

HFA_PATTERN = re.compile(r"HOME\s+ADVANTAGE=\[\s*([0-9]+\.\d+)\s*\]", re.IGNORECASE)


def _ensure_suffix(base: Path, suffix: str) -> Path:
    if base.suffix:
        return base.with_suffix(suffix)
    return base.parent / (base.name + suffix)


@dataclass
class SagarinRecord:
    rank: int
    team_raw: str
    pr: float
    pr_rank: int
    sos: Optional[float]
    sos_rank: Optional[int]


def _decode_response(resp: requests.Response) -> str:
    encoding = resp.apparent_encoding or resp.encoding
    fallbacks = [encoding, "cp1252", "latin-1", "utf-8"]
    content = resp.content
    for enc in fallbacks:
        if not enc:
            continue
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="replace")


def fetch_html(url: str = SAGARIN_URL) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return _decode_response(resp)


def read_local_html(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    from html import unescape

    text = unescape(html)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    return text


def parse_hfa(text: str) -> Optional[float]:
    match = HFA_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_page_stamp(lines: Iterable[str]) -> Optional[str]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("HOME ADVANTAGE"):
            break
        if "SAGARIN" in stripped.upper() or "NFL" in stripped.upper():
            return stripped
    return None


def _clean_team_raw(team_raw: str) -> str:
    return re.sub(r"[\*\+]+$", "", team_raw.strip())


def extract_table_week(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    pattern = re.compile(r"NFL\s+(\d{4})\s+through games of\s+.*?Week\s+(\d+)", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            season = int(match.group(1))
            week = int(match.group(2))
            return season, week, line.strip()
    return None, None, None




def _parse_rank_line(line: str) -> Optional[SagarinRecord]:
    sanitized = line.replace('=', ' ').replace("\u00a0", " ")
    stripped = sanitized.strip()
    if not stripped or not stripped[0].isdigit():
        return None
    upper = sanitized.upper()
    if '(NFC' not in upper and '(AFC' not in upper:
        return None
    tokens = stripped.split()
    if len(tokens) < 2:
        return None
    rank = int(tokens[0])
    team_tokens: List[str] = []
    pr: Optional[float] = None
    for token in tokens[1:]:
        if re.fullmatch(r'-?\d+\.\d+', token):
            pr = float(token)
            break
        team_tokens.append(token)
    if pr is None:
        return None
    team_raw = _clean_team_raw(' '.join(team_tokens))
    sos = sos_rank = None
    sos_match = re.search(r'(-?\d+\.\d+)\(\s*(\d+)\s*\)', sanitized)
    if sos_match:
        sos = float(sos_match.group(1))
        sos_rank = int(sos_match.group(2))
    return SagarinRecord(rank=rank, team_raw=team_raw, pr=pr, pr_rank=rank, sos=sos, sos_rank=sos_rank)


def parse_line_regex(line: str) -> Optional[SagarinRecord]:
    return _parse_rank_line(line)


def parse_line_split(line: str) -> Optional[SagarinRecord]:
    return _parse_rank_line(line)


def parse_table_lines(text: str) -> List[SagarinRecord]:
    records: List[SagarinRecord] = []
    seen_ranks: set[int] = set()
    seen_teams: set[str] = set()
    for line in text.splitlines():
        parsed = parse_line_regex(line)
        if parsed is None:
            parsed = parse_line_split(line)
        if not parsed:
            continue
        pr_rank = int(parsed.pr_rank)
        raw_rank = int(parsed.rank)
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


def validate_records(df: DataFrame) -> None:
    errors: List[str] = []

    if len(df) != 32:
        errors.append(f"Count != 32 (found {len(df)})")

    ranks = df["pr_rank"].tolist()
    expected = set(range(1, 33))
    missing = expected.difference(ranks)
    duplicates = [r for r in set(ranks) if ranks.count(r) > 1]
    if missing or duplicates:
        errors.append(f"Ranks invalid; missing={sorted(missing)} duplicates={sorted(duplicates)}")

    if not df["pr"].apply(lambda v: pd.notna(v) and abs(v * 100 - round(v * 100)) < 1e-6).all():
        errors.append("PR values missing or not two-decimal precision")

    if df["team_norm"].nunique() != 32:
        errors.append("Team normalization failed (team_norm not unique)")

    if df["team_norm"].isna().any():
        unmapped = df[df["team_norm"].isna()]["team_raw"].tolist()
        errors.append(f"Unmapped teams: {unmapped}")

    sos_present = df["sos"].notna().sum()
    if sos_present == 0:
        print("NOTE: SoS values not present in source; skipping SoS coverage check.")
    else:
        coverage = sos_present / len(df)
        ranks_valid = df[df["sos"].notna()]["sos_rank"].apply(lambda v: pd.notna(v) and 1 <= int(v) <= 32).all()
        precision_ok = df[df["sos"].notna()]["sos"].apply(lambda v: abs(v * 100 - round(v * 100)) < 1e-6).all()
        if coverage < 0.9 or not ranks_valid or not precision_ok:
            errors.append(f"SoS coverage check failed (coverage={coverage:.2%}, ranks_valid={ranks_valid}, precision_ok={precision_ok})")

    if errors:
        raise ValueError("; ".join(errors))


def records_to_dataframe(
    records: List[SagarinRecord],
    season: int,
    week: int,
    hfa: Optional[float],
    source_url: str,
    page_stamp: Optional[str],
) -> DataFrame:
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows = []
    unmapped = []
    for rec in records:
        team_norm = normalize_team_display(rec.team_raw)
        if not team_norm:
            team_norm = rec.team_raw
            unmapped.append(rec.team_raw)
        rows.append(
            {
                "season": season,
                "week": week,
                "team_raw": rec.team_raw,
                "team_norm": team_norm or rec.team_raw,
                "team": team_norm or rec.team_raw,
                "pr": round(rec.pr, 2),
                "pr_rank": rec.pr_rank,
                "sos": None if rec.sos is None else round(rec.sos, 2),
                "sos_rank": rec.sos_rank,
                "hfa": hfa,
                "source_url": source_url,
                "fetched_at": fetched_at,
                "page_stamp": page_stamp,
            }
        )
    if unmapped:
        unique_warn = sorted(set(unmapped))
        print(f"WARNING: team names needing review -> {unique_warn}")
    df = pd.DataFrame(rows)
    return df.sort_values("pr_rank").reset_index(drop=True)


def write_outputs(df: DataFrame, out_base: Path) -> None:
    csv_path = _ensure_suffix(out_base, ".csv")
    jsonl_path = _ensure_suffix(out_base, ".jsonl")
    write_csv(df, csv_path)
    write_jsonl(df.to_dict(orient="records"), jsonl_path)


def acceptance_summary(df: DataFrame) -> None:
    print("=== SAGARIN ACCEPTANCE CHECKS ===")
    print(f"Count: {len(df)}")
    unique_ranks = sorted(df['pr_rank'].tolist())
    print(f"Ranks coverage: {unique_ranks[:5]} ... {unique_ranks[-5:]}")
    pr_ok = df["pr"].apply(lambda v: abs(v * 100 - round(v * 100)) < 1e-6).all()
    print(f"PR precision ok: {pr_ok}")
    print(f"Team_norm unique: {df['team_norm'].nunique()} of {len(df)}")
    sos_count = df["sos"].notna().sum()
    if sos_count:
        coverage = sos_count / len(df)
        print(f"SoS coverage: {sos_count}/{len(df)} ({coverage:.1%})")
    else:
        print("SoS coverage: source omitted values")


def fetch_sagarin_week(
    season: int,
    week: int,
    out_base: Optional[Path] = None,
    local_html: Optional[Path] = None,
) -> dict:
    out_dir = ensure_out_dir()
    base = out_base or out_dir / f"sagarin_nfl_{season}_wk{week}"
    source_url = SAGARIN_URL
    try:
        if local_html:
            text = read_local_html(local_html)
            source_url = str(Path(local_html).resolve())
        else:
            text = fetch_html()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Sagarin page: {exc}") from exc

    stripped = strip_html(text)
    season_from_page, page_week, header_line = extract_table_week(stripped)
    effective_week = page_week or week
    effective_season = season_from_page or season
    if season_from_page and season_from_page != season:
        warnings.warn(
            f"Requested season {season} but Sagarin page reports season {season_from_page}. Using page season.",
            RuntimeWarning,
        )
    if page_week and page_week != week:
        warnings.warn(
            f"Requested week {week} but Sagarin page reports week {page_week}. Using page week in outputs.",
            RuntimeWarning,
        )
    records = parse_table_lines(stripped)

    if len(records) != 32:
        debug_path = _ensure_suffix(base, "_raw.txt")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(stripped, encoding="utf-8")
        raise RuntimeError(f"Parsed {len(records)} records; wrote debug text to {debug_path}")

    hfa = parse_hfa(stripped)
    page_stamp = header_line or parse_page_stamp(stripped.splitlines())
    df = records_to_dataframe(records, effective_season, effective_week, hfa, source_url, page_stamp)

    validate_records(df)
    output_base = base
    if page_week and page_week != week:
        prefix = base.stem.split('_wk')[0]
        output_base = base.with_name(f"{prefix}_wk{page_week}_[wk{week}_requested]")
    write_outputs(df, output_base)
    acceptance_summary(df)
    print(f"Sagarin page season/week: {effective_season} Week {effective_week}")

    return {
        "csv_path": _ensure_suffix(output_base, ".csv"),
        "jsonl_path": _ensure_suffix(output_base, ".jsonl"),
        "hfa": hfa,
        "count": len(df),
        "page_week": effective_week,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch weekly NFL Sagarin ratings.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output basename (default /out/sagarin_nfl_<season>_wk<week>)",
    )
    parser.add_argument(
        "--local-html",
        type=str,
        default=None,
        help="Path to saved HTML file for offline parsing.",
    )
    args = parser.parse_args()
    out_base = Path(args.out) if args.out else None
    local_html = Path(args.local_html) if args.local_html else None
    fetch_sagarin_week(args.season, args.week, out_base=out_base, local_html=local_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

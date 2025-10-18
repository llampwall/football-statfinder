"""Fetch weekly College Football Sagarin ratings into CSV/JSONL artifacts.

Purpose:
    Scrape Jeff Sagarin's College Football ratings, normalize team names, and
    emit per-week outputs that mirror the NFL workflow. The scraper keeps only
    FBS programs, rounds precision, and fails loudly when acceptance checks
    miss the expected coverage.
Inputs:
    Season/week CLI flags, optional local HTML path for offline parsing.
Outputs:
    out/cfb/<season>_week<week>/sagarin_cfb_<season>_wk<week>.{csv,jsonl}
Source of truth:
    http://sagarin.com/sports/cfsend.htm (plain-text HTML table).
Example:
    python -m src.fetch_sagarin_week_cfb --season 2025 --week 8
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from pandas import DataFrame

from src.common.io_utils import ensure_out_dir, write_csv, write_jsonl
from src.common.team_names_cfb import normalize_team_name_cfb

SAGARIN_URL = "http://sagarin.com/sports/cfsend.htm"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

HFA_PATTERN = re.compile(r"HOME\s+ADVANTAGE=\[\s*([0-9]+\.\d+)\s*\]", re.IGNORECASE)
HEADER_PATTERN = re.compile(r"COLLEGE\s+FOOTBALL\s+(\d{4}).*WEEK\s+(\d+)", re.IGNORECASE)
LINE_PATTERN = re.compile(r"^\s*(\d+)\s+(.+?)\s+([A-Z]{1,2})\s*=\s*(-?\d+\.\d+)", re.ASCII)
SCHEDULE_PATTERN = re.compile(r"(-?\d+\.\d+)\(\s*(\d+)\s*\)")

FBS_CLASSIFICATION = "A"
MIN_FBS_ROWS = 120
MAX_FBS_ROWS = 140


@dataclass
class SagarinRecord:
    """Lightweight container for raw Sagarin table rows."""

    rank: int
    team_raw: str
    classification: str
    pr: float
    sos: Optional[float]
    sos_rank: Optional[int]


def cfb_week_dir(season: int, week: int) -> Path:
    """Return the per-week CFB output directory."""
    out_dir = ensure_out_dir() / "cfb" / f"{season}_week{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_suffix(base: Path, suffix: str) -> Path:
    if base.suffix:
        return base.with_suffix(suffix)
    return base.parent / (base.name + suffix)


def _decode_response(resp: requests.Response) -> str:
    encoding = resp.apparent_encoding or resp.encoding
    fallbacks = [encoding, "cp1252", "latin-1", "utf-8"]
    data = resp.content
    for enc in fallbacks:
        if not enc:
            continue
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


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
        if "COLLEGE FOOTBALL" in stripped.upper():
            return stripped
    return None


def extract_table_week(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for line in text.splitlines():
        match = HEADER_PATTERN.search(line)
        if match:
            season = int(match.group(1))
            week = int(match.group(2))
            return season, week, line.strip()
    return None, None, None


def _clean_team_raw(team_raw: str) -> str:
    return re.sub(r"[\*\+\^\#]+$", "", team_raw.strip())


def parse_table_lines(text: str) -> List[SagarinRecord]:
    records: List[SagarinRecord] = []
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
        match = LINE_PATTERN.match(line)
        if not match:
            continue
        rank = int(match.group(1))
        team_raw = _clean_team_raw(match.group(2))
        classification = match.group(3)
        if team_raw in seen:
            continue
        seen.add(team_raw)
        try:
            pr = float(match.group(4))
        except ValueError:
            continue
        sos = sos_rank = None
        sched_match = SCHEDULE_PATTERN.search(line)
        if sched_match:
            try:
                sos = float(sched_match.group(1))
                sos_rank = int(sched_match.group(2))
            except ValueError:
                sos = None
                sos_rank = None
        records.append(
            SagarinRecord(
                rank=rank,
                team_raw=team_raw,
                classification=classification,
                pr=pr,
                sos=sos,
                sos_rank=sos_rank,
            )
        )
    return records


def records_to_dataframe(
    records: List[SagarinRecord],
    season: int,
    week: int,
    hfa: Optional[float],
    source_url: str,
    page_stamp: Optional[str],
) -> Tuple[DataFrame, List[str]]:
    fbs_records = [rec for rec in records if rec.classification == FBS_CLASSIFICATION]
    fbs_records.sort(key=lambda rec: (rec.rank, rec.team_raw))
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows: List[dict] = []
    unmapped: List[str] = []
    for idx, rec in enumerate(fbs_records, start=1):
        team_norm = normalize_team_name_cfb(rec.team_raw)
        if not team_norm:
            unmapped.append(rec.team_raw)
            team_norm = rec.team_raw
        rows.append(
            {
                "season": season,
                "week": week,
                "team_raw": rec.team_raw,
                "team_norm": team_norm,
                "team": team_norm,
                "pr": round(rec.pr, 2),
                "pr_rank": idx,
                "sos": None if rec.sos is None else round(rec.sos, 2),
                "sos_rank": rec.sos_rank,
                "hfa": hfa,
                "source_url": source_url,
                "fetched_at": fetched_at,
                "page_stamp": page_stamp,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pr_rank").reset_index(drop=True)
    return df, unmapped


def validate_records(df: DataFrame, unmapped: List[str]) -> List[str]:
    errors: List[str] = []
    count = len(df)
    if not (MIN_FBS_ROWS <= count <= MAX_FBS_ROWS):
        errors.append(f"Count {count} outside expected range [{MIN_FBS_ROWS}, {MAX_FBS_ROWS}]")
    expected_ranks = list(range(1, count + 1))
    ranks = df["pr_rank"].tolist()
    if ranks != expected_ranks:
        errors.append("pr_rank not contiguous 1..N")
    if df["pr_rank"].duplicated().any():
        dupes = df[df["pr_rank"].duplicated()]["pr_rank"].tolist()
        errors.append(f"Duplicate pr_rank values: {dupes}")
    team_norm_unique = df["team_norm"].nunique()
    if team_norm_unique != count:
        errors.append(f"team_norm unique count {team_norm_unique} != total rows {count}")
    if df["team_norm"].isna().any() or (df["team_norm"].astype(str).str.strip() == "").any():
        errors.append("team_norm contains blank values")
    if unmapped:
        errors.append(f"Unmapped teams: {sorted(set(unmapped))}")
    pr_precision_ok = df["pr"].apply(lambda val: pd.notna(val) and abs(val * 100 - round(val * 100)) < 1e-6).all()
    if not pr_precision_ok:
        errors.append("pr values missing or not two-decimal precision")
    sos_nonnull = df["sos"].dropna()
    if not sos_nonnull.empty:
        sos_precision_ok = sos_nonnull.apply(lambda val: abs(val * 100 - round(val * 100)) < 1e-6).all()
        if not sos_precision_ok:
            errors.append("sos values not two-decimal precision")
    return errors


def write_outputs(df: DataFrame, out_base: Path) -> dict:
    csv_path = _ensure_suffix(out_base, ".csv")
    jsonl_path = _ensure_suffix(out_base, ".jsonl")
    write_csv(df, csv_path)
    write_jsonl(df.to_dict(orient="records"), jsonl_path)
    return {"csv_path": csv_path, "jsonl_path": jsonl_path}


def acceptance_summary(df: DataFrame, hfa: Optional[float], page_info: Tuple[Optional[int], Optional[int]]) -> None:
    page_season, page_week = page_info
    print("=== CFB SAGARIN ACCEPTANCE ===")
    print(f"Rows: {len(df)} (expected {MIN_FBS_ROWS}-{MAX_FBS_ROWS})")
    print(f"pr_rank coverage: {df['pr_rank'].min() if not df.empty else 'n/a'} .. {df['pr_rank'].max() if not df.empty else 'n/a'}")
    print(f"Unique team_norm: {df['team_norm'].nunique()}")
    if hfa is not None:
        print(f"Home-field advantage: {hfa:.2f}")
    else:
        print("Home-field advantage: unavailable")
    page_season_str = page_season if page_season is not None else "unknown"
    page_week_str = page_week if page_week is not None else "unknown"
    print(f"Sagarin page season/week: {page_season_str} Week {page_week_str}")


def write_failure_artifacts(out_base: Path, stripped_text: str, receipt: dict) -> Tuple[Path, Path]:
    raw_path = out_base.parent / f"{out_base.name}_raw.txt"
    receipt_path = out_base.parent / f"{out_base.name}_receipt.json"
    raw_path.write_text(stripped_text, encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return raw_path, receipt_path


def fetch_sagarin_week_cfb(
    season: int,
    week: int,
    local_html: Optional[Path] = None,
) -> dict:
    out_dir = cfb_week_dir(season, week)
    out_base = out_dir / f"sagarin_cfb_{season}_wk{week}"
    source_url = SAGARIN_URL
    try:
        if local_html:
            text = read_local_html(local_html)
            source_url = str(local_html.resolve())
        else:
            text = fetch_html()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Sagarin CFB page: {exc}") from exc

    stripped = strip_html(text)
    page_season, page_week, header_line = extract_table_week(stripped)
    effective_season = page_season or season
    effective_week = page_week or week
    if page_season and page_season != season:
        warnings.warn(
            f"Requested season {season} but page reports {page_season}; using page season in outputs.",
            RuntimeWarning,
        )
    if page_week and page_week != week:
        warnings.warn(
            f"Requested week {week} but page reports {page_week}; using page week in outputs.",
            RuntimeWarning,
        )
    records = parse_table_lines(stripped)
    if not records:
        receipt = {
            "error": "No table rows parsed",
            "season": season,
            "week": week,
            "source_url": source_url,
        }
        write_failure_artifacts(out_base, stripped, receipt)
        raise SystemExit("FAIL: CFB Sagarin parse produced zero rows (see receipt).")

    hfa = parse_hfa(stripped)
    page_stamp = header_line or parse_page_stamp(stripped.splitlines())
    df, unmapped = records_to_dataframe(
        records,
        effective_season,
        effective_week,
        hfa,
        source_url,
        page_stamp,
    )

    errors = validate_records(df, unmapped)
    if errors:
        receipt = {
            "season": effective_season,
            "week": effective_week,
            "source_url": source_url,
            "row_count": len(df),
            "unique_team_norm": df["team_norm"].nunique() if not df.empty else 0,
            "unmapped_team_raw": sorted(set(unmapped)),
            "errors": errors,
        }
        write_failure_artifacts(out_base, stripped, receipt)
        raise SystemExit(f"FAIL: CFB Sagarin acceptance failed -> {'; '.join(errors)}")

    outputs = write_outputs(df, out_base)
    acceptance_summary(df, hfa, (page_season, page_week))
    print(f"PASS: wrote Sagarin CSV to {outputs['csv_path']}")
    print(f"PASS: wrote Sagarin JSONL to {outputs['jsonl_path']}")
    return {
        "csv_path": outputs["csv_path"],
        "jsonl_path": outputs["jsonl_path"],
        "count": len(df),
        "hfa": hfa,
        "page_season": page_season,
        "page_week": page_week,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch weekly CFB Sagarin ratings.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--local-html",
        type=str,
        default=None,
        help="Path to saved HTML file for offline parsing.",
    )
    args = parser.parse_args()
    local_html = Path(args.local_html).expanduser() if args.local_html else None
    fetch_sagarin_week_cfb(args.season, args.week, local_html=local_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

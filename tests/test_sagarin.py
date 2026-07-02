"""Tests for football_statfinder.sources.sagarin and .sagarin_parsers.

No archived pages exist under data/sagarin/raw/ (checked nfl/ and cfb/), so
these tests use synthetic pages built from the parsers' expected line format;
the NFL line-parser cases reuse the real-page line shapes from the season-1
test suite. No network: the HTML fetch is injected via ``fetch_html_fn``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from football_statfinder import paths
from football_statfinder.common.jsonl import read_jsonl
from football_statfinder.common.team_names import TEAM_ABBR_TO_FULL
from football_statfinder.config import Settings
from football_statfinder.leagues import CFB, NFL
from football_statfinder.sources import sagarin, sagarin_parsers


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    out_root = tmp_path / "out"
    monkeypatch.setattr(paths, "OUT_ROOT", out_root)
    monkeypatch.setattr(paths, "MASTER_ROOT", out_root / "master")
    monkeypatch.setattr(paths, "STAGING_ROOT", out_root / "staging")
    monkeypatch.setattr(paths, "SAGARIN_RAW_ROOT", tmp_path / "data" / "sagarin" / "raw")
    monkeypatch.setattr(paths, "STATE_PATH", out_root / "state" / "current_week.json")
    return tmp_path


def _nfl_team_names() -> list:
    names = sorted(set(TEAM_ABBR_TO_FULL.values()))
    assert len(names) == 32
    return names


def make_nfl_page(team_count: int = 32, season: int = 2026, week: int = 5, pr_shift: float = 0.0) -> str:
    names = _nfl_team_names()[:team_count]
    lines = [f"NFL {season} through games of October 4 Sunday - Week {week}"]
    for i, name in enumerate(names, start=1):
        pr = 30.0 - i * 0.25 + pr_shift
        sos = 25.0 - i * 0.05
        lines.append(
            f" {i:>2}  {name:<26}=  {pr:6.2f}    4   1   0   {sos:6.2f}(  {i:>2})"
            f"   2  1  0 |   3  1  0  (AFC EAST)"
        )
    lines.append("HOME ADVANTAGE=[  2.14]")
    return "<html><pre>" + "\n".join(lines) + "</pre></html>"


def make_cfb_page(fbs_count: int = 125, season: int = 2026, week: int = 10, fcs_count: int = 3) -> str:
    lines = [f"COLLEGE FOOTBALL {season} through results of NOVEMBER 7 - WEEK {week}"]
    rank = 0
    for i in range(1, fbs_count + 1):
        rank += 1
        name = f"Team Alpha{i:03d}"
        pr = 99.0 - i * 0.25
        sos = 60.0 - i * 0.05
        lines.append(
            f" {rank:>3}  {name:<24} A  =  {pr:6.2f}   10  2  0   {sos:6.2f}(  {i + 5:>3})"
        )
    for j in range(1, fcs_count + 1):
        rank += 1
        lines.append(
            f" {rank:>3}  Fcs Beta{j:03d}             B  =  {40.0 - j:6.2f}    5  5  0   {30.0:6.2f}( 200)"
        )
    lines.append("CONFERENCE AVERAGES")
    lines.append("HOME ADVANTAGE=[  2.51]")
    return "<html><pre>" + "\n".join(lines) + "</pre></html>"


# --- pure parsers ---------------------------------------------------------------


def test_parse_nfl_line_real_format():
    line = (
        " 1  Kansas City Chiefs        =  26.83    4   1   0   21.34(   7)   2  1  0 |   3  1  0 |"
        "   26.29    2 |   27.39    1 |   27.21    1 |   26.00    2  (AFC WEST)"
    )
    record = sagarin_parsers.parse_nfl_line(line)
    assert record is not None
    assert record.rank == 1
    assert record.team_raw == "Kansas City Chiefs"
    assert record.pr == 26.83
    assert record.pr_rank == 1
    assert record.sos == 21.34
    assert record.sos_rank == 7


def test_parse_nfl_line_strips_trailing_symbols():
    line = (
        " 27  Los Angeles Chargers*     =  20.79    3   2   0   20.93(  11)   2  1  0 |   2  1  0 |"
        "   20.66   15 |   20.91   18  (AFC WEST)"
    )
    record = sagarin_parsers.parse_nfl_line(line)
    assert record is not None
    assert record.team_raw == "Los Angeles Chargers"
    assert record.pr == 20.79
    assert record.sos_rank == 11


def test_parse_nfl_table_full_page():
    text = sagarin_parsers.strip_html(make_nfl_page())
    records = sagarin_parsers.parse_nfl_table(text)
    assert len(records) == 32
    assert [r.pr_rank for r in records] == list(range(1, 33))
    season, week, header = sagarin_parsers.extract_nfl_table_week(text)
    assert (season, week) == (2026, 5)
    assert header is not None
    assert sagarin_parsers.parse_hfa(text) == 2.14


def test_parse_cfb_table_filters_fcs_and_reranks():
    text = sagarin_parsers.strip_html(make_cfb_page(fbs_count=5, fcs_count=2))
    raw_records = sagarin_parsers.parse_cfb_table(text)
    assert len(raw_records) == 7  # both classifications parsed
    rated = sagarin_parsers.select_fbs_and_rank(raw_records)
    assert len(rated) == 5  # FCS (classification B) filtered out
    assert [r.pr_rank for r in rated] == [1, 2, 3, 4, 5]
    assert all("Fcs" not in r.team_raw for r in rated)
    season, week, _ = sagarin_parsers.extract_cfb_table_week(text)
    assert (season, week) == (2026, 10)
    assert sagarin_parsers.parse_hfa(text) == 2.51


# --- staging engine: happy paths ---------------------------------------------------


def test_nfl_staging_happy_path(sandbox):
    result = sagarin.run_sagarin_staging(
        NFL, 2026, 5, Settings(), fetch_html_fn=lambda url: make_nfl_page()
    )
    assert result.teams_parsed == 32
    assert result.teams_selected == 32
    assert result.hfa == 2.14
    # weekly snapshot in the unified week dir
    assert result.weekly_csv == paths.week_dir("nfl", 2026, 5) / "sagarin_nfl_2026_wk5.csv"
    weekly = pd.read_csv(result.weekly_csv)
    assert len(weekly) == 32
    assert set(weekly["season"]) == {2026} and set(weekly["week"]) == {5}
    rows = read_jsonl(result.weekly_jsonl)
    assert len(rows.rows) == 32 and rows.skipped == 0
    # append-only ledger
    ledger = read_jsonl(paths.sagarin_staging_dir("nfl") / "2026.jsonl")
    assert len(ledger.rows) == 32
    # master via the one upsert path
    master = pd.read_csv(paths.sagarin_master_csv("nfl"))
    assert len(master) == 32
    assert set(master["league"]) == {"NFL"}
    # raw HTML archived
    archived = list(paths.sagarin_raw_html_dir("nfl").glob("*.html"))
    assert len(archived) == 1
    assert "_2026_wk5_" in archived[0].name


def test_cfb_staging_happy_path(sandbox):
    result = sagarin.run_sagarin_staging(
        CFB, 2026, 10, Settings(), fetch_html_fn=lambda url: make_cfb_page(fbs_count=125)
    )
    assert result.teams_parsed == 125
    assert result.teams_selected == 125
    master = pd.read_csv(paths.sagarin_master_csv("cfb"))
    assert len(master) == 125
    assert set(master["league"]) == {"CFB"}
    archived = list(paths.sagarin_raw_html_dir("cfb").glob("*.html"))
    assert len(archived) == 1


def test_latest_per_team_wins_and_master_idempotent(sandbox):
    sagarin.run_sagarin_staging(NFL, 2026, 5, Settings(), fetch_html_fn=lambda url: make_nfl_page())
    result = sagarin.run_sagarin_staging(
        NFL, 2026, 5, Settings(), fetch_html_fn=lambda url: make_nfl_page(pr_shift=1.0)
    )
    ledger = read_jsonl(result.staging_path)
    assert len(ledger.rows) == 64  # append-only: both runs retained
    weekly = pd.read_csv(result.weekly_csv)
    assert len(weekly) == 32  # latest-per-team selection
    top = weekly.sort_values("pr_rank").iloc[0]
    assert top["pr"] == pytest.approx(30.75)  # second run's shifted value won
    master = pd.read_csv(paths.sagarin_master_csv("nfl"))
    assert len(master) == 32  # upsert keyed on league/season/week/team


# --- validation gates: fail loud with a receipt -------------------------------------


def test_nfl_validation_fails_on_31_teams(sandbox):
    with pytest.raises(sagarin.SagarinValidationError, match="Count != 32"):
        sagarin.run_sagarin_staging(
            NFL, 2026, 5, Settings(), fetch_html_fn=lambda url: make_nfl_page(team_count=31)
        )
    receipts = list(paths.sagarin_staging_dir("nfl").glob("failure_*.receipt.json"))
    assert len(receipts) == 1
    # nothing staged, no weekly snapshot, no master
    assert not (paths.sagarin_staging_dir("nfl") / "2026.jsonl").exists()
    assert not (paths.week_dir("nfl", 2026, 5) / "sagarin_nfl_2026_wk5.csv").exists()
    assert not paths.sagarin_master_csv("nfl").exists()


def test_cfb_validation_fails_below_range(sandbox):
    with pytest.raises(sagarin.SagarinValidationError, match="outside expected range"):
        sagarin.run_sagarin_staging(
            CFB, 2026, 10, Settings(), fetch_html_fn=lambda url: make_cfb_page(fbs_count=119)
        )
    receipts = list(paths.sagarin_staging_dir("cfb").glob("failure_*.receipt.json"))
    assert len(receipts) == 1
    assert not paths.sagarin_master_csv("cfb").exists()


def test_cfb_validation_fails_above_range(sandbox):
    with pytest.raises(sagarin.SagarinValidationError, match="outside expected range"):
        sagarin.run_sagarin_staging(
            CFB, 2026, 10, Settings(), fetch_html_fn=lambda url: make_cfb_page(fbs_count=141)
        )


def test_fetch_failure_raises_never_soft_fails(sandbox):
    def boom(url):
        raise ConnectionError("sagarin.com unreachable")

    with pytest.raises(RuntimeError, match="Failed to fetch Sagarin page"):
        sagarin.run_sagarin_staging(NFL, 2026, 5, Settings(), fetch_html_fn=boom)
    assert not (paths.sagarin_staging_dir("nfl") / "2026.jsonl").exists()


def test_raw_html_archived_before_parse_failure(sandbox):
    with pytest.raises(sagarin.SagarinValidationError):
        sagarin.run_sagarin_staging(
            NFL, 2026, 5, Settings(), fetch_html_fn=lambda url: "<html>not a ratings page</html>"
        )
    archived = list(paths.sagarin_raw_html_dir("nfl").glob("*.html"))
    assert len(archived) == 1  # archive happened before parsing failed


# --- season/week stamping (bugs 14/15) -----------------------------------------------


def test_requested_season_week_stamp_paths_and_content(sandbox):
    # Page reports 2025 week 9; the caller (current-week service) says 2026 week 12.
    page = make_nfl_page(season=2025, week=9)
    result = sagarin.run_sagarin_staging(NFL, 2026, 12, Settings(), fetch_html_fn=lambda url: page)
    assert (result.page_season, result.page_week) == (2025, 9)
    assert (result.season, result.week) == (2026, 12)
    # paths stamped with requested values
    assert result.weekly_csv == paths.week_dir("nfl", 2026, 12) / "sagarin_nfl_2026_wk12.csv"
    assert result.staging_path == paths.sagarin_staging_dir("nfl") / "2026.jsonl"
    archived = list(paths.sagarin_raw_html_dir("nfl").glob("*.html"))
    assert "_2026_wk12_" in archived[0].name
    # content stamped with requested values, never max(page_season, year)
    weekly = pd.read_csv(result.weekly_csv)
    assert set(weekly["season"]) == {2026} and set(weekly["week"]) == {12}
    ledger = read_jsonl(result.staging_path)
    assert {r["season"] for r in ledger.rows} == {2026}
    assert {r["week"] for r in ledger.rows} == {12}
    master = pd.read_csv(paths.sagarin_master_csv("nfl"))
    assert set(master["season"]) == {2026} and set(master["week"]) == {12}

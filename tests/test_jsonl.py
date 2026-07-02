"""Tests for counted-skip JSONL reading."""

from __future__ import annotations

import pytest

from football_statfinder.common.jsonl import JsonlError, iter_jsonl, read_jsonl


def _write(tmp_path, text: str):
    path = tmp_path / "rows.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_jsonl_counts_bad_lines(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n{truncated\n\n{"b": 2}\n')
    result = read_jsonl(path)
    assert [row for row in result.rows] == [{"a": 1}, {"b": 2}]
    assert result.skipped == 1


def test_read_jsonl_skips_non_objects(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n[1, 2, 3]\n42\n')
    result = read_jsonl(path)
    assert result.rows == [{"a": 1}]
    assert result.skipped == 2


def test_read_jsonl_missing_file_is_empty(tmp_path):
    result = read_jsonl(tmp_path / "nope.jsonl")
    assert result.rows == []
    assert result.skipped == 0


def test_strict_mode_raises_with_location(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n{bad\n')
    with pytest.raises(JsonlError, match=r"rows\.jsonl:2"):
        read_jsonl(path, strict=True)


def test_iter_jsonl_yields_lazily(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n{"b": 2}\n')
    rows = list(iter_jsonl(path))
    assert rows == [{"a": 1}, {"b": 2}]

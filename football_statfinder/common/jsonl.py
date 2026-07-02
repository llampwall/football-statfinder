"""JSONL reading with counted, logged skips.

Every season-1 JSONL reader swallowed decode errors with a bare ``continue``
(REBUILD.md bug 10), so corrupted lines dropped data silently. This is now the
only way the pipeline reads JSONL: bad lines are still skipped (a truncated
staging append must not kill a refresh) but they are counted and logged, and
callers that must not lose data can pass ``strict=True``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Union

logger = logging.getLogger(__name__)


class JsonlError(RuntimeError):
    """Raised in strict mode when a line fails to decode."""


@dataclass
class JsonlReadResult:
    rows: List[dict] = field(default_factory=list)
    skipped: int = 0


def iter_jsonl(path: Union[str, Path], *, strict: bool = False) -> Iterator[dict]:
    """Yield decoded objects; skip-and-log bad lines (raise when strict)."""
    target = Path(path)
    skipped = 0
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                if strict:
                    raise JsonlError(f"{target}:{line_no}: {exc}") from exc
                skipped += 1
                logger.warning("skipping undecodable JSONL line %s:%s", target, line_no)
                continue
            if isinstance(payload, dict):
                yield payload
            else:
                skipped += 1
                logger.warning("skipping non-object JSONL line %s:%s", target, line_no)
    if skipped:
        logger.warning("%s: skipped %d bad JSONL line(s)", target, skipped)


def read_jsonl(path: Union[str, Path], *, strict: bool = False) -> JsonlReadResult:
    """Read a whole JSONL file, returning rows plus the skipped-line count."""
    target = Path(path)
    result = JsonlReadResult()
    if not target.exists():
        return result
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                if strict:
                    raise JsonlError(f"{target}:{line_no}: {exc}") from exc
                result.skipped += 1
                logger.warning("skipping undecodable JSONL line %s:%s", target, line_no)
                continue
            if isinstance(payload, dict):
                result.rows.append(payload)
            else:
                result.skipped += 1
                logger.warning("skipping non-object JSONL line %s:%s", target, line_no)
    if result.skipped:
        logger.warning("%s: skipped %d bad JSONL line(s)", target, result.skipped)
    return result


__all__ = ["JsonlError", "JsonlReadResult", "iter_jsonl", "read_jsonl"]

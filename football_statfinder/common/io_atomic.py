"""Atomic write helpers for staging workflows.

Purpose & scope:
    Provide tiny utilities that write text/CSV/JSONL files via tmp-replace so
    staging tasks can update artifacts safely without partially-written files.

Spec anchors:
    - /context/global_week_and_provider_decoupling.md (C, E, F, I)

Invariants:
    * Parent directories are created before writing.
    * Writes occur via ``path.tmp`` followed by ``os.replace`` for atomicity.
    * Functions accept either pandas DataFrames or row iterables for CSV.

Side effects:
    * Writes files directly to disk; callers supply resolved target paths.

Do not:
    * Use these helpers for append-only flows (open in ``a`` instead).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import pandas as pd

JsonLike = Mapping[str, object]
CsvInput = Union[pd.DataFrame, Sequence[Mapping[str, object]]]


def _tmp_path(path: Path) -> Path:
    """Return a sibling temporary path for atomic writes."""
    return path.with_suffix(path.suffix + ".tmp")


def write_atomic_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8") -> None:
    """Write string content to ``path`` atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    with tmp.open("w", encoding=encoding) as handle:
        handle.write(content)
    os.replace(tmp, target)


def write_atomic_jsonl(path: Union[str, Path], rows: Iterable[JsonLike]) -> None:
    """Write iterable of dictionaries as JSON Lines atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(tmp, target)


def write_atomic_csv(path: Union[str, Path], data: CsvInput) -> None:
    """Write a DataFrame or sequence of dicts to CSV atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    if isinstance(data, pd.DataFrame):
        data.to_csv(tmp, index=False)
    else:
        pd.DataFrame(list(data)).to_csv(tmp, index=False)
    os.replace(tmp, target)


def write_atomic_json(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    """Serialize mapping to JSON (utf-8) atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


__all__ = ["write_atomic_text", "write_atomic_jsonl", "write_atomic_csv", "write_atomic_json"]

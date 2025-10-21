"""Shared IO helpers for downloading datasets and writing outputs.

Purpose:
    Centralise filesystem and HTTP utilities so scripts stay focused on data logic.
Inputs:
    Remote CSV URLs, iterables of rows, requested environment variable keys.
Outputs:
    Pandas DataFrames, JSONL/CSV files within /out, environment key/value mappings.
Source(s) of truth:
    Live nflverse/team feeds for downloads; repository /out directory for writes.
Example:
    >>> df = download_csv("https://example.com/data.csv")
    >>> write_jsonl([{'a': 1}], ensure_out_dir() / "demo.jsonl")
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback path
    load_dotenv = None  # type: ignore

RepositoryPath = Path(__file__).resolve().parents[2]
OUT = RepositoryPath / "out"

_ENV_LOADED = False

def load_env_once(override: bool = False) -> None:
    """Load .env into os.environ once (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = RepositoryPath / ".env"
    if load_dotenv and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=override)
    _ENV_LOADED = True

def getenv(key: str, default: str | None = None) -> str | None:
    """Project-safe getenv that ensures .env is loaded once."""
    load_env_once(override=False)
    return os.environ.get(key, default)


def download_csv(url: str) -> pd.DataFrame:
    """Download a CSV from a URL and return it as a DataFrame."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def ensure_out_dir() -> Path:
    """Ensure the repository /out directory exists and return its Path."""
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def week_out_dir(season: int, week: int) -> Path:
    """Return the per-week output directory, creating it if necessary."""
    directory = OUT / f"{season}_week{week}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_jsonl(rows: Iterable[Mapping], path: Union[str, Path]) -> None:
    """Write iterable of dictionaries to JSON Lines file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str))
            fh.write("\n")


def write_csv(df_or_rows: Union[pd.DataFrame, Sequence[Mapping]], path: Union[str, Path]) -> None:
    """Write a DataFrame or sequence of dicts to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(target, index=False)
    else:
        pd.DataFrame(list(df_or_rows)).to_csv(target, index=False)


def read_env(keys: Sequence[str]) -> dict:
    """Return environment variables (loaded from .env if python-dotenv is available)."""
    env_path = RepositoryPath / ".env"
    if load_dotenv and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    values = {}
    for key in keys:
        values[key] = os.environ.get(key)
    return values


__all__ = [
    "download_csv",
    "ensure_out_dir",
    "week_out_dir",
    "write_jsonl",
    "write_csv",
    "read_env",
]

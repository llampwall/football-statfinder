"""Structured run summaries.

Replaces the season-1 print-and-grep observability (a runner script scraping
stdout for a ``NOTIFY:`` line with three regex variants). Each refresh run
appends stage records to a ``RunSummary`` and writes one machine-readable JSON
status file per league run; the runner and the Discord notifier read that file
instead of parsing logs. Logging is standard ``logging`` so levels and
formatting stay configurable.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .common.io_atomic import write_atomic_json
from .paths import OUT_ROOT


def setup_logging(level: int = logging.INFO) -> None:
    """Console logging with UTC timestamps; idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


@dataclass
class StageResult:
    name: str
    ok: bool
    duration_s: float
    counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RunSummary:
    """Collects per-stage outcomes for one league refresh run."""

    league: str
    season: Optional[int] = None
    week: Optional[int] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    stages: List[StageResult] = field(default_factory=list)

    def add(self, stage: StageResult) -> None:
        self.stages.append(stage)

    @property
    def ok(self) -> bool:
        return all(stage.ok for stage in self.stages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "league": self.league,
            "season": self.season,
            "week": self.week,
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stages": [
                {
                    "name": s.name,
                    "ok": s.ok,
                    "duration_s": round(s.duration_s, 3),
                    "counts": s.counts,
                    "notes": s.notes,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }

    def write(self) -> None:
        """Persist to ``out/state/run_summary_{league}.json`` atomically."""
        target = OUT_ROOT / "state" / f"run_summary_{self.league.lower()}.json"
        write_atomic_json(target, self.to_dict())

    def notify_line(self, rows: int) -> str:
        """The one machine-readable NOTIFY line the runner contract expects."""
        return f"NOTIFY: {self.league.upper()} refresh complete week={self.season}-{self.week} rows={rows}"


__all__ = ["RunSummary", "StageResult", "setup_logging"]

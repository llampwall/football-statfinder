"""Run CFB + NFL refresh sequentially, capture logs, persist run metadata, and notify."""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import getenv, write_atomic_json  # noqa: E402

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
NOTE_PREFIXES = ("BACKFILL_MERGE(", "BACKFILL_REPAIR(", "ODDS_REPROMOTE(")


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def run_refresh(module: str, league_tag: str) -> Dict[str, object]:
    cmd = [sys.executable, "-m", module]
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    raw_lines: List[str] = []
    stripped_lines: List[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        print(raw, end="")
        raw = raw.rstrip("\r\n")
        raw_lines.append(raw)
        stripped = strip_ansi(raw)
        stripped_lines.append(stripped)
    proc.wait()
    seconds = time.perf_counter() - start

    notify_line: Optional[str] = None
    notes: List[str] = []

    target_prefix = f"NOTIFY: {league_tag}"

    for line in stripped_lines:
        if notify_line is None and line.startswith(target_prefix):
            notify_line = line
        if line.startswith(NOTE_PREFIXES):
            notes.append(line)

    return {
        "rc": proc.returncode,
        "seconds": seconds,
        "notify": notify_line,
        "notes": notes,
    }


def ensure_logs_dir() -> Path:
    logs_dir = ROOT / "out" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def write_index_line(path: Path, ts_iso: str, cfb_ok: bool, nfl_ok: bool, cfb_sec: float, nfl_sec: float) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        if not exists:
            handle.write("ts_utc\tok_cfb\tok_nfl\tsec_cfb\tsec_nfl\n")
        handle.write(f"{ts_iso}\t{int(cfb_ok)}\t{int(nfl_ok)}\t{cfb_sec:.2f}\t{nfl_sec:.2f}\n")


def post_discord(webhook_url: str, message: str) -> None:
    # Discord: content <= 2000 chars
    payload = json.dumps({"content": str(message)[:1900]}).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        # Avoid Cloudflare 1010 (Python-urllib UA gets blocked)
        "User-Agent": "curl/8.5.0"
    }
    req = Request(webhook_url, data=payload, headers=headers)
    try:
        with urlopen(req, timeout=10) as resp:
            resp.read()
    except HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Discord HTTPError {e.code}: {body}")


def summarize_notes(sections: Iterable[List[str]], limit: int = 3) -> List[str]:
    combined: List[str] = []
    for group in sections:
        for line in group:
            if line not in combined:
                combined.append(line)
            if len(combined) >= limit:
                return combined[:limit]
    return combined[:limit]


def main() -> None:
    logs_dir = ensure_logs_dir()
    ts = datetime.now(timezone.utc)
    ts_iso = ts.isoformat().replace("+00:00", "Z")
    ts_for_filename = ts_iso.replace(":", "-")

    cfb_result = run_refresh("src.refresh_week_data_cfb", "CFB")
    nfl_result = run_refresh("src.refresh_week_data_nfl", "NFL")

    cfb_ok = cfb_result["rc"] == 0 and bool(cfb_result["notify"])
    nfl_ok = nfl_result["rc"] == 0 and bool(nfl_result["notify"])

    payload = {
        "ts_utc": ts_iso,
        "cfb": {
            "ok": cfb_ok,
            "seconds": round(float(cfb_result["seconds"]), 3),
            "notify": cfb_result["notify"],
            "notes": cfb_result["notes"],
        },
        "nfl": {
            "ok": nfl_ok,
            "seconds": round(float(nfl_result["seconds"]), 3),
            "notify": nfl_result["notify"],
            "notes": nfl_result["notes"],
        },
        "host": socket.gethostname(),
    }

    json_path = logs_dir / f"refresh_{ts_for_filename}.json"
    write_atomic_json(json_path, payload)

    index_path = logs_dir / "refresh_index.tsv"
    write_index_line(
        index_path,
        ts_iso,
        cfb_ok,
        nfl_ok,
        float(cfb_result["seconds"]),
        float(nfl_result["seconds"]),
    )

    webhook = (getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if webhook:
        note_lines = summarize_notes([cfb_result["notes"], nfl_result["notes"]])
        note_block = "\n".join(f"- {line}" for line in note_lines) if note_lines else "- (none)"
        message = (
            f"REFRESH SUMMARY (UTC {ts_iso})\n"
            f"CFB → {cfb_result['notify'] or 'missing NOTIFY'}\n"
            f"NFL → {nfl_result['notify'] or 'missing NOTIFY'}\n"
            "notes:\n"
            f"{note_block}"
        )
        try:
            post_discord(webhook, message)
        except URLError as exc:
            print(f"WARN: Discord post failed: {exc}", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"WARN: Discord post failed: {exc}", file=sys.stderr)

    status_line = (
        f"RUNNER NOTIFY: CFB={'ok' if cfb_ok else 'err'} "
        f"NFL={'ok' if nfl_ok else 'err'} "
        f"log={json_path.as_posix()}"
    )
    print(status_line)

    if not (cfb_ok and nfl_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()

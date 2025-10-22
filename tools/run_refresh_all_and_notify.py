"""Run CFB + NFL refresh sequentially, capture logs, persist run metadata, and notify."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import getenv, write_atomic_json  # noqa: E402

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
NOTE_PREFIXES = ("BACKFILL_MERGE(", "BACKFILL_REPAIR(", "ODDS_REPROMOTE(")
NOTE_SUBSTRINGS = ("ODDS_FETCH_ERROR", "EXCEPTION", "Traceback", "HTTPError")
PROMOTED_RE = re.compile(r"promoted=(\d+)", re.IGNORECASE)
NOTIFY_WEEK_RE = re.compile(r"week=(\d{4})-(\d+)", re.IGNORECASE)
NOTIFY_ROWS_RE = re.compile(r"rows=(\d+)", re.IGNORECASE)


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Global Week refresh and post summary.")
    parser.add_argument("--verbose", action="store_true", help="Emit full notes (legacy format).")
    return parser.parse_args(argv)


def _should_capture_line(line: str) -> bool:
    if line.startswith(NOTE_PREFIXES):
        return True
    if any(token in line for token in NOTE_SUBSTRINGS):
        return True
    match = PROMOTED_RE.search(line)
    if match:
        try:
            return int(match.group(1)) > 0
        except ValueError:
            return False
    return False


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
        if _should_capture_line(line):
            notes.append(line)

    tail_line = stripped_lines[-1] if stripped_lines else ""

    return {
        "rc": proc.returncode,
        "seconds": seconds,
        "notify": notify_line,
        "notes": notes,
        "tail": tail_line,
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
    trimmed = str(message)[:1900]
    payload = json.dumps({"content": trimmed}).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "curl/8.5.0",
    }
    req = Request(webhook_url, data=payload, headers=headers)
    try:
        with urlopen(req, timeout=10) as resp:
            resp.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Discord HTTPError {error.code}: {body}")


def note_has_keyword(line: str) -> bool:
    if any(token in line for token in NOTE_SUBSTRINGS):
        return True
    match = PROMOTED_RE.search(line)
    if match:
        try:
            return int(match.group(1)) > 0
        except ValueError:
            return False
    return False


def dedupe_notes(*note_groups: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for group in note_groups:
        for line in group:
            if not line or line in seen:
                continue
            seen.add(line)
            output.append(line)
    return output


def limit_notes(notes: List[str], limit: int = 8) -> List[str]:
    if len(notes) <= limit:
        return notes
    remaining = len(notes) - limit
    return notes[:limit] + [f"... (+{remaining} more)"]


def parse_notify_details(notify: Optional[str]) -> Dict[str, Optional[str]]:
    week = None
    rows: Optional[str] = None
    if notify:
        week_match = NOTIFY_WEEK_RE.search(notify)
        if week_match:
            week = f"{week_match.group(1)}-{week_match.group(2)}"
        rows_match = NOTIFY_ROWS_RE.search(notify)
        if rows_match:
            rows = rows_match.group(1)
    return {"week": week, "rows": rows}


def format_league_line(tag: str, result: Dict[str, object]) -> str:
    seconds = float(result["seconds"])
    rc = int(result["rc"])
    notify = result["notify"]
    details = parse_notify_details(notify)
    ok = rc == 0 and bool(notify)
    if ok:
        week = details["week"] or "unknown"
        rows = details["rows"] or "?"
        return f"✅ {tag} week={week} rows={rows} in {seconds:.1f}s"
    reasons: List[str] = []
    if rc != 0:
        reasons.append(f"exit={rc}")
    if not notify:
        reasons.append("missing NOTIFY")
    reason_text = ", ".join(reasons) if reasons else "unknown issue"
    return f"❌ {tag} ({reason_text}) in {seconds:.1f}s"


def build_verbose_message(ts_iso: str, cfb_result: Dict[str, object], nfl_result: Dict[str, object], json_path: Path) -> str:
    note_lines = dedupe_notes(cfb_result["notes"], nfl_result["notes"])
    note_summary = "\n".join(f"- {line}" for line in limit_notes(note_lines, 20)) if note_lines else "- (none)"
    return (
        f"REFRESH SUMMARY (UTC {ts_iso})\n"
        f"CFB → {cfb_result['notify'] or 'missing NOTIFY'}\n"
        f"NFL → {nfl_result['notify'] or 'missing NOTIFY'}\n"
        "notes:\n"
        f"{note_summary}\n"
        f"log: {json_path.as_posix()}"
    )


def build_compact_message(
    ts_iso: str,
    cfb_result: Dict[str, object],
    nfl_result: Dict[str, object],
    json_path: Path,
    show_notes: bool,
) -> str:
    cfb_ok = cfb_result["rc"] == 0 and bool(cfb_result["notify"])
    nfl_ok = nfl_result["rc"] == 0 and bool(nfl_result["notify"])
    header_icon = "✅" if (cfb_ok and nfl_ok) else "⚠️"

    lines = [
        f"REFRESH {header_icon} (UTC {ts_iso})",
        format_league_line("CFB", cfb_result),
        format_league_line("NFL", nfl_result),
    ]

    if show_notes:
        combined_notes = dedupe_notes(
            cfb_result["notes"],
            nfl_result["notes"],
        )
        extra: List[str] = []
        if not cfb_ok and cfb_result.get("tail"):
            extra.append(f"stderr(tail): {cfb_result['tail']}")
        if not nfl_ok and nfl_result.get("tail"):
            extra.append(f"stderr(tail): {nfl_result['tail']}")
        combined_notes.extend(note for note in extra if note)
        combined_notes = dedupe_notes(combined_notes)
        limited = limit_notes(combined_notes, 8)
        if limited:
            lines.append("")
            lines.append("notes:")
            lines.extend(f"- {line}" for line in limited)

    lines.append(f"log: {json_path.as_posix()}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

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
        show_notes = args.verbose
        if not show_notes:
            combined_notes = dedupe_notes(
                cfb_result["notes"],
                nfl_result["notes"],
            )
            show_notes = (not cfb_ok or not nfl_ok) or any(note_has_keyword(line) for line in combined_notes)
        if args.verbose:
            message = build_verbose_message(ts_iso, cfb_result, nfl_result, json_path)
        else:
            message = build_compact_message(ts_iso, cfb_result, nfl_result, json_path, show_notes)
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

# tools/run_refresh_cfb_and_notify.py
"""
Runs the CFB refresh (global-week) and posts to Discord only when summary changes.

Requires:
  - Env DISCORD_WEBHOOK_URL = your webhook for #codex-notify
  - Python on PATH
Writes:
  - out/state/last_refresh_summary_cfb.json (for delta detection)
"""

import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
from src.common.io_utils import read_env

# Ensure repo root is on sys.path so "import src.…" works no matter where we run from
ROOT = Path(__file__).resolve().parents[1]  # repo/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

RE_LINE_PROMO = re.compile(r"^CFB ODDS PROMOTION:\s*week=(\d{4})-(\d+)\s+promoted=(\d+)\b.*$")
RE_LINE_DONE  = re.compile(r"^NOTIFY:\s*CFB refresh complete week=(\d{4})-(\d+)\s+rows=(\d+)\s+odds_promoted=(\d+)\b")

STATE_PATH = os.path.join("out", "state", "last_refresh_summary_cfb.json")

def post_to_discord(webhook_url: str, content: str) -> None:
    data = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as _:
        pass

def load_last():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_current(obj):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)

def main():
    env = read_env(["DISCORD_WEBHOOK_URL"])
    webhook = env.get("DISCORD_WEBHOOK_URL").strip()
    # webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print("WARN: DISCORD_WEBHOOK_URL not set; running refresh without notify.", file=sys.stderr)

    # Run the refresh
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.refresh_week_data_cfb"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    promo = None
    done  = None
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line)
        lines.append(line)
        m1 = RE_LINE_PROMO.match(line)
        if m1:
            y, w, promoted = m1.groups()
            promo = {"season": int(y), "week": int(w), "promoted": int(promoted)}
        m2 = RE_LINE_DONE.match(line)
        if m2:
            y, w, rows, promoted = m2.groups()
            done = {"season": int(y), "week": int(w), "rows": int(rows), "promoted": int(promoted)}

    proc.wait()
    rc = proc.returncode

    # Build summary
    summary = {
        "rc": rc,
        "ts": int(time.time()),
        "promo": promo or {},
        "done":  done  or {},
    }

    # Decide whether to notify
    last = load_last()
    should_notify = False
    reason = []

    if rc != 0:
        should_notify = True
        reason.append("refresh failed")

    if summary.get("done") != last.get("done"):
        should_notify = True
        reason.append("summary changed")

    # Strong signal: odds promoted increased or flipped from 0 -> >0
    if (summary.get("done", {}).get("promoted", 0) or 0) != (last.get("done", {}).get("promoted", 0) or 0):
        should_notify = True
        reason.append("odds_promoted changed")

    save_current(summary)

    if webhook and should_notify:
        content = (
            f"**CFB Refresh** — {', '.join(reason)}\n"
            f"week={summary.get('done',{}).get('season','?')}-{summary.get('done',{}).get('week','?')}, "
            f"rows={summary.get('done',{}).get('rows','?')}, "
            f"odds_promoted={summary.get('done',{}).get('promoted','?')}\n"
            "```text\n" +
            "\n".join([s for s in lines if s.startswith(("CFB ODDS STAGING", "Sagarin(CFB)", "Scores(CFB)", "CFB ODDS PROMOTION", "NOTIFY: CFB refresh"))]) +
            "\n```"
        )
        try:
            post_to_discord(webhook, content)
            print("NOTIFY: Discord posted.")
        except Exception as e:
            print(f"WARN: Discord post failed: {e}", file=sys.stderr)

    sys.exit(rc)

if __name__ == "__main__":
    main()

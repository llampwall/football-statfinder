# AGENTS.md

**Mission**  
Ship features and fixes **safely and deterministically** under Policy + Canon.

**Scope**  
All code paths in this repo; prefer surgical patches; never break logging/NOTIFY contracts.

**Must Read Inputs**  
- Canon: `context/ats_api_backfill_spec.md`, `context/ats_backfill_flow.md`, `context/merge_summary_global_week.md`
- Guidelines: `context/CODEX_RULES.md`
- (When relevant) SPEC/UI notes

**Guardrails**
- No schema/path changes without updating canon.
- Atomic writes; idempotent re-runs; UTC timestamps.
- Keep “Surgical Patch” surface area small; update docs when contracts change.
- Env vars come from getenv() (src.common.io_utils); defaults safe; no silent “magic” flags.
- One NOTIFY: per league run.
- Clear acceptance steps before code.

**Deliverables**  
1) **Plan** (files to touch, risks, acceptance checks) before code.  
2) **Change** (minimal diff; no schema churn).  
3) **Diff Note** (any log/contract/env changes).  
4) **Transcript** excerpt proving Acceptance checks.

**Acceptance checks (verify before “done”)**
- Logs include: computed CurrentWeek banner; staging/ratings/backfill/ATS summaries; promotion counts.
- Exactly one final line:  
  `NOTIFY: <LEAGUE> refresh complete week=<season>-<week> rows=<n>`
- Run discord notify to alert me
- Evidence: include console transcript snippet per Run/Evidence policy.

**Runbook**  
- compile: `python -m compileall <touched modules>`
- main full auto refresh: `python tools/run_refresh_all_and_notify`
- NFL manual refresh: `python -m src.refresh_week_data_nfl [--season S --week W]`
- CFB manual refresh: `python -m src.refresh_week_data_cfb [--season S --week W]`
- Notify me of task completion: `powershell -NoProfile -ExecutionPolicy Bypass -File .webhooks\discord_notify.ps1 -Message "<one sentence summary of the work done>"`


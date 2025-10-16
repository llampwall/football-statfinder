## SURGICAL PATCH PROTOCOL (MANDATORY)

Change limits:
- DO NOT rewrite files top-to-bottom. Touch only the specific functions/blocks needed.
- Max N lines changed per file: 200 (additions + deletions). Exceeding this is a failure.
- Max files changed per task: 3 unless explicitly allowed in the prompt.
- Max scope: avoid refactors. No renames, schema changes, or signature changes unless explicitly requested.

Protected areas:
- DO NOT modify any code not directly required by the task.
- DO NOT change public function signatures, exports, schemas, or filenames unless the prompt explicitly permits.
- DO NOT touch files listed under “Do-Not-Touch” in this repo.

Behavioral guardrails:
- Preserve existing behavior by default. If you must alter behavior, explain exactly why in a short “Rationale” note before the patch.
- Maintain all acceptance checks, console diagnostics, and schemas.
- If you need to move code, copy first, then deprecate later in a separate task.

Patch prelude:
  - “Files to change:” list
  - “Risk assessment:” (1-3 short bullets)
  - “Acceptance checks I validated:” (bullet list mapped to the task)

Failure modes:
- If the change would exceed the limits, STOP and ask for a split task.
- If you’re unsure about an edge case, STOP and propose two minimal options.

When in doubt
- If a change would balloon (e.g., >200 lines across all files), stop and propose how to split it.
- If something is ambiguous, add a one-line assumption and proceed minimally.

**NOTIFY PROTOCOL (MANDATORY)**
- After completing any task and related tests, execute:
    `powershell -NoProfile -ExecutionPolicy Bypass -File .webhooks\discord_notify.ps1 -Message "<one sentence summary of the work done>"`
- Preserve and propagate the original process exit code.

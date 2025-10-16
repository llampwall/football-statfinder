## SURGICAL PATCH PROTOCOL (MANDATORY)

You DO NOT rewrite whole files. You ONLY produce minimal diffs.

Output format:
- For each file you touch, emit a unified diff (git-style) fenced in ```diff.
- Include enough @@ context so the patch applies cleanly.
- If creating a new file, include the full file within a diff header that starts with: 
  --- /dev/null
  +++ b/<path>
- Never include unrelated whitespace reflows, re-orderings, or “style cleanups”.

Change limits:
- Max N lines changed per file: 80 (additions + deletions). Exceeding this is a failure.
- Max files changed per task: 3 unless explicitly allowed in the prompt.

Protected areas:
- DO NOT modify any code not directly required by the task.
- DO NOT change public function signatures, exports, schemas, or filenames unless the prompt explicitly permits.
- DO NOT touch files listed under “Do-Not-Touch” in this repo.

Behavioral guardrails:
- Preserve existing behavior by default. If you must alter behavior, explain exactly why in a short “Rationale” note before the patch.
- Maintain all acceptance checks, console diagnostics, and schemas.
- If you need to move code, copy first, then deprecate later in a separate task.

Patch prelude:
- Before the diff, print:
  - “Files to change:” list
  - “Risk assessment:” (1-3 short bullets)
  - “Acceptance checks I validated:” (bullet list mapped to the task)

Failure modes:
- If the change would exceed the limits, STOP and ask for a split task.
- If you’re unsure about an edge case, STOP and propose two minimal options.

**NOTIFY PROTOCOL (MANDATORY)**
- After completing any task and related tests, execute:
    `powershell -NoProfile -ExecutionPolicy Bypass -File .webhooks\discord_notify.ps1 -Message "<one sentence summary of the work done>"`
- Preserve and propagate the original process exit code.

"""Refuse a commit that touches `models/` unless its message declares the retrain.

    python .githooks/check_retrain_declaration.py <path-to-commit-message-file>

Why this exists
---------------
Three commits in this repository's history are titled *"Refactor code structure
for improved readability and maintainability"* and each one contains a production
retrain:

    82b45ee  2026-08-??   retrain
    f2645a0  2026-08-15   retrain -- 30 artifacts, AND a re-read of the one-shot
                          test block in models/volatility/vol_metrics.json
    d61d033  2026-08-??   retrain

The damage was not the retraining. It was that nobody could see it had happened.
The checksum guard in `tests/fixtures/*_protected_sha256.json` catches a moved
artifact, but only *afterwards*, and only if someone runs the suite and reads the
failure rather than re-baselining it away. By then the retrain is already history
under a message describing something else, and in `f2645a0` it had already spent
a one-shot evaluation block that the methodology reserves for a single report.

This check moves the detection to the moment the commit is written, where it is
still free to fix.

Why `commit-msg` and not `pre-commit`
-------------------------------------
The rule needs two facts at once: WHICH PATHS are staged, and WHAT THE MESSAGE
SAYS. `pre-commit` runs before a message exists, so it can only see the first.
`commit-msg` is the earliest hook that sees both, so that is where this runs. The
name is the only thing that differs from a "pre-commit check" -- it still fails
the commit before anything is written to the branch.

The rule
--------
If any staged path is under `models/`, the message must carry a line beginning
with `RETRAIN:` followed by something. Otherwise the commit is refused.

`RETRAIN:` with nothing after it does not count -- the point is a declaration a
reader can act on, not a token that silences a hook. Comment lines are stripped
first, exactly as git strips them, so a `RETRAIN:` that would never appear in the
final message cannot satisfy the check.

Merge commits are exempt. A merge carries changes that were already committed on
another branch, where this same check ran; failing the merge would only punish
the person integrating.

Limits, stated honestly
-----------------------
`git commit --no-verify` bypasses this, as it bypasses every hook. This guard is
here to stop the accident -- the retrain nobody noticed they were committing --
not to stop a determined author. The checksum fixtures remain the second line of
defence for exactly that reason.
"""

from __future__ import annotations

import subprocess
import sys

DECLARATION = "RETRAIN:"
WATCHED_PREFIX = "models/"


def staged_paths() -> list[str]:
    """Paths staged for this commit, as forward-slash repo-relative strings."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in out.stdout.splitlines() if line.strip()]


def watched(paths: list[str]) -> list[str]:
    """The staged paths this rule cares about: anything under `models/`."""
    return sorted(p for p in paths if p.startswith(WATCHED_PREFIX))


def strip_comments(message: str) -> str:
    """Drop the lines git itself drops, so the check sees the real message.

    Without this, a `RETRAIN:` written on a `#` line would satisfy the hook and
    then vanish from the commit -- the exact invisibility this guard exists to
    prevent.
    """
    kept = [ln for ln in message.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(kept)


def declares_retrain(message: str) -> bool:
    """True if some line declares a retrain AND says something about it."""
    for line in strip_comments(message).splitlines():
        stripped = line.strip()
        if stripped.startswith(DECLARATION) and stripped[len(DECLARATION):].strip():
            return True
    return False


def is_merge() -> bool:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "MERGE_HEAD"], capture_output=True, text=True
    )
    if out.returncode != 0:
        return False
    import os

    return os.path.exists(out.stdout.strip())


def explain(offenders: list[str]) -> str:
    shown = offenders[:12]
    listing = "\n".join(f"    {p}" for p in shown)
    if len(offenders) > len(shown):
        listing += f"\n    ... and {len(offenders) - len(shown)} more"
    return f"""
COMMIT REFUSED -- this commit touches models/ but does not declare a retrain.

  {len(offenders)} staged path(s) under {WATCHED_PREFIX}:
{listing}

  Add a line to the commit message that starts with {DECLARATION} and says what
  was retrained and why. For example:

      RETRAIN: volatility 5-seed ensemble, refit on the 8,605-row set

  Three commits in this repository titled "Refactor code structure for improved
  readability and maintainability" each hid a production retrain. One of them
  (f2645a0) also re-read the one-shot test block in vol_metrics.json, spending an
  evaluation block the methodology reserves for a single final report. Nobody saw
  it for four days because the message described something else.

  If you did NOT mean to retrain anything, unstage the artifacts instead:

      git restore --staged models/

  {DECLARATION} with nothing after it does not count.
"""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_retrain_declaration.py <commit-message-file>", file=sys.stderr)
        return 2

    if is_merge():
        return 0

    offenders = watched(staged_paths())
    if not offenders:
        return 0

    try:
        with open(argv[1], encoding="utf-8", errors="replace") as fh:
            message = fh.read()
    except OSError as exc:
        print(f"could not read the commit message file: {exc}", file=sys.stderr)
        return 2

    if declares_retrain(message):
        return 0

    print(explain(offenders), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

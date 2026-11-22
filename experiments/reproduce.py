"""
experiments/reproduce.py - cross-platform twin of `make reproduce-fast`.

Re-evaluates the committed PPO models on the committed held-out sets for all
four regimes, regenerates the statistical summaries, and re-evaluates the 30
multi-seed models without retraining. No API key, no training, about a minute
on a laptop CPU. Works where GNU Make is not installed (for example Windows).

Run from the repository root:

    python -m experiments.reproduce            # regenerate the results
    python -m experiments.reproduce --check    # ... then confirm nothing changed

With --check, the committed results are the reference: the run passes only if
`git status` shows no change under experiments/, models/ or data/ afterwards.
Keep the Makefile's reproduce-fast target and STEPS below in step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("raw", "calibrated", "regime", "regime_raw")
WATCHED = ("experiments", "models", "data")

STEPS: list[list[str]] = []
for _ds in DATASETS:
    STEPS.append(["agent.evaluate", "--dataset", _ds])
    STEPS.append(["experiments.statistical_analysis", "--dataset", _ds])
STEPS.append([
    "experiments.multi_seed", "--datasets", "raw", "regime",
    "--variants", "A", "B", "C", "--seeds", "0", "1", "2", "3", "4", "--eval-only",
])


def changed_files() -> list[str] | None:
    """Files under WATCHED that differ from the last commit, or None without git."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *WATCHED],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true",
        help="After the run, fail unless no file under experiments/, models/ or "
             "data/ differs from the last commit.",
    )
    args = parser.parse_args()

    if args.check:
        before = changed_files()
        if before is None:
            raise SystemExit("--check needs a git checkout with git on PATH.")
        if before:
            raise SystemExit(
                "--check needs a clean tree under experiments/, models/ and data/ "
                "to compare against; these already differ:\n  " + "\n  ".join(before)
            )

    start = time.time()
    for i, step in enumerate(STEPS, start=1):
        print(f"=== [{i}/{len(STEPS)}] python -m {' '.join(step)}", flush=True)
        result = subprocess.run([sys.executable, "-m", *step], cwd=REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Step failed (exit {result.returncode}): python -m {' '.join(step)}")
    print(f"\nAll {len(STEPS)} steps finished in {time.time() - start:.0f} s. "
          "See experiments/results/.")

    if args.check:
        after = changed_files() or []
        if after:
            print("\nCHECK FAILED: the rerun changed committed files:")
            for line in after:
                print("  " + line)
            print("Inspect with `git diff`; restore with `git checkout -- experiments models data`.")
            raise SystemExit(1)
        print("\nCHECK PASSED: every regenerated file is identical to the committed one.")


if __name__ == "__main__":
    main()

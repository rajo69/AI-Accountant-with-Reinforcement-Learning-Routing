"""
experiments/multi_seed.py — multi-seed robustness sweep.

For each (dataset, variant, seed), trains a PPO policy and evaluates it on the
matching held-out set. Reports mean ± std of headline metrics across seeds.

This exists to answer the standard reviewer question "what about seed
variance?" The canonical headline numbers in the README come from a single
seed (42); this sweep tests whether the tier-level policy learned by PPO is
robust across different random-seed initialisations of the PPO trainer.

By default, sweeps 5 seeds x 3 variants x {raw, regime} = 30 training runs
(~10 min each at 100k timesteps), then evaluates each. The per-seed models
are NOT committed to git — this file produces only a summary JSON + Markdown
for each regime.

Usage
-----
    # Full default sweep (~5 hours of training on a consumer CPU):
    python -m experiments.multi_seed

    # Narrower sweep:
    python -m experiments.multi_seed --datasets raw --seeds 0 1 2 --variants A

    # Skip the training step (models must already exist under the multi-seed
    # naming convention — useful when the sweep was interrupted):
    python -m experiments.multi_seed --eval-only

Output
------
    experiments/results/multi_seed_summary_{dataset}.json
    experiments/results/multi_seed_summary_{dataset}.md
    models/trained/ppo_variant_{V}{dataset_tag}_seed{N}.zip
    models/trained/best_ppo_variant_{V}{dataset_tag}_seed{N}/
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINED_DIR = REPO_ROOT / "models" / "trained"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DATA_EVAL_DIR = REPO_ROOT / "data" / "evaluation"

DATASET_TAGS = {
    "raw":         "",
    "calibrated":  "_calibrated",
    "regime":      "_regime",
    "regime_raw":  "_regime_raw",
}
EVAL_FILES = {
    "raw":         DATA_EVAL_DIR / "held_out_set.json",
    "calibrated":  DATA_EVAL_DIR / "held_out_set_calibrated.json",
    "regime":      DATA_EVAL_DIR / "held_out_set_regime.json",
    "regime_raw":  DATA_EVAL_DIR / "held_out_set_regime_raw.json",
}

METRIC_KEYS = (
    "overall_routing_accuracy",
    "auto_approval_precision",
    "auto_approval_rate",
    "error_rate_auto",
)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_VARIANTS = ["A", "B", "C"]
DEFAULT_DATASETS = ["raw", "regime"]


def train_one(variant: str, dataset: str, seed: int) -> tuple[bool, float]:
    """Subprocess a single training run. Returns (success, wallclock_seconds)."""
    suffix = f"_seed{seed}"
    model_path = TRAINED_DIR / f"ppo_variant_{variant}{DATASET_TAGS[dataset]}{suffix}.zip"

    cmd = [
        sys.executable, "-m", "agent.train",
        "--reward", variant,
        "--dataset", dataset,
        "--seed", str(seed),
        "--suffix", suffix,
    ]

    start = time.time()
    print(f"[multi_seed] Training variant={variant} dataset={dataset} seed={seed}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    wallclock = time.time() - start

    if result.returncode != 0:
        print(f"[multi_seed] FAILED variant={variant} seed={seed} dataset={dataset}")
        print(result.stderr[-2000:])
        return False, wallclock
    if not model_path.exists():
        print(f"[multi_seed] FAILED: model not written: {model_path}")
        return False, wallclock
    print(f"[multi_seed]   done in {wallclock:.0f}s -> {model_path.name}", flush=True)
    return True, wallclock


def evaluate_one(variant: str, dataset: str, seed: int) -> dict[str, Any]:
    """Evaluate one (variant, seed, dataset) model on its held-out set.

    Uses the same metric functions as agent.evaluate so numbers are directly
    comparable. Returns a flat dict of the headline metrics.
    """
    # Lazy import — avoids loading SB3 when running --help or aggregation-only.
    from agent.evaluate import evaluate_ppo, load_held_out_set

    records = load_held_out_set(EVAL_FILES[dataset])
    tag = f"{DATASET_TAGS[dataset]}_seed{seed}"
    metrics = evaluate_ppo(records, variant, tag=tag)

    # Surface the key metrics in a flat shape for easy aggregation.
    return {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "action_totals": metrics.get("action_totals"),
        **{k: metrics.get(k) for k in METRIC_KEYS},
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean / std / min / max of each metric, grouped by variant."""
    out: dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        variant_summary: dict[str, Any] = {
            "n_seeds": len(subset),
            "seeds": sorted(r["seed"] for r in subset),
        }
        for key in METRIC_KEYS:
            vals = [r[key] for r in subset if r[key] is not None]
            if not vals:
                variant_summary[key] = {"mean": None, "std": None, "min": None, "max": None}
                continue
            variant_summary[key] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
        # Action totals should be invariant across seeds on this problem.
        # Report the set of distinct action-totals observed.
        unique_totals = {json.dumps(r["action_totals"], sort_keys=True) for r in subset}
        variant_summary["distinct_action_totals"] = [json.loads(t) for t in unique_totals]
        out[variant] = variant_summary
    return out


def write_markdown(dataset: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    """Emit a human-readable markdown table for the README."""
    seeds_used = sorted({r["seed"] for r in rows})
    lines = []
    lines.append(f"# Multi-seed robustness — dataset `{dataset}`")
    lines.append("")
    lines.append(
        f"Each row reports mean across {len(seeds_used)} seeds "
        f"({', '.join(str(s) for s in seeds_used)}), with the population "
        "standard deviation in parentheses. Identical action totals across "
        "all seeds (where listed) indicate that the tier-level policy is "
        "seed-invariant for the given (variant, regime)."
    )
    lines.append("")
    lines.append("| Variant | Routing accuracy | Auto-precision | Auto-rate | Auto-error-rate | Distinct action totals |")
    lines.append("|---|:---:|:---:|:---:|:---:|---|")
    for variant in sorted(summary):
        s = summary[variant]
        def cell(k: str) -> str:
            v = s[k]
            if v["mean"] is None:
                return "N/A"
            return f"{v['mean']*100:.1f}% ({v['std']*100:.2f}pp)"
        totals = s["distinct_action_totals"]
        if len(totals) == 1:
            t = totals[0]
            totals_str = f"AUTO={t.get('AUTO_APPROVE')}, SURFACE={t.get('SURFACE_FOR_REVIEW')}, REJECT={t.get('REJECT_FOR_MANUAL')} (all seeds)"
        else:
            totals_str = f"{len(totals)} distinct outcomes across seeds"
        lines.append(
            f"| PPO-{variant} | {cell('overall_routing_accuracy')} | "
            f"{cell('auto_approval_precision')} | "
            f"{cell('auto_approval_rate')} | "
            f"{cell('error_rate_auto')} | {totals_str} |"
        )
    lines.append("")
    lines.append(f"Per-seed raw numbers are in `multi_seed_summary_{dataset}.json`.")
    md_path = RESULTS_DIR / f"multi_seed_summary_{dataset}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def run_sweep(
    datasets: list[str],
    variants: list[str],
    seeds: list[int],
    eval_only: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(datasets) * len(variants) * len(seeds)
    run_i = 0
    overall_start = time.time()

    for dataset in datasets:
        dataset_rows: list[dict[str, Any]] = []
        train_failures: list[tuple[str, int]] = []

        for variant in variants:
            for seed in seeds:
                run_i += 1

                if not eval_only:
                    print(f"\n[multi_seed] ==== run {run_i}/{total_runs} ====", flush=True)
                    ok, secs = train_one(variant, dataset, seed)
                    if not ok:
                        train_failures.append((variant, seed))
                        continue
                try:
                    metrics = evaluate_one(variant, dataset, seed)
                    dataset_rows.append(metrics)
                    print(
                        f"[multi_seed]   eval: routing={metrics['overall_routing_accuracy']:.4f}, "
                        f"auto_rate={metrics['auto_approval_rate']:.4f}, "
                        f"action_totals={metrics['action_totals']}",
                        flush=True,
                    )
                except FileNotFoundError as e:
                    print(f"[multi_seed] FAILED evaluation: {e}")
                    train_failures.append((variant, seed))

        # Summary for this dataset
        summary = aggregate(dataset_rows)
        payload = {
            "dataset": dataset,
            "n_seeds": len(seeds),
            "seeds_requested": seeds,
            "variants": variants,
            "total_runs": total_runs,
            "train_failures": [{"variant": v, "seed": s} for v, s in train_failures],
            "per_run": dataset_rows,
            "summary": summary,
        }
        json_path = RESULTS_DIR / f"multi_seed_summary_{dataset}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path = write_markdown(dataset, dataset_rows, summary)
        print(f"[multi_seed] wrote {json_path.name} and {md_path.name}", flush=True)

    wall = time.time() - overall_start
    print(f"\n[multi_seed] total wallclock: {wall/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed robustness sweep.")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASET_TAGS))
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; only evaluate existing models.")
    args = ap.parse_args()
    run_sweep(args.datasets, args.variants, args.seeds, args.eval_only)


if __name__ == "__main__":
    main()

"""
agent/evaluate.py — Evaluation and comparison of routing policies.

Evaluates the hand-tuned baseline and all three PPO variants on the held-out
set. Produces a comparison table to stdout and saves structured results to
experiments/results/evaluation_results.json.

Usage:
    python -m agent.evaluate

Metrics computed per policy:
    - overall_routing_accuracy:   fraction of transactions routed to the
                                  "optimal" action given is_correct flag
    - auto_approval_precision:    of auto-approved txns, fraction that were
                                  correct (is_correct=True)
    - auto_approval_rate:         fraction of all txns routed to AUTO_APPROVE
    - unnecessary_escalation_rate: fraction of all txns that were correct AND
                                   surfaced for review (wasted accountant time)
    - error_rate_auto:            fraction of auto-approved txns that were wrong
    - error_rate_by_tier:         error_rate_auto broken down by difficulty tier
    - routing_confusion_matrix:   3x3 confusion over routing actions
      rows=difficulty_tier (easy/medium/hard), columns=action (0/1/2)

"Optimal" routing decision for each transaction:
    is_correct=True  → AUTO_APPROVE (0) is optimal
    is_correct=False → SURFACE_FOR_REVIEW (1) is optimal (REJECT is also
                       acceptable but costlier; we score 1 as the baseline
                       "correct" human escalation)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from agent.baseline import make_baseline_policy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINED_DIR = REPO_ROOT / "models" / "trained"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# Held-out sets: "raw" uses original Claude Haiku confidences; "calibrated"
# uses the Platt-scaled output from experiments/calibrate.py.
EVAL_DATASETS = {
    "raw":        REPO_ROOT / "data" / "evaluation" / "held_out_set.json",
    "calibrated": REPO_ROOT / "data" / "evaluation" / "held_out_set_calibrated.json",
}

DIFFICULTY_NAMES = {0: "easy", 1: "medium", 2: "hard"}
ACTION_NAMES = {0: "AUTO_APPROVE", 1: "SURFACE_FOR_REVIEW", 2: "REJECT_FOR_MANUAL"}

# Optimal action given is_correct:
#   correct prediction → auto-approve is optimal
#   wrong prediction   → surface for review is optimal (escalation warranted)
OPTIMAL_ACTION = {True: 0, False: 1}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_held_out_set(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    # Normalise: held_out records may have features nested or flat
    records = []
    for rec in data:
        features = rec.get("features", {})
        records.append({
            "transaction_id": rec["transaction_id"],
            "confidence_score": float(rec.get("confidence_score", features.get("confidence_score", 0.0))),
            "amount_normalised": float(features.get("amount_normalised", rec.get("amount_normalised", 0.0))),
            "difficulty_tier": int(rec.get("difficulty_tier", features.get("difficulty_tier", 0))),
            "category_entropy": float(features.get("category_entropy", rec.get("category_entropy", 0.0))),
            "is_correct": bool(rec["is_correct"]),
        })
    return records


def make_obs(rec: dict, reward_variant: str) -> np.ndarray:
    """Build observation array matching RoutingEnv._make_obs()."""
    obs = [
        rec["confidence_score"],
        rec["amount_normalised"],
        float(rec["difficulty_tier"]),
        rec["category_entropy"],
    ]
    if reward_variant == "B":
        obs.append(0.5)  # neutral accountant load for evaluation
    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    records: list[dict],
    actions: list[int],
    policy_name: str,
) -> dict[str, Any]:
    """Compute all evaluation metrics for a set of routing decisions."""
    n = len(records)
    assert len(actions) == n

    # Per-tier accumulators: {tier: {action: count}}
    tier_action_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Per-tier auto-approval errors: {tier: [wrong_auto_approvals]}
    tier_auto_errors: dict[int, list[bool]] = defaultdict(list)

    optimal_count = 0
    auto_correct = 0    # auto-approved AND is_correct=True
    auto_wrong = 0      # auto-approved AND is_correct=False
    unnecessary_esc = 0 # is_correct=True AND SURFACE_FOR_REVIEW

    for rec, action in zip(records, actions):
        tier = rec["difficulty_tier"]
        is_correct = rec["is_correct"]

        tier_action_counts[tier][action] += 1

        if action == OPTIMAL_ACTION[is_correct]:
            optimal_count += 1

        if action == 0:  # AUTO_APPROVE
            if is_correct:
                auto_correct += 1
            else:
                auto_wrong += 1
            tier_auto_errors[tier].append(not is_correct)

        if action == 1 and is_correct:
            unnecessary_esc += 1

    total_auto = auto_correct + auto_wrong
    auto_precision = auto_correct / total_auto if total_auto > 0 else float("nan")
    error_rate_auto = auto_wrong / total_auto if total_auto > 0 else float("nan")

    # Per-tier error rate
    error_rate_by_tier: dict[str, float] = {}
    for tier in range(3):
        errs = tier_auto_errors[tier]
        if errs:
            error_rate_by_tier[DIFFICULTY_NAMES[tier]] = sum(errs) / len(errs)
        else:
            error_rate_by_tier[DIFFICULTY_NAMES[tier]] = float("nan")

    # Routing action distribution: {tier_name: {action_name: count}}
    action_distribution: dict[str, dict[str, int]] = {}
    for tier in range(3):
        action_distribution[DIFFICULTY_NAMES[tier]] = {
            ACTION_NAMES[a]: tier_action_counts[tier][a]
            for a in range(3)
        }

    # Flat action distribution totals
    action_totals = {ACTION_NAMES[a]: sum(tier_action_counts[t][a] for t in range(3)) for a in range(3)}

    return {
        "policy": policy_name,
        "n_transactions": n,
        "overall_routing_accuracy": round(optimal_count / n, 4),
        "auto_approval_precision": round(auto_precision, 4) if not (isinstance(auto_precision, float) and auto_precision != auto_precision) else None,
        "auto_approval_rate": round(total_auto / n, 4),
        "unnecessary_escalation_rate": round(unnecessary_esc / n, 4),
        "error_rate_auto": round(error_rate_auto, 4) if not (isinstance(error_rate_auto, float) and error_rate_auto != error_rate_auto) else None,
        "error_rate_by_tier": {k: (round(v, 4) if v == v else None) for k, v in error_rate_by_tier.items()},
        "action_distribution_by_tier": action_distribution,
        "action_totals": action_totals,
    }


# ---------------------------------------------------------------------------
# Policy runners
# ---------------------------------------------------------------------------

def evaluate_baseline(records: list[dict]) -> dict[str, Any]:
    policy = make_baseline_policy()
    actions = []
    for rec in records:
        obs = make_obs(rec, "A")  # baseline uses 4-dim obs
        action, _ = policy.predict(obs)
        actions.append(int(action))
    return compute_metrics(records, actions, policy_name="Baseline (threshold 0.85/0.50)")


def evaluate_ppo(records: list[dict], variant: str, tag: str = "") -> dict[str, Any]:
    model_path = TRAINED_DIR / f"ppo_variant_{variant}{tag}.zip"
    if not model_path.exists():
        print(f"  [WARN] Model not found: {model_path} — skipping variant {variant}")
        return {"policy": f"PPO Variant {variant}", "error": "model not found"}

    model = PPO.load(str(model_path))
    actions = []
    for rec in records:
        obs = make_obs(rec, variant)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
    return compute_metrics(records, actions, policy_name=f"PPO Variant {variant}")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

METRIC_LABELS = [
    ("overall_routing_accuracy",   "Routing Accuracy"),
    ("auto_approval_precision",    "Auto-Approval Precision"),
    ("auto_approval_rate",         "Auto-Approval Rate"),
    ("unnecessary_escalation_rate","Unnecessary Escalation Rate"),
    ("error_rate_auto",            "Error Rate (auto-approved)"),
]


def fmt(val: float | None) -> str:
    if val is None:
        return "  N/A  "
    return f"{val:.1%}"


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    policies = [r["policy"] for r in results]
    col_width = max(len(p) for p in policies) + 2

    header = f"{'Metric':<35}" + "".join(f"{p:>{col_width}}" for p in policies)
    print("\n" + "=" * len(header))
    print("ROUTING POLICY COMPARISON — HELD-OUT SET")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key, label in METRIC_LABELS:
        row = f"{label:<35}"
        for r in results:
            val = r.get(key)
            row += f"{fmt(val):>{col_width}}"
        print(row)

    # Per-tier error rates
    print("-" * len(header))
    print("Error Rate (auto-approved) by Difficulty Tier:")
    for tier in ("easy", "medium", "hard"):
        row = f"  {tier.capitalize():<33}"
        for r in results:
            by_tier = r.get("error_rate_by_tier", {})
            val = by_tier.get(tier)
            row += f"{fmt(val):>{col_width}}"
        print(row)

    # Action distribution totals
    print("-" * len(header))
    print("Action Distribution (totals):")
    for action_name in ("AUTO_APPROVE", "SURFACE_FOR_REVIEW", "REJECT_FOR_MANUAL"):
        row = f"  {action_name:<33}"
        for r in results:
            totals = r.get("action_totals", {})
            count = totals.get(action_name, 0)
            row += f"{count:>{col_width-2}}  " if col_width > 2 else f"{count:>{col_width}}"
        print(row)

    print("=" * len(header))
    print(f"  Evaluation set: {results[0]['n_transactions']} held-out transactions\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and PPO routing policies on the held-out set."
    )
    parser.add_argument(
        "--dataset",
        choices=list(EVAL_DATASETS),
        default="raw",
        help="Which held-out set and PPO models to evaluate. 'raw' uses the "
             "original Claude Haiku confidences and ppo_variant_{A,B,C}.zip; "
             "'calibrated' uses the Platt-scaled eval set and "
             "ppo_variant_{A,B,C}_calibrated.zip.",
    )
    args = parser.parse_args()
    dataset: str = args.dataset
    held_out_path = EVAL_DATASETS[dataset]
    tag = "" if dataset == "raw" else f"_{dataset}"
    results_filename = f"evaluation_results{tag}.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}  ({held_out_path})")
    print("Loading held-out evaluation set...")
    records = load_held_out_set(held_out_path)
    print(f"  Loaded {len(records)} transactions")

    tier_counts: defaultdict[int, int] = defaultdict(int)
    correct_counts: defaultdict[int, int] = defaultdict(int)
    for rec in records:
        tier_counts[rec["difficulty_tier"]] += 1
        if rec["is_correct"]:
            correct_counts[rec["difficulty_tier"]] += 1

    print("  Distribution:")
    for tier in range(3):
        n = tier_counts[tier]
        c = correct_counts[tier]
        print(f"    {DIFFICULTY_NAMES[tier]:8s}: {n} txns, {c} correct ({c/n:.0%} correct by agent)")

    print("\nEvaluating policies...")
    all_results = []

    print("  [1/4] Baseline (hand-tuned thresholds)...")
    all_results.append(evaluate_baseline(records))

    for i, variant in enumerate(["A", "B", "C"], start=2):
        print(f"  [{i}/4] PPO Variant {variant}...")
        all_results.append(evaluate_ppo(records, variant, tag=tag))

    print_comparison_table(all_results)

    # Save results
    output = {
        "dataset": dataset,
        "held_out_set_path": str(held_out_path),
        "n_transactions": len(records),
        "policies": all_results,
    }
    results_path = RESULTS_DIR / results_filename
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()

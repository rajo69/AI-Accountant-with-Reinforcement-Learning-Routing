"""
experiments/regime_probe.py — Data-regime probe for the EV-invariance hypothesis.

Context
-------
The calibration probe (experiments/calibrate.py) showed that Platt-scaled
confidence did not resolve the A/B/C convergence. The README Key Finding #4
advances a working hypothesis: at the observed per-tier accuracies (training
set: easy 82.6% / medium 52.4% / hard 62.6%), the EV-optimal tier-level
action is the same for all three reward variants, so reward-table variation
alone cannot produce different policies.

This probe tests that hypothesis directly. We reshape the *calibrated*
training and eval sets so the easy tier's accuracy moves from 82.6% into the
divergence band where Variant A prefers AUTO_APPROVE but Variant C prefers
SURFACE_FOR_REVIEW. Medium and hard tiers are kept at their observed
accuracies.

Divergence-band derivation (at Variant B's expected accountant load E[L]=0.5)
-----------------------------------------------------------------------------
For a transaction with per-tier correct-probability p:

    EV(AUTO   | variant)  = p * (+1.0) + (1-p) * (penalty)
    EV(SURFACE| variant)  = p * (-0.3 * k) + (1-p) * (+0.3)

where penalty is -2.0 for A, -2.0 for B, -5.0 for C; and k is the SURFACE
scaling — 1.0 for A and C, (1 + E[L]) = 1.5 for B at average load.

Break-even p above which AUTO is preferred:

    A:  p > 2.3 / 3.6  = 0.6389
    B:  p > 2.3 / 3.75 = 0.6133  (at E[L]=0.5)
    C:  p > 5.3 / 6.6  = 0.8030

In p in (0.6389, 0.8030), A and B prefer AUTO but C prefers SURFACE.

Experiment
----------
Target easy-tier accuracy = 0.72 (middle of the A-vs-C divergence band).
Medium and hard are preserved. We start from the calibrated dataset
(transactions_calibrated.jsonl and held_out_set_calibrated.json) because
the calibration probe already isolated the signal-quality variable;
reshaping accuracy on top of calibrated scores isolates the accuracy-regime
variable.

Subsampling procedure (deterministic with RANDOM_STATE):
  - For the easy tier, include ALL wrong-prediction transactions plus a random
    subsample of correct-prediction transactions sized so the resulting
    accuracy is exactly 0.72.
  - Medium and hard tiers: pass through unchanged.
  - Applied identically to training (705 -> ~583) and eval (177 -> ~160).

If, after retraining PPO A/B/C on the reshaped training data, the three
variants produce different action distributions on the reshaped eval easy
tier, the EV-invariance hypothesis is supported. If they still converge,
the hypothesis fails.

Run:
    python -m experiments.regime_probe

Outputs:
    data/synthetic/transactions_regime.jsonl
    data/evaluation/held_out_set_regime.json
    experiments/results/regime_probe_report.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source/output configurations. "calibrated" is the primary probe (uses the
# Platt-scaled signal from experiments/calibrate.py). "raw" is a robustness
# check on the original Claude Haiku confidence, demonstrating the regime
# effect is not a calibration-specific artefact.
SOURCES = {
    "calibrated": {
        "train_in":  REPO_ROOT / "data" / "synthetic" / "transactions_calibrated.jsonl",
        "eval_in":   REPO_ROOT / "data" / "evaluation" / "held_out_set_calibrated.json",
        "train_out": REPO_ROOT / "data" / "synthetic" / "transactions_regime.jsonl",
        "eval_out":  REPO_ROOT / "data" / "evaluation" / "held_out_set_regime.json",
        "report_out": REPO_ROOT / "experiments" / "results" / "regime_probe_report.json",
    },
    "raw": {
        "train_in":  REPO_ROOT / "data" / "synthetic" / "transactions.jsonl",
        "eval_in":   REPO_ROOT / "data" / "evaluation" / "held_out_set.json",
        "train_out": REPO_ROOT / "data" / "synthetic" / "transactions_regime_raw.jsonl",
        "eval_out":  REPO_ROOT / "data" / "evaluation" / "held_out_set_regime_raw.json",
        "report_out": REPO_ROOT / "experiments" / "results" / "regime_probe_raw_report.json",
    },
}

RANDOM_STATE = 42

# Target per-tier accuracies. None means "pass through unchanged".
TARGETS = {
    0: 0.72,   # easy: shift from 82.6% into the (0.64, 0.80) divergence band
    1: None,   # medium: keep as-is
    2: None,   # hard: keep as-is
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tier(r: dict) -> int:
    return int(r.get("difficulty_tier", r.get("features", {}).get("difficulty_tier", 0)))


def is_correct(r: dict) -> bool:
    return bool(r["is_correct"])


def reshape(
    records: list[dict], target_easy_acc: float, rng: random.Random
) -> tuple[list[dict], dict[int, dict[str, int]]]:
    """
    Reshape records so tier 0 (easy) has the target accuracy, other tiers
    pass through. Returns (reshaped_records, stats_by_tier).
    """
    by_tier: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for r in records:
        by_tier[tier(r)].append(r)

    stats: dict[int, dict[str, int]] = {}
    out: list[dict] = []

    for t_idx, recs in by_tier.items():
        if TARGETS[t_idx] is None:
            out.extend(recs)
            correct = sum(1 for r in recs if is_correct(r))
            stats[t_idx] = {"n": len(recs), "correct": correct}
            continue

        p_target = float(TARGETS[t_idx])  # type: ignore[arg-type]
        wrong = [r for r in recs if not is_correct(r)]
        correct = [r for r in recs if is_correct(r)]

        # Include all wrong. Subsample correct so correct / (correct + wrong) = p_target.
        # correct_needed = p_target * (correct_needed + len(wrong))
        # correct_needed * (1 - p_target) = p_target * len(wrong)
        # correct_needed = p_target * len(wrong) / (1 - p_target)
        correct_needed = int(round(p_target * len(wrong) / (1.0 - p_target)))
        if correct_needed > len(correct):
            # Not enough correct records to hit the target; take all and flag.
            sampled_correct = correct
        else:
            sampled_correct = rng.sample(correct, correct_needed)

        reshaped = list(wrong) + list(sampled_correct)
        rng.shuffle(reshaped)
        out.extend(reshaped)
        stats[t_idx] = {"n": len(reshaped), "correct": len(sampled_correct)}

    return out, stats


def summarise(records: list[dict]) -> dict[str, Any]:
    by_tier: dict[int, dict[str, int]] = {0: {"n": 0, "correct": 0},
                                           1: {"n": 0, "correct": 0},
                                           2: {"n": 0, "correct": 0}}
    for r in records:
        t = tier(r)
        by_tier[t]["n"] += 1
        if is_correct(r):
            by_tier[t]["correct"] += 1

    names = {0: "easy", 1: "medium", 2: "hard"}
    return {
        names[t]: {
            "n": by_tier[t]["n"],
            "correct": by_tier[t]["correct"],
            "accuracy": (by_tier[t]["correct"] / by_tier[t]["n"])
                          if by_tier[t]["n"] else None,
        }
        for t in (0, 1, 2)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reshape training and eval data to target per-tier accuracies "
                    "in the EV-divergence band, to test the EV-invariance hypothesis."
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES),
        default="calibrated",
        help="Which signal to reshape. 'calibrated' (default) uses the Platt-"
             "scaled dataset from experiments/calibrate.py; 'raw' uses the "
             "original Claude Haiku confidence as a robustness check.",
    )
    args = parser.parse_args()
    cfg = SOURCES[args.source]
    train_in: Path = cfg["train_in"]
    eval_in: Path = cfg["eval_in"]
    train_out: Path = cfg["train_out"]
    eval_out: Path = cfg["eval_out"]
    report_out: Path = cfg["report_out"]
    source: str = args.source
    dataset_tag = "regime" if source == "calibrated" else "regime_raw"

    print(f"Loading {source} datasets...")
    train_records = load_jsonl(train_in)
    eval_records = load_json(eval_in)
    print(f"  Train: {len(train_records)}  Eval: {len(eval_records)}")
    print()

    print("Pre-reshape per-tier accuracies:")
    pre_train = summarise(train_records)
    pre_eval = summarise(eval_records)
    for split_name, s in [("train", pre_train), ("eval", pre_eval)]:
        print(f"  [{split_name}]")
        for name, d in s.items():
            acc = f"{d['accuracy']:.3f}" if d["accuracy"] is not None else "N/A"
            print(f"    {name:<6} n={d['n']:3}  correct={d['correct']:3}  acc={acc}")
    print()

    rng_train = random.Random(RANDOM_STATE)
    rng_eval = random.Random(RANDOM_STATE + 1)

    train_records_out, _ = reshape(train_records, TARGETS[0], rng_train)
    eval_records_out, _ = reshape(eval_records, TARGETS[0], rng_eval)

    print("Post-reshape per-tier accuracies:")
    post_train = summarise(train_records_out)
    post_eval = summarise(eval_records_out)
    for split_name, s in [("train", post_train), ("eval", post_eval)]:
        print(f"  [{split_name}]")
        for name, d in s.items():
            acc = f"{d['accuracy']:.3f}" if d["accuracy"] is not None else "N/A"
            print(f"    {name:<6} n={d['n']:3}  correct={d['correct']:3}  acc={acc}")
    print()

    print("Writing regime-reshaped datasets...")
    with open(train_out, "w", encoding="utf-8") as f:
        for rec in train_records_out:
            f.write(json.dumps(rec) + "\n")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_records_out, f, indent=2)
    print(f"  {train_out}")
    print(f"  {eval_out}")

    report = {
        "config": {
            "source": source,
            "source_train": str(train_in),
            "source_eval": str(eval_in),
            "target_easy_accuracy": TARGETS[0],
            "random_state": RANDOM_STATE,
            "ev_divergence_band": {
                "variant_a_min_p": 0.6389,
                "variant_b_min_p_at_E_load_0p5": 0.6133,
                "variant_c_min_p": 0.8030,
                "rationale": "With target easy accuracy = 0.72, Variants A and B "
                             "should prefer AUTO_APPROVE on easy tier while Variant "
                             "C should prefer SURFACE_FOR_REVIEW. Medium and hard "
                             "tiers are unchanged (expected SURFACE for all "
                             "variants).",
            },
        },
        "pre_reshape": {"train": pre_train, "eval": pre_eval},
        "post_reshape": {"train": post_train, "eval": post_eval},
        "n_train_before_after": [len(train_records), len(train_records_out)],
        "n_eval_before_after": [len(eval_records), len(eval_records_out)],
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  {report_out}")

    print(f"\nDone. Next step (use --dataset {dataset_tag}):")
    print(f"  python -m agent.train    --reward A --dataset {dataset_tag}")
    print(f"  python -m agent.train    --reward B --dataset {dataset_tag}")
    print(f"  python -m agent.train    --reward C --dataset {dataset_tag}")
    print(f"  python -m agent.evaluate           --dataset {dataset_tag}")
    print(f"  python -m experiments.statistical_analysis --dataset {dataset_tag}")


if __name__ == "__main__":
    main()

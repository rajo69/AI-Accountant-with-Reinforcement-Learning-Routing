"""
experiments/calibrate.py — Calibration probe for the RL routing policy.

Hypothesis (from README Key Finding #4):
    The reason all three PPO reward variants (A/B/C) converge to the same
    policy is that raw Claude Haiku confidence scores cluster at round values
    (0.95 / 0.85 / 0.75) regardless of correctness, providing insufficient
    within-tier variance for reward shaping to produce different policies.
    The binding constraint is confidence-score calibration, not the RL method.

This script tests that hypothesis by replacing the raw confidence score with a
well-calibrated one produced by a Platt-scaled logistic regression
(scikit-learn). The logistic regression is fit on the training set's features
[raw_confidence, amount_normalised, difficulty_tier (one-hot)] -> is_correct.
To avoid label leakage on the training set, the calibrated confidences for
training transactions are produced via 5-fold stratified-by-tier cross-
validation. The eval set uses predictions from a model fit on the full
training set.

The outputs are two new datasets:
    data/synthetic/transactions_calibrated.jsonl
    data/evaluation/held_out_set_calibrated.json

Each record has the same schema as the input dataset but with:
    - confidence_score            replaced by the calibrated score
    - features.confidence_score   replaced by the calibrated score
    - confidence_score_raw        retained (original, for reference)

Downstream training (`python -m agent.train --reward A --dataset calibrated`)
and evaluation (`python -m agent.evaluate --dataset calibrated`) read the
calibrated files via the --dataset flag.

A small calibration report is printed and saved to
experiments/results/calibration_report.json.

Run:
    python -m experiments.calibrate

No command-line arguments — paths are deterministic from REPO_ROOT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "synthetic" / "transactions.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "held_out_set.json"
TRAIN_OUT_PATH = REPO_ROOT / "data" / "synthetic" / "transactions_calibrated.jsonl"
EVAL_OUT_PATH = REPO_ROOT / "data" / "evaluation" / "held_out_set_calibrated.json"
REPORT_PATH = REPO_ROOT / "experiments" / "results" / "calibration_report.json"

N_SPLITS = 5
RANDOM_STATE = 42


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(records: list[dict]) -> np.ndarray:
    """
    Feature matrix: [raw_confidence, amount_normalised, tier_is_easy,
    tier_is_medium, tier_is_hard]. Using one-hot for tier avoids treating
    it as ordinal.
    """
    rows = []
    for r in records:
        feats = r.get("features", {})
        conf = float(r.get("confidence_score", feats.get("confidence_score", 0.0)))
        amt = float(feats.get("amount_normalised", 0.0))
        tier = int(r.get("difficulty_tier", feats.get("difficulty_tier", 0)))
        tier_oh = [1.0 if tier == t else 0.0 for t in (0, 1, 2)]
        rows.append([conf, amt, *tier_oh])
    return np.asarray(rows, dtype=np.float64)


def get_labels(records: list[dict]) -> np.ndarray:
    return np.asarray([1 if r["is_correct"] else 0 for r in records], dtype=np.int64)


def get_tiers(records: list[dict]) -> np.ndarray:
    return np.asarray([int(r.get("difficulty_tier", 0)) for r in records], dtype=np.int64)


def oof_calibrated_scores(
    X: np.ndarray, y: np.ndarray, strata: np.ndarray
) -> np.ndarray:
    """
    Produce out-of-fold calibrated P(is_correct=1) for each training row,
    via 5-fold CV stratified by difficulty tier.
    """
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = np.zeros(len(X), dtype=np.float64)
    for fold, (train_idx, holdout_idx) in enumerate(kf.split(X, strata)):
        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE,
        )
        clf.fit(X[train_idx], y[train_idx])
        out[holdout_idx] = clf.predict_proba(X[holdout_idx])[:, 1]
    return out


def full_fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs",
        max_iter=1000, random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf


def rewrite_record(rec: dict, calibrated: float) -> dict:
    new = dict(rec)
    raw = float(rec.get("confidence_score", rec.get("features", {}).get("confidence_score", 0.0)))
    new["confidence_score_raw"] = raw
    new["confidence_score"] = float(calibrated)
    if "features" in new and isinstance(new["features"], dict):
        new["features"] = dict(new["features"])
        new["features"]["confidence_score_raw"] = raw
        new["features"]["confidence_score"] = float(calibrated)
    return new


def summarise_per_tier(
    label: str, scores: np.ndarray, tiers: np.ndarray
) -> dict[str, Any]:
    names = {0: "easy", 1: "medium", 2: "hard"}
    out: dict[str, Any] = {"label": label}
    for t_idx, name in names.items():
        mask = tiers == t_idx
        if mask.sum() == 0:
            continue
        s = scores[mask]
        out[name] = {
            "n": int(mask.sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p50": float(np.median(s)),
            "max": float(s.max()),
            "unique_values": int(len(np.unique(np.round(s, 4)))),
        }
    return out


def main() -> None:
    print("Loading data...")
    train_records = load_jsonl(TRAIN_PATH)
    eval_records = load_json(EVAL_PATH)
    print(f"  Train: {len(train_records)}  Eval: {len(eval_records)}")

    X_train = build_features(train_records)
    y_train = get_labels(train_records)
    tiers_train = get_tiers(train_records)
    X_eval = build_features(eval_records)
    y_eval = get_labels(eval_records)
    tiers_eval = get_tiers(eval_records)

    # Baseline training-set diagnostics on raw confidence
    raw_train = X_train[:, 0]
    raw_eval = X_eval[:, 0]
    print("\nRaw confidence diagnostics:")
    print(f"  train overall:  mean={raw_train.mean():.3f}  std={raw_train.std():.3f}  "
          f"unique_rounded={len(np.unique(np.round(raw_train, 2)))}")

    # Out-of-fold calibrated scores for training set
    print("\nFitting out-of-fold calibrator on training set (5-fold, stratified by tier)...")
    cal_train = oof_calibrated_scores(X_train, y_train, tiers_train)

    # Full-training model applied to eval
    print("Fitting final calibrator on full training set, applying to eval set...")
    clf = full_fit(X_train, y_train)
    cal_eval = clf.predict_proba(X_eval)[:, 1]

    # Diagnostics
    print("\nCalibrated-confidence within-tier spread (higher = better separability):")
    raw_summary = {
        "train": summarise_per_tier("raw_train", raw_train, tiers_train),
        "eval":  summarise_per_tier("raw_eval",  raw_eval,  tiers_eval),
    }
    cal_summary = {
        "train": summarise_per_tier("cal_train", cal_train, tiers_train),
        "eval":  summarise_per_tier("cal_eval",  cal_eval,  tiers_eval),
    }
    for split in ("train", "eval"):
        print(f"  [{split}]")
        for tier_name in ("easy", "medium", "hard"):
            raw = raw_summary[split].get(tier_name)
            cal = cal_summary[split].get(tier_name)
            if raw and cal:
                print(
                    f"    {tier_name:<6} n={raw['n']:3}  "
                    f"raw mean={raw['mean']:.3f} std={raw['std']:.3f} "
                    f"uniq={raw['unique_values']:2}  |  "
                    f"cal mean={cal['mean']:.3f} std={cal['std']:.3f} "
                    f"uniq={cal['unique_values']:3}"
                )

    # Fit-quality metrics
    from sklearn.metrics import brier_score_loss, log_loss
    raw_brier_eval = brier_score_loss(y_eval, raw_eval)
    cal_brier_eval = brier_score_loss(y_eval, cal_eval)
    raw_logloss_eval = log_loss(y_eval, np.clip(raw_eval, 1e-6, 1 - 1e-6))
    cal_logloss_eval = log_loss(y_eval, np.clip(cal_eval, 1e-6, 1 - 1e-6))
    print("\nEval-set proper scoring rules (lower is better):")
    print(f"  Brier  raw={raw_brier_eval:.4f}  cal={cal_brier_eval:.4f}")
    print(f"  LogLoss raw={raw_logloss_eval:.4f}  cal={cal_logloss_eval:.4f}")

    # Write calibrated datasets
    print("\nWriting calibrated datasets...")
    with open(TRAIN_OUT_PATH, "w", encoding="utf-8") as f:
        for rec, s in zip(train_records, cal_train):
            f.write(json.dumps(rewrite_record(rec, s)) + "\n")
    with open(EVAL_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([rewrite_record(r, s) for r, s in zip(eval_records, cal_eval)], f, indent=2)
    print(f"  {TRAIN_OUT_PATH}")
    print(f"  {EVAL_OUT_PATH}")

    # Report
    report = {
        "config": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "calibrator": "sklearn LogisticRegression(C=1.0, penalty='l2')",
            "features": ["raw_confidence", "amount_normalised",
                         "tier_easy", "tier_medium", "tier_hard"],
        },
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        "raw_summary": raw_summary,
        "calibrated_summary": cal_summary,
        "eval_scoring": {
            "brier_raw": raw_brier_eval,
            "brier_calibrated": cal_brier_eval,
            "log_loss_raw": raw_logloss_eval,
            "log_loss_calibrated": cal_logloss_eval,
        },
        "calibrator_coef": clf.coef_.tolist(),
        "calibrator_intercept": clf.intercept_.tolist(),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  {REPORT_PATH}")

    print("\nDone. Next step:")
    print("  python -m agent.train    --reward A --dataset calibrated")
    print("  python -m agent.train    --reward B --dataset calibrated")
    print("  python -m agent.train    --reward C --dataset calibrated")
    print("  python -m agent.evaluate           --dataset calibrated")


if __name__ == "__main__":
    main()

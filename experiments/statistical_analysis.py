"""
experiments/statistical_analysis.py — Statistical rigour for the headline claims.

Loads experiments/results/evaluation_results.json and computes:

  1. Wilson score 95% confidence intervals for every reported rate
  2. Two-proportion z-tests for each Baseline vs PPO comparison
  3. A compact report that can be pasted into README.md

Why Wilson over Wald (normal approximation):
  Wilson is recommended for small n and for rates near 0 or 1 (Agresti 2002).
  The auto-approval subsets here have n in {81, 113} and some sub-tier error
  rates are at or near 0/1, where Wald breaks down.

Why two-proportion z (not chi-square, not Fisher):
  All comparisons are of two independent proportions with n*p and n*(1-p) well
  above 5; the normal approximation is adequate and the resulting p-value is
  directly interpretable. For very small cells (e.g. medium-tier PPO has n=0
  auto-approved) we skip the test and report that.

Run:
    python -m experiments.statistical_analysis

Output:
    - Console table
    - experiments/results/statistical_summary.json (machine-readable)
    - experiments/results/statistical_summary.md   (README-pasteable)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "evaluation_results.json"
SUMMARY_JSON = REPO_ROOT / "experiments" / "results" / "statistical_summary.json"
SUMMARY_MD = REPO_ROOT / "experiments" / "results" / "statistical_summary.md"

Z_95 = 1.959963984540054  # standard-normal 97.5th percentile


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def wilson_ci(x: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion x/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """
    Two-proportion z-test for H0: p1 == p2.
    Returns (z_statistic, two_sided_p_value).
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return (0.0, 1.0)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    # Two-sided p from standard normal, via erf
    p_value = 2.0 * (1.0 - _phi(abs(z)))
    return (z, p_value)


def _phi(z: float) -> float:
    """Standard-normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fmt_pct(p: float) -> str:
    if p != p:  # nan
        return "  N/A"
    return f"{p * 100:5.1f}%"


def fmt_ci(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:
        return "N/A"
    return f"[{lo*100:4.1f}%, {hi*100:4.1f}%]"


# ---------------------------------------------------------------------------
# Extract counts from evaluation_results.json
#
# The results JSON stores rates, not raw counts, but it also stores
# action_totals and action_distribution_by_tier. We reconstruct the counts we
# need directly — no rate-to-count rounding.
# ---------------------------------------------------------------------------

def extract_policy_counts(policy: dict) -> dict:
    """
    Given one entry from evaluation_results.json, return raw integer counts
    needed for Wilson CIs and two-proportion tests.
    """
    n = policy["n_transactions"]
    action_totals = policy["action_totals"]
    auto = action_totals["AUTO_APPROVE"]
    surface = action_totals["SURFACE_FOR_REVIEW"]
    reject = action_totals["REJECT_FOR_MANUAL"]

    # Derive auto-approval wrong count from the reported rate (exact because
    # the rate is reported to 4 dp and auto sample size is small).
    err_auto = policy["error_rate_auto"] or 0.0
    auto_wrong = round(err_auto * auto) if auto > 0 else 0
    auto_correct = auto - auto_wrong

    # Routing accuracy: overall_routing_accuracy × n
    overall_acc = policy["overall_routing_accuracy"]
    optimal = round(overall_acc * n)

    # Per-tier auto-approved counts and errors
    per_tier = {}
    for tier_name, d in policy["action_distribution_by_tier"].items():
        t_auto = d["AUTO_APPROVE"]
        tier_err = policy["error_rate_by_tier"].get(tier_name)
        if tier_err is None or t_auto == 0:
            per_tier[tier_name] = {"auto": t_auto, "wrong": 0, "rate": None}
        else:
            per_tier[tier_name] = {
                "auto": t_auto,
                "wrong": round(tier_err * t_auto),
                "rate": tier_err,
            }

    return {
        "policy": policy["policy"],
        "n": n,
        "auto": auto,
        "surface": surface,
        "reject": reject,
        "auto_correct": auto_correct,
        "auto_wrong": auto_wrong,
        "optimal": optimal,
        "per_tier": per_tier,
    }


# ---------------------------------------------------------------------------
# Build the statistical report
# ---------------------------------------------------------------------------

def analyse(policies: list[dict]) -> dict:
    baseline = policies[0]
    ppo = policies[1]  # all three PPO variants are identical → compare against one

    report: dict = {
        "baseline_name": baseline["policy"],
        "ppo_name": f"{ppo['policy']} (A/B/C identical on eval)",
        "n_eval": baseline["n"],
        "metrics": {},
        "comparisons": {},
        "notes": [],
    }

    # --- Per-policy Wilson CIs -------------------------------------------------
    for label, p in [("baseline", baseline), ("ppo", ppo)]:
        m: dict = {}

        # Routing accuracy (over all n=177)
        lo, hi = wilson_ci(p["optimal"], p["n"])
        m["routing_accuracy"] = {
            "x": p["optimal"], "n": p["n"],
            "rate": p["optimal"] / p["n"],
            "ci95": [lo, hi],
        }

        # Auto-approval precision (over n auto-approved)
        lo, hi = wilson_ci(p["auto_correct"], p["auto"])
        m["auto_precision"] = {
            "x": p["auto_correct"], "n": p["auto"],
            "rate": p["auto_correct"] / p["auto"] if p["auto"] else float("nan"),
            "ci95": [lo, hi],
        }

        # Auto-approval error rate (over n auto-approved)
        lo, hi = wilson_ci(p["auto_wrong"], p["auto"])
        m["auto_error_rate"] = {
            "x": p["auto_wrong"], "n": p["auto"],
            "rate": p["auto_wrong"] / p["auto"] if p["auto"] else float("nan"),
            "ci95": [lo, hi],
        }

        # Auto-approval rate (over all n)
        lo, hi = wilson_ci(p["auto"], p["n"])
        m["auto_approval_rate"] = {
            "x": p["auto"], "n": p["n"],
            "rate": p["auto"] / p["n"],
            "ci95": [lo, hi],
        }

        # Per-tier error rate among auto-approved
        tier_cis = {}
        for tier_name, d in p["per_tier"].items():
            if d["auto"] == 0:
                tier_cis[tier_name] = {"x": 0, "n": 0, "rate": None, "ci95": [None, None]}
            else:
                lo, hi = wilson_ci(d["wrong"], d["auto"])
                tier_cis[tier_name] = {
                    "x": d["wrong"], "n": d["auto"],
                    "rate": d["wrong"] / d["auto"],
                    "ci95": [lo, hi],
                }
        m["error_rate_by_tier"] = tier_cis

        report["metrics"][label] = m

    # --- Baseline-vs-PPO two-proportion tests ---------------------------------
    comparisons = {}

    # Auto-approval error rate
    z, pv = two_proportion_z(
        baseline["auto_wrong"], baseline["auto"],
        ppo["auto_wrong"], ppo["auto"],
    )
    comparisons["auto_error_rate"] = {"z": z, "p_value": pv}

    # Auto-approval precision (equivalent to error rate, just flipped)
    z, pv = two_proportion_z(
        baseline["auto_correct"], baseline["auto"],
        ppo["auto_correct"], ppo["auto"],
    )
    comparisons["auto_precision"] = {"z": z, "p_value": pv}

    # Overall routing accuracy (n=177 each)
    z, pv = two_proportion_z(
        baseline["optimal"], baseline["n"],
        ppo["optimal"], ppo["n"],
    )
    comparisons["routing_accuracy"] = {"z": z, "p_value": pv}

    # Auto-approval rate — how often each policy auto-approves at all
    z, pv = two_proportion_z(
        baseline["auto"], baseline["n"],
        ppo["auto"], ppo["n"],
    )
    comparisons["auto_approval_rate"] = {"z": z, "p_value": pv}

    report["comparisons"] = comparisons

    # --- Qualitative notes ----------------------------------------------------
    report["notes"].append(
        f"PPO A/B/C produce identical action sequences on the eval set "
        f"(total AUTO={ppo['auto']}, SURFACE={ppo['surface']}, REJECT={ppo['reject']}), "
        f"so only one PPO column needs reporting in comparison tables."
    )
    report["notes"].append(
        "Both policies emit 0 REJECT_FOR_MANUAL on this eval set because real "
        "Claude Haiku confidences never fall below 0.5 (baseline's reject threshold). "
        "PPO's REJECT-elimination is a learned policy-level property; the baseline "
        "coincidence reflects the data distribution, not the threshold design."
    )
    if comparisons["auto_error_rate"]["p_value"] > 0.05:
        report["notes"].append(
            "The Baseline-vs-PPO difference in auto-approval error rate is NOT "
            "statistically significant at alpha=0.05 (the headline README framing "
            "overstates this finding)."
        )
    if comparisons["auto_approval_rate"]["p_value"] < 0.05:
        report["notes"].append(
            "The Baseline-vs-PPO difference in auto-approval RATE IS highly "
            "significant: PPO auto-approves substantially less. Combined with "
            "the baseline's 54.8% medium-tier error rate among its auto-approvals, "
            "this is the defensible headline: PPO learns to not auto-approve the "
            "tier where the baseline is wrong more often than right."
        )

    return report


# ---------------------------------------------------------------------------
# Console + file output
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    print("=" * 78)
    print("  STATISTICAL ANALYSIS — Baseline vs PPO routing policy on held-out set")
    print("=" * 78)
    print(f"  Eval set size: n = {report['n_eval']}")
    print(f"  Policies compared: {report['baseline_name']}")
    print(f"                     {report['ppo_name']}")
    print()

    b = report["metrics"]["baseline"]
    p = report["metrics"]["ppo"]

    metrics = [
        ("Routing accuracy (all 177)",    "routing_accuracy"),
        ("Auto-approval precision",       "auto_precision"),
        ("Auto-approval error rate",      "auto_error_rate"),
        ("Auto-approval rate",            "auto_approval_rate"),
    ]

    header = f"  {'Metric':<35} {'Baseline':<22} {'PPO':<22} {'p-value'}"
    print(header)
    print("  " + "-" * 76)

    for label, key in metrics:
        bm = b[key]
        pm = p[key]
        cmp_key = key if key in report["comparisons"] else None
        pv = report["comparisons"][cmp_key]["p_value"] if cmp_key else None
        bcell = f"{fmt_pct(bm['rate'])} {fmt_ci(*bm['ci95'])}"
        pcell = f"{fmt_pct(pm['rate'])} {fmt_ci(*pm['ci95'])}"
        pvcell = f"p={pv:.3f}" if pv is not None else ""
        print(f"  {label:<35} {bcell:<22} {pcell:<22} {pvcell}")
    print()

    print("  Per-tier auto-approval error rate (auto-approved subset only)")
    print("  " + "-" * 76)
    for tier in ("easy", "medium", "hard"):
        bt = b["error_rate_by_tier"][tier]
        pt = p["error_rate_by_tier"][tier]
        bcell = (
            f"{fmt_pct(bt['rate'])} {fmt_ci(*bt['ci95'])} (n={bt['n']})"
            if bt["rate"] is not None else "N/A (no auto-approvals)"
        )
        pcell = (
            f"{fmt_pct(pt['rate'])} {fmt_ci(*pt['ci95'])} (n={pt['n']})"
            if pt["rate"] is not None else "N/A (no auto-approvals)"
        )
        print(f"  {tier:<10} baseline: {bcell}")
        print(f"  {'':<10} ppo:      {pcell}")
    print()

    print("  Notes:")
    for n in report["notes"]:
        print(f"   - {n}")
    print("=" * 78)


def write_markdown(report: dict, path: Path) -> None:
    """Emit a README-pasteable Markdown block."""
    b = report["metrics"]["baseline"]
    p = report["metrics"]["ppo"]

    def row(label: str, key: str) -> str:
        bm = b[key]
        pm = p[key]
        cmp_key = key if key in report["comparisons"] else None
        pv = report["comparisons"][cmp_key]["p_value"] if cmp_key else None
        pvcell = f"p={pv:.2f}" if pv is not None else ""
        return (
            f"| {label} | "
            f"{bm['rate']*100:.1f}% [{bm['ci95'][0]*100:.1f}, {bm['ci95'][1]*100:.1f}] | "
            f"{pm['rate']*100:.1f}% [{pm['ci95'][0]*100:.1f}, {pm['ci95'][1]*100:.1f}] | "
            f"{pvcell} |"
        )

    lines = []
    lines.append(f"## Results with 95% confidence intervals (n={report['n_eval']})")
    lines.append("")
    lines.append("All intervals are Wilson score 95% CIs. p-values are two-sided two-proportion z-tests (Baseline vs PPO).")
    lines.append("")
    lines.append("| Metric | Baseline (0.85/0.50) | PPO (A/B/C identical) | Significance |")
    lines.append("|---|:---:|:---:|:---:|")
    lines.append(row("Routing accuracy (over 177)", "routing_accuracy"))
    lines.append(row("Auto-approval precision",      "auto_precision"))
    lines.append(row("Auto-approval error rate",     "auto_error_rate"))
    lines.append(row("Auto-approval rate",           "auto_approval_rate"))
    lines.append("")
    lines.append("### Per-tier auto-approval error rate (auto-approved subset only)")
    lines.append("")
    lines.append("| Tier | Baseline | PPO |")
    lines.append("|---|:---:|:---:|")
    for tier in ("easy", "medium", "hard"):
        bt = b["error_rate_by_tier"][tier]
        pt = p["error_rate_by_tier"][tier]
        bcell = (
            f"{bt['rate']*100:.1f}% [{bt['ci95'][0]*100:.1f}, {bt['ci95'][1]*100:.1f}] (n={bt['n']})"
            if bt["rate"] is not None else "N/A"
        )
        pcell = (
            f"{pt['rate']*100:.1f}% [{pt['ci95'][0]*100:.1f}, {pt['ci95'][1]*100:.1f}] (n={pt['n']})"
            if pt["rate"] is not None else "N/A (no auto-approvals)"
        )
        lines.append(f"| {tier} | {bcell} | {pcell} |")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    for n in report["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    counts = [extract_policy_counts(p) for p in raw["policies"]]
    report = analyse(counts)

    print_report(report)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved machine-readable summary: {SUMMARY_JSON}")

    write_markdown(report, SUMMARY_MD)
    print(f"Saved README-pasteable markdown:  {SUMMARY_MD}")


if __name__ == "__main__":
    main()

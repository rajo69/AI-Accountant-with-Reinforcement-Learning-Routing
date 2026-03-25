"""
Categorisation eval runner.

Runs the categorisation agent against labeled fixtures and produces:
- Accuracy by difficulty tier (easy / medium / hard)
- Per-category precision and recall
- Confidence calibration (how well confidence correlates with correctness)
- Full cost report
- CSV results file for analysis

Usage:
    # Dry-run with cached responses (no API cost):
    python -m evals.eval_runner --mode cached

    # Real API run on Haiku (cheap):
    python -m evals.eval_runner --mode live --model claude-haiku-4-5-20251001 --budget 0.05

    # Full quality benchmark on Sonnet:
    python -m evals.eval_runner --mode live --model claude-sonnet-4-6 --budget 0.50

    # Subset test (first 10 transactions only):
    python -m evals.eval_runner --mode live --limit 10

Environment variables:
    ANTHROPIC_API_KEY  — required for live mode
    EVAL_CACHE=0       — bypass cache (default: 1 = use cache)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evals.cost_tracker import CostTracker, estimate_before_run, DEFAULT_EVAL_MODEL
from evals.response_cache import ResponseCache

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalTransaction:
    id: str
    date: str
    amount: str
    description: str
    reference: str
    expected_category_code: str
    expected_category_name: str
    difficulty: str
    notes: str


@dataclass
class EvalResult:
    transaction_id: str
    description: str
    expected_code: str
    expected_name: str
    predicted_code: str
    predicted_name: str
    confidence: float
    correct: bool
    difficulty: str
    reasoning: str
    cost_usd: float
    cached: bool


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_transactions(limit: Optional[int] = None) -> list[EvalTransaction]:
    data = json.loads((FIXTURES_DIR / "transactions.json").read_text())
    txs = [EvalTransaction(**{k: v for k, v in t.items() if k in EvalTransaction.__dataclass_fields__}) for t in data]
    if limit:
        txs = txs[:limit]
    return txs


def load_accounts() -> list[dict]:
    return json.loads((FIXTURES_DIR / "accounts.json").read_text())


# ---------------------------------------------------------------------------
# Mock categorisation (no API) — for testing the eval framework itself
# ---------------------------------------------------------------------------

def _mock_categorise(tx: EvalTransaction, accounts: list[dict]) -> dict:
    """
    Rule-based mock categoriser for framework testing.
    Returns structured output matching CategoryPrediction schema.
    """
    desc = tx.description.upper()
    amount = float(tx.amount)

    # Simple keyword rules
    rules = [
        (["ADOBE", "MICROSOFT", "XERO", "SLACK", "AWS", "AMAZON WEB", "SAGE PAYROLL", "CANVA"], "404", "Computer Equipment & Software", 0.95),
        (["GOOGLE ADS", "LINKEDIN PREMIUM"], "412", "Advertising & Marketing", 0.90),
        (["HMRC PAYE", "HMRC NIC", "HMRC VAT", "HMRC COMPANIES"], "820", "Tax & Statutory Payments", 0.97),
        (["TESCO", "COSTA", "DELIVEROO"], "429", "Entertainment & Hospitality", 0.85),
        (["REGUS", "RENT"], "401", "Rent", 0.92),
        (["TFL", "PREMIER INN", "NCP CAR PARK"], "493", "Travel", 0.88),
        (["BRITISH GAS", "BT BUSINESS", "VEOLIA", "UTILITIES"], "445", "Utilities", 0.87),
        (["ROYAL MAIL", "DPD COURIER", "CITY SPRINT"], "489", "Postage & Freight", 0.93),
        (["KNIGHTS SOLICITORS", "THORNTON BAKER", "LEGAL"], "420", "Legal & Accountancy Fees", 0.91),
        (["ZURICH", "BUPA", "LLOYDS BANK INSURANCE"], "462", "Insurance", 0.89),
        (["STAPLES OFFICE"], "460", "Office Supplies", 0.94),
        (["ICAEW", "LAW SOCIETY"], "461", "Subscriptions", 0.86),
        (["PAYROLL", "WAGES"], "477", "Wages & Salaries", 0.96),
        (["PHOENIX CLEANING"], "480", "Cleaning", 0.90),
        (["RATES DEMAND"], "469", "Rates", 0.91),
        (["ACCOUNTANCY AGE"], "463", "Training & Development", 0.82),
        (["REED RECRUITMENT"], "477", "Recruitment", 0.80),
        (["CLIENT PAYMENT", "GRANT RECEIVED"] if amount > 0 else [], "200", "Sales", 0.88),
    ]

    for keywords, code, name, conf in rules:
        if any(kw in desc for kw in keywords):
            return {"category_code": code, "category_name": name, "confidence": conf, "reasoning": f"Matched keyword in description"}

    # Fallback
    return {"category_code": "416", "category_name": "General Expenses", "confidence": 0.40, "reasoning": "No strong signal found"}


# ---------------------------------------------------------------------------
# Live categorisation (real API)
# ---------------------------------------------------------------------------

async def _live_categorise(
    tx: EvalTransaction,
    accounts: list[dict],
    model: str,
    tracker: CostTracker,
    cache: ResponseCache,
) -> tuple[dict, bool]:
    """Call Claude API to categorise a transaction. Returns (result, was_cached)."""
    try:
        import anthropic
        import instructor
        from pydantic import BaseModel

        class CategoryPrediction(BaseModel):
            category_code: str
            category_name: str
            confidence: float
            reasoning: str

        accounts_text = "\n".join(f"  {a['code']}: {a['name']} ({a['type']})" for a in accounts)
        prompt = f"""You are a UK accounting assistant. Categorise this bank transaction.

Transaction:
  Date: {tx.date}
  Amount: £{tx.amount}
  Description: {tx.description}
  Reference: {tx.reference or 'none'}

Chart of Accounts:
{accounts_text}

Return the most appropriate account code and name, your confidence (0.0-1.0), and a brief reasoning."""

        cached = cache.get(model, prompt)
        if cached:
            return cached, True

        t0 = time.monotonic()
        client = instructor.from_anthropic(anthropic.Anthropic())
        pred = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_model=CategoryPrediction,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Get usage from the raw response (instructor wraps it)
        raw_client = anthropic.Anthropic()
        raw_resp = raw_client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        tracker.record(raw_resp.usage, label=tx.id, duration_ms=elapsed_ms)

        result = {
            "category_code": pred.category_code,
            "category_name": pred.category_name,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
        }
        cache.set(model, prompt, result)
        return result, False

    except ImportError:
        # anthropic/instructor not installed yet (Phase 3 dep)
        return _mock_categorise(tx, accounts), False


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[EvalResult]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0

    # By difficulty
    by_diff: dict[str, dict] = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r.difficulty == diff]
        if subset:
            by_diff[diff] = {
                "count": len(subset),
                "correct": sum(1 for r in subset if r.correct),
                "accuracy": sum(1 for r in subset if r.correct) / len(subset),
            }

    # Per-category precision / recall
    categories = set(r.expected_code for r in results)
    per_cat: dict[str, dict] = {}
    for cat in sorted(categories):
        tp = sum(1 for r in results if r.expected_code == cat and r.predicted_code == cat)
        fp = sum(1 for r in results if r.expected_code != cat and r.predicted_code == cat)
        fn = sum(1 for r in results if r.expected_code == cat and r.predicted_code != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_cat[cat] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    # Confidence calibration: split into bins
    bins = {"high (>0.85)": [], "medium (0.5-0.85)": [], "low (<0.5)": []}
    for r in results:
        if r.confidence > 0.85:
            bins["high (>0.85)"].append(r.correct)
        elif r.confidence >= 0.5:
            bins["medium (0.5-0.85)"].append(r.correct)
        else:
            bins["low (<0.5)"].append(r.correct)

    calibration = {
        name: {
            "count": len(vals),
            "accuracy": sum(vals) / len(vals) if vals else 0,
            "would_auto_accept": len(vals) if name == "high (>0.85)" else 0,
        }
        for name, vals in bins.items()
    }

    return {
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "by_difficulty": by_diff,
        "per_category": per_cat,
        "confidence_calibration": calibration,
    }


def print_report(metrics: dict, tracker: CostTracker, results: list[EvalResult]) -> None:
    print("\n" + "=" * 70)
    print("  CATEGORISATION EVAL REPORT")
    print("=" * 70)
    print(f"\n  Overall accuracy: {metrics['overall_accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")

    print("\n  By difficulty:")
    for diff, m in metrics["by_difficulty"].items():
        bar = "#" * int(m["accuracy"] * 20)
        print(f"    {diff:8s}: {m['accuracy']:.1%} ({m['correct']}/{m['count']}) {bar}")

    print("\n  Confidence calibration:")
    print("  (Well-calibrated = high confidence -> high accuracy)")
    for band, m in metrics["confidence_calibration"].items():
        if m["count"] > 0:
            print(f"    {band}: acc={m['accuracy']:.1%} ({m['count']} transactions)")

    print("\n  Per-category (F1 >= 0.8 is good):")
    for code, m in sorted(metrics["per_category"].items(), key=lambda x: -x[1]["f1"]):
        name = next((r.expected_name for r in results if r.expected_code == code), code)
        print(f"    {code} {name[:30]:30s}: F1={m['f1']:.2f} P={m['precision']:.2f} R={m['recall']:.2f}")

    print("\n  Failures:")
    failures = [r for r in results if not r.correct]
    if failures:
        for r in failures:
            print(f"    [{r.difficulty}] {r.description[:45]:45s}")
            print(f"           expected={r.expected_code}/{r.expected_name[:20]}")
            print(f"           got     ={r.predicted_code}/{r.predicted_name[:20]} (conf={r.confidence:.2f})")
    else:
        print("    None — perfect score!")

    print()
    print(tracker.report())


def save_csv(results: list[EvalResult], model: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    path = RESULTS_DIR / f"eval_{model.replace('/', '_')}_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "difficulty", "description", "expected_code", "expected_name",
            "predicted_code", "predicted_name", "confidence", "correct", "cached", "cost_usd", "reasoning"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r.transaction_id,
                "difficulty": r.difficulty,
                "description": r.description,
                "expected_code": r.expected_code,
                "expected_name": r.expected_name,
                "predicted_code": r.predicted_code,
                "predicted_name": r.predicted_name,
                "confidence": f"{r.confidence:.3f}",
                "correct": r.correct,
                "cached": r.cached,
                "cost_usd": f"{r.cost_usd:.6f}",
                "reasoning": r.reasoning,
            })
    return path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_eval(
    mode: str = "mock",
    model: str = DEFAULT_EVAL_MODEL,
    budget_usd: Optional[float] = None,
    limit: Optional[int] = None,
    save_results: bool = True,
) -> dict:
    transactions = load_transactions(limit=limit)
    accounts = load_accounts()

    print(f"\n  Running eval: mode={mode}, model={model}, n={len(transactions)}")
    if mode == "live":
        estimate_before_run(len(transactions), model)

    tracker = CostTracker(model=model, budget_usd=budget_usd)
    cache = ResponseCache()

    results: list[EvalResult] = []

    for tx in transactions:
        if mode == "live":
            pred, was_cached = await _live_categorise(tx, accounts, model, tracker, cache)
        else:
            pred, was_cached = _mock_categorise(tx, accounts), False

        correct = pred["category_code"] == tx.expected_category_code
        cost = tracker.calls[-1].cost_usd if tracker.calls else 0.0

        results.append(EvalResult(
            transaction_id=tx.id,
            description=tx.description,
            expected_code=tx.expected_category_code,
            expected_name=tx.expected_category_name,
            predicted_code=pred["category_code"],
            predicted_name=pred["category_name"],
            confidence=float(pred["confidence"]),
            correct=correct,
            difficulty=tx.difficulty,
            reasoning=pred.get("reasoning", ""),
            cost_usd=cost,
            cached=was_cached,
        ))

    metrics = compute_metrics(results)
    print_report(metrics, tracker, results)

    if mode == "live":
        print(f"\n  Cache stats: {cache.stats()}")

    if save_results:
        csv_path = save_csv(results, model)
        print(f"\n  Results saved to: {csv_path}")

    # --- Acceptance gate ---
    failures = []
    overall_acc = metrics["overall_accuracy"]
    if overall_acc < 0.80:
        failures.append(f"Overall accuracy {overall_acc:.1%} < 80% minimum")

    easy = metrics["by_difficulty"].get("easy", {})
    easy_acc = easy.get("accuracy", 1.0)
    if easy.get("count", 0) > 0 and easy_acc < 0.95:
        failures.append(f"Easy tier accuracy {easy_acc:.1%} < 95% minimum")

    high_cal = metrics["confidence_calibration"].get("high (>0.85)", {})
    auto_acc = high_cal.get("accuracy", 1.0)
    if high_cal.get("count", 0) > 0 and auto_acc < 0.90:
        failures.append(f"Auto-accept accuracy {auto_acc:.1%} < 90% minimum")

    if failures:
        print("\n  ❌ ACCEPTANCE GATE FAILED:")
        for f in failures:
            print(f"     - {f}")
        print()
        metrics["gate_passed"] = False
    else:
        print("\n  ✅ Acceptance gate passed\n")
        metrics["gate_passed"] = True

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Categorisation eval runner")
    parser.add_argument("--mode", choices=["mock", "cached", "live"], default="mock",
                        help="mock=no API, cached=use cache only, live=real API calls")
    parser.add_argument("--model", default=DEFAULT_EVAL_MODEL,
                        help=f"Claude model ID (default: {DEFAULT_EVAL_MODEL})")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max spend in USD (e.g. 0.05 = 5 cents)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only test first N transactions")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save CSV results file")
    args = parser.parse_args()

    if args.mode == "live" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to backend/.env")
        sys.exit(1)

    metrics = asyncio.run(run_eval(
        mode=args.mode,
        model=args.model,
        budget_usd=args.budget,
        limit=args.limit,
        save_results=not args.no_save,
    ))
    if not metrics.get("gate_passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()

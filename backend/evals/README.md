# Eval Framework

Evaluate the AI categorisation agent before deploying. Measures accuracy,
per-category F1, confidence calibration, and API cost.

---

## Quick Start

```bash
cd backend

# 1. Dry run — no API calls, uses rule-based mock (free, instant)
python -m evals.eval_runner --mode mock

# 2. Cached run — uses real API once, caches to disk, free on reruns
python -m evals.eval_runner --mode live --budget 0.05

# 3. Rerun against cache (zero cost after first run)
python -m evals.eval_runner --mode live --budget 0.05  # hits cache

# 4. Clear cache and force fresh API calls
python -c "from evals.response_cache import ResponseCache; print(ResponseCache().clear(), 'files deleted')"

# 5. Benchmark: compare Haiku vs Sonnet
python -m evals.eval_runner --mode live --model claude-haiku-4-5-20251001 --budget 0.05
python -m evals.eval_runner --mode live --model claude-sonnet-4-6 --budget 0.50

# 6. Quick smoke test on 10 transactions
python -m evals.eval_runner --mode live --limit 10 --budget 0.01
```

---

## Eval Modes

| Mode | API calls | Cost | When to use |
|------|-----------|------|-------------|
| `mock` | None | $0 | Testing the eval framework itself |
| `live` (cache hit) | None | $0 | Reruns after first live run |
| `live` (cache miss) | Yes | See below | First run or after cache clear |

---

## Expected Costs

All costs use the 50-transaction fixture set.

| Model | Est. cost | Notes |
|-------|-----------|-------|
| claude-haiku-4-5-20251001 | ~$0.001 | Use for all development evals |
| claude-sonnet-4-6 | ~$0.005 | Use for pre-deployment quality check |

After the first live run, **reruns are free** (responses are cached to `evals/.cache/`).

---

## What the Report Shows

```
Overall accuracy: 88.0% (44/50)

By difficulty:
  easy    : 95.8% (23/24) ████████████████████
  medium  : 81.3% (13/16) ████████████████
  hard    :  80.0% (8/10) ████████████████

Confidence calibration:
  high (>0.85):   acc=95.0% (40 transactions)  ← auto-accept threshold
  medium (0.5-0.85): acc=71.4% (7 transactions) ← shown to accountant
  low (<0.5):     acc=33.3% (3 transactions)    ← flagged for review

Per-category (F1 ≥ 0.8 is good):
  404 Computer Equipment & Software: F1=0.95 P=0.95 R=0.95
  820 Tax & Statutory Payments:      F1=1.00 P=1.00 R=1.00
  ...
```

---

## Fixture Set

`fixtures/transactions.json` — 50 labeled UK SME bank transactions:
- 24 **easy** (unambiguous: HMRC, well-known SaaS, payroll)
- 16 **medium** (require context: professional memberships, dual-category spends)
- 10 **hard** (industry-specific, balance sheet vs P&L edge cases)

`fixtures/accounts.json` — 20 standard UK chart of accounts codes.

To add more fixtures, append to `transactions.json` following the same schema.

---

## Acceptance Criteria (pre-deployment)

Before going live with any version of the categorisation agent:

| Metric | Minimum | Target |
|--------|---------|--------|
| Overall accuracy | 80% | 90%+ |
| Easy accuracy | 95% | 100% |
| Auto-accept accuracy (conf > 0.85) | 90% | 95% |
| Cost per transaction | < $0.01 | < $0.005 |
| F1 on core categories (HMRC, payroll, software) | 0.90 | 0.95+ |

If the agent doesn't meet these thresholds, **do not deploy**. Tune prompts,
add few-shot examples, or adjust confidence thresholds first.

---

## Cost Architecture

See `docs/ARCHITECTURE.md` Section 9 for the full cost model.

**Key strategy**: Use `claude-haiku-4-5-20251001` for all eval runs (~10x cheaper
than Sonnet). Only switch to `claude-sonnet-4-6` for a final pre-deployment
quality check. Cache all responses to disk — the second run always costs $0.

---

## Files

```
evals/
├── __init__.py
├── README.md               ← this file
├── cost_tracker.py         ← token counting + cost estimation + budget guard
├── response_cache.py       ← disk cache for API responses (keyed by SHA-256(model+prompt))
├── eval_runner.py          ← main runner: loads fixtures, calls agent, prints metrics
├── fixtures/
│   ├── transactions.json   ← 50 labeled UK transactions
│   └── accounts.json       ← 20 chart of accounts codes
├── results/                ← CSV results from each run (auto-created)
└── .cache/                 ← cached API responses (auto-created, gitignored)
```

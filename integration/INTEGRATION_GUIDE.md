# Integration Guide: Swapping in the Learned Router

This guide explains exactly how to replace the hand-tuned confidence threshold
logic in the parent project's CategoriserAgent with the LearnedRouter.

## Prerequisites

This guide assumes you have the parent
[AI Accountant](https://github.com/rajo69/agentic-ai-accounting) repository
cloned alongside this one. All `backend/...` paths below refer to files in the
parent repository, not in this repository. This repository no longer vendors
the parent code.

---

## What to change

The routing logic lives in **`backend/app/agents/categoriser.py`** (in the
parent repository) in the `decide` node of the LangGraph graph.

### Current code (hand-tuned thresholds)

```python
# agents/categoriser.py: decide node (current)
def decide(state: CategoriserState) -> CategoriserState:
    confidence = state["prediction"]["confidence"]

    if confidence > 0.85:
        status = "auto_categorised"
    elif confidence > 0.50:
        status = "suggested"
    else:
        status = "needs_review"

    # ... update transaction in DB and log to AuditLog
    state["status"] = status
    return state
```

### Replacement code (LearnedRouter)

```python
# agents/categoriser.py: decide node (with LearnedRouter)
from integration.router import LearnedRouter

# Create once at module level (loads model on import)
_router = LearnedRouter(model_variant="C")   # PPO-C: most conservative reward (-5.0)


def decide(state: CategoriserState) -> CategoriserState:
    confidence = state["prediction"]["confidence"]
    transaction = state["transaction_data"]

    decision = _router.route(
        confidence_score=confidence,
        transaction_features={
            "amount": float(transaction.get("amount", 0)),
            "difficulty_tier": 1,     # or derive from context
            "category_entropy": 0.0,  # Claude API doesn't return top-k probs
        },
    )

    status = decision.status  # "auto_categorised" | "suggested" | "needs_review"

    # Existing DB update logic unchanged ...

    # Add router audit data to the existing AuditLog entry
    # Merge into ai_decision_data alongside the LLM reasoning text:
    existing_audit_data = state.get("audit_data", {})
    existing_audit_data.update(decision.audit_data)
    state["audit_data"] = existing_audit_data

    state["status"] = status
    return state
```

---

## What does NOT change

- The CategoriserAgent's `fetch_context`, `classify`, `validate`, and `explain`
  nodes are completely unchanged.
- The `AuditLog` schema is unchanged: `decision.audit_data` is stored in the
  existing `ai_decision_data` JSONB column alongside the LLM reasoning text.
- All API routes, database models, and frontend components are unchanged.
- The fallback is automatic: if `models/trained/ppo_variant_C.zip` is missing,
  the router logs a warning and uses the original 0.85/0.50 thresholds.

---

## Deploying the model file

The trained model (`models/trained/ppo_variant_C.zip`, ~140KB) is committed
to this repository. Copy it to the parent project at the same relative path:

```bash
cp models/trained/ppo_variant_C.zip /path/to/parent-project/models/trained/
```

Or set `MODEL_PATH` in the parent project's `.env` to point to the file
and update the `LearnedRouter()` constructor call accordingly.

---

## Switching between variants

| Variant | Use case | Reward design |
|---------|----------|---------------|
| A | High-throughput, error-tolerant | -2.0 for an accepted wrong prediction |
| B | Load-sensitive (vary by queue depth) | A, with the review penalty scaled by accountant load |
| C | Compliance-critical | -5.0 for an accepted wrong prediction |
| baseline | No ML, pure thresholds | Fixed 0.85 / 0.50 thresholds |

Change `model_variant="C"` in the `_router = LearnedRouter(...)` line, or
set the `ROUTING_MODEL_VARIANT` environment variable.

---

## Verifying the integration

Run this repository's router tests, which cover model loading, the threshold
fallback, the audit data and the interface match with the baseline:

```bash
python -m pytest integration/tests -v
```

The router decides which status a prediction receives (`auto_categorised`,
`suggested` or `needs_review`); it does not change the predicted category, so
the parent project's categorisation eval
(`python -m evals.eval_runner --mode mock`, run from `backend/`) is unaffected
by the swap.

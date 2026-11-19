# RL_ROUTING_PROJECT.md: Research Framing

> Historical note: this file began as the master prompt used to scaffold this
> repository. The phase-by-phase execution instructions have been removed as
> they are now stale; what remains is the research framing, which is still
> load-bearing for how this project is presented. For current status see
> `PROGRESS.md`; for reproduction instructions see `README.md`.

---

## Context and motivation

This repository is a research extension of the [AI Accountant](https://github.com/rajo69/agentic-ai-accounting),
a deployed FastAPI + LangGraph review layer that proposes a category for every
transaction, for UK accountants connected via the Xero API. Its core
component is a `CategoriserAgent` that uses Anthropic's Claude API to predict
a category for each bank transaction with a confidence score in [0, 1].

### The research question

The parent project's `CategoriserAgent` uses three hand-tuned confidence
thresholds:

- above 0.85 → auto-approve
- 0.50 to 0.85 → surface for accountant review
- below 0.50 → flag for manual decision

These thresholds were set by hand as defaults. They are fixed: they do not adapt to accountant workload, transaction
difficulty, or the asymmetric stakes of different misclassifications. The
natural research question: **can a lightweight reinforcement learning agent
learn better routing thresholds than hand tuning?**

This project builds and evaluates exactly that. A PPO agent observes the
CategoriserAgent's confidence score plus transaction features, chooses a
routing action, and receives reward based on whether the categorisation was
in fact correct. We compare the learned policy against the hand-tuned
baseline using routing accuracy, auto-approval precision, and workload
reduction as metrics. Three reward variants encode different operational
priorities (typical / workload-weighted / compliance-critical).

### Why this matters beyond this application

Deciding when an AI component should act on its own and when it should hand a
case to a person is an open question for agentic AI systems. This repository
studies one small, synthetic instance of it; the findings are a case study,
not a general result.

### What the repository now reports (as of 2026-04-21)

See `README.md` for full numbers with confidence intervals. In summary:

- On the natural evaluation regime, all three PPO variants learn the same
  rule: auto-approve if and only if the difficulty tier is easy. The tier is a
  label from the seed fixtures that the thresholds do not see. Routing
  accuracy and precision do not differ from the hand-tuned baseline.
- The learned policy auto-approves 45.8% of held-out transactions against
  63.8% for the thresholds. Counting each seed scenario once, the gap has a
  95% interval of 4.0 to 32.8 points (exact p = 0.032, 0.096 after a Holm
  correction). It comes from PPO reviewing the whole medium-difficulty tier,
  where 17 of the baseline's 31 auto-approvals disagree with the labels.
- A calibration probe (logistic recalibration on confidence, amount and tier)
  improves Brier and log-loss, mostly by learning the tier base rates, and
  leaves the shared policy unchanged.
- A constructed-regime probe (easy-tier accuracy moved into the band where
  Variant C's EV-optimal tier action differs from A/B's) produces the
  expected divergence on both calibrated and raw confidence. This is
  consistent with an expected-value explanation for the natural-regime
  convergence.
- A five-seed robustness sweep (3 variants x 5 seeds x {raw, regime} = 30
  retrainings) shows the tier-level policy does not depend on the training
  seed: every retrained model matches the action totals of the canonical
  seed=42 policy, with population standard deviation 0.00 percentage points
  across every headline metric on both regimes.

### Integration with the parent project

See `integration/INTEGRATION_GUIDE.md` for the exact code change required in
the parent project's `CategoriserAgent.decide` node to swap in the learned
router. The trained model (`models/trained/ppo_variant_C.zip`) is committed
to this repository and is ~140 KB.

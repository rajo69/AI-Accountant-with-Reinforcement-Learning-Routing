# RL_ROUTING_PROJECT.md — Research Framing

> Historical note: this file began as the master prompt used to scaffold this
> repository. The phase-by-phase execution instructions have been removed as
> they are now stale; what remains is the research framing, which is still
> load-bearing for how this project is presented. For current status see
> `PROGRESS.md`; for reproduction instructions see `README.md`.

---

## Context and motivation

This repository is a research extension of the [AI Accountant](https://github.com/rajo69/agentic-ai-accounting),
a production-grade FastAPI + LangGraph system that automates transaction
categorisation for UK accountants connected via the Xero API. Its core
component is a `CategoriserAgent` that uses Anthropic's Claude API to predict
a category for each bank transaction with a confidence score in [0, 1].

### The research question

The parent project's `CategoriserAgent` uses three hand-tuned confidence
thresholds:

- above 0.85 → auto-approve
- 0.50–0.85 → surface for accountant review
- below 0.50 → flag for manual decision

These thresholds were chosen by the developer as reasonable defaults. They
are arbitrary — they do not adapt to accountant workload, transaction
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

Confidence-gated autonomy — deciding when an AI should act versus escalate to
a human — is an unsolved problem across agentic AI systems broadly. Findings
here are relevant to clinical decision support, autonomous compliance
checking, and any multi-agent pipeline where human oversight must be
calibrated against throughput.

### What the repository now reports (as of 2026-04-21)

See `README.md` for full numbers with confidence intervals. In summary:

- On the natural evaluation regime, all three PPO variants converge to the
  same action sequence. Accuracy and precision gains over the hand-tuned
  baseline are not statistically significant at n=177.
- The auto-approval *rate* difference IS highly significant (p<0.001), driven
  by PPO's refusal to auto-approve medium-difficulty transactions where the
  baseline has 54.8% error.
- A calibration probe (Platt-scaled logistic regression) improves Brier and
  log-loss substantially but does not break variant convergence.
- A constructed-regime probe (easy-tier accuracy moved into the band where
  Variant C's EV-optimal tier action differs from A/B's) produces the
  expected divergence on both calibrated and raw confidence. This supports
  an expected-value-invariance explanation for the natural-regime
  convergence.
- A five-seed robustness sweep (3 variants x 5 seeds x {raw, regime} = 30
  trainings) confirms the tier-level policy is seed-invariant on this
  problem: every trained model produces an action sequence identical to
  the canonical seed=42 policy, with population standard deviation 0.00
  percentage points across every headline metric on both regimes.

### Integration with the parent project

See `integration/INTEGRATION_GUIDE.md` for the exact code change required in
the parent project's `CategoriserAgent.decide` node to swap in the learned
router. The trained model (`models/trained/ppo_variant_C.zip`) is committed
to this repository and is ~140 KB.

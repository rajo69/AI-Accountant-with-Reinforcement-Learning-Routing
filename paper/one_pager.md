# Learned Routing Under Uncalibrated LLM Confidence: a PPO Case Study on an LLM Accounting Pipeline

**Rajarshi Nandi** · `einkatz65@gmail.com` · [github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing](https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing) · April 2026

*This is the 1-page companion note to the repository. For the full methodology, references, and evaluation scope, see the repository README.*

---

## Problem

Many agentic pipelines use fixed confidence thresholds to decide when an LLM-based component may act on its own and when it should hand a case to a person. These thresholds (e.g. `confidence > 0.85 ⇒ auto-approve`) are hand-tuned and do not adapt to input difficulty, stakes, or operator load.

## Question

Can a lightweight PPO policy, observing the confidence score plus a few transaction features, route an LLM accounting agent's decisions better than its hand-tuned thresholds?

## Setup

The parent system is an AI bookkeeping assistant for UK accountants. A `CategoriserAgent` (Claude Haiku via Instructor) assigns a category and a confidence score to each Xero bank transaction; a hand-tuned router then picks one of three actions: `AUTO_APPROVE`, `SURFACE_FOR_REVIEW`, or `REJECT_FOR_MANUAL`. The study uses a Gymnasium environment (a contextual bandit, since an action never changes the next transaction) whose state holds the confidence, the log amount and a difficulty-tier label from the seed fixtures; the thresholds see the confidence only. The data are 882 synthetic transactions: 50 seed scenarios written with Claude plus 832 variants, 244 of them template copies from a silent generator fallback, all scored by Claude Haiku 4.5, with labels no person checked. The 80/20 split is by transaction, so all 49 scenarios in the 177-item held-out set also appear in training. Three PPO variants (Stable-Baselines3 2.7.1, 100k steps, seed=42) correspond to three deployment priorities: a typical firm (A), a workload-weighted high-volume firm (B), and a compliance-critical setting (C) with a 5× penalty on silent auto-approval errors.

## Results (held-out n=177; Wilson 95% CIs, which assume independent items)

| Metric | Baseline | PPO (A/B/C identical) |
|---|:---:|:---:|
| Routing accuracy (all 177) | 66.7% [59.4, 73.2] | 63.3% [56.0, 70.0] |
| Auto-approval precision | 72.6% [63.7, 79.9] | 77.8% [67.6, 85.5] |
| Auto-approval error rate | 27.4% [20.1, 36.3] | 22.2% [14.5, 32.4] |
| **Auto-approval rate** | **63.8% [56.5, 70.6]** | **45.8% [38.6, 53.1]** |

All three PPO variants learned the same policy: auto-approve every easy-tier transaction and review every medium-tier and hard-tier one, whatever the confidence. Counting each seed scenario once, the auto-approval gap of 18.1 points has a 95% interval of 4.0 to 32.8 points (exact p = 0.032; 0.096 after a Holm correction over three comparisons). Routing accuracy does not differ between the policies, and neither does better than approving everything (62.1%). The baseline auto-approves 31 medium-tier transactions, 17 of which disagree with the reference labels; PPO auto-approves none of them, because it reviews the whole tier.

## Interpretation

The learned policy is a rule on a label the thresholds never see. What it found is that the difficulty tier predicts Haiku's mistakes better than Haiku's stated confidence, which took only 14 distinct values and separated right from wrong answers weakly (AUROC 0.66). A live transaction has no tier, so this is not yet a deployable router.

Two probes test the explanation. A **calibration probe** (logistic recalibration on [confidence, amount, tier_onehot] → is_correct; not Platt scaling, which uses the confidence alone) improves proper scoring rules on eval (Brier 0.295 → 0.216, close to the 0.219 that tier base rates alone give) and leaves the shared A/B/C policy unchanged. A **regime probe** removes correct easy-tier items until easy-tier accuracy is 0.72, inside the band (0.64, 0.80) where Variants A and B prefer AUTO_APPROVE but Variant C prefers SURFACE_FOR_REVIEW. On the reshaped set A and B auto-approve the whole easy tier and C auto-approves nothing, as the break-even arithmetic predicts, on raw and recalibrated confidence alike. This is consistent with a simple account: reward tables change the policy when they disagree on the expected-reward-optimal action for a tier at the observed accuracies. On the natural data the margins are thin (the easy tier sits 2.3 points above C's break-even).

## Next experiments

Split by scenario and give every policy the same inputs; compare against a calibrated break-even threshold and a learning-to-defer model; check the labels by hand; use multi-sample self-consistency for a richer uncertainty signal; extend to the parent pipeline's reconciliation agent. A five-seed robustness sweep (30 retrainings over seeds 0 to 4, three variants, two datasets) matched every canonical run's action totals and metrics; see the repository README's "Multi-seed robustness" subsection.

## Code, data, reproducibility

Full code, held-out eval set, trained PPO models, and a reproducible statistical-analysis script are at the GitHub link above (MIT). The item-level numbers on this page regenerate via `python -m agent.evaluate` followed by `python -m experiments.statistical_analysis`; the scenario-level interval comes from a separate re-analysis described in the README.

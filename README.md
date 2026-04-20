# AI Accountant: Learned Routing via Reinforcement Learning

> A research extension investigating whether a learned routing policy can
> outperform hand-tuned confidence thresholds in a production multi-agent
> accounting pipeline.

[![CI](https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing/actions/workflows/ci.yml/badge.svg)](https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Research Question

Can a lightweight Proximal Policy Optimisation (PPO) agent learn to make better
routing decisions than a fixed confidence threshold in a multi-agent AI pipeline?
Specifically: given a transaction categorisation agent that outputs a confidence
score, should the system auto-approve the prediction, surface it for human review,
or flag it for manual re-categorisation? Can a learned policy calibrate this
decision more effectively than a hand-chosen threshold?

This question matters because confidence-gated autonomy (the problem of deciding
when an AI should act versus escalate to a human) is a core unsolved problem
across production agentic systems [1, 10]. The findings here bear directly on
clinical decision support, autonomous compliance checking, and any pipeline where
human oversight must be calibrated against throughput.

---

## Background

### The Parent Project

The AI Accountant is a production-grade FastAPI and LangGraph system that
automates transaction categorisation for UK accountants connected via the Xero
API. Its core component is a `CategoriserAgent` that uses Anthropic's Claude API
(via the Instructor library) to predict a category for each bank transaction,
along with a confidence score between 0 and 1.

### The Problem with Hand-Tuned Thresholds

After classification, the agent applies three fixed routing thresholds inherited
from the developer's initial intuition:

```python
if confidence > 0.85:   status = "auto_categorised"   # write to ledger
elif confidence > 0.50: status = "suggested"           # add to review queue
else:                   status = "needs_review"        # flag for manual work
```

These values (0.85 and 0.50) are arbitrary. They do not adapt to:
- the accountant's current workload or available capacity
- the inherent difficulty of the transaction (ambiguous vendor descriptions)
- the stakes of a misclassification (VAT-sensitive categories vs. petty cash)

A natural research question arises: if we model routing as a decision problem and
give an agent feedback on whether its routing decisions were correct, can it learn
a better policy than a hard-coded threshold? The problem is closely related to the
"learning to defer" literature, where a classifier learns to route uncertain inputs
to a human expert [1, 9].

### This Project

This repository builds and evaluates that learned policy. A PPO agent [2] observes
the CategoriserAgent's confidence score and transaction features, chooses a
routing action, and receives reward based on whether the routing decision was
appropriate. We train three variants with different reward designs (see
Methodology) and compare against the hand-tuned baseline on a held-out evaluation
set.

---

## Methodology

### Environment Design

The routing task is framed as a finite-horizon Markov Decision Process. Each
episode is one full pass through the transaction dataset (shuffled). At each
step the agent sees one transaction and must choose an action. The environment
is implemented using the Gymnasium interface [4].

**State space** (4-dimensional continuous vector, 5 for Variant B):

| Feature | Range | Description |
|---------|-------|-------------|
| `confidence_score` | [0, 1] | Primary routing signal from CategoriserAgent |
| `amount_normalised` | [0, 1] | log1p(\|amount\|) / log1p(50,000) |
| `difficulty_tier` | {0, 1, 2} | easy / medium / hard from eval framework |
| `category_entropy` | [0, 1] | Entropy over top-k predictions (0 currently) |
| `accountant_load` | [0, 1] | Queue pressure (Variant B only) |

**Action space** (Discrete(3)):

| Action | Index | Effect |
|--------|-------|--------|
| AUTO_APPROVE | 0 | Accept prediction; write to ledger without review |
| SURFACE_FOR_REVIEW | 1 | Add to accountant review queue |
| REJECT_FOR_MANUAL | 2 | Flag for full manual re-categorisation |

### Reward Function Variants

Three variants were designed to model different deployment priorities. The
asymmetries are not arbitrary; each value encodes a specific cost judgement
grounded in the reward shaping literature [5].

**Variant A (Binary asymmetric, typical firm):**

| Situation | Reward | Rationale |
|-----------|--------|-----------|
| AUTO_APPROVE, correct | +1.0 | Ideal outcome; no human time spent |
| AUTO_APPROVE, wrong | -2.0 | Silent ledger error; 2:1 asymmetry models audit risk |
| SURFACE_FOR_REVIEW, correct | -0.3 | Unnecessary escalation wastes accountant time |
| SURFACE_FOR_REVIEW, wrong | +0.3 | Warranted escalation; positive but modest |
| REJECT, correct | -1.0 | Discards good prediction; costlier than escalation |
| REJECT, wrong | +0.5 | Warranted rejection; worth more than mere escalation |

**Variant B (Workload-weighted, high-volume firm):**
Identical to A except the SURFACE_FOR_REVIEW unnecessary penalty scales with
`accountant_load`: `-0.3 x (1 + load)`. At full load the penalty doubles,
teaching the agent to be more conservative about unnecessary escalation when the
queue is already stressed.

**Variant C (Conservative, compliance-critical, e.g. HMRC audit prep):**
Identical to A except AUTO_APPROVE wrong is **-5.0** instead of -2.0. This
models the catastrophic risk of a silent misclassification in a regulated context
(VAT return restatement, grant claw-back, covenant breach).

### Synthetic Data Generation

The training dataset was generated from the 50 labelled seed transactions in the
parent project's evaluation framework by
[`environment/transaction_simulator.py`](environment/transaction_simulator.py):

1. The Claude API generated 10 variations of each seed transaction, varying
   amount, date, merchant phrasing, and description while preserving category.
2. Each synthetic transaction was processed by a simplified CategoriserAgent
   prompt (same Claude model as production, without pgvector few-shot context)
   to obtain real Claude confidence scores. See `data/synthetic/README.md`.
3. Final dataset: **882 transactions** (705 training / 177 evaluation), stratified
   by difficulty tier across all 50 seed categories.

| Tier | Train | Eval | Agent Accuracy (eval) |
|------|-------|------|---------------|
| Easy | 322 | 81 | 78% |
| Medium | 252 | 63 | 48% |
| Hard | 131 | 33 | 52% |

**Effective unit of diversity.** The 882 transactions derive from **50 hand-
labelled seed scenarios × ~18 Claude-generated variants each**, not from 882
independent draws. Variants within a seed share vendor family and category; they
differ in amount, date, merchant phrasing, and reference. Claims about
statistical power on this dataset should be read accordingly — small-n caveats
in the eval set (177 total, 33 hard) compound with this seed-level dependence.
Additional variation from a larger, seed-independent distribution is listed as
future work.

### Training

All three variants were trained using PPO [2] as implemented in Stable Baselines3 [3].

- **Algorithm:** PPO (Stable Baselines3 v2.7.1)
- **Hyperparameters:** lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
  gamma=0.99, lambda=0.95, clip_range=0.2, [64, 64] MLP
- **Duration:** 100,000 timesteps per variant (seed=42)
- **Hardware:** Consumer CPU (approximately 10-25 min per variant)

### Evaluation

The held-out 177 transactions were never seen during training (stratified 80/20
split). All four policies (baseline + 3 PPO variants) were evaluated
deterministically on identical data.

**Optimal routing** for a given transaction:
- `is_correct=True`: AUTO_APPROVE is optimal
- `is_correct=False`: SURFACE_FOR_REVIEW is optimal

This definition penalises REJECT_FOR_MANUAL even for wrong predictions (SURFACE
is less costly). See Key Findings and Limitations for how this affects
interpretation.

### Statistical methods

All headline rates are reported with Wilson score 95% confidence intervals.
Policy comparisons use two-sided two-proportion z-tests (pooled variance).
Per-tier rates are reported with their subset sample size; small-n tiers
(e.g., baseline auto-approvals in the hard tier, n=10) have wide intervals and
are flagged rather than interpreted. The analysis is fully reproducible from
the committed `experiments/results/evaluation_results.json` via
`python -m experiments.statistical_analysis`, which writes
`statistical_summary.json` (machine-readable) and `statistical_summary.md`
(human-readable) alongside the evaluation file.

---

## Results

*Run with real Claude Haiku confidence scores from the CategoriserAgent. All
three PPO variants produce identical action sequences on the held-out set, so
they are reported as a single "PPO" column.*

### Headline metrics (n=177)

Ranges are Wilson score 95% confidence intervals. P-values are two-sided
two-proportion z-tests comparing Baseline and PPO. See `experiments/statistical_analysis.py`.

| Metric | Baseline (0.85/0.50) | PPO (A/B/C identical) | Significance |
|---|:---:|:---:|:---:|
| Routing accuracy (over 177)   | 66.7% [59.4%, 73.2%] | 63.3% [56.0%, 70.0%] | p=0.50 |
| Auto-approval precision       | 72.6% [63.7%, 79.9%] | 77.8% [67.6%, 85.5%] | p=0.41 |
| Auto-approval error rate      | 27.4% [20.1%, 36.3%] | 22.2% [14.5%, 32.4%] | p=0.41 |
| **Auto-approval rate**        | **63.8% [56.5%, 70.6%]** | **45.8% [38.6%, 53.1%]** | **p=0.001** |

Of the four metrics, only the auto-approval *rate* differs significantly at
n=177. The apparent accuracy and precision differences are directionally
favourable to PPO but underpowered on this eval set.

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| Easy   | 19.4% [12.0%, 30.0%] (n=72) | 22.2% [14.5%, 32.4%] (n=81) |
| Medium | **54.8% [37.8%, 70.8%]** (n=31) | N/A — PPO auto-approves zero |
| Hard   | 0.0% [0.0%, 27.8%] (n=10)   | N/A — PPO auto-approves zero |

The single most consequential row is Medium: the baseline auto-approves 31
medium-tier transactions with a **54.8% error rate** (more wrong than right),
while PPO auto-approves zero. This is what the significant auto-approval-rate
difference is buying.

Analysis figures: [`experiments/results/figures/`](experiments/results/figures/).
Reproducible CIs and p-values: `python -m experiments.statistical_analysis`.

### Calibration probe

To test whether the variant convergence in Key Finding #1 is explained by
uncalibrated confidence, we re-trained all three PPO variants on a Platt-scaled
calibrated confidence signal (scikit-learn logistic regression fit on
`[raw_confidence, amount_normalised, tier_onehot]` via 5-fold stratified CV
for training-set scores; full-training-set fit for eval scores). The calibrator
materially improves proper scoring rules on the eval set and produces a
continuous-looking signal rather than the ~7-value clusters of raw Claude
confidence.

| Signal | Brier (↓) | Log-loss (↓) | Unique values on eval (easy / medium / hard) |
|---|:---:|:---:|:---:|
| Raw Claude Haiku confidence | 0.295 | 0.910 | 7 / 7 / 4 |
| Platt-scaled (this probe) | **0.215** | **0.616** | 74 / 60 / 32 |

All three PPO variants nevertheless produce **identical** action sequences on
the calibrated eval set: the same 81 auto-approvals / 96 surfacings / 0 rejects
as under raw confidence, the same 63.3% routing accuracy, and the same 77.8%
auto-approval precision. The same-policy-collapse observed in Key Finding #1
is not resolved by this particular calibration approach.

The hand-tuned baseline, applied to calibrated confidences without retuning
its 0.85/0.50 thresholds, drops to 42.4% routing accuracy and 8.5%
auto-approval rate (it auto-approves only 15 transactions, all correct, and
now emits 18 rejects). This is a mis-tuned-baseline artefact and not a fair
head-to-head with PPO on the calibrated regime; the meaningful comparison here
is PPO-A vs PPO-B vs PPO-C, which are equal.

A working hypothesis for why the variants remain equal under calibration: at
the observed per-tier agent accuracies (easy 82.6% / medium 52.4% / hard 62.6%
on the training set), the EV-optimal tier-level action is the same for all
three reward variants — easy-tier expected reward from AUTO_APPROVE is
positive even under Variant C's 5:1 loss ratio, and medium- and hard-tier
expected reward from AUTO_APPROVE is negative even under Variant A's 2:1
ratio. A data regime with per-tier accuracies in a narrower band
(e.g. easy-tier accuracy ~0.75, medium-tier ~0.75) would be expected to
change this calculus; we list this as future work.

Reproduce the probe: `python -m experiments.calibrate` followed by
`python -m agent.train --reward {A,B,C} --dataset calibrated` and
`python -m agent.evaluate --dataset calibrated`. Full diagnostics in
`experiments/results/calibration_report.json` and
`experiments/results/statistical_summary_calibrated.md`.

### Regime probe (EV-invariance test)

The calibration probe's candidate explanation (Key Finding #4) predicts that
all three PPO variants converge because the observed per-tier accuracies
(easy 82.6% / medium 52.4% / hard 62.6%) place every tier outside the A-vs-C
divergence band (0.64, 0.80) derived from the reward tables. We test this
directly by reshaping the easy tier to 72% accuracy (via subsampling;
medium and hard kept unchanged) and retraining all three PPO variants on the
matching reshaped training set (72% easy / 52% medium / 63% hard, seed=42,
100k steps). We run the probe on **both** the Platt-calibrated and raw
confidence regimes as a robustness check.

**EV break-even thresholds** (accuracy above which AUTO_APPROVE beats
SURFACE_FOR_REVIEW for each variant; B evaluated at E[accountant_load]=0.5):

| Variant | Break-even p |
|---|:---:|
| A (AUTO penalty −2) | 0.639 |
| B (E[load]=0.5)     | 0.613 |
| C (AUTO penalty −5) | 0.803 |

At p=0.72, A and B should prefer AUTO_APPROVE on easy tier; C should prefer
SURFACE_FOR_REVIEW. All three should surface medium and hard.

**Result: Variant C diverges from Variants A and B exactly as predicted, on
both signal regimes.** Per-variant action totals (AUTO / SURFACE / REJECT) on
the reshaped eval set (n=160):

| Variant | Calibrated regime | Raw regime |
|---|:---:|:---:|
| PPO-A | 64 / 96 / 0 | 64 / 96 / 0 |
| PPO-B | 64 / 96 / 0 | 64 / 96 / 0 |
| **PPO-C** | **0 / 160 / 0** | **0 / 160 / 0** |

PPO-A and PPO-B each auto-approve all 64 easy-tier transactions (and zero
others); PPO-C auto-approves zero and surfaces all 160. The result replicates
identically across the two signal regimes, which rules out calibration as the
primary driver and supports the EV-invariance mechanism.

Variants A and B remain indistinguishable at this sample size because their
break-even thresholds differ by only 0.026; A-vs-B divergence would require a
tier with accuracy precisely in (0.613, 0.639), narrower than the sampling
noise allows at n≈64 per tier. We flag this as future work.

This probe confirms the EV-invariance explanation from Key Finding #4:
reward-driven policies differ when and only when the reward tables produce
different EV-optimal tier-level actions at the observed per-tier accuracies.
On the natural data regime, all three reward tables agreed; on the constructed
regime where reward tables disagree, the policies diverge accordingly. Full
reproduction: `python -m experiments.regime_probe --source {calibrated,raw}`
then the corresponding train/evaluate/stats commands with
`--dataset {regime,regime_raw}`.

---

## Key Findings

1. **All PPO variants eliminated REJECT_FOR_MANUAL and converged to the same
   policy.** With real Claude API confidence scores, all three reward variants
   learned identical tier-based routing: auto-approve all easy-tier transactions,
   surface all medium and hard for review. The intended A/B/C differentiation did
   not materialise. The convergence persists after Platt-scaled calibration of
   the confidence signal (see the Calibration probe subsection under Results);
   calibration alone does not produce variant divergence in this probe.

2. **Real confidence scores render the fixed-threshold baseline significantly
   worse.** With mock scores the baseline auto-approved 37.3% at 10.6% error.
   With real scores the same thresholds auto-approve 63.8% at 27.4% error,
   because Claude Haiku outputs confidence values at or above 0.85 for nearly
   all transactions, including many incorrect medium-tier predictions (54.8%
   error rate among those auto-approved). This finding is consistent with
   broader evidence that verbally elicited LLM confidence scores are
   systematically overconfident and poorly calibrated [6, 7, 8].

3. **PPO's apparent accuracy and precision gains are not statistically
   significant at n=177, but PPO learns a qualitatively different and safer
   strategy.** Two-proportion z-tests on the held-out set show no significant
   difference in overall routing accuracy (p=0.50), auto-approval precision
   (p=0.41), or auto-approval error rate (p=0.41) — the confidence intervals
   overlap heavily. What *is* highly significant is the auto-approval *rate*:
   63.8% [56.5%, 70.6%] for the baseline versus 45.8% [38.6%, 53.1%] for PPO
   (p=0.001). The per-tier breakdown shows the mechanism: the baseline
   auto-approves 31 medium-tier transactions at a 54.8% [37.8%, 70.8%] error
   rate — more wrong than right — while PPO auto-approves zero medium-tier.
   The defensible contribution is not "PPO improves accuracy" but "PPO learns
   to not auto-approve the tier where the baseline is dangerously wrong."

4. **Confidence calibration is necessary but not sufficient for variant
   divergence.** Single-sample Claude Haiku confidence scores are poorly
   calibrated [6, 7, 8], and Platt-scaled calibration on the same features
   materially improves proper scoring rules (Brier 0.295 → 0.215 on eval).
   But applying that calibrated signal does not by itself break the
   A/B/C convergence (see Calibration probe). A candidate explanation,
   supported by the observed per-tier accuracies (82.6% / 52.4% / 62.6% on the
   training set), is that the EV-optimal tier-level action is the same for all
   three reward variants in this accuracy regime — easy-tier AUTO_APPROVE
   remains positive-EV even under Variant C's 5:1 loss ratio, and medium- and
   hard-tier AUTO_APPROVE remain negative-EV even under Variant A's 2:1 ratio.
   Whether variants would diverge under richer uncertainty signals (multi-sample
   self-consistency, top-k log-probabilities) or under data regimes with
   different per-tier accuracy profiles is an open empirical question we flag
   as future work.

5. **REJECT elimination is a learned policy-level property.** All three PPO
   variants assign zero probability to REJECT_FOR_MANUAL under their trained
   policies, regardless of input confidence; this dominance of
   SURFACE_FOR_REVIEW over REJECT_FOR_MANUAL holds across both mock and real
   confidence-score regimes during training. The hand-tuned baseline also emits
   zero rejects on the held-out set, but for a different reason: real Claude
   Haiku confidences never fall below the baseline's 0.5 reject threshold.
   The PPO result is therefore stronger than the eval-set coincidence suggests
   — PPO would continue to never reject even if confidence scores dropped below
   0.5, whereas the baseline would begin rejecting. A data regime that exposes
   lower-confidence transactions would separate these two policies on this
   axis.

---

## Limitations and Future Work

**Calibration alone did not resolve variant convergence.** Real Claude Haiku
confidence scores cluster at round values (0.95, 0.85, 0.75) regardless of
difficulty tier [6, 7, 8], and Platt-scaled calibration on available features
improves proper scoring rules but does not change the learned tier-based
policy (Calibration probe, above). A candidate explanation is that the
per-tier accuracy structure (easy 82.6% / medium 52.4% / hard 62.6%) makes the
tier-level EV-optimal action invariant across reward variants in this regime.
Testing this requires either richer uncertainty signals (multi-sample
self-consistency, top-k log-probabilities) or data with different per-tier
accuracies; both are listed below.

**All reward variants converged to the same policy** with real scores, removing
the intended A/B/C tradeoff surface. Longer training (500k to 1M steps) or harder
reward gradients may be needed to recover differentiation.

**Small evaluation set.** 177 transactions is insufficient for statistically
robust conclusions (especially hard tier, n=33). All results are indicative.

**No real feedback loop.** The reward signal is derived from a synthetic
`is_correct` flag, not from actual accountant corrections. Production deployment
would require online learning from genuine human feedback, as studied in the
learning-to-defer literature [1, 9].

**Future work:** Multi-sample self-consistency or top-k log-probability
confidence estimation to test richer signals than Platt scaling; a probe on
a dataset with deliberately different per-tier agent accuracies to test the
EV-invariance explanation for variant convergence; online learning from
accountant corrections; multi-load evaluation for Variant B; longer training;
extending to the ReconcilerAgent (which also uses fixed thresholds).

---

## Implications for Agentic AI Systems

Two practical implications can be stated with the evidence collected here; a
third often-claimed implication was not supported and is flagged honestly.

**Supported.** Reward-driven routing can discover dominated actions in the
routing action space without explicit specification: all three PPO variants
learn to avoid REJECT_FOR_MANUAL entirely, regardless of input confidence. For
a designer of a routing layer, this suggests that asymmetric reward tables are
a useful way to surface which actions are practically dominated, even on
relatively small training sets.

**Supported.** Learned routing can change the operating point on the
coverage/precision curve in a way that hand-tuned thresholds do not naturally
expose. On the held-out set, the hand-tuned baseline auto-approves 63.8% of
transactions at 72.6% precision, while the PPO policies auto-approve 45.8% at
77.8%; the difference in coverage is statistically significant (p=0.001). The
tradeoff is not an improvement in one number but a principled choice of
operating point, which is useful when downstream costs of false auto-approvals
and unnecessary escalations are asymmetric and known.

**Not supported by this experiment.** That reward-function design
*automatically* produces differentiated policies. All three reward variants in
this study produced identical action distributions on the eval set under both
raw and Platt-scaled calibrated confidences. A plausible explanation (EV-
invariance at the observed per-tier accuracies) suggests differentiation may
appear in other data regimes, but we did not demonstrate this here. Readers
considering reward shaping as a policy-tuning knob should not assume that
varying the reward table is sufficient to change the learned policy — whether
it does depends on the empirical error structure of the underlying classifier.

As confidence-gated autonomy becomes a standard component of agentic AI
pipelines (including systems that route queries between different models [10]),
the methodology described here offers a reproducible, data-driven baseline
for evaluating learned vs hand-tuned routing, and a template for the kind of
calibration-probe and EV analysis that should accompany such comparisons [1].

---

## Repository Structure

```
.
├── environment/            Gymnasium routing environment and reward variants
├── agent/                  PPO training, evaluation, and baseline policy
├── data/                   Synthetic training data and held-out eval set
├── experiments/            Results, figures, and analysis notebook
├── models/trained/         Committed PPO model files (~140KB each)
├── integration/            LearnedRouter drop-in and integration guide
├── api/                    FastAPI routing service
└── .github/workflows/      CI (test+lint+type-check) and manual retrain
```

See full annotated tree in [PROGRESS.md](PROGRESS.md).

---

## Quickstart

```bash
git clone https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing.git
cd AI-Accountant-with-Reinforcement-Learning-Routing
pip install -r requirements.txt

python -m agent.evaluate          # evaluate pre-trained models on held-out set
uvicorn api.main:app --reload     # serve the routing API at localhost:8000/docs
```

To retrain from scratch:

```bash
python -m agent.train --reward A  # approximately 10 min on CPU
python -m agent.train --reward B
python -m agent.train --reward C
```

---

## Running the API

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"confidence_score": 0.88, "transaction_features": {"amount": 450.0}}'
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway, Docker, and production instructions.

---

## References

[1] H. Mozannar and D. Sontag, "Consistent Estimators for Learning to Defer to
an Expert," in *Proceedings of the 37th International Conference on Machine
Learning (ICML)*, PMLR vol. 119, 2020, pp. 7076-7087.
[arXiv:2006.01862](https://arxiv.org/abs/2006.01862)

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal
Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[3] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann,
"Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal
of Machine Learning Research*, vol. 22, no. 268, pp. 1-8, 2021.
[jmlr.org/papers/v22/20-1364](https://jmlr.org/papers/v22/20-1364.html)

[4] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu,
et al., "Gymnasium: A Standard Interface for Reinforcement Learning
Environments," *arXiv preprint arXiv:2407.17032*, 2024.
[arXiv:2407.17032](https://arxiv.org/abs/2407.17032)

[5] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping," in *Proceedings of
the 16th International Conference on Machine Learning (ICML)*, 1999,
pp. 278-287.

[6] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, et al.,
"Language Models (Mostly) Know What They Know," *arXiv preprint
arXiv:2207.05221*, 2022.
[arXiv:2207.05221](https://arxiv.org/abs/2207.05221)

[7] M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, and B. Hooi, "Can LLMs Express
Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs,"
in *Proceedings of the 12th International Conference on Learning Representations
(ICLR)*, 2024.
[arXiv:2306.13063](https://arxiv.org/abs/2306.13063)

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern
Neural Networks," in *Proceedings of the 34th International Conference on
Machine Learning (ICML)*, PMLR vol. 70, 2017, pp. 1321-1330.
[arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

[9] N. Okati, A. De, and M. Gomez-Rodriguez, "Differentiable Learning Under
Triage," in *Advances in Neural Information Processing Systems 34 (NeurIPS)*,
2021.
[arXiv:2103.08902](https://arxiv.org/abs/2103.08902)

[10] I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez,
M. W. Kadous, and I. Stoica, "RouteLLM: Learning to Route LLMs with Preference
Data," in *Proceedings of the 13th International Conference on Learning
Representations (ICLR)*, 2025.
[arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

---

## Citing This Work

```bibtex
@misc{nandi2026rlrouting,
  author       = {Rajarshi Nandi},
  title        = {{AI Accountant: Learned Routing via Reinforcement Learning}},
  year         = {2026},
  howpublished = {\url{https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing}},
  note         = {Research extension: PPO-based confidence-gated routing for
                  multi-agent accounting pipelines.}
}
```

---

## Licence

MIT (c) 2026 Rajarshi Nandi

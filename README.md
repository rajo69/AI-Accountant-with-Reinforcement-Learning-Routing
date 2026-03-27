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
parent project's evaluation framework:

1. The Claude API generated 10 variations of each seed transaction, varying
   amount, date, merchant phrasing, and description while preserving category.
2. Each synthetic transaction was processed by the CategoriserAgent to obtain
   real confidence scores. (In the current repository, mock confidence scores
   are used due to API cost; see Limitations.)
3. Final dataset: **882 transactions** (705 training / 177 evaluation), stratified
   by difficulty tier across all 50 seed categories.

| Tier | Train | Eval | Agent Accuracy |
|------|-------|------|---------------|
| Easy | ~258 | 81 | ~90% |
| Medium | ~202 | 63 | ~70% |
| Hard | ~105 | 33 | ~52% |

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

---

## Results

*Run with real Claude Haiku confidence scores from the CategoriserAgent.*

| Policy | Routing Accuracy | Auto-Approval Precision | Auto-Approval Rate | Error Rate |
|--------|:---:|:---:|:---:|:---:|
| Baseline (0.85/0.50) | **66.7%** | 72.6% | 63.8% | 27.4% |
| PPO Variant A | 63.3% | **77.8%** | 45.8% | **22.2%** |
| PPO Variant B | 63.3% | **77.8%** | 45.8% | **22.2%** |
| PPO Variant C | 63.3% | **77.8%** | 45.8% | **22.2%** |

**Error rate by difficulty tier (auto-approved transactions only):**

| Tier | Baseline | PPO-A | PPO-B | PPO-C |
|------|:---:|:---:|:---:|:---:|
| Easy | 19.4% | 22.2% | 22.2% | 22.2% |
| Medium | 54.8% | N/A (none auto-approved) | N/A | N/A |
| Hard | 0.0% | N/A | N/A | N/A |

All PPO variants surface 100% of medium and hard transactions for review.

Analysis figures: [`experiments/results/figures/`](experiments/results/figures/)

---

## Key Findings

1. **All PPO variants eliminated REJECT_FOR_MANUAL and converged to the same
   policy.** With real Claude API confidence scores, all three reward variants
   learned identical tier-based routing: auto-approve all easy-tier transactions,
   surface all medium and hard for review. The intended A/B/C differentiation did
   not materialise. Real confidence scores (clustering at 0.95/0.85/0.75) provide
   less tier separation than mock scores, leaving less room for reward shaping to
   produce different behaviours.

2. **Real confidence scores render the fixed-threshold baseline significantly
   worse.** With mock scores the baseline auto-approved 37.3% at 10.6% error.
   With real scores the same thresholds auto-approve 63.8% at 27.4% error,
   because Claude Haiku outputs confidence values at or above 0.85 for nearly
   all transactions, including many incorrect medium-tier predictions (54.8%
   error rate among those auto-approved). This finding is consistent with
   broader evidence that verbally elicited LLM confidence scores are
   systematically overconfident and poorly calibrated [6, 7, 8].

3. **PPO variants beat the baseline on the metrics that matter most.** Despite
   lower routing accuracy (63.3% vs 66.7%), all PPO variants achieve better
   precision (77.8% vs 72.6%) and lower error rate (22.2% vs 27.4%). They do
   this by learning to refuse auto-approval of medium-tier transactions, which
   the threshold baseline incorrectly auto-approves at high volume.

4. **The binding constraint is confidence score calibration, not the RL method.**
   Single-sample Claude Haiku confidence scores are poorly calibrated: high
   values appear regardless of whether the prediction is correct [6, 7, 8]. A
   well-calibrated uncertainty signal (multi-sample prompting, or a classifier
   with a native probability output) would give the policy enough signal to learn
   fine-grained intra-tier routing. The RL infrastructure is sound; the signal
   quality limits it.

5. **REJECT elimination is robust across both data regimes.** The dominance of
   SURFACE_FOR_REVIEW over REJECT_FOR_MANUAL holds whether training uses mock or
   real confidence scores. This is the most reliable finding: reward-driven routing
   consistently eliminates the costliest escalation action without explicit guidance.

---

## Limitations and Future Work

**Confidence score calibration is the binding constraint.** Real Claude Haiku
confidence scores cluster at round values (0.95, 0.85, 0.75) regardless of
difficulty tier [6, 7, 8]. The resulting within-tier variance is too small for
the policy to learn confidence-based routing at 100k training steps. All three
PPO variants converge to the same tier-based strategy. A well-calibrated
uncertainty estimate from multi-sample prompting, top-k probability distributions,
or a native classifier would unlock finer-grained routing.

**All reward variants converged to the same policy** with real scores, removing
the intended A/B/C tradeoff surface. Longer training (500k to 1M steps) or harder
reward gradients may be needed to recover differentiation.

**Small evaluation set.** 177 transactions is insufficient for statistically
robust conclusions (especially hard tier, n=33). All results are indicative.

**No real feedback loop.** The reward signal is derived from a synthetic
`is_correct` flag, not from actual accountant corrections. Production deployment
would require online learning from genuine human feedback, as studied in the
learning-to-defer literature [1, 9].

**Future work:** Multi-sample confidence estimation; online learning from
accountant corrections; multi-load evaluation for Variant B; longer training;
extending to the ReconcilerAgent (which also uses fixed thresholds).

---

## Implications for Agentic AI Systems

The central result (that reward-driven routing discovers action dominance
relationships without explicit specification) has a direct practical implication:
the action space of your routing layer likely contains dominated actions that a
hand-tuned policy uses suboptimally. A learned policy will discover and eliminate
them.

The accuracy-vs-precision tradeoff between Variant A and Variant C demonstrates
that reward function design is not merely a hyperparameter but an operational
policy decision. Making this tradeoff explicit and parameterisable, rather than
encoded opaquely in a threshold value, is a step toward responsible deployment
of autonomous systems in high-stakes domains. As confidence-gated autonomy
becomes a standard component of agentic AI pipelines (including systems that
route queries between different models [10]), the methodology described here
offers a principled, data-driven alternative to hand-tuned thresholds [1].

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

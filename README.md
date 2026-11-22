# AI Accountant: Learned Routing via Reinforcement Learning

> PPO routing policies, trained in a Gymnasium environment with Stable-Baselines3,
> that decide when an LLM accounting pipeline should accept a prediction, ask a
> person, or flag it for manual work, compared with the pipeline's hand-tuned
> confidence thresholds on 882 synthetic transactions. Every learned policy
> reduced to one rule: auto-approve the easy-difficulty transactions and send
> the rest for review. The write-up below sets out why, and what that does and
> does not show. Ships with a FastAPI routing service and a `LearnedRouter`
> demonstration.

[![CI](https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing/actions/workflows/ci.yml/badge.svg)](https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Summary of Findings

- **The learned policy is a rule on a label.** All three reward variants
  learned the same policy: auto-approve a transaction if and only if its
  difficulty tier is "easy", otherwise send it for review. On the held-out data
  its decisions do not change with Claude Haiku's stated confidence. The tier is
  a label from the seed fixtures that the threshold baseline never sees and that
  a live transaction does not carry, so the comparison is between a tier rule
  and a confidence rule rather than a like-for-like test.
- **It auto-approves less, at similar routing accuracy.** On 177 held-out
  transactions the learned policy auto-approved 45.8% against 63.8% for the
  fixed 0.85/0.50 thresholds. The held-out transactions are new variants of the
  same 49 seed scenarios used in training, so the scenario is the honest unit:
  counting each scenario once, the gap of about 18 points has a 95% interval of
  4.0 to 32.8 points and an exact p of 0.032, or 0.096 after a Holm correction
  for three comparisons. Routing accuracy was 63.3% for the learned policy and
  66.7% for the thresholds, not distinguishable from each other or from
  approving everything (62.1%).
- **Haiku's stated confidence carried little signal.** It took only 14 distinct
  values across 882 transactions, ran well above its agreement with the
  reference labels (held-out mean 0.90 against 62% agreement) and separated
  right from wrong answers weakly (AUROC 0.66). A logistic recalibration using
  confidence, amount and tier cut the Brier score from 0.295 to 0.216, about what
  predicting each tier's training accuracy gives (0.219), and left the policy
  unchanged.
- **The reward designs agree because of where the tier accuracies fall.** All
  three reward tables gave the same rule on the natural data, and 30 retrainings
  (five extra seeds for each of three variants on two datasets) matched their
  canonical runs. A break-even analysis explains the agreement, and a
  constructed data regime that moves the easy tier between the break-evens
  separates the variants as that arithmetic predicts.
- **The data are synthetic.** The 882 transactions are 50 seed scenarios,
  written with Claude for the parent app's tests, plus 832 variants, of which
  244 are template copies from a silent fallback in the generator. Labels come
  from the same fixtures and were not checked by a person.

---

## Research Question

Can a lightweight Proximal Policy Optimisation (PPO) agent learn to make better
routing decisions than a fixed confidence threshold in a multi-agent AI pipeline?
Specifically: given a transaction categorisation agent that outputs a confidence
score, should the system auto-approve the prediction, surface it for human review,
or flag it for manual re-categorisation? Can a learned policy calibrate this
decision more effectively than a hand-chosen threshold?

Deciding when an AI component may act on its own and when it should hand a case
to a person is an open question for LLM pipelines, studied under learning to
defer and model routing [1, 9, 10]. This repository looks at one small instance
of it, bookkeeping categorisation, where the cost of each kind of mistake can be
written down. The results come from one synthetic dataset and are a case study,
not a general result.

---

## Background

### The Parent Project

The [AI Accountant](https://github.com/rajo69/agentic-ai-accounting) is a
deployed, open-source FastAPI and LangGraph review layer for UK small businesses
and bookkeepers on Xero: it proposes a category for every transaction, with a
confidence tier that tells the reviewer where to look. Its core component is the
categoriser workflow (the CategoriserAgent below) that uses Anthropic's Claude API
(via the Instructor library) to predict a category for each bank transaction,
along with a confidence score between 0 and 1.

### The Problem with Hand-Tuned Thresholds

After classification, the agent applies three fixed routing thresholds carried
over from the parent project's initial hand-tuning:

```python
if confidence > 0.85:   status = "auto_categorised"   # auto-accept, no review queue
elif confidence > 0.50: status = "suggested"           # add to review queue
else:                   status = "needs_review"        # flag for manual work
```

These thresholds (0.85 and 0.50) are fixed and were not learned from data. They do not adapt to:
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
the CategoriserAgent's confidence score, the transaction amount and a
difficulty-tier label, chooses a routing action, and receives a reward that
depends on whether the prediction it routed was right. We train three variants
with different reward designs (see Methodology) and compare them on a held-out
evaluation set against the hand-tuned baseline, which sees the confidence score
only.

---

## Methodology

### Environment Design

The environment uses the Gymnasium interface [4]. Each episode is one shuffled
pass through the training transactions; at each step the agent sees one
transaction and chooses an action. An action earns a reward for that
transaction alone and does not change which transaction comes next, so the task
is a contextual bandit rather than a sequential decision problem. PPO serves
here as a general-purpose learner for it.

**State space** (4-dimensional continuous vector, 5 for Variant B):

| Feature | Range | Description |
|---------|-------|-------------|
| `confidence_score` | [0, 1] | Claude Haiku's stated confidence (CategoriserAgent) |
| `amount_normalised` | [0, 1] | log1p(\|amount\|) / log1p(50,000) |
| `difficulty_tier` | {0, 1, 2} | easy / medium / hard label from the seed fixtures |
| `category_entropy` | [0, 1] | Entropy over top-k predictions (0 currently) |
| `accountant_load` | [0, 1] | Queue pressure (Variant B only) |

The threshold baseline sees `confidence_score` only. `difficulty_tier` is a
label written for the parent project's test fixtures, not something a live
transaction carries, and it turned out to drive every learned policy (see
Results).

**Action space** (Discrete(3)):

| Action | Index | Effect |
|--------|-------|--------|
| AUTO_APPROVE | 0 | Accept prediction without review (auto-accept tier) |
| SURFACE_FOR_REVIEW | 1 | Add to accountant review queue |
| REJECT_FOR_MANUAL | 2 | Flag for full manual re-categorisation |

### Reward Function Variants

Three variants were designed to model different deployment priorities. Each
value is a hand-set cost judgement. The tables are direct cost choices, unlike
the potential-based shaping terms that leave the optimal policy unchanged [5],
so changing them can change the learned policy (see the regime probe).

**Variant A (Binary asymmetric, typical firm):**

| Situation | Reward | Rationale |
|-----------|--------|-----------|
| AUTO_APPROVE, correct | +1.0 | Ideal outcome; no human time spent |
| AUTO_APPROVE, wrong | -2.0 | Silent miscategorisation; 2:1 asymmetry models audit risk |
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

### Training Data Generation

The dataset was generated from the 50 labelled seed transactions in the parent
project's evaluation fixtures (themselves written with Claude) by
[`environment/transaction_simulator.py`](environment/transaction_simulator.py):

1. For each seed, the simulator asked Claude Haiku 4.5 for 12, 20 or 40 variants
   (easy, medium and hard seeds respectively), varying amount, date, merchant
   phrasing and reference while keeping the category. For 16 seeds (12 easy,
   3 medium, 1 hard) that call failed and the simulator fell back, without
   stopping, to template copies: the seed description followed by " VAR1",
   " VAR2" and so on, with the amount jittered by up to 20%. These 244 rows
   (187 training, 57 held out) are in the committed data, and within each of
   those seeds every row has the same correctness.
2. Each transaction was categorised by a simplified CategoriserAgent prompt
   (same Claude model as the parent app, without its pgvector few-shot context),
   which returns a category and a stated confidence. Two rows, one training and
   one held out, carry a score from the simulator's mock fallback instead of
   Claude's. See `data/generated/README.md`.
3. Final dataset: **882 synthetic transactions** (50 seeds and 832 variants;
   705 training / 177 held out), split 80/20 by transaction within each
   difficulty tier. The held-out set covers 49 of the 50 seeds.

| Tier | Train | Held out | Categoriser agreement with labels (held out) |
|------|-------|------|---------------|
| Easy | 322 | 81 | 78% |
| Medium | 252 | 63 | 48% |
| Hard | 131 | 33 | 52% |

**Labels.** Categories and difficulty tiers come from the seed fixtures and were
not checked by a person. For three seeds (t022, t032, t041) the labelled
account has a different name in the chart of accounts the categoriser was
shown, and 81 of their 83 rows are scored wrong. Several other "errors" are
arguable, for example software subscriptions predicted as Subscriptions against
a label of Computer Equipment & Software. Accuracy and error rates in this
README are agreement with these reference labels.

**Effective unit of diversity.** The 882 transactions come from 50 seed
scenarios, not from 882 independent draws. Variants within a seed share vendor
family and category, and whether Haiku is right is mostly a property of the
scenario (intra-scenario correlation 0.76 on the held-out set). The 177
held-out items therefore carry roughly the information of 59 independent ones,
and the headline comparison under Results is also reported with the scenario
as the unit.

### Training

All three variants were trained using PPO [2] as implemented in Stable Baselines3 [3].

- **Algorithm:** PPO (Stable Baselines3 v2.7.1)
- **Hyperparameters:** lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
  gamma=0.99, lambda=0.95, clip_range=0.2, [64, 64] MLP
- **Duration:** 100,000 timesteps per variant (seed=42)
- **Hardware:** Consumer CPU (about 9 to 13 minutes per variant)

### Evaluation

The 177 held-out transactions were not used in training, but the split is by
transaction, not by scenario: every held-out item is a variant of a seed
scenario that also has variants in training (49 of 49 scenarios), and 57 of
them are template copies. The results are therefore estimates for new variants
of familiar scenarios, not for unfamiliar kinds of transaction. All four
policies (baseline + 3 PPO variants) were evaluated deterministically on the
same items.

**Optimal routing** for a given transaction:
- `is_correct=True`: AUTO_APPROVE is optimal
- `is_correct=False`: SURFACE_FOR_REVIEW is optimal

Routing accuracy, the share of items that received this action, is a simple
summary rather than the objective the policies were trained on. It counts
REJECT_FOR_MANUAL as wrong even for a wrong prediction, although every reward
table pays more for that (+0.5 against +0.3 for review), and it weighs a silent
wrong auto-approval and an unnecessary review equally, which no reward table
does. Neither policy chose REJECT on the held-out data (Key Finding 5), so the
first point does not affect the numbers below.

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

These item-level tests treat the two policies as independent samples of 177
independent items. In fact both policies route the same items, and the items
cluster in 49 seed scenarios; the paired, scenario-level comparison under
Results allows for both and is the one to quote.

---

## Results

*Run with real Claude Haiku confidence scores from the CategoriserAgent. All
three PPO variants make the same decision on every held-out item, so they are
reported as a single "PPO" column.*

### What the policy learned

Every PPO variant auto-approves all 81 easy-tier items and sends all 63
medium-tier and 33 hard-tier items for review: the rule "auto-approve if and
only if the difficulty tier is easy". Within a tier the decision does not change
with confidence. Easy items at confidence 0.75 are auto-approved, and the 31
medium-tier and 10 hard-tier items above 0.85 all go to review. The thresholds
never see the tier, so the tables below compare a tier rule with a confidence
rule.

### Headline metrics (n=177)

Ranges are Wilson score 95% confidence intervals. They assume independent items
and are roughly half as wide as intervals that respect the scenario structure.
See `experiments/statistical_analysis.py`.

| Metric | Baseline (0.85/0.50) | PPO (A/B/C identical) |
|---|:---:|:---:|
| Routing accuracy (over 177)   | 66.7% [59.4%, 73.2%] | 63.3% [56.0%, 70.0%] |
| Auto-approval precision       | 72.6% [63.7%, 79.9%] | 77.8% [67.6%, 85.5%] |
| Auto-approval error rate      | 27.4% [20.1%, 36.3%] | 22.2% [14.5%, 32.4%] |
| **Auto-approval rate**        | **63.8% [56.5%, 70.6%]** | **45.8% [38.6%, 53.1%]** |

For reference, approving everything would route 62.1% of items correctly,
because that is how often Haiku agreed with the labels.

### Comparing the two policies

The item-level z-tests in `statistical_summary.md` ignore the pairing and the
scenario structure (see Statistical methods). A re-analysis of the committed
`held_out_set.json` and `evaluation_results.json` that keeps the pairing and
counts each scenario once (exact sign-flip test over scenarios; scenario
bootstrap with 10,000 resamples) gives:

| Comparison (baseline minus PPO) | Difference | Scenario-level 95% interval | Scenario-level p |
|---|:---:|:---:|:---:|
| Auto-approval rate | +18.1 points | +4.0 to +32.8 | 0.032 (0.096 after Holm over three tests) |
| Routing accuracy | +3.4 points | -12.3 to +20.1 | 0.760 |
| Auto-approval precision | -5.2 points | -19.5 to +7.7 | about 0.43 |

The direction of the auto-approval gap is clear; its size is uncertain, and it
does not survive a correction for the three headline comparisons. The policies
disagree on 50 items: the baseline alone auto-approves 41 (31 medium-tier items
from 10 scenarios, and 10 hard-tier items from one scenario that Haiku always
gets right), and PPO alone auto-approves 9 easy-tier items. Routing accuracy and
precision do not differ. This re-analysis was run outside the repository's
pipeline, so `make reproduce-fast` does not regenerate this table.

### Per-tier auto-approval error rate (auto-approved subset only)

| Tier | Baseline | PPO |
|---|:---:|:---:|
| Easy   | 19.4% [12.0%, 30.0%] (n=72) | 22.2% [14.5%, 32.4%] (n=81) |
| Medium | **54.8% [37.8%, 70.8%]** (n=31) | N/A: PPO auto-approves zero |
| Hard   | 0.0% [0.0%, 27.8%] (n=10, one scenario) | N/A: PPO auto-approves zero |

The medium tier is where the policies differ most. The baseline auto-approves
31 medium-tier items, 17 of which disagree with the reference labels; PPO
auto-approves none, because it reviews the whole tier. Inside that tier
confidence carries no signal: medium items above 0.85 disagreed with their
labels 17 times in 31, and those at 0.85 or below 16 times in 32 (Fisher exact
p = 0.80). The gap therefore reflects the tier label, not a better reading of
confidence. The 17 come from six scenarios, several with arguable or
chart-inconsistent labels.

Reproducible Wilson intervals and item-level p-values:
`python -m experiments.statistical_analysis`.

### Calibration probe

To test whether the variant convergence in Key Finding 1 is explained by
uncalibrated confidence, we re-trained all three PPO variants on a recalibrated
confidence signal: a scikit-learn logistic regression of correctness on
`[raw_confidence, amount_normalised, tier_onehot]`, with 5-fold stratified CV
for training-set scores and a full-training-set fit for eval scores. This is a
logistic recalibration with tier and amount as extra inputs, not Platt scaling,
which maps the confidence alone. It improves proper scoring rules on the eval
set and produces a continuous-looking signal rather than the ~7-value clusters
of raw Claude confidence.

| Signal | Brier (↓) | Log-loss (↓) | Unique values on eval (easy / medium / hard) |
|---|:---:|:---:|:---:|
| Raw Claude Haiku confidence | 0.295 | 0.910 | 7 / 7 / 4 |
| Logistic recalibration (this probe) | **0.216** | **0.616** | 74 / 60 / 32 |

Most of the gain comes from the tier. Predicting each tier's training accuracy
alone gives a Brier score of 0.219, and recalibrating the confidence alone about
0.22. The eval items are variants of training scenarios, so all of these are
in-scenario figures; on scenarios the recalibrator has not seen
(leave-one-scenario-out) it does worse than a constant prediction.

All three PPO variants again make the same decisions on the recalibrated eval
set: the same 81 auto-approvals / 96 surfacings / 0 rejects as under raw
confidence, the same 63.3% routing accuracy, and the same 77.8% auto-approval
precision. That is expected, since the policy follows the tier and the
recalibrated score largely encodes the tier.

The hand-tuned baseline, applied to calibrated confidences without retuning
its 0.85/0.50 thresholds, drops to 42.4% routing accuracy and 8.5%
auto-approval rate (it auto-approves only 15 transactions, all correct, and
now emits 18 rejects). This is a mis-tuned-baseline artefact and not a fair
head-to-head with PPO on the calibrated regime; the meaningful comparison here
is PPO-A vs PPO-B vs PPO-C, which are equal.

A working hypothesis for why the variants remain equal under calibration: at
the observed per-tier agent accuracies (easy 82.6% / medium 52.4% / hard 62.6%
on the training set), the EV-optimal tier-level action is the same for all
three reward variants: easy-tier expected reward from AUTO_APPROVE is
positive even under Variant C's 5:1 loss ratio, and medium- and hard-tier
expected reward from AUTO_APPROVE is negative even under Variant A's 2:1
ratio. A data regime with per-tier accuracies in a narrower band
(e.g. easy-tier accuracy ~0.75, medium-tier ~0.75) would be expected to
change this calculus; the regime probe below tests that.

Reproduce the probe: `python -m experiments.calibrate` followed by
`python -m agent.train --reward {A,B,C} --dataset calibrated` and
`python -m agent.evaluate --dataset calibrated`. Full diagnostics in
`experiments/results/calibration_report.json` and
`experiments/results/statistical_summary_calibrated.md`.

### Regime probe (EV-invariance test)

The calibration probe's candidate explanation (Key Finding 4) predicts that
all three PPO variants converge because the observed per-tier accuracies
(easy 82.6% / medium 52.4% / hard 62.6%) place every tier outside the A-vs-C
divergence band (0.64, 0.80) derived from the reward tables. We test this by
removing correct easy-tier items from both the training and evaluation sets
until easy-tier accuracy is 72% (medium and hard kept unchanged), then
retraining all three PPO variants on the reshaped training set (72% easy / 52%
medium / 63% hard, seed=42, 100k steps). We run the probe on **both** the
recalibrated and raw confidence signals as a robustness check.

**EV break-even thresholds** (accuracy above which AUTO_APPROVE beats
SURFACE_FOR_REVIEW for each variant; B evaluated at E[accountant_load]=0.5):

| Variant | Break-even p |
|---|:---:|
| A (AUTO penalty −2) | 0.639 |
| B (E[load]=0.5)     | 0.613 |
| C (AUTO penalty −5) | 0.803 |

At p=0.72, A and B should prefer AUTO_APPROVE on easy tier; C should prefer
SURFACE_FOR_REVIEW. All three should surface medium and hard.

On the natural data the margins are thin. Easy-tier training accuracy (82.6%)
is 2.3 points above C's break-even, and hard-tier accuracy (62.6%) is 1.3
points below A's, a figure that rests largely on seed t041, whose label the
categoriser could not express. On the held-out data easy-tier accuracy is
77.8%, below C's break-even, so PPO-C scores worse there by its own reward than
sending everything for review (-0.149 against -0.073 per item).

**Result: Variant C separates from Variants A and B as the break-even
arithmetic predicts, on both signals.** Per-variant action totals (AUTO /
SURFACE / REJECT) on the reshaped eval set (n=160):

| Variant | Calibrated regime | Raw regime |
|---|:---:|:---:|
| PPO-A | 64 / 96 / 0 | 64 / 96 / 0 |
| PPO-B | 64 / 96 / 0 | 64 / 96 / 0 |
| **PPO-C** | **0 / 160 / 0** | **0 / 160 / 0** |

PPO-A and PPO-B each auto-approve all 64 easy-tier transactions (and zero
others); PPO-C auto-approves zero and surfaces all 160. The split is the same
on raw and recalibrated confidence, so recalibration is not what drives it.

Variants A and B remain indistinguishable at this sample size because their
break-even thresholds differ by only 0.026; A-vs-B divergence would require a
tier with accuracy precisely in (0.613, 0.639), narrower than the sampling
noise allows at n≈64 per tier.

The probe is consistent with the break-even explanation in Key Finding 4: on
the natural data all three reward tables imply the same tier-level actions and
the policies agree; on the constructed data they disagree and the policies
split. Because the probe builds the data so that tier-level arithmetic decides
the answer, it mainly checks that PPO finds that optimum, and two regimes cannot
show that the policies differ only when the tables disagree. Full
reproduction: `python -m experiments.regime_probe --source {calibrated,raw}`
then the corresponding train/evaluate/stats commands with
`--dataset {regime,regime_raw}`.

### Multi-seed robustness

To test whether the headline policy is sensitive to random-seed
initialisation (the standard reviewer hygiene question), we retrained
every (variant, regime) combination with five additional seeds (0 to 4,
alongside the canonical seed=42 used elsewhere in this README). All other
hyperparameters, the training dataset, and the held-out evaluation set
were held constant: 30 retrainings in total (3 variants × 5 seeds × 2 regimes).

| Regime | PPO-A | PPO-B | PPO-C | Distinct action totals across 5 seeds |
|---|:---:|:---:|:---:|---|
| Raw    | 63.3% (0.00pp) | 63.3% (0.00pp) | 63.3% (0.00pp) | 1: (81 / 96 / 0) for all three variants |
| Regime | 59.4% (0.00pp) | 59.4% (0.00pp) | 41.9% (0.00pp) | A/B: 1 of (64 / 96 / 0); C: 1 of (0 / 160 / 0) |

Rows show routing accuracy (mean across 5 seeds, with population standard
deviation in percentage points). All other headline metrics (auto-approval
precision, rate, and error rate) are identically invariant across seeds.
See [`experiments/results/multi_seed_summary_raw.md`](experiments/results/multi_seed_summary_raw.md)
and [`multi_seed_summary_regime.md`](experiments/results/multi_seed_summary_regime.md)
for per-seed numbers.

**Result: the tier-level policy does not depend on the training seed.**
Every one of the 30 retrainings produced the same action totals and metrics as
its canonical (seed=42) counterpart. Per-item actions were saved only for the
seed=42 models, so item-level agreement for the others is inferred from those
totals. Standard deviations are 0.00 percentage points across every headline
metric, on every regime, for every variant. The seeds vary only PPO's
initialisation and sampling while the data and the split stay fixed, so this
shows that the optimiser reliably finds the tier rule. It
says nothing about sampling uncertainty, which the scenario-level comparison
under Results addresses.

Reproduce via `make multi-seed` (~5 hours on a consumer CPU) or
`python -m experiments.multi_seed --datasets raw regime --variants A B C
--seeds 0 1 2 3 4`.

---

## Key Findings

1. **All three reward variants learned the same tier rule.** With real Claude
   Haiku confidence scores, Variants A, B and C all learned to auto-approve
   every easy-tier transaction and send every medium-tier and hard-tier one for
   review, whatever the confidence. The difficulty tier is a label from the seed
   fixtures that the thresholds do not see, so the learned policy and the
   baseline did not have the same information. The rule survives recalibration
   of the confidence (Calibration probe) and 30 retrainings (Multi-seed
   robustness), and the break-even analysis explains why the three reward
   tables agree (Key Finding 4).

2. **Real confidence scores moved the fixed thresholds to a very different
   operating point.** With mock scores (an earlier run on different data) the
   baseline auto-approved 37.3% at 10.6% error. With real scores the same
   thresholds auto-approve 63.8% at 27.4% error, because Claude Haiku gives 0.85
   or more to about four in five transactions (723 of 882), including many
   medium-tier predictions that disagree with their labels. Haiku's confidence
   took only 14 distinct values, averaged 0.90 on held-out items that agreed
   with their labels 62% of the time, and separated right from wrong answers
   weakly (AUROC 0.66 overall, 0.48 within the held-out medium tier). Part of
   that gap is label noise: without the three seeds whose labels the
   categoriser could not express, mean confidence across the remaining 799
   rows is 0.91 against 74% agreement. This is consistent with broader
   evidence that verbally elicited LLM confidence scores are overconfident and
   poorly calibrated [6, 7, 8].

3. **The learned policy auto-approves less, at similar routing accuracy.** It
   auto-approved 45.8% of held-out items against the baseline's 63.8%. Counting
   each scenario once, the 18-point gap has a 95% interval of 4.0 to 32.8 points
   and an exact p of 0.032, which becomes 0.096 after a Holm correction for
   three comparisons: the direction is clear and the size uncertain. Routing
   accuracy (63.3% against 66.7%) and auto-approval precision (77.8% against
   72.6%) do not differ. The gap sits in the medium tier, where the baseline
   auto-approves 31 items and 17 of them disagree with the reference labels,
   while PPO reviews the whole tier. PPO made fewer wrong auto-approvals (18
   against 31) but sent more correct predictions to review (47 against 28), and
   neither count difference holds up at the scenario level. What the policy
   found is that the tier label predicts Haiku's mistakes better than Haiku's
   own confidence does.

4. **The reward variants agree because of where the tier accuracies fall,
   not because of calibration.** A logistic recalibration (confidence, amount
   and tier) improves the held-out Brier score from 0.295 to 0.216, mostly by
   learning each tier's base rate, and leaves the shared A/B/C policy
   unchanged (see Calibration probe). The explanation lies in the per-tier
   accuracies (82.6% / 52.4% / 62.6% on the training set): each tier falls on
   the same side of all three variants' break-even points (A 0.639, B 0.613,
   C 0.803), so the expected-reward-optimal tier-level action is the same for
   all three. Easy-tier AUTO_APPROVE stays positive-EV even under Variant C's
   5:1 loss ratio, and medium-tier and hard-tier AUTO_APPROVE stay negative-EV
   even under Variant A's 2:1 ratio. The constructed regime probe, which puts
   easy-tier accuracy at 72%, splits C from A and B on raw as well as
   recalibrated confidence, so calibration is not needed for the variants to
   differ. The probe is consistent with this explanation, though the
   natural-data margins are thin (Regime probe).

5. **No policy chose REJECT_FOR_MANUAL, and the reward tables explain why.**
   Rejecting beats sending for review only when the chance that the prediction
   is right is below 0.22 under Variants A and C (at most 0.33 under B), and no
   tier or confidence band in this data is below 0.47. Never rejecting is
   therefore the expected-reward-optimal choice here, built into the tables
   rather than discovered. The trained policies give REJECT a small probability
   (at most 0.06 on held-out inputs) but never choose it. The baseline also
   emits no rejects on the held-out set, because Haiku's confidence never falls
   below its 0.5 threshold (the lowest held-out value is 0.55). How either
   policy would behave on lower-confidence data is untested; since the learned
   policy does not respond to confidence, lower scores alone would not make it
   reject.

---

## Evaluation Scope

**Same inputs.** The learned policy saw a difficulty label that the baseline
did not, and that a live transaction does not carry. A fair comparison gives
every policy the same inputs and includes the textbook rule for known costs: a
threshold on an estimated probability that the prediction is right, set at the
cost break-even. This study does neither.

**Why reinforcement learning?** An action never changes which transaction comes
next, and the environment reveals whether every prediction was right, including
auto-approved ones. With that full feedback the task is a contextual bandit, and
a supervised estimate of that probability plus the break-even rule would do the
same job. A reinforcement learning or bandit formulation earns its place under
partial feedback, where only reviewed items reveal whether the prediction was
right.

**Confidence signal.** Real Claude Haiku confidence scores cluster at round
values (0.95, 0.85, 0.75) regardless of difficulty tier [6, 7, 8]. Logistic
recalibration with tier and amount improves proper scoring rules without
changing the learned tier-based policy (Calibration probe, above), and the
regime probe ties that policy to the per-tier accuracy structure (easy 82.6% /
medium 52.4% / hard 62.6%).

**Data and labels.** The 882 transactions are synthetic, 244 of the variants
are template copies, two rows carry mock scores, and the labels come from
Claude-written fixtures that no person checked (see Training Data Generation).

**Evaluation set.** 177 held-out variants of 49 scenarios that also appear in
training (33 hard items). Rates are reported with item-level Wilson intervals,
small-n tiers are flagged, and the headline comparison is also given at the
scenario level.

**Reward signal.** The reward comes from each prediction's `is_correct` flag.
In deployment the reward would come from accountant corrections, learned
online, as studied in the learning-to-defer literature [1, 9].

**Next experiments.** Split by scenario rather than by transaction; give every
policy the same inputs, with a calibrated break-even threshold and a
learning-to-defer model [1, 9] as baselines; check the labels by hand and
regenerate the 16 template seeds; report expected cost under each reward table
and risk-coverage curves; use multi-sample self-consistency or top-k
log-probabilities as a richer uncertainty signal than one stated confidence;
learn online from accountant corrections under partial feedback; evaluate
Variant B at several loads; extend to the ReconcilerAgent (which also uses
fixed thresholds).

---

## Implications for Agentic AI Systems

Three cautious lessons follow from this study.

**A learned router will use whatever input best separates easy cases from hard
ones.** Here that was a difficulty label, which a live transaction does not
have. Before crediting a learned policy over a threshold, check that both saw
the same inputs.

**One stated confidence score may be too coarse to gate automation.** Haiku's
score took 14 values and barely separated right from wrong answers, so no
threshold on it reproduces the learned operating point: a threshold of 0.90
would still auto-approve 112 held-out items, at 72.3% precision. The
difficulty tier carried more of the signal than the confidence did.

**A new reward table changes the policy only when it changes the
expected-reward-optimal action.** The three tables gave the same policy on the
natural data and different policies on the constructed regime, as the
break-even arithmetic predicts. Per-tier accuracies tell a designer in advance
whether changing the table is likely to change the learned policy.

The evidence comes from one small synthetic dataset. The code is a
reproducible starting point for a fairer comparison of learned and hand-tuned
routing [1, 10], not evidence that learned routing is better.

---

## Repository Structure

```
.
├── environment/            Gymnasium routing environment and reward variants
├── agent/                  PPO training, evaluation, and baseline policy
├── data/                   Training data and held-out eval set
├── experiments/            Results, figures, and analysis notebook
├── models/trained/         Committed PPO model files (~140KB each)
├── integration/            LearnedRouter drop-in and integration guide
├── api/                    FastAPI routing service
└── .github/workflows/      CI (test+lint+type-check) and manual retrain
```

See full annotated tree in [PROGRESS.md](PROGRESS.md).

---

## Quickstart

### One-command reproduction (Docker)

No local Python install needed. Regenerates the evaluation results and
item-level statistics in this README and in `paper/one_pager.md` from the
committed PPO models and datasets, for all four regimes (raw, calibrated,
regime, regime_raw). The scenario-level comparison under Results comes from a
separate re-analysis and is not regenerated here.

```bash
git clone https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing.git
cd AI-Accountant-with-Reinforcement-Learning-Routing
docker build -f Dockerfile.reproduce -t rl-routing-reproduce .
docker run --rm rl-routing-reproduce
```

Build takes ~5 minutes (dominated by the PyTorch dependency); `reproduce-fast`
runs in ~1 minute. To also retrain every variant from scratch (~4 hours):

```bash
docker run --rm rl-routing-reproduce make reproduce-full
```

To extract the regenerated `experiments/results/` files back to the host:

```bash
docker run --rm -v "$PWD/experiments/results":/app/experiments/results \
    rl-routing-reproduce
```

### Local Python install

```bash
git clone https://github.com/rajo69/AI-Accountant-with-Reinforcement-Learning-Routing.git
cd AI-Accountant-with-Reinforcement-Learning-Routing
pip install -r requirements.txt

python -m agent.evaluate          # evaluate pre-trained models on held-out set
python -m experiments.statistical_analysis   # regenerate CIs and p-values
uvicorn api.main:app --reload     # serve the routing API at localhost:8000/docs
```

To rerun the evaluation for all four regimes and confirm that every
regenerated file matches the committed one (no API key, about a minute on a
laptop CPU; `make reproduce-fast` runs the same steps where GNU Make is
installed):

```bash
python -m experiments.reproduce --check
```

To retrain from scratch (this replaces the committed models; add
`--suffix _try1` to write new files beside them instead):

```bash
python -m agent.train --reward A  # approximately 10 min on CPU
python -m agent.train --reward B
python -m agent.train --reward C
# or: make reproduce-full  (all variants, all regimes)
```

[REPRODUCING.md](REPRODUCING.md) covers setting up from an empty folder
(Windows included), trying changes without overwriting the committed results,
which command replaces which file, and regenerating the dataset.

---

## Running the API

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"confidence_score": 0.88, "transaction_features": {"amount": 450.0, "difficulty_tier": 1}}'
```

The learned policies decide on `difficulty_tier` (0 easy, 1 medium, 2 hard).
If it is left out, the API assumes 1, and every PPO variant then returns
`suggested` (review) whatever the confidence. A live transaction has no tier,
so the service demonstrates the integration mechanics; it is not a deployable
router until the tier is replaced by a real difficulty estimate.

See [`integration/INTEGRATION_GUIDE.md`](integration/INTEGRATION_GUIDE.md) for
instructions on swapping the learned router into the
[parent project](https://github.com/rajo69/agentic-ai-accounting).

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

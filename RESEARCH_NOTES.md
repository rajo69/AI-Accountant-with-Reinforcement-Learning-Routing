# Research Notes

A running log of design decisions, things tried that did not work, and
observations during development. Written for PhD interview preparation —
these are the choices you would be asked to defend.

---

## Reward Function Design Decisions

### Why asymmetric penalties for AUTO_APPROVE?

The 2:1 ratio in Variant A (correct=+1.0, wrong=−2.0) was derived from the
existing confidence threshold: the parent project auto-approves above 0.85,
which implicitly assumes the agent is right at least ~67% of the time at
that threshold for auto-approval to be net-positive. A 2:1 loss-to-gain ratio
produces exactly that break-even at 67% confidence, aligning the RL incentive
with the threshold logic's implicit assumption.

Variant C's −5.0 was chosen to set the break-even at ~83% (P(correct) > 5/6),
which is close to the 0.85 hand-tuned threshold. The expectation was that the
Variant C policy would learn an effective threshold near 0.85. On the raw-
confidence and Platt-calibrated eval sets PPO-C's action distribution is
identical to PPO-A and PPO-B (81/96/0), because the observed per-tier
accuracies (easy 82.6% / medium 52.4% / hard 62.6%) happen to place every
tier outside C's divergence band, so C's EV-optimal tier-level action
coincides with A's and B's. The regime probe (Results → Regime probe in
the README) verifies this by reshaping easy-tier accuracy to 0.72, where
C's EV-optimal action flips to SURFACE while A/B still prefer AUTO —
producing the expected divergence.

### Why is SURFACE_FOR_REVIEW penalised symmetrically (+0.3 / −0.3)?

Early reward designs penalised unnecessary escalation less than correct
escalation was rewarded. This produced policies that escalated everything —
a "safe default" that technically maximised reward while making the system
useless. Symmetric penalties (+0.3 / −0.3) force the agent to discriminate:
escalation is only net-positive if the agent would have been wrong otherwise.

### Why does REJECT get worse treatment than SURFACE?

REJECT_FOR_MANUAL was given −1.0 for incorrect rejection (vs −0.3 for
SURFACE unnecessary escalation) because REJECT triggers a full manual workflow
(re-categorisation from scratch) while SURFACE only requires the accountant to
review a pre-formed suggestion. The cost differential reflects real workflow
cost. The +0.5 for correct rejection (vs +0.3 for correct escalation) rewards
the harder, more decisive call appropriately.

---

## Environment Design Decisions

### Why episode = full dataset pass (not single transactions)?

Early testing used single-transaction episodes (episode length = 1). This
produced unstable training because the discount factor γ never mattered (no
multi-step return). Full-dataset episodes (length ~705) allow the policy to
learn from return propagation and give the Monitor wrapper meaningful episode
statistics for TensorBoard.

### Why include difficulty_tier as a feature when it's not available in production?

In a real deployment, `difficulty_tier` is not directly observable — it comes
from the eval framework, not from transaction features. It was included because
(a) it is a reasonable proxy for inherent transaction ambiguity and (b) it
provides a discrete signal that the 4-dim state space can actually learn from,
given the mock confidence scores. In a production re-run with real confidence
scores, difficulty_tier could be dropped or replaced with a learned ambiguity
score derived from the CategoriserAgent's top-k probabilities.

### Why category_entropy = 0.0 for all transactions?

The Claude API returns a single category prediction, not a distribution over
categories. To get true entropy, one would need to either (a) run the prompt
multiple times with temperature > 0 or (b) use a classifier that natively
outputs a probability distribution. Neither was done for cost reasons.
The feature is retained in the state space as a placeholder — future work
with multi-sample prompting could populate it.

---

## Training Observations

### Variant A convergence (mock-score run)

Variant A converged very quickly — by ~20k timesteps the policy had essentially
reached its maximum eval reward (291.1) and stayed there. The action distribution
plot shows it learned a pure tier-based strategy: auto-approve easy+medium,
surface hard. The training curve is essentially flat after 20k steps, suggesting
100k was 5x more than needed for this dataset size. This is worth noting in
interviews — it suggests the environment is too simple for the model capacity.

### Variant B identical to Variant A on eval set (mock-score run)

The workload-sensitive penalty was a deliberate design choice to make the agent
learn load-responsive behaviour. The identical eval results suggest either:
(a) 100k steps was insufficient to learn load-sensitive routing that deviates
    from Variant A's default, or
(b) at neutral load (0.5, used in evaluation), the load-adjusted penalty
    (-0.3 × 1.5 = -0.45) is not different enough from Variant A's (-0.3)
    to change routing decisions on a dataset where routing is primarily
    determined by difficulty_tier.

To distinguish (a) from (b), one would need to evaluate Variant B at load=0.0
and load=1.0 separately. This is future work.

### Variant C reward scale mismatch (mock-score run)

Variant C's final training mean reward (67.3) vs Variant A (291.1) is striking.
This is expected: by never auto-approving medium transactions (45.8% auto-rate
vs 81.4% for A), Variant C foregoes many +1.0 rewards. The training objective
and the evaluation objective are different — Variant C is "doing well" at its
own objective (minimising false auto-approvals) even though its cumulative
episode reward is lower. This is a common confusion in RL: a lower reward does
not mean a worse policy; it means the agent is optimising a different target.

---

## Real Confidence Score Run Observations (2026-03-26)

The simulator was re-run with `dry_run=False` and a real `ANTHROPIC_API_KEY`.
This section documents the new findings.

### Claude Haiku outputs round-number confidence scores

The real API call (via Instructor → `CategoryPrediction(confidence: float)`)
consistently produced values like 0.95, 0.85, 0.75, 0.65 rather than continuous
values. Of 705 training transactions:
- Easy tier: 204/322 scored 0.95 exactly; top-8 distinct values cover 99% of tier
- Medium tier: 131/252 scored 0.95; clearly high-confidence even for 48%-accuracy tier
- Hard tier: 65/131 scored 0.75; still quite high for 52%-accuracy tier

Average confidence by tier: easy=0.938, medium=0.883, hard=0.820. The separation
(0.118 between easy and hard) is much smaller than mock scores (0.90 - 0.38 = 0.52).
This is the fundamental difference between runs.

**Why Haiku outputs round numbers:** Haiku is an instruction-following model
generating a confidence float as text. It tends to round to the nearest 5% or 10%.
This is a well-known LLM calibration failure — the model doesn't natively output
a probability from a softmax; it generates a number based on its training distribution
for what "confidence" numbers look like.

### All three reward variants converge to the same policy

With real scores, Variants A, B, and C all trained to: auto-approve all easy (81/81),
surface all medium and hard. Training rewards diverged (A=140.5, B=110.9, C=−27.5)
but eval behaviour was identical. The Variant C reward of −27.5 means the policy
surfaces most correct easy-tier transactions (−0.3 each), choosing conservatism even
for the cases it would have auto-approved profitably. However, the eval showed it still
auto-approves 81 easy transactions — so the negative training reward reflects surfacing
easy transactions during the training episodes, while the eval (deterministic, fixed
seed) converges to the same action.

The convergence across variants on raw and calibrated confidence is not evidence of a
single dominant strategy in general — it is evidence that the observed per-tier
accuracies (82.6% / 52.4% / 62.6%) place every tier outside the A-vs-C EV-divergence
band (0.64, 0.80). The subsequent regime probe (see Results → Regime probe in the
README) confirms this: reshaping easy-tier accuracy to 0.72 (inside the divergence
band) produces the expected A/B vs C variant divergence exactly as the EV break-even
math predicts, with Variant C refusing all auto-approvals while A and B auto-approve
the entire easy tier. The prior mock-score run's apparent "Variant C at 45.8% auto-
rate, 9.9% error" distinct-point result is now understood as an artefact of the mock
score distribution and not load-bearing.

### Baseline degradation with real scores

The hand-tuned 0.85 threshold was calibrated for the mock score distribution (easy≈0.90,
medium≈0.65, hard≈0.38) — the developer's intuition was essentially correct about what
CategoriserAgent outputs would look like. With real scores, nearly everything exceeds
0.85, so the threshold auto-approves 63.8% of transactions including medium ones with
54.8% error rates. The baseline's confidence threshold is miscalibrated for real Claude
output.

This is the most practically important finding: the reason the hand-tuned threshold
existed in the first place was precisely because nobody knew what real confidence
distributions would look like. The 0.85 value was a guess. With real data, it turns
out the model is overconfident, and the threshold needs to be much higher (~0.95) or
replaced with a non-confidence-based gate (like difficulty_tier) to avoid the
auto-approval volume the PPO agents correctly reject.

---

## Data Generation Decisions

### Why commit the confidence scores rather than regenerate them in CI?

Generating confidence scores requires running 882 transactions through
the CategoriserAgent, which calls the Anthropic API at ~$0.01–0.05 per
transaction. Total cost: ~$9–44. This is acceptable for a one-time research
run but not for a public repository where CI would re-run it on every push.

The committed data files (`data/synthetic/transactions.jsonl` and
`data/evaluation/held_out_set.json`) use **real** Claude Haiku confidence scores
from a one-time API run (2026-03-26). These are committed to the repo so that
results are reproducible without re-running the API. CI does not regenerate
them; the simulator is run manually with `ANTHROPIC_API_KEY` when the seed
fixture or model changes.

The real scores turned out to cluster at round values (0.95, 0.85, 0.75)
rather than varying continuously. This is documented in the Real Confidence
Score Run Observations section above and in Limitations. The practical
consequence — policies still learn tier-based rather than confidence-based
routing — is the same as with mock scores, but the mechanism is different:
mock scores had no intra-tier variance by construction; real scores have low
intra-tier variance because Haiku is poorly calibrated.

### Why 80/20 train/eval split?

The parent project's eval framework uses 50 labelled transactions; 20% of ~882
gives 177 held-out transactions (81 easy, 63 medium, 33 hard). The split is
stratified by difficulty tier to ensure the eval set is representative.
Hard tier is under-powered (n=33), which is why hard-tier error rates should
be treated with caution.

---

## Integration Design Decisions

### Why PPO-C as the default in LearnedRouter?

On the original mock-score run Phase 3 evaluation PPO-C showed the best precision
and lowest error rate and was selected as the production default. On the subsequent
real-confidence and Platt-calibrated evaluations PPO-A, PPO-B, and PPO-C produce
identical action sequences, so the choice of default among the three is immaterial
on the natural data regime. PPO-C remains the integration default because it is
the most conservative under data regimes that fall into its divergence band (as the
regime probe demonstrates — there C auto-approves nothing while A/B auto-approve
the easy tier). For an accounting deployment where false auto-approvals carry
outsized cost, the more conservative default is the defensible choice, even when
it is empirically equal on the natural-regime eval.

### Why a module-level singleton for LearnedRouter in the API?

SB3's `model.predict()` is a read-only operation (no state mutation). A single
LearnedRouter instance can safely serve all concurrent requests without locking.
This avoids the overhead of loading a 140KB zip file on each request. The
`POST /policy/reload` endpoint provides hot-swap capability without restart.

### Why not store routing decisions in a dedicated table?

The parent project already has an `AuditLog` table with a `ai_decision_data`
JSONB column. Storing routing decisions there (merged with the LLM reasoning
text) keeps all AI decision audit data in one place, makes the integration
minimal, and avoids schema changes to the parent project's database.

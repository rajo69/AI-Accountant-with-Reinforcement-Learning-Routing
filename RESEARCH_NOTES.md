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
Variant C policy would learn an effective threshold near 0.85. The result
(PPO-C auto-approves all easy tier at ~90% confidence and nothing else) was
consistent with this expectation.

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

### Variant A convergence

Variant A converged very quickly — by ~20k timesteps the policy had essentially
reached its maximum eval reward (291.1) and stayed there. The action distribution
plot shows it learned a pure tier-based strategy: auto-approve easy+medium,
surface hard. The training curve is essentially flat after 20k steps, suggesting
100k was 5x more than needed for this dataset size. This is worth noting in
interviews — it suggests the environment is too simple for the model capacity.

### Variant B identical to Variant A on eval set

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

### Variant C reward scale mismatch

Variant C's final training mean reward (67.3) vs Variant A (291.1) is striking.
This is expected: by never auto-approving medium transactions (45.8% auto-rate
vs 81.4% for A), Variant C foregoes many +1.0 rewards. The training objective
and the evaluation objective are different — Variant C is "doing well" at its
own objective (minimising false auto-approvals) even though its cumulative
episode reward is lower. This is a common confusion in RL: a lower reward does
not mean a worse policy; it means the agent is optimising a different target.

---

## Data Generation Decisions

### Why use mock confidence scores in the repository?

Generating real confidence scores requires running 882 transactions through
the CategoriserAgent, which calls the Anthropic API at ~$0.01–0.05 per
transaction. Total cost: ~$9–44. This is acceptable for a research project
but not for a public repository where CI would re-run it on every push.
The mock scores (easy=0.91, medium=0.66, hard=0.38) are the mean values
from the dry-run output and are clearly documented as such in the data card.

The practical consequence — policies learning tier-based rather than
confidence-based routing — is documented explicitly in Limitations and
is arguably the most important finding: it shows that data quality is the
binding constraint on this approach, not the RL method itself.

### Why 80/20 train/eval split?

The parent project's eval framework uses 50 labelled transactions; 20% of ~882
gives 177 held-out transactions (81 easy, 63 medium, 33 hard). The split is
stratified by difficulty tier to ensure the eval set is representative.
Hard tier is under-powered (n=33), which is why hard-tier error rates should
be treated with caution.

---

## Integration Design Decisions

### Why PPO-C as the default in LearnedRouter?

Phase 3 evaluation showed PPO-C achieves the best precision (90.1%) and lowest
error rate (9.9%) — the metrics that matter most in an accounting context where
incorrect auto-approvals are the most damaging errors. PPO-A/B's higher routing
accuracy (75.1%) comes at the cost of auto-approving medium transactions with a
30% error rate, which is too high for financial data. PPO-C's profile (46%
auto-rate, 90% precision) is strictly safer for production.

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

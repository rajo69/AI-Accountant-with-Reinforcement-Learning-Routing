"""
Reward functions for the RL routing agent.

This module is the intellectual core of the project. Every numerical value here
represents a deliberate, documented design decision. The asymmetries are not
arbitrary — they reflect the real-world cost structure of an accounting pipeline
where different types of routing errors have very different consequences.

Background
----------
The CategoriserAgent outputs a confidence score (0-1) and a predicted category.
The routing policy must decide what to do with this prediction:

    Action 0 — AUTO_APPROVE:        Accept the prediction, write it to the ledger.
    Action 1 — SURFACE_FOR_REVIEW:  Send to the accountant queue. A human decides.
    Action 2 — REJECT_FOR_MANUAL:   Flag for manual re-categorisation from scratch.

"Correct" means the CategoriserAgent's prediction matched the true category.
An action is "correct" when it leads to the best outcome for that prediction quality.

Specifically:
    - AUTO_APPROVE is "correct" when the prediction was right (is_correct=True)
    - SURFACE_FOR_REVIEW is "correct" when the prediction was wrong (is_correct=False)
      — i.e., escalation was warranted because the agent would have made a mistake
    - REJECT is "correct" when the prediction was wrong (is_correct=False)

Why Three Variants?
-------------------
A single reward function embeds a single set of priorities. Different deployment
environments have different priorities:

    Variant A: A typical firm where errors are costly but not catastrophic.
    Variant B: A high-volume firm where accountant capacity is a bottleneck.
    Variant C: A compliance-critical environment (regulated industries, HMRC audits)
               where a single incorrect auto-approval could trigger a full investigation.

Training the same PPO policy under each variant and comparing results shows which
reward design matters most and whether the learned behaviour differs meaningfully.
This multi-variant approach is the primary research contribution.
"""

from __future__ import annotations

from typing import Callable

RewardFn = Callable[..., float]

# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------
AUTO_APPROVE = 0        # Accept agent prediction without human review
SURFACE_FOR_REVIEW = 1  # Route to accountant queue for human decision
REJECT_FOR_MANUAL = 2   # Flag for full manual re-categorisation

ACTION_NAMES = {
    AUTO_APPROVE: "AUTO_APPROVE",
    SURFACE_FOR_REVIEW: "SURFACE_FOR_REVIEW",
    REJECT_FOR_MANUAL: "REJECT_FOR_MANUAL",
}


# ---------------------------------------------------------------------------
# Variant A — Binary reward
# ---------------------------------------------------------------------------

def reward_a(action: int, is_correct: bool, **kwargs: float) -> float:
    """
    Variant A: Binary asymmetric reward.

    Design rationale for each value:
    ─────────────────────────────────

    AUTO_APPROVE correct (+1.0):
        The ideal outcome. Agent was confident AND right. No human time was spent
        and the ledger entry is accurate. Maximum reward.

    AUTO_APPROVE incorrect (-2.0):
        The worst outcome. The agent was wrong but we wrote it to the ledger
        without human review. This creates a silent error that may not be caught
        until the quarter-end audit, requiring correction entries and wasted
        accountant time. The penalty is DOUBLE the reward to create a strong
        asymmetry — false confidence is far more damaging than over-escalation.
        Asymmetry ratio: 2:1 (mirrors the >0.85 auto-approval threshold, which
        itself implies the agent must be right ~85%+ of the time to break even).

    SURFACE_FOR_REVIEW correct (+0.3):
        Escalation was warranted (agent would have been wrong), but it still
        costs accountant time and adds latency. Positive but modest — we want
        the agent to prefer auto-approval when genuinely confident, not use
        escalation as a safe default for everything.

    SURFACE_FOR_REVIEW unnecessary (-0.3):
        Agent was right; escalation was a false alarm that wasted accountant time.
        Penalty matches the correct-escalation reward in magnitude: unnecessary
        escalation is as bad as missing a necessary escalation. This symmetric
        treatment avoids the degenerate policy of escalating everything.

    REJECT correct (+0.5):
        Flagging for manual re-categorisation was right (agent was badly wrong),
        and the harder intervention is worth more reward than mere escalation (+0.3).
        Manual re-categorisation costs the accountant more time than reviewing a
        suggestion, hence the moderate reward — it's the right call but expensive.

    REJECT incorrect (-1.0):
        Agent was right but we threw away its prediction and forced manual work.
        Worse than unnecessary escalation (-0.3) because manual re-categorisation
        is costlier than queue review. This prevents the agent from defaulting to
        REJECT as a conservative "safe" option.

    Args:
        action: Routing action (0=AUTO_APPROVE, 1=SURFACE_FOR_REVIEW, 2=REJECT)
        is_correct: Whether CategoriserAgent's prediction matched true category
        **kwargs: Ignored (allows uniform calling convention across variants)

    Returns:
        Scalar reward.
    """
    if action == AUTO_APPROVE:
        return 1.0 if is_correct else -2.0

    if action == SURFACE_FOR_REVIEW:
        # Correct escalation: agent wrong, escalation warranted
        # Unnecessary escalation: agent right, escalation wasted time
        return 0.3 if not is_correct else -0.3

    # REJECT_FOR_MANUAL
    return 0.5 if not is_correct else -1.0


# ---------------------------------------------------------------------------
# Variant B — Workload-weighted reward
# ---------------------------------------------------------------------------

def reward_b(action: int, is_correct: bool, accountant_load: float = 0.0, **kwargs: float) -> float:
    """
    Variant B: Workload-weighted reward.

    Identical to Variant A except that the penalty for unnecessary escalation
    scales with the current accountant workload (0-1).

    Design rationale:
    ─────────────────

    The core insight: the cost of adding a transaction to an accountant's review
    queue is NOT constant. When the accountant queue is empty, an unnecessary
    escalation is a minor inconvenience (-0.3). When the queue is already full,
    the same unnecessary escalation could push important items past a deadline,
    requiring overtime or delaying client deliverables. The marginal cost of
    queue work is superlinear under load.

    Penalty formula:
        -0.3 * (1 + accountant_load)

        accountant_load=0.0 → -0.30  (same as Variant A, queue is empty)
        accountant_load=0.5 → -0.45  (moderate queue, 50% extra penalty)
        accountant_load=1.0 → -0.60  (full queue, double the base penalty)

    Why this specific formula?
        Linear scaling keeps the penalty bounded and interpretable. An exponential
        penalty (e.g., -0.3 * exp(load)) would likely cause the agent to never
        escalate at high loads, which is dangerous for compliance. The linear form
        maintains a positive expected return for correct escalation at all load levels
        (correct escalation still earns +0.3 regardless of load).

    The accountant_load is injected into the observation state, so the agent can
    LEARN to behave differently when load is high — more conservative about unnecessary
    escalation, favouring auto-approval for borderline cases.

    All other reward values are identical to Variant A.

    Args:
        action: Routing action (0, 1, or 2)
        is_correct: Whether CategoriserAgent's prediction matched true category
        accountant_load: Current accountant queue pressure (0.0=empty, 1.0=full)
        **kwargs: Ignored

    Returns:
        Scalar reward.
    """
    if action == AUTO_APPROVE:
        return 1.0 if is_correct else -2.0

    if action == SURFACE_FOR_REVIEW:
        if not is_correct:
            # Correct escalation: reward is load-independent.
            # We always want to surface genuinely uncertain transactions.
            return 0.3
        else:
            # Unnecessary escalation: penalty scales with accountant load.
            # At full load, this is twice as bad as at zero load.
            return -0.3 * (1.0 + float(accountant_load))

    # REJECT_FOR_MANUAL — load does not affect reject reward/penalty
    # (manual re-categorisation bypasses the accountant queue entirely)
    return 0.5 if not is_correct else -1.0


# ---------------------------------------------------------------------------
# Variant C — Conservative (compliance-critical) reward
# ---------------------------------------------------------------------------

def reward_c(action: int, is_correct: bool, **kwargs: float) -> float:
    """
    Variant C: Conservative reward for high-stakes compliance environments.

    Identical to Variant A except AUTO_APPROVE incorrect is -5.0 instead of -2.0.

    Design rationale:
    ─────────────────

    In a regulated accounting environment (e.g., a firm preparing accounts for an
    HMRC audit, or managing restricted grant funding), a single incorrect
    auto-categorisation can:

        - Trigger a full audit of the client's VAT returns
        - Misstate the P&L, requiring restatement and correction entries
        - Breach covenants in a loan agreement that reference specific cost lines
        - Violate grant reporting requirements, triggering claw-back

    The -5.0 penalty models this catastrophic risk. A rational agent under Variant C
    will be MUCH more conservative about AUTO_APPROVE, preferring SURFACE_FOR_REVIEW
    for any borderline case. We expect the learned policy to raise the effective
    auto-approval threshold significantly above the hand-tuned 0.85 baseline.

    Why -5.0 specifically?
        This creates a 5:1 loss-to-gain ratio for auto-approval decisions.
        Expected value of AUTO_APPROVE is positive only when P(correct) > 5/6 ≈ 0.833.
        Since the hand-tuned threshold of 0.85 was chosen as the auto-approval
        boundary, Variant C should produce a learned threshold near or above 0.85.
        If the RL policy learns a LOWER threshold under Variant C, that would be a
        surprising finding worth investigating (suggesting the confidence score is
        better calibrated than assumed, or that the policy is using other state
        features to make better-than-confidence-alone decisions).

    All other values match Variant A.

    Args:
        action: Routing action (0, 1, or 2)
        is_correct: Whether CategoriserAgent's prediction matched true category
        **kwargs: Ignored

    Returns:
        Scalar reward.
    """
    if action == AUTO_APPROVE:
        # CHANGED from Variant A: -2.0 → -5.0
        # The catastrophic penalty for false auto-approval in compliance contexts.
        return 1.0 if is_correct else -5.0

    if action == SURFACE_FOR_REVIEW:
        return 0.3 if not is_correct else -0.3

    # REJECT_FOR_MANUAL
    return 0.5 if not is_correct else -1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS: dict[str, RewardFn] = {
    "A": reward_a,
    "B": reward_b,
    "C": reward_c,
}


def get_reward_function(variant: str) -> RewardFn:
    """Return the reward function for a given variant string ('A', 'B', or 'C')."""
    if variant not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward variant '{variant}'. Choose from: {list(REWARD_FUNCTIONS)}")
    return REWARD_FUNCTIONS[variant]

"""
integration/tests/test_router.py — Tests for LearnedRouter.

Tests:
    1. test_learned_router_returns_valid_action
    2. test_fallback_to_baseline_when_model_missing
    3. test_router_logs_to_audit_table
    4. test_router_interface_matches_baseline_interface
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.baseline import make_baseline_policy
from integration.router import (
    AUTO_APPROVE,
    SURFACE_FOR_REVIEW,
    REJECT_FOR_MANUAL,
    LearnedRouter,
    RoutingDecision,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Sample transaction features for all tests
SAMPLE_FEATURES = {
    "amount": 450.0,
    "difficulty_tier": 0,   # easy
    "category_entropy": 0.0,
    "accountant_load": 0.5,
}

VALID_STATUSES = {AUTO_APPROVE, SURFACE_FOR_REVIEW, REJECT_FOR_MANUAL}


# ---------------------------------------------------------------------------
# Test 1: learned router returns a valid action
# ---------------------------------------------------------------------------

def test_learned_router_returns_valid_action():
    """
    LearnedRouter.route() must return a RoutingDecision whose status is one
    of the three valid values and whose action_index is in {0, 1, 2}.
    """
    router = LearnedRouter(model_variant="C")

    for confidence in [0.95, 0.72, 0.30]:
        decision = router.route(
            confidence_score=confidence,
            transaction_features=SAMPLE_FEATURES,
        )
        assert isinstance(decision, RoutingDecision), "route() must return RoutingDecision"
        assert decision.status in VALID_STATUSES, (
            f"status '{decision.status}' not in valid set {VALID_STATUSES}"
        )
        assert decision.action_index in (0, 1, 2), (
            f"action_index {decision.action_index} not in {{0, 1, 2}}"
        )
        assert 0.0 <= decision.confidence_score <= 1.0
        assert decision.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Test 2: fallback to baseline when model file missing
# ---------------------------------------------------------------------------

def test_fallback_to_baseline_when_model_missing(tmp_path):
    """
    When the model file does not exist, LearnedRouter must silently activate
    the fallback ThresholdPolicy and set is_using_fallback=True. The routing
    decisions must match what the ThresholdPolicy would produce directly.
    """
    # Point the router at an empty temp directory (no model files)
    router = LearnedRouter(model_variant="A", models_dir=tmp_path)

    assert router.is_using_fallback, (
        "Router must activate fallback when model file is missing"
    )
    assert router._policy_used == "threshold_baseline"

    # Decisions must match the hand-tuned ThresholdPolicy output
    baseline = make_baseline_policy()
    test_confidences = [0.95, 0.70, 0.30]

    for conf in test_confidences:
        decision = router.route(
            confidence_score=conf,
            transaction_features=SAMPLE_FEATURES,
        )
        obs = np.array([conf, 0.5, 0.0, 0.0], dtype=np.float32)
        expected_action, _ = baseline.predict(obs)
        assert decision.action_index == int(expected_action), (
            f"Fallback router returned action {decision.action_index} but "
            f"ThresholdPolicy returned {expected_action} for confidence={conf}"
        )


# ---------------------------------------------------------------------------
# Test 3: router writes structured audit data
# ---------------------------------------------------------------------------

def test_router_logs_to_audit_table():
    """
    RoutingDecision.audit_data must be a non-empty dict containing all fields
    required for storage in AuditLog.ai_decision_data, including:
      - router.policy_used
      - router.action_index
      - router.action_name
      - router.confidence_score
      - router.observation
      - router.latency_ms
      - router.using_fallback
    """
    router = LearnedRouter(model_variant="C")
    decision = router.route(
        confidence_score=0.88,
        transaction_features=SAMPLE_FEATURES,
    )

    audit = decision.audit_data
    assert isinstance(audit, dict), "audit_data must be a dict"
    assert "router" in audit, "audit_data must have a 'router' key"

    r = audit["router"]
    required_fields = [
        "policy_used",
        "action_index",
        "action_name",
        "confidence_score",
        "observation",
        "latency_ms",
        "using_fallback",
    ]
    for field in required_fields:
        assert field in r, f"audit_data['router'] missing required field '{field}'"

    assert isinstance(r["observation"], list), "observation must be serialisable as list"
    assert r["action_index"] in (0, 1, 2)
    assert r["action_name"] in ("AUTO_APPROVE", "SURFACE_FOR_REVIEW", "REJECT_FOR_MANUAL")
    assert isinstance(r["latency_ms"], float)
    assert isinstance(r["using_fallback"], bool)


# ---------------------------------------------------------------------------
# Test 4: LearnedRouter and ThresholdPolicy share the same predict() interface
# ---------------------------------------------------------------------------

def test_router_interface_matches_baseline_interface():
    """
    LearnedRouter.route() and ThresholdPolicy.predict() must both accept a
    confidence score and return a compatible action. The RoutingDecision.status
    field must be a string in VALID_STATUSES for any input.

    This test verifies interface compatibility, not identical outputs —
    the PPO policy is allowed to make different (ideally better) decisions
    than the baseline.
    """
    router = LearnedRouter(model_variant="C")
    baseline = make_baseline_policy()

    test_cases = [
        {"confidence_score": 0.92, "expected_baseline_action": 0},  # > 0.85 → AUTO_APPROVE
        {"confidence_score": 0.65, "expected_baseline_action": 1},  # 0.50–0.85 → SURFACE
        {"confidence_score": 0.30, "expected_baseline_action": 2},  # < 0.50 → REJECT
    ]

    for case in test_cases:
        conf = case["confidence_score"]

        # Baseline interface: predict(obs) → (action, state)
        obs = np.array([conf, 0.35, 0.0, 0.0], dtype=np.float32)
        baseline_action, _ = baseline.predict(obs)
        assert int(baseline_action) == case["expected_baseline_action"], (
            f"ThresholdPolicy returned unexpected action for conf={conf}"
        )

        # Router interface: route(conf, features) → RoutingDecision
        decision = router.route(
            confidence_score=conf,
            transaction_features=SAMPLE_FEATURES,
        )
        # Interface check: result must be a valid RoutingDecision
        assert isinstance(decision, RoutingDecision)
        assert decision.status in VALID_STATUSES
        # Both must produce a valid action (not necessarily the same one)
        assert decision.action_index in (0, 1, 2)

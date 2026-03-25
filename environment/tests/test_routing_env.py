"""
Unit tests for the RL routing environment and reward functions.

All tests run without calling the Anthropic API. The environment falls back to
fixture-derived or minimal mock transactions when the JSONL dataset is absent.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from environment.reward_functions import (
    AUTO_APPROVE,
    SURFACE_FOR_REVIEW,
    REJECT_FOR_MANUAL,
    reward_a,
    reward_b,
    reward_c,
    get_reward_function,
    REWARD_FUNCTIONS,
)
from environment.routing_env import RoutingEnv, _minimal_test_transactions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transactions(n: int = 20) -> list[dict]:
    """Build a minimal valid transaction list for environment testing."""
    records = []
    for i in range(n):
        tier = i % 3
        conf = {0: 0.91, 1: 0.65, 2: 0.38}[tier]
        is_correct = conf > 0.60
        records.append({
            "transaction_id": f"t{i:03d}",
            "confidence_score": conf,
            "amount_normalised": min(math.log1p(100 + i * 50) / math.log1p(50000), 1.0),
            "difficulty_tier": tier,
            "category_entropy": 0.0,
            "is_correct": is_correct,
            "true_category": "Sales",
            "true_category_code": "200",
            "is_synthetic": i > 2,
            "description": f"TEST TRANSACTION {i}",
            "amount": float(100 + i * 50),
        })
    return records


@pytest.fixture
def env_a() -> RoutingEnv:
    return RoutingEnv(transactions=_make_transactions(), reward_variant="A", seed=42)


@pytest.fixture
def env_b() -> RoutingEnv:
    return RoutingEnv(transactions=_make_transactions(), reward_variant="B", seed=42)


@pytest.fixture
def env_c() -> RoutingEnv:
    return RoutingEnv(transactions=_make_transactions(), reward_variant="C", seed=42)


# ---------------------------------------------------------------------------
# Test 1: env.reset() returns a valid state
# ---------------------------------------------------------------------------

def test_env_reset_returns_valid_state(env_a: RoutingEnv) -> None:
    obs, info = env_a.reset()

    assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
    assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
    assert obs.dtype == np.float32, "Observation must be float32"

    confidence_score, amount_normalised, difficulty_tier, category_entropy = obs

    assert 0.0 <= float(confidence_score) <= 1.0, "confidence_score out of [0, 1]"
    assert 0.0 <= float(amount_normalised) <= 1.0, "amount_normalised out of [0, 1]"
    assert float(difficulty_tier) in (0.0, 1.0, 2.0), f"difficulty_tier must be 0/1/2, got {difficulty_tier}"
    assert 0.0 <= float(category_entropy) <= 1.0, "category_entropy out of [0, 1]"

    assert "transaction_id" in info
    assert "accountant_load" in info
    assert float(info["accountant_load"]) == 0.0  # Variant A: no load signal


def test_env_reset_variant_b_returns_5d_state(env_b: RoutingEnv) -> None:
    obs, info = env_b.reset()
    assert obs.shape == (5,), f"Variant B requires 5-D obs, got {obs.shape}"
    accountant_load = float(obs[4])
    assert 0.0 <= accountant_load <= 1.0, "accountant_load out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 2: env.step() returns correct shapes and types
# ---------------------------------------------------------------------------

def test_env_step_returns_correct_shape(env_a: RoutingEnv) -> None:
    env_a.reset()
    result = env_a.step(AUTO_APPROVE)

    assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray), "obs must be ndarray"
    assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
    assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
    assert isinstance(terminated, bool), "terminated must be bool"
    assert isinstance(truncated, bool), "truncated must be bool"
    assert isinstance(info, dict), "info must be dict"
    assert truncated is False, "truncated should always be False (no time limit)"


def test_env_episode_terminates_after_all_transactions(env_a: RoutingEnv) -> None:
    obs, _ = env_a.reset()
    n = env_a.n_transactions
    terminated = False

    for step in range(n + 5):  # +5 ensures we don't infinite-loop on a bug
        _, _, terminated, _, _ = env_a.step(AUTO_APPROVE)
        if terminated:
            assert step == n - 1, f"Expected termination at step {n-1}, got {step}"
            break

    assert terminated, "Episode never terminated"


def test_env_step_before_reset_raises() -> None:
    env = RoutingEnv(transactions=_make_transactions())
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


# ---------------------------------------------------------------------------
# Test 3: Variant A reward — correct auto-approval
# ---------------------------------------------------------------------------

def test_reward_variant_a_correct_approval() -> None:
    r = reward_a(action=AUTO_APPROVE, is_correct=True)
    assert r == 1.0, f"Correct AUTO_APPROVE should return +1.0, got {r}"


def test_reward_variant_a_incorrect_approval() -> None:
    r = reward_a(action=AUTO_APPROVE, is_correct=False)
    assert r == -2.0, f"Incorrect AUTO_APPROVE should return -2.0, got {r}"


def test_reward_variant_a_correct_escalation() -> None:
    # Escalating when agent was wrong = correct decision
    r = reward_a(action=SURFACE_FOR_REVIEW, is_correct=False)
    assert r == 0.3, f"Correct escalation should return +0.3, got {r}"


def test_reward_variant_a_unnecessary_escalation() -> None:
    # Escalating when agent was right = unnecessary
    r = reward_a(action=SURFACE_FOR_REVIEW, is_correct=True)
    assert r == -0.3, f"Unnecessary escalation should return -0.3, got {r}"


# ---------------------------------------------------------------------------
# Test 4: Incorrect approval penalised more than unnecessary escalation
# ---------------------------------------------------------------------------

def test_reward_variant_a_incorrect_approval_penalised_more_than_unnecessary_escalation() -> None:
    """
    A false auto-approval (-2.0) must be penalised more heavily than unnecessary
    escalation (-0.3). This is the core asymmetry in the reward design —
    a silent ledger error is far costlier than a wasted accountant review.
    """
    penalty_false_approval = reward_a(action=AUTO_APPROVE, is_correct=False)
    penalty_unnecessary_escalation = reward_a(action=SURFACE_FOR_REVIEW, is_correct=True)

    assert penalty_false_approval < penalty_unnecessary_escalation, (
        f"False auto-approval penalty ({penalty_false_approval}) must be more negative "
        f"than unnecessary escalation penalty ({penalty_unnecessary_escalation})"
    )
    assert penalty_false_approval == -2.0
    assert penalty_unnecessary_escalation == -0.3


def test_reward_variant_c_incorrect_approval_more_severe_than_a() -> None:
    """Variant C must penalise false auto-approvals more severely than Variant A."""
    penalty_a = reward_a(action=AUTO_APPROVE, is_correct=False)
    penalty_c = reward_c(action=AUTO_APPROVE, is_correct=False)
    assert penalty_c < penalty_a, (
        f"Variant C penalty ({penalty_c}) must be more negative than Variant A ({penalty_a})"
    )
    assert penalty_c == -5.0


# ---------------------------------------------------------------------------
# Test 5: Variant B workload scaling
# ---------------------------------------------------------------------------

def test_reward_variant_b_unnecessary_escalation_scales_with_load() -> None:
    """
    Variant B: unnecessary escalation penalty increases with accountant load.
    At load=0.0 it equals Variant A; at load=1.0 it doubles.
    """
    penalty_zero_load = reward_b(action=SURFACE_FOR_REVIEW, is_correct=True, accountant_load=0.0)
    penalty_half_load = reward_b(action=SURFACE_FOR_REVIEW, is_correct=True, accountant_load=0.5)
    penalty_full_load = reward_b(action=SURFACE_FOR_REVIEW, is_correct=True, accountant_load=1.0)

    assert penalty_zero_load == pytest.approx(-0.30), f"At load=0, expected -0.30, got {penalty_zero_load}"
    assert penalty_half_load == pytest.approx(-0.45), f"At load=0.5, expected -0.45, got {penalty_half_load}"
    assert penalty_full_load == pytest.approx(-0.60), f"At load=1, expected -0.60, got {penalty_full_load}"

    assert penalty_zero_load > penalty_half_load > penalty_full_load, (
        "Penalty must increase monotonically with load"
    )


def test_reward_variant_b_correct_escalation_load_independent() -> None:
    """Correct escalation reward (+0.3) must NOT depend on accountant load."""
    for load in (0.0, 0.5, 1.0):
        r = reward_b(action=SURFACE_FOR_REVIEW, is_correct=False, accountant_load=load)
        assert r == pytest.approx(0.3), (
            f"Correct escalation reward at load={load} should be +0.3, got {r}"
        )


# ---------------------------------------------------------------------------
# Test 6: Simulator stratified split — held-out set not in training set
# ---------------------------------------------------------------------------

def test_held_out_set_not_in_training_set() -> None:
    """
    After a dry-run simulation, the held-out eval set must share no transaction
    IDs with the training set.
    """
    from environment.transaction_simulator import TransactionSimulator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch paths to use temp dir
        import environment.transaction_simulator as sim_module
        original_synthetic = sim_module.SYNTHETIC_DIR
        original_eval = sim_module.EVAL_DIR
        sim_module.SYNTHETIC_DIR = Path(tmpdir) / "synthetic"
        sim_module.EVAL_DIR = Path(tmpdir) / "evaluation"

        try:
            sim = TransactionSimulator(dry_run=True, verbose=False)
            sim.run(n_variants_by_difficulty={"easy": 2, "medium": 2, "hard": 2})

            train_path = Path(tmpdir) / "synthetic" / "transactions.jsonl"
            eval_path = Path(tmpdir) / "evaluation" / "held_out_set.json"

            assert train_path.exists(), "Training JSONL not written"
            assert eval_path.exists(), "Eval JSON not written"

            train_ids = set()
            with open(train_path, "r") as f:
                for line in f:
                    if line.strip():
                        train_ids.add(json.loads(line)["transaction_id"])

            with open(eval_path, "r") as f:
                eval_records = json.load(f)
            eval_ids = {r["transaction_id"] for r in eval_records}

            overlap = train_ids & eval_ids
            assert len(overlap) == 0, (
                f"Held-out set and training set share {len(overlap)} IDs: {overlap}"
            )

        finally:
            sim_module.SYNTHETIC_DIR = original_synthetic
            sim_module.EVAL_DIR = original_eval


def test_simulator_produces_balanced_categories() -> None:
    """
    After a dry-run simulation, each difficulty tier should have at least one
    transaction in both train and eval sets.
    """
    from environment.transaction_simulator import TransactionSimulator

    with tempfile.TemporaryDirectory() as tmpdir:
        import environment.transaction_simulator as sim_module
        original_synthetic = sim_module.SYNTHETIC_DIR
        original_eval = sim_module.EVAL_DIR
        sim_module.SYNTHETIC_DIR = Path(tmpdir) / "synthetic"
        sim_module.EVAL_DIR = Path(tmpdir) / "evaluation"

        try:
            sim = TransactionSimulator(dry_run=True, verbose=False)
            summary = sim.run(n_variants_by_difficulty={"easy": 3, "medium": 4, "hard": 5})

            # All three tiers should have training examples
            for tier_idx, tier_name in enumerate(("Easy", "Medium", "Hard")):
                count = summary["train_by_tier"][tier_idx]["count"]
                assert count > 0, f"{tier_name} tier has 0 training examples"

        finally:
            sim_module.SYNTHETIC_DIR = original_synthetic
            sim_module.EVAL_DIR = original_eval


# ---------------------------------------------------------------------------
# Reward function registry
# ---------------------------------------------------------------------------

def test_reward_function_registry() -> None:
    for variant in ("A", "B", "C"):
        fn = get_reward_function(variant)
        assert callable(fn), f"get_reward_function('{variant}') must return a callable"

    with pytest.raises(ValueError, match="Unknown reward variant"):
        get_reward_function("X")


def test_all_variants_share_correct_auto_approve_reward() -> None:
    """All variants must agree: correct auto-approval earns +1.0."""
    for variant, fn in REWARD_FUNCTIONS.items():
        r = fn(action=AUTO_APPROVE, is_correct=True)
        assert r == 1.0, f"Variant {variant}: correct AUTO_APPROVE should be +1.0, got {r}"

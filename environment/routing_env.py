"""
RoutingEnv — Gymnasium-compatible environment for transaction routing decisions.

The environment presents the agent with one transaction at a time and asks it to
choose a routing action. An episode is one full pass through the transaction dataset
(shuffled). The agent receives a scalar reward after each action.

State space
-----------
Variants A and C (4-dimensional):
    [confidence_score, amount_normalised, difficulty_tier, category_entropy]

Variant B (5-dimensional — adds accountant load to the state):
    [confidence_score, amount_normalised, difficulty_tier, category_entropy, accountant_load]

Feature details:
    confidence_score  (float, 0-1): The CategoriserAgent's output confidence.
                                    This is the primary routing signal.
    amount_normalised (float, 0-1): Transaction amount, log-scaled.
                                    log1p(|amount_£|) / log1p(50000).
                                    Captures amount magnitude without extreme values
                                    dominating (a £100 and £50,000 transaction are
                                    distinguishable but the difference is compressed).
    difficulty_tier   (float, 0/1/2): easy=0, medium=1, hard=2.
                                    Proxy for inherent transaction ambiguity.
    category_entropy  (float, 0-1): Entropy over top-k category probabilities.
                                    Currently 0.0 for all transactions (Claude API
                                    returns a single prediction, not top-k probs).
                                    Retained for future use with multi-sample prompting.
    accountant_load   (float, 0-1): [Variant B only] Simulated accountant queue pressure.
                                    Drawn from U(0,1) at the start of each episode,
                                    constant within an episode.

Action space
------------
Discrete(3):
    0 = AUTO_APPROVE      — write the CategoriserAgent's prediction to the ledger
    1 = SURFACE_FOR_REVIEW — add to accountant review queue
    2 = REJECT_FOR_MANUAL  — flag for full manual re-categorisation

Reward
------
See environment/reward_functions.py for the full rationale. Reward depends on both
the action chosen and whether the agent's prediction was correct (is_correct flag
stored in the transaction data).

Episode structure
-----------------
Each episode iterates through ALL transactions in the dataset in a shuffled order.
Episode terminates (terminated=True) when the last transaction is processed.
This means episode length == dataset size (typically 700+ training transactions).

SB3 compatibility: The environment is registered as a standard Gymnasium Env and
can be wrapped in Monitor, VecEnv, etc.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.reward_functions import (
    ACTION_NAMES,
    RewardFn,
    get_reward_function,
)

# Maximum £ amount used for log-normalisation.
# Transactions above this are clipped to 1.0 (very rare for typical SME activity).
_MAX_AMOUNT_LOG = math.log1p(50_000.0)

# Difficulty tier → integer mapping
_DIFFICULTY_MAP: dict[str, int] = {"easy": 0, "medium": 1, "hard": 2}


class RoutingEnv(gym.Env):
    """
    Gymnasium environment for RL-based transaction routing.

    Args:
        transactions: Optional list of pre-loaded transaction dicts.
            If None, data is loaded from data_path.
        data_path: Path to JSONL file produced by transaction_simulator.py.
            Used only when transactions is None.
        reward_variant: Which reward design to use — 'A', 'B', or 'C'.
        seed: Random seed for episode shuffling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transactions: list[dict] | None = None,
        data_path: str = "data/synthetic/transactions.jsonl",
        reward_variant: str = "A",
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if reward_variant not in ("A", "B", "C"):
            raise ValueError(f"reward_variant must be 'A', 'B', or 'C', got '{reward_variant}'")

        self.reward_variant = reward_variant
        self._reward_fn: RewardFn = get_reward_function(reward_variant)

        # Load transaction dataset
        if transactions is not None:
            self._transactions = transactions
        else:
            self._transactions = _load_transactions(data_path)

        if not self._transactions:
            raise ValueError(
                "No transactions loaded. Run `python -m environment.transaction_simulator` "
                "first, or provide transactions directly."
            )

        # ── Observation space ────────────────────────────────────────────────
        # Variant B adds accountant_load as a 5th dimension so the agent can
        # learn load-sensitive routing behaviour.
        obs_dim = 5 if reward_variant == "B" else 4
        low: np.ndarray = np.zeros(obs_dim, dtype=np.float32)
        high = np.array(
            [1.0, 1.0, 2.0, 1.0] + ([1.0] if reward_variant == "B" else []),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action space ─────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(3)

        # ── Episode state ────────────────────────────────────────────────────
        self._rng = random.Random(seed)
        self._shuffled: list[dict] = []
        self._step_idx: int = 0
        self._current_tx: dict | None = None

        # Variant B: accountant load is drawn once per episode
        self._accountant_load: float = 0.0

        # Running episode statistics (exposed in terminal info dict)
        self._ep_rewards: list[float] = []
        self._ep_action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)

        # Shuffle dataset for this episode
        self._shuffled = list(self._transactions)
        self._rng.shuffle(self._shuffled)
        self._step_idx = 0

        # Draw accountant load for this episode (Variant B only)
        self._accountant_load = self._rng.random() if self.reward_variant == "B" else 0.0

        # Reset episode tracking
        self._ep_rewards = []
        self._ep_action_counts = {0: 0, 1: 0, 2: 0}

        self._current_tx = self._shuffled[0]
        obs = self._make_obs(self._current_tx)
        info: dict[str, Any] = {
            "transaction_id": self._current_tx.get("transaction_id"),
            "accountant_load": self._accountant_load,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_tx is None:
            raise RuntimeError("Call reset() before the first step().")

        tx = self._current_tx
        action = int(action)

        # ── Reward ────────────────────────────────────────────────────────────
        reward = float(
            self._reward_fn(
                action=action,
                is_correct=bool(tx["is_correct"]),
                accountant_load=self._accountant_load,
            )
        )
        self._ep_rewards.append(reward)
        self._ep_action_counts[action] = self._ep_action_counts.get(action, 0) + 1

        # ── Advance ───────────────────────────────────────────────────────────
        self._step_idx += 1
        terminated = self._step_idx >= len(self._shuffled)
        truncated = False

        if not terminated:
            self._current_tx = self._shuffled[self._step_idx]
            next_obs = self._make_obs(self._current_tx)
        else:
            # Terminal step: return the last observation again (convention)
            next_obs = self._make_obs(tx)

        info: dict[str, Any] = {
            "transaction_id": tx.get("transaction_id"),
            "action_name": ACTION_NAMES.get(action, str(action)),
            "is_correct": tx["is_correct"],
            "confidence_score": tx.get("confidence_score"),
            "difficulty_tier": tx.get("difficulty_tier"),
            "accountant_load": self._accountant_load,
        }

        if terminated:
            n = len(self._ep_rewards)
            info["episode"] = {
                "r": sum(self._ep_rewards),
                "l": n,
                "mean_reward": sum(self._ep_rewards) / n if n else 0.0,
                "action_counts": dict(self._ep_action_counts),
                "auto_approve_rate": self._ep_action_counts.get(0, 0) / n if n else 0.0,
            }

        return next_obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._current_tx is None:
            print("[RoutingEnv] not yet reset")
            return
        tx = self._current_tx
        tier_name = ["easy", "medium", "hard"][int(tx.get("difficulty_tier", 0))]
        print(
            f"[RoutingEnv] step={self._step_idx}/{len(self._shuffled)} "
            f"id={tx.get('transaction_id', '?')} "
            f"conf={tx.get('confidence_score', 0):.3f} "
            f"tier={tier_name} correct={tx.get('is_correct')}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_obs(self, tx: dict) -> np.ndarray:
        """Build the observation array from a transaction record."""
        obs = [
            float(tx.get("confidence_score", 0.0)),
            float(tx.get("amount_normalised", 0.0)),
            float(tx.get("difficulty_tier", 0)),
            float(tx.get("category_entropy", 0.0)),
        ]
        if self.reward_variant == "B":
            obs.append(float(self._accountant_load))
        return np.array(obs, dtype=np.float32)

    @property
    def n_transactions(self) -> int:
        """Total number of transactions in the dataset."""
        return len(self._transactions)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_transactions(data_path: str) -> list[dict]:
    """
    Load transactions from a JSONL file produced by transaction_simulator.py.

    Falls back to fixture-derived mock data if the JSONL file doesn't exist yet,
    so the environment can be used for unit testing before the simulator is run.
    """
    path = Path(data_path)
    if path.exists():
        return _read_jsonl(path)

    # Seed fallback for testing
    fixture_path = Path("data/seeds/transactions.json")
    if fixture_path.exists():
        return _transactions_from_fixtures(fixture_path)

    # Minimal in-memory fallback for isolated unit tests
    return _minimal_test_transactions()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _transactions_from_fixtures(fixture_path: Path) -> list[dict]:
    """
    Build RoutingEnv-compatible records from the eval fixture JSON.
    Uses mock confidence scores (no API call). For testing only.
    """
    with open(fixture_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for tx in raw:
        amount = abs(float(tx["amount"]))
        difficulty = tx.get("difficulty", "easy")
        tier = _DIFFICULTY_MAP.get(difficulty, 0)
        conf = _mock_confidence(tier)
        records.append({
            "transaction_id": tx["id"],
            "confidence_score": conf,
            "amount_normalised": min(math.log1p(amount) / _MAX_AMOUNT_LOG, 1.0),
            "difficulty_tier": tier,
            "category_entropy": 0.0,
            "is_correct": conf > 0.60,  # mock: high confidence treated as correct
            "true_category": tx["expected_category_name"],
            "true_category_code": tx["expected_category_code"],
            "is_synthetic": False,
            "description": tx["description"],
            "amount": float(tx["amount"]),
        })
    return records


def _mock_confidence(tier: int) -> float:
    """Deterministic mock confidence by difficulty tier (for testing only)."""
    return {0: 0.91, 1: 0.66, 2: 0.38}.get(tier, 0.5)


def _minimal_test_transactions() -> list[dict]:
    """Six hand-crafted records for unit tests with no file dependencies."""
    return [
        {
            "transaction_id": f"mock_{i}",
            "confidence_score": c,
            "amount_normalised": 0.35,
            "difficulty_tier": d,
            "category_entropy": 0.0,
            "is_correct": ic,
            "true_category": "Sales",
            "true_category_code": "200",
            "is_synthetic": False,
            "description": "TEST TRANSACTION",
            "amount": 500.0,
        }
        for i, (c, d, ic) in enumerate([
            (0.93, 0, True),
            (0.71, 1, True),
            (0.36, 2, False),
            (0.89, 0, True),
            (0.55, 1, False),
            (0.96, 0, True),
        ])
    ]

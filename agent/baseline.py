"""
agent/baseline.py — Hand-tuned threshold routing policy.

This module implements the original CategoriserAgent routing logic as a policy
class with the same interface as a Stable Baselines3 policy. It is the
comparison baseline for all PPO variants.

The parent project uses three fixed confidence thresholds:
    confidence > 0.85  → AUTO_APPROVE       (action 0)
    confidence 0.50–0.85 → SURFACE_FOR_REVIEW (action 1)
    confidence < 0.50  → REJECT_FOR_MANUAL   (action 2)

These values are documented in CLAUDE.md under "Confidence thresholds" and
are hard-coded in agents/categoriser.py in the parent project.

Interface compatibility
-----------------------
The `predict()` method matches the SB3 interface:
    predict(observation, state, episode_start, deterministic)
    → (action_array, state)

This lets evaluation code call baseline and PPO models identically.

Configuration
-------------
Thresholds can be overridden via environment variables for experiments:
    ROUTING_AUTO_THRESHOLD   (default: 0.85)
    ROUTING_REVIEW_THRESHOLD (default: 0.50)

This enables ablation studies without changing code.
"""

from __future__ import annotations

import os
import numpy as np


# ---------------------------------------------------------------------------
# Threshold defaults — match the parent project's CategoriserAgent exactly
# ---------------------------------------------------------------------------
_DEFAULT_AUTO_THRESHOLD: float = 0.85    # > this → AUTO_APPROVE
_DEFAULT_REVIEW_THRESHOLD: float = 0.50  # > this → SURFACE_FOR_REVIEW, else REJECT


class ThresholdPolicy:
    """
    Hand-tuned confidence-threshold routing policy.

    Reproduces the exact logic from the parent project's CategoriserAgent:
        confidence > auto_threshold    → AUTO_APPROVE (0)
        confidence > review_threshold  → SURFACE_FOR_REVIEW (1)
        otherwise                      → REJECT_FOR_MANUAL (2)

    The confidence score is always the FIRST element of the observation
    vector (index 0), regardless of reward variant (A, B, or C).

    Args:
        auto_threshold: Confidence above which the prediction is auto-approved.
            Defaults to ROUTING_AUTO_THRESHOLD env var or 0.85.
        review_threshold: Confidence above which the transaction is surfaced
            for human review (but not auto-approved). Below this → reject.
            Defaults to ROUTING_REVIEW_THRESHOLD env var or 0.50.
    """

    def __init__(
        self,
        auto_threshold: float | None = None,
        review_threshold: float | None = None,
    ) -> None:
        self.auto_threshold = (
            auto_threshold
            if auto_threshold is not None
            else float(os.environ.get("ROUTING_AUTO_THRESHOLD", _DEFAULT_AUTO_THRESHOLD))
        )
        self.review_threshold = (
            review_threshold
            if review_threshold is not None
            else float(os.environ.get("ROUTING_REVIEW_THRESHOLD", _DEFAULT_REVIEW_THRESHOLD))
        )

        if not (0.0 <= self.review_threshold < self.auto_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: review={self.review_threshold}, "
                f"auto={self.auto_threshold}. Must satisfy 0 ≤ review < auto ≤ 1."
            )

    def predict(
        self,
        observation: np.ndarray,
        state: object = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, object]:
        """
        Predict a routing action from an observation.

        Compatible with the SB3 policy.predict() interface so that evaluation
        code can call baseline and PPO models identically.

        Args:
            observation: Numpy array of shape (obs_dim,) or (n_envs, obs_dim).
                The confidence score is always at index 0.
            state: Unused (kept for interface compatibility with recurrent policies).
            episode_start: Unused (kept for interface compatibility).
            deterministic: Unused (threshold policy is always deterministic).

        Returns:
            (actions, state) — actions is a numpy array of int, state is None.
        """
        obs = np.atleast_2d(observation)  # ensure (n_envs, obs_dim)
        confidence = obs[:, 0]            # confidence score is always feature 0

        actions = np.where(
            confidence > self.auto_threshold,
            0,   # AUTO_APPROVE
            np.where(
                confidence > self.review_threshold,
                1,   # SURFACE_FOR_REVIEW
                2,   # REJECT_FOR_MANUAL
            ),
        ).astype(int)

        # If input was 1D (single observation), return 1D action
        if np.ndim(observation) == 1:
            return actions.squeeze(), state

        return actions, state

    def __repr__(self) -> str:
        return (
            f"ThresholdPolicy("
            f"auto_threshold={self.auto_threshold}, "
            f"review_threshold={self.review_threshold})"
        )


# ---------------------------------------------------------------------------
# Convenience factory — matches the parent project's defaults exactly
# ---------------------------------------------------------------------------

def make_baseline_policy() -> ThresholdPolicy:
    """
    Return a ThresholdPolicy using the parent project's original thresholds
    (0.85 auto-approve, 0.50 review boundary).

    This is the policy that the PPO agents are being compared against.
    """
    return ThresholdPolicy(
        auto_threshold=_DEFAULT_AUTO_THRESHOLD,
        review_threshold=_DEFAULT_REVIEW_THRESHOLD,
    )

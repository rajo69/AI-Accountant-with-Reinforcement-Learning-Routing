"""
integration/router.py — Drop-in replacement for the parent project's threshold
routing logic.

This module provides LearnedRouter, a class with an identical interface to the
existing hand-tuned threshold router in the parent project's CategoriserAgent.
It loads the best-performing trained PPO policy (Variant C, per Phase 3 results)
and exposes a single route() method that returns a RoutingDecision.

Graceful degradation
--------------------
If the model file is not found at startup, LearnedRouter automatically falls back
to the hand-tuned ThresholdPolicy. This guarantees the parent project never
breaks due to a missing model file — the fallback produces exactly the same
routing behaviour as the current production system.

Audit logging
-------------
Every routing decision is written to a structured audit dict compatible with the
parent project's AuditLog.ai_decision_data JSONB column. The parent project's
audit infrastructure (agents/categoriser.py → AuditLog) must call
router.route() and store the returned RoutingDecision.audit_data.

Interface compatibility
-----------------------
LearnedRouter.route() accepts the same arguments as the implicit threshold routing
in the parent project's categoriser decide node:
    route(confidence_score: float, transaction_features: dict) -> RoutingDecision

This is a drop-in: replace the if/elif confidence threshold block in
agents/categoriser.py with a LearnedRouter().route() call. See
INTEGRATION_GUIDE.md for the exact diff.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.baseline import ThresholdPolicy, make_baseline_policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolved relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINED_DIR = REPO_ROOT / "models" / "trained"

# Best model determined by Phase 3 evaluation: PPO Variant C achieves the best
# balance of precision (90.1%) and error rate (9.9%) — the recommended default
# for compliance contexts. Override via ROUTING_MODEL_VARIANT env var.
DEFAULT_MODEL_VARIANT = "C"

# Action constants — match parent project's CategoriserAgent decide node
AUTO_APPROVE = "auto_categorised"       # confidence > 0.85 in parent project
SURFACE_FOR_REVIEW = "suggested"        # 0.50–0.85 in parent project
REJECT_FOR_MANUAL = "needs_review"      # < 0.50 in parent project

# Internal action index → parent project status string
_ACTION_TO_STATUS: dict[int, str] = {
    0: AUTO_APPROVE,
    1: SURFACE_FOR_REVIEW,
    2: REJECT_FOR_MANUAL,
}

# Max amount for log-normalisation — must match RoutingEnv._MAX_AMOUNT_LOG
_MAX_AMOUNT_LOG = math.log1p(50_000.0)


@dataclass
class RoutingDecision:
    """
    Return value of LearnedRouter.route().

    Fields match the parent project's categoriser decide node output so the
    integration is a drop-in replacement.
    """
    status: str                      # "auto_categorised" | "suggested" | "needs_review"
    action_index: int                # 0, 1, or 2
    confidence_score: float          # original confidence from CategoriserAgent
    policy_used: str                 # "ppo_variant_C" or "threshold_baseline"
    model_variant: str               # "A", "B", "C", or "baseline"
    latency_ms: float                # routing inference time
    audit_data: dict[str, Any] = field(default_factory=dict)
    """Structured dict ready for AuditLog.ai_decision_data storage."""


class LearnedRouter:
    """
    PPO-based routing policy as a drop-in for the parent project's threshold logic.

    Usage:
        router = LearnedRouter()               # loads PPO-C by default
        decision = router.route(
            confidence_score=0.72,
            transaction_features={
                "amount": 450.00,
                "difficulty_tier": 1,
                "category_entropy": 0.0,
            },
        )
        # decision.status is "auto_categorised" | "suggested" | "needs_review"
        # decision.audit_data is ready for AuditLog.ai_decision_data

    Args:
        model_variant: Which trained PPO variant to load ("A", "B", or "C").
            Defaults to "C" (best precision in Phase 3 evaluation).
            Override with env var ROUTING_MODEL_VARIANT.
        models_dir: Directory containing ppo_variant_*.zip files.
            Defaults to models/trained/ in the repo root.
    """

    def __init__(
        self,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        models_dir: Path | None = None,
    ) -> None:
        import os
        model_variant = os.environ.get("ROUTING_MODEL_VARIANT", model_variant)

        self.model_variant = model_variant
        self._models_dir = models_dir or TRAINED_DIR
        self._policy_used = f"ppo_variant_{model_variant}"
        self._model: Any = None  # SB3 PPO | None
        self._fallback: ThresholdPolicy = make_baseline_policy()
        self._using_fallback: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model, setting _using_fallback=True if not found."""
        model_path = self._models_dir / f"ppo_variant_{self.model_variant}.zip"
        if not model_path.exists():
            logger.warning(
                "Trained model not found at %s — falling back to hand-tuned "
                "threshold policy (0.85/0.50). Routing behaviour unchanged from "
                "current production system.",
                model_path,
            )
            self._using_fallback = True
            self._policy_used = "threshold_baseline"
            return

        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(model_path))
            logger.info("Loaded PPO routing model: variant=%s path=%s", self.model_variant, model_path)
        except Exception as exc:
            logger.error("Failed to load PPO model (%s): %s — falling back to baseline", model_path, exc)
            self._using_fallback = True
            self._policy_used = "threshold_baseline"

    def reload(self) -> bool:
        """
        Hot-reload the model from disk without restarting the process.

        Returns True if reload succeeded, False if fallback was activated.
        Used by the POST /policy/reload API endpoint.
        """
        self._model = None
        self._using_fallback = False
        self._policy_used = f"ppo_variant_{self.model_variant}"
        self._load_model()
        return not self._using_fallback

    def route(
        self,
        confidence_score: float,
        transaction_features: dict[str, Any],
    ) -> RoutingDecision:
        """
        Determine routing action for a transaction.

        Args:
            confidence_score: Float 0–1 from CategoriserAgent output.
            transaction_features: Dict with optional keys:
                - amount (float): Transaction amount in GBP. Used for log-normalisation.
                - difficulty_tier (int): 0=easy, 1=medium, 2=hard. From eval framework.
                - category_entropy (float): Entropy over top-k categories. Pass 0.0
                  if not available (Claude API returns single prediction).
                - accountant_load (float): Queue pressure 0–1. Used by Variant B only.
                  Pass 0.5 for neutral/unknown load.

        Returns:
            RoutingDecision with status, audit_data, and inference metadata.
        """
        t0 = time.perf_counter()

        obs = self._build_observation(confidence_score, transaction_features)

        if self._using_fallback:
            action_arr, _ = self._fallback.predict(obs)
            action = int(action_arr)
        else:
            action_arr, _ = self._model.predict(obs, deterministic=True)
            action = int(action_arr)

        latency_ms = (time.perf_counter() - t0) * 1000
        status = _ACTION_TO_STATUS[action]

        audit_data = {
            "router": {
                "policy_used": self._policy_used,
                "model_variant": self.model_variant if not self._using_fallback else "baseline",
                "using_fallback": self._using_fallback,
                "action_index": action,
                "action_name": ["AUTO_APPROVE", "SURFACE_FOR_REVIEW", "REJECT_FOR_MANUAL"][action],
                "confidence_score": round(confidence_score, 6),
                "observation": obs.tolist(),
                "latency_ms": round(latency_ms, 3),
            }
        }

        return RoutingDecision(
            status=status,
            action_index=action,
            confidence_score=confidence_score,
            policy_used=self._policy_used,
            model_variant=self.model_variant if not self._using_fallback else "baseline",
            latency_ms=round(latency_ms, 3),
            audit_data=audit_data,
        )

    def _build_observation(
        self,
        confidence_score: float,
        features: dict[str, Any],
    ) -> np.ndarray:
        """Build the observation vector from transaction features."""
        amount = abs(float(features.get("amount", 0.0)))
        amount_normalised = min(math.log1p(amount) / _MAX_AMOUNT_LOG, 1.0)

        obs = [
            float(confidence_score),
            float(amount_normalised),
            float(features.get("difficulty_tier", 1)),   # default to medium
            float(features.get("category_entropy", 0.0)),
        ]

        # Variant B requires a 5th dimension: accountant_load
        if self.model_variant == "B":
            obs.append(float(features.get("accountant_load", 0.5)))

        return np.array(obs, dtype=np.float32)

    @property
    def is_using_fallback(self) -> bool:
        """True if the router is currently using the hand-tuned baseline."""
        return self._using_fallback

    @property
    def policy_info(self) -> dict[str, Any]:
        """Metadata dict for the GET /policy/info endpoint."""
        import json
        from datetime import datetime

        meta_path = REPO_ROOT / "experiments" / "results" / f"training_meta_{self.model_variant}.json"
        training_meta: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                training_meta = json.load(f)

        eval_path = REPO_ROOT / "experiments" / "results" / "evaluation_results.json"
        eval_metrics: dict = {}
        if eval_path.exists():
            with open(eval_path) as f:
                results = json.load(f)
            variant_key = f"PPO Variant {self.model_variant}"
            for p in results.get("policies", []):
                if p.get("policy") == variant_key:
                    eval_metrics = {
                        "routing_accuracy": p.get("overall_routing_accuracy"),
                        "auto_approval_precision": p.get("auto_approval_precision"),
                        "auto_approval_rate": p.get("auto_approval_rate"),
                        "error_rate_auto": p.get("error_rate_auto"),
                    }
                    break

        return {
            "policy": self._policy_used,
            "model_variant": self.model_variant,
            "using_fallback": self._using_fallback,
            "training": training_meta,
            "evaluation": eval_metrics,
            "loaded_at": datetime.utcnow().isoformat() + "Z",
        }

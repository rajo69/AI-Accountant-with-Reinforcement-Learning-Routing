"""
api/schemas.py — Pydantic v2 request/response models for the routing API.

Follows the same validation style as the parent project (Pydantic v2,
model_config = ConfigDict(from_attributes=True)).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    """
    Features extracted from a transaction at routing time.
    All fields are optional — the router uses sensible defaults for missing values.
    """
    model_config = ConfigDict(from_attributes=True)

    amount: float = Field(
        default=0.0,
        description="Transaction amount in GBP (absolute value used for normalisation).",
    )
    difficulty_tier: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Difficulty tier: 0=easy, 1=medium, 2=hard.",
    )
    category_entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Entropy over top-k category probabilities. Pass 0.0 if unavailable.",
    )
    accountant_load: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Accountant queue pressure (0=empty, 1=full). Used by Variant B only.",
    )


class RouteRequest(BaseModel):
    """Request body for POST /route."""
    model_config = ConfigDict(from_attributes=True)

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from the CategoriserAgent (0–1).",
    )
    transaction_features: TransactionFeatures = Field(
        default_factory=TransactionFeatures,
        description="Transaction features used by the routing policy.",
    )
    transaction_id: str | None = Field(
        default=None,
        description="Optional transaction ID for audit logging.",
    )


class RouteResponse(BaseModel):
    """Response body for POST /route."""
    model_config = ConfigDict(from_attributes=True)

    status: str = Field(
        description=(
            "Routing decision: 'auto_categorised' | 'suggested' | 'needs_review'. "
            "Drop-in replacement for the CategoriserAgent's decide-node output."
        ),
    )
    action_index: int = Field(description="Raw action index: 0=AUTO_APPROVE, 1=SURFACE_FOR_REVIEW, 2=REJECT.")
    confidence_score: float = Field(description="The confidence score that was routed.")
    policy_used: str = Field(description="Which policy made the decision.")
    model_variant: str = Field(description="Model variant ('A', 'B', 'C', or 'baseline').")
    latency_ms: float = Field(description="Router inference latency in milliseconds.")
    audit_data: dict[str, Any] = Field(
        description="Structured audit dict ready for AuditLog.ai_decision_data storage.",
    )


class PolicyInfo(BaseModel):
    """Response body for GET /policy/info."""
    model_config = ConfigDict(from_attributes=True)

    policy: str
    model_variant: str
    using_fallback: bool
    training: dict[str, Any]
    evaluation: dict[str, Any]
    loaded_at: str


class ReloadResponse(BaseModel):
    """Response body for POST /policy/reload."""
    model_config = ConfigDict(from_attributes=True)

    success: bool
    using_fallback: bool
    message: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    model_config = ConfigDict(from_attributes=True)

    status: str
    version: str
    policy: str
    using_fallback: bool

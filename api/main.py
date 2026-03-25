"""
api/main.py — FastAPI application exposing the RL routing policy.

Endpoints
---------
POST /route         Route a transaction using the loaded PPO policy.
GET  /policy/info   Return metadata about the current policy (variant, training, eval metrics).
GET  /health        Standard health check.
POST /policy/reload Hot-reload the policy from disk without restart.

Architecture consistency
------------------------
This API follows the same patterns as the parent project (app/main.py):
- Lifespan handler for startup/shutdown
- CORS middleware (allow_origins=["*"], allow_credentials=False)
- FastAPI with Pydantic v2 response models
- No business logic in the route handlers — everything delegated to the router

The router singleton is created at startup and shared across requests.
Thread safety: SB3 model.predict() is stateless (read-only), so sharing is safe.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PolicyInfo, ReloadResponse, RouteRequest, RouteResponse
from integration.router import LearnedRouter

# ---------------------------------------------------------------------------
# Application-level router singleton — created once at startup
# ---------------------------------------------------------------------------
_router: LearnedRouter | None = None

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router
    _router = LearnedRouter()
    yield
    _router = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Accountant — RL Routing API",
    version=VERSION,
    description=(
        "Exposes a learned PPO routing policy as a drop-in replacement for "
        "the hand-tuned confidence thresholds in the AI Accountant's CategoriserAgent. "
        "See /docs for interactive API documentation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def get_router() -> LearnedRouter:
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialised — server starting up.")
    return _router


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Standard health check. Returns 200 if the service is ready."""
    router = get_router()
    return HealthResponse(
        status="healthy",
        version=VERSION,
        policy=router._policy_used,
        using_fallback=router.is_using_fallback,
    )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
def route_transaction(request: RouteRequest) -> RouteResponse:
    """
    Route a transaction using the loaded PPO policy.

    Accepts a confidence score and optional transaction features.
    Returns the routing decision (status), audit data, and inference metadata.

    The returned `status` field is a drop-in replacement for the parent
    project's CategoriserAgent decide-node output:
    - `auto_categorised` — write prediction to ledger without review
    - `suggested`        — surface for accountant review
    - `needs_review`     — flag for manual re-categorisation
    """
    router = get_router()
    decision = router.route(
        confidence_score=request.confidence_score,
        transaction_features=request.transaction_features.model_dump(),
    )
    return RouteResponse(
        status=decision.status,
        action_index=decision.action_index,
        confidence_score=decision.confidence_score,
        policy_used=decision.policy_used,
        model_variant=decision.model_variant,
        latency_ms=decision.latency_ms,
        audit_data=decision.audit_data,
    )


@app.get("/policy/info", response_model=PolicyInfo, tags=["Policy"])
def policy_info() -> PolicyInfo:
    """
    Return metadata about the currently loaded routing policy.

    Includes training configuration, evaluation metrics from Phase 3,
    and whether the fallback (hand-tuned threshold) is active.
    """
    router = get_router()
    info = router.policy_info
    return PolicyInfo(**info)


@app.post("/policy/reload", response_model=ReloadResponse, tags=["Policy"])
def reload_policy() -> ReloadResponse:
    """
    Hot-reload the routing policy from disk.

    Use this after deploying a retrained model to pick up the new weights
    without restarting the server. Returns whether the reload succeeded
    or whether the fallback was activated.
    """
    router = get_router()
    success = router.reload()
    if success:
        message = f"Policy reloaded successfully: ppo_variant_{router.model_variant}"
    else:
        message = "Model file not found — falling back to hand-tuned threshold policy."
    return ReloadResponse(
        success=success,
        using_fallback=router.is_using_fallback,
        message=message,
    )

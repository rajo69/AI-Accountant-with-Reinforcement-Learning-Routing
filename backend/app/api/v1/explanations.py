"""Explanation API — returns the full XAI package for a transaction."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.session import get_current_org
from app.models.database import AuditLog, Organisation, Transaction

router = APIRouter(prefix="/api/v1", tags=["explanations"])


@router.get("/transactions/{transaction_id}/explanation")
async def get_explanation(
    transaction_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Return the full XAI explanation package for a transaction.

    Includes: prediction, confidence, top features, risk score, risk label,
    fired fuzzy rules, explanation text, and audit history.
    """
    # Fetch transaction (scoped to org)
    tx_result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = tx_result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Fetch audit logs for this transaction (most recent first)
    logs_result = await db.execute(
        select(AuditLog)
        .where(
            AuditLog.entity_type == "transaction",
            AuditLog.entity_id == transaction_id,
        )
        .order_by(desc(AuditLog.created_at))
    )
    audit_logs = list(logs_result.scalars().all())

    # Find the most recent AI categorisation log that has XAI data
    xai_data = None
    risk_data = None
    prediction_info = None

    for log in audit_logs:
        if log.action == "ai_categorise" and log.ai_decision_data:
            dec = log.ai_decision_data
            if not xai_data and "xai" in dec:
                xai_data = dec["xai"]
            if not risk_data and "risk" in dec:
                risk_data = dec["risk"]
            if not prediction_info:
                prediction_info = {
                    "category": (log.new_value or {}).get("category"),
                    "confidence": float(log.ai_confidence) if log.ai_confidence else None,
                    "reasoning": log.ai_explanation,
                    "model": log.ai_model,
                    "category_code": dec.get("category_code"),
                }
            if xai_data and risk_data and prediction_info:
                break

    # Defaults if no XAI data stored yet
    if not xai_data:
        xai_data = {
            "top_features": [],
            "explanation_text": prediction_info["reasoning"] if prediction_info else None,
            "model_type": "llm",
        }
    if not risk_data:
        risk_data = {
            "risk_score": None,
            "risk_label": None,
            "fired_rules": [],
            "input_values": {},
        }

    audit_history = [
        {
            "id": str(log.id),
            "action": log.action,
            "ai_model": log.ai_model,
            "ai_confidence": float(log.ai_confidence) if log.ai_confidence else None,
            "ai_explanation": log.ai_explanation,
            "new_value": log.new_value,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in audit_logs
    ]

    return {
        "transaction_id": str(tx.id),
        "category": tx.category,
        "category_confidence": float(tx.category_confidence) if tx.category_confidence else None,
        "categorisation_status": tx.categorisation_status,
        "prediction": prediction_info,
        "xai": xai_data,
        "risk": risk_data,
        "audit_history": audit_history,
    }

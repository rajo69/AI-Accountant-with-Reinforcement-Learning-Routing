"""LangGraph transaction categorisation agent.

Flow: fetch_context → classify → validate → decide
"""
import asyncio
import logging
from decimal import Decimal
from typing import Optional, TypedDict
from uuid import UUID

import instructor
from anthropic import AsyncAnthropic
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import Account, AuditLog, Transaction
from app.models.schemas import BatchCategoriseResponse
from app.services.embedding_service import embed_transaction, find_similar_transactions
from app.xai.explainer import explain_categorisation
from app.xai.fuzzy_engine import compute_fuzzy_inputs, compute_risk_score

logger = logging.getLogger(__name__)

LLM_MODEL = "claude-haiku-4-5-20251001"


class CategoriserState(TypedDict):
    transaction_id: str
    transaction_data: dict
    chart_of_accounts: list[dict]
    similar_examples: list[dict]
    prediction: Optional[dict]
    status: str
    audit_log_id: Optional[str]


class CategoryPrediction(BaseModel):
    category_code: str
    category_name: str
    confidence: float
    reasoning: str


def build_categoriser_graph(db: AsyncSession):
    """Build and compile a LangGraph categorisation workflow with DB injected via closure."""

    async def fetch_context(state: CategoriserState) -> dict:
        org_id = UUID(state["transaction_data"]["organisation_id"])

        result = await db.execute(
            select(Account).where(Account.organisation_id == org_id)
        )
        accounts = list(result.scalars().all())
        chart = [
            {"code": a.code, "name": a.name, "type": a.type}
            for a in accounts
        ]

        description = state["transaction_data"].get("description", "")
        similar = await find_similar_transactions(description, org_id, db=db)
        examples = [
            {
                "description": t.description,
                "amount": str(t.amount),
                "category": t.category,
                "date": str(t.date),
            }
            for t in similar
            if t.category
        ]

        return {"chart_of_accounts": chart, "similar_examples": examples}

    async def classify(state: CategoriserState) -> dict:
        if not state["chart_of_accounts"]:
            return {
                "prediction": None,
                "status": "classified",
            }

        client = instructor.from_anthropic(
            AsyncAnthropic(api_key=settings.anthropic_api_key)
        )

        tx = state["transaction_data"]
        accounts_text = "\n".join(
            f"  - Code: {a['code'] or 'N/A'}, Name: {a['name']}, Type: {a['type']}"
            for a in state["chart_of_accounts"]
        )
        examples_text = "\n".join(
            f"  - '{e['description']}' (£{e['amount']}) → {e['category']}"
            for e in state["similar_examples"][:5]
        ) or "  (none yet)"

        amount = Decimal(str(tx["amount"]))
        direction = "debit (expense)" if amount < 0 else "credit (income)"

        prompt = (
            "You are an expert UK accountant using Xero.\n\n"
            "Categorise this bank transaction into the correct account from the chart of accounts.\n\n"
            f"Transaction:\n"
            f"- Date: {tx['date']}\n"
            f"- Amount: £{abs(amount):.2f} ({direction})\n"
            f"- Description: {tx['description']}\n"
            f"- Reference: {tx.get('reference') or 'N/A'}\n\n"
            f"Chart of Accounts (pick ONE):\n{accounts_text}\n\n"
            f"Similar transactions already categorised:\n{examples_text}\n\n"
            "Return the exact account code and name from the chart, confidence (0.0-1.0), "
            "and brief reasoning (1-2 sentences)."
        )

        prediction = await client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            response_model=CategoryPrediction,
        )

        return {
            "prediction": {
                "category_code": prediction.category_code,
                "category_name": prediction.category_name,
                "confidence": float(prediction.confidence),
                "reasoning": prediction.reasoning,
            },
            "status": "classified",
        }

    async def validate(state: CategoriserState) -> dict:
        prediction = state.get("prediction")
        if not prediction:
            return {"status": "validated"}

        valid_codes = {a["code"] for a in state["chart_of_accounts"] if a.get("code")}
        valid_names = {a["name"] for a in state["chart_of_accounts"]}
        confidence = prediction.get("confidence", 0.0)

        code_ok = prediction["category_code"] in valid_codes
        name_ok = prediction["category_name"] in valid_names
        confidence_ok = 0.0 <= confidence <= 1.0

        if not (code_ok or name_ok) or not confidence_ok:
            return {
                "prediction": {
                    **prediction,
                    "confidence": 0.0,
                    "reasoning": "validation failed: predicted account not in chart of accounts",
                },
                "status": "validated",
            }

        return {"status": "validated"}

    async def decide(state: CategoriserState) -> dict:
        prediction = state.get("prediction")
        tx_data = state["transaction_data"]
        tx_id = UUID(state["transaction_id"])
        org_id = UUID(tx_data["organisation_id"])

        confidence = prediction["confidence"] if prediction else 0.0

        if not prediction or confidence == 0.0:
            final_status = "needs_review"
        elif confidence > 0.85:
            final_status = "auto_categorised"
        elif confidence >= 0.5:
            final_status = "suggested"
        else:
            final_status = "needs_review"

        result = await db.execute(select(Transaction).where(Transaction.id == tx_id))
        tx = result.scalar_one_or_none()

        if tx and prediction:
            tx.category = prediction["category_name"]
            tx.category_confidence = Decimal(str(round(confidence, 4)))
            tx.categorisation_status = final_status
            tx.embedding = await embed_transaction(tx)

        audit = AuditLog(
            organisation_id=org_id,
            action="ai_categorise",
            entity_type="transaction",
            entity_id=tx_id,
            new_value={
                "category": prediction["category_name"] if prediction else None,
                "status": final_status,
            },
            ai_model=LLM_MODEL,
            ai_confidence=Decimal(str(round(confidence, 4))) if prediction else None,
            ai_explanation=prediction["reasoning"] if prediction else "No prediction generated",
            ai_decision_data={
                "category_code": prediction["category_code"] if prediction else None,
                "similar_examples_count": len(state.get("similar_examples", [])),
            },
        )
        db.add(audit)
        await db.flush()  # get audit.id without closing the session
        audit_log_id = str(audit.id)
        await db.commit()

        return {"status": final_status, "audit_log_id": audit_log_id}

    async def explain(state: CategoriserState) -> dict:
        """Generate XAI explanation and append it to the AuditLog ai_decision_data."""
        prediction = state.get("prediction")
        audit_log_id = state.get("audit_log_id")
        if not prediction or not audit_log_id:
            return {}

        tx_data = state["transaction_data"]
        org_id = UUID(tx_data["organisation_id"])

        try:
            xai_result = await explain_categorisation(
                transaction=tx_data,
                prediction=prediction,
                similar_examples=state.get("similar_examples", []),
                org_id=org_id,
                db=db,
            )
        except Exception:
            logger.exception("XAI explain_categorisation failed")
            xai_result = {"top_features": [], "explanation_text": prediction.get("reasoning", ""), "model_type": "llm"}

        try:
            fuzzy_inputs = await compute_fuzzy_inputs(tx_data, prediction, org_id, db)
            risk_result = compute_risk_score(**fuzzy_inputs)
        except Exception:
            logger.exception("Fuzzy risk scoring failed")
            risk_result = {"risk_score": 0.5, "risk_label": "medium", "fired_rules": [], "input_values": {}}

        # Update the AuditLog with XAI data
        try:
            res = await db.execute(
                select(AuditLog).where(AuditLog.id == UUID(audit_log_id))
            )
            audit = res.scalar_one_or_none()
            if audit:
                existing = dict(audit.ai_decision_data or {})
                existing["xai"] = xai_result
                existing["risk"] = risk_result
                audit.ai_decision_data = existing
                await db.commit()
        except Exception:
            logger.exception("Failed to update AuditLog with XAI data")

        return {}

    workflow = StateGraph(CategoriserState)
    workflow.add_node("fetch_context", fetch_context)
    workflow.add_node("classify", classify)
    workflow.add_node("validate", validate)
    workflow.add_node("decide", decide)
    workflow.add_node("explain", explain)

    workflow.set_entry_point("fetch_context")
    workflow.add_edge("fetch_context", "classify")
    workflow.add_edge("classify", "validate")
    workflow.add_edge("validate", "decide")
    workflow.add_edge("decide", "explain")
    workflow.add_edge("explain", END)

    return workflow.compile()


async def categorise_batch(org_id: UUID, db: AsyncSession) -> BatchCategoriseResponse:
    """Categorise all uncategorised transactions for an organisation."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.organisation_id == org_id,
            Transaction.categorisation_status == "uncategorised",
        )
    )
    transactions = list(result.scalars().all())

    graph = build_categoriser_graph(db)
    semaphore = asyncio.Semaphore(5)
    counts: dict[str, int] = {
        "auto_categorised": 0,
        "suggested": 0,
        "needs_review": 0,
        "errors": 0,
    }

    async def process_one(tx: Transaction) -> None:
        async with semaphore:
            try:
                initial_state: CategoriserState = {
                    "transaction_id": str(tx.id),
                    "transaction_data": {
                        "organisation_id": str(tx.organisation_id),
                        "date": str(tx.date),
                        "amount": str(tx.amount),
                        "description": tx.description,
                        "reference": tx.reference,
                    },
                    "chart_of_accounts": [],
                    "similar_examples": [],
                    "prediction": None,
                    "status": "pending",
                    "audit_log_id": None,
                }
                final = await graph.ainvoke(initial_state)
                status = final.get("status", "errors")
                if status in counts:
                    counts[status] += 1
                else:
                    counts["errors"] += 1
            except Exception:
                logger.exception("Error categorising transaction %s", tx.id)
                counts["errors"] += 1

    await asyncio.gather(*[process_one(tx) for tx in transactions])

    return BatchCategoriseResponse(
        auto_categorised=counts["auto_categorised"],
        suggested=counts["suggested"],
        needs_review=counts["needs_review"],
        errors=counts["errors"],
        total_processed=len(transactions),
    )

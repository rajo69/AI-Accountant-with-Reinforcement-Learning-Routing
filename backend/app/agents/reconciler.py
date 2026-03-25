"""LangGraph bank reconciliation agent.

Flow: find_candidates → score_candidates → decide → explain → save
"""
import asyncio
import logging
from datetime import date as date_type, timedelta
from decimal import Decimal
from typing import Optional, TypedDict
from uuid import UUID

from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import AuditLog, BankStatement, Transaction
from app.models.schemas import ReconcileBatchResponse

logger = logging.getLogger(__name__)

_AMOUNT_TOLERANCE = Decimal("0.01")
_DATE_WINDOW_DAYS = 7  # covers 5 business days
_AUTO_MATCH_THRESHOLD = 0.9
_SUGGEST_THRESHOLD = 0.6


class ReconcilerState(TypedDict):
    bank_statement_id: str
    bank_statement_data: dict
    candidates: list[dict]
    best_match: Optional[dict]
    match_confidence: float
    explanation: str
    status: str


# ---------------------------------------------------------------------------
# Pure scoring helpers (exported so the API detail view can reuse them)
# ---------------------------------------------------------------------------

def compute_amount_score(tx_amount: Decimal, bs_amount: Decimal) -> float:
    """Return 1.0 for exact match, 0.8 if within 1%, else 0.0."""
    diff = abs(tx_amount - bs_amount)
    if diff <= Decimal("0.001"):
        return 1.0
    if bs_amount != Decimal("0") and diff / abs(bs_amount) <= Decimal("0.01"):
        return 0.8
    return 0.0


def compute_date_score(tx_date: date_type, bs_date: date_type) -> float:
    """1.0 for same day, minus 0.15 per day, minimum 0.0."""
    day_diff = abs((tx_date - bs_date).days)
    return max(0.0, 1.0 - day_diff * 0.15)


def compute_description_score(tx_desc: str, bs_desc: str) -> float:
    """RapidFuzz token_sort_ratio normalised to 0-1 (case-insensitive)."""
    return fuzz.token_sort_ratio((tx_desc or "").lower(), (bs_desc or "").lower()) / 100.0


def compute_combined_score(
    amount_score: float, date_score: float, description_score: float
) -> float:
    """Weighted combination: amount 50%, date 20%, description 30%."""
    return amount_score * 0.5 + date_score * 0.2 + description_score * 0.3


# ---------------------------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------------------------

def build_reconciler_graph(db: AsyncSession):
    """Build and compile a LangGraph reconciliation workflow with DB injected via closure."""

    async def find_candidates(state: ReconcilerState) -> dict:
        bs_data = state["bank_statement_data"]
        org_id = UUID(bs_data["organisation_id"])
        bs_amount = Decimal(str(bs_data["amount"]))
        bs_date = date_type.fromisoformat(bs_data["date"])

        result = await db.execute(
            select(Transaction)
            .where(
                Transaction.organisation_id == org_id,
                Transaction.amount.between(
                    bs_amount - _AMOUNT_TOLERANCE,
                    bs_amount + _AMOUNT_TOLERANCE,
                ),
                Transaction.date.between(
                    bs_date - timedelta(days=_DATE_WINDOW_DAYS),
                    bs_date + timedelta(days=_DATE_WINDOW_DAYS),
                ),
                Transaction.is_reconciled == False,  # noqa: E712
            )
            .limit(10)
        )
        transactions = list(result.scalars().all())

        candidates = [
            {
                "transaction_id": str(t.id),
                "description": t.description or "",
                "date": str(t.date),
                "amount": str(t.amount),
            }
            for t in transactions
        ]
        return {"candidates": candidates}

    async def score_candidates(state: ReconcilerState) -> dict:
        bs_data = state["bank_statement_data"]
        bs_amount = Decimal(str(bs_data["amount"]))
        bs_date = date_type.fromisoformat(bs_data["date"])
        bs_desc = bs_data.get("description", "")

        scored = []
        for c in state["candidates"]:
            tx_amount = Decimal(str(c["amount"]))
            tx_date = date_type.fromisoformat(c["date"])

            amount_score = compute_amount_score(tx_amount, bs_amount)
            date_score = compute_date_score(tx_date, bs_date)
            desc_score = compute_description_score(c["description"], bs_desc)
            combined = compute_combined_score(amount_score, date_score, desc_score)

            scored.append(
                {
                    **c,
                    "amount_score": round(amount_score, 4),
                    "date_score": round(date_score, 4),
                    "description_score": round(desc_score, 4),
                    "combined_score": round(combined, 4),
                }
            )

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return {"candidates": scored}

    async def decide(state: ReconcilerState) -> dict:
        candidates = state["candidates"]

        if not candidates:
            return {
                "best_match": None,
                "match_confidence": 0.0,
                "status": "needs_review",
            }

        top = candidates[0]
        top_score = top["combined_score"]
        close_count = sum(
            1 for c in candidates if top_score - c["combined_score"] <= 0.05
        )

        if top_score >= _AUTO_MATCH_THRESHOLD and close_count == 1:
            status = "auto_matched"
            best_match = top
        elif top_score >= _SUGGEST_THRESHOLD:
            # Multiple candidates within 0.05 → ambiguous, flag for review
            status = "needs_review" if close_count > 1 else "suggested"
            best_match = top
        else:
            status = "needs_review"
            best_match = None

        return {
            "best_match": best_match,
            "match_confidence": top_score,
            "status": status,
        }

    async def explain(state: ReconcilerState) -> dict:
        best_match = state["best_match"]
        bs_data = state["bank_statement_data"]

        if not best_match:
            return {"explanation": "No suitable match found — manual review required."}

        bs_date = date_type.fromisoformat(bs_data["date"])
        tx_date = date_type.fromisoformat(best_match["date"])
        day_diff = abs((tx_date - bs_date).days)

        amount_qualifier = (
            "exactly" if best_match["amount_score"] >= 1.0 else "within 1%"
        )
        desc_pct = int(best_match["description_score"] * 100)
        day_text = f"{day_diff} day{'s' if day_diff != 1 else ''}"

        explanation = (
            f"Matched to '{best_match['description']}' on {best_match['date']}: "
            f"amount matches {amount_qualifier} "
            f"(£{abs(Decimal(best_match['amount'])):.2f}), "
            f"dated {day_text} apart, "
            f"description {desc_pct}% similar."
        )
        return {"explanation": explanation}

    async def save(state: ReconcilerState) -> dict:
        bs_id = UUID(state["bank_statement_id"])
        bs_data = state["bank_statement_data"]
        org_id = UUID(bs_data["organisation_id"])
        best_match = state["best_match"]
        final_status = state["status"]
        confidence = state["match_confidence"]
        explanation = state["explanation"]

        result = await db.execute(
            select(BankStatement).where(BankStatement.id == bs_id)
        )
        bs = result.scalar_one_or_none()

        if bs:
            if best_match:
                bs.matched_transaction_id = UUID(best_match["transaction_id"])
                bs.match_confidence = Decimal(str(round(confidence, 4)))
            bs.match_status = final_status

            # For auto-matched, also mark the transaction as reconciled
            if final_status == "auto_matched" and best_match:
                tx_result = await db.execute(
                    select(Transaction).where(
                        Transaction.id == UUID(best_match["transaction_id"])
                    )
                )
                tx = tx_result.scalar_one_or_none()
                if tx:
                    tx.is_reconciled = True

        audit = AuditLog(
            organisation_id=org_id,
            action="ai_reconcile",
            entity_type="bank_statement",
            entity_id=bs_id,
            new_value={
                "match_status": final_status,
                "matched_transaction_id": (
                    best_match["transaction_id"] if best_match else None
                ),
                "confidence": round(confidence, 4),
            },
            ai_confidence=(
                Decimal(str(round(confidence, 4))) if confidence else None
            ),
            ai_explanation=explanation,
            ai_decision_data={
                "candidates_count": len(state["candidates"]),
                "top_scores": [
                    {
                        "transaction_id": c["transaction_id"],
                        "combined_score": c["combined_score"],
                    }
                    for c in state["candidates"][:3]
                ],
            },
        )
        db.add(audit)
        await db.commit()

        return {"status": final_status}

    workflow = StateGraph(ReconcilerState)
    workflow.add_node("find_candidates", find_candidates)
    workflow.add_node("score_candidates", score_candidates)
    workflow.add_node("decide", decide)
    workflow.add_node("explain", explain)
    workflow.add_node("save", save)

    workflow.set_entry_point("find_candidates")
    workflow.add_edge("find_candidates", "score_candidates")
    workflow.add_edge("score_candidates", "decide")
    workflow.add_edge("decide", "explain")
    workflow.add_edge("explain", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def reconcile_batch(org_id: UUID, db: AsyncSession) -> ReconcileBatchResponse:
    """Reconcile all unmatched bank statements for an organisation."""
    result = await db.execute(
        select(BankStatement).where(
            BankStatement.organisation_id == org_id,
            BankStatement.match_status == "unmatched",
        )
    )
    statements = list(result.scalars().all())

    graph = build_reconciler_graph(db)
    semaphore = asyncio.Semaphore(5)
    counts: dict[str, int] = {
        "auto_matched": 0,
        "suggested": 0,
        "needs_review": 0,
        "errors": 0,
    }

    async def process_one(bs: BankStatement) -> None:
        async with semaphore:
            try:
                initial_state: ReconcilerState = {
                    "bank_statement_id": str(bs.id),
                    "bank_statement_data": {
                        "organisation_id": str(bs.organisation_id),
                        "date": str(bs.date),
                        "amount": str(bs.amount),
                        "description": bs.description or "",
                        "reference": bs.reference or "",
                    },
                    "candidates": [],
                    "best_match": None,
                    "match_confidence": 0.0,
                    "explanation": "",
                    "status": "pending",
                }
                final = await graph.ainvoke(initial_state)
                status = final.get("status", "errors")
                if status in counts:
                    counts[status] += 1
                else:
                    counts["errors"] += 1
            except Exception:
                logger.exception("Error reconciling bank statement %s", bs.id)
                counts["errors"] += 1

    await asyncio.gather(*[process_one(bs) for bs in statements])

    return ReconcileBatchResponse(
        auto_matched=counts["auto_matched"],
        suggested=counts["suggested"],
        needs_review=counts["needs_review"],
        errors=counts["errors"],
        total_processed=len(statements),
    )

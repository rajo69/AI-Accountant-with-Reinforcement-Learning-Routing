"""Bank reconciliation routes."""
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.reconciler import (
    compute_amount_score,
    compute_combined_score,
    compute_date_score,
    compute_description_score,
    reconcile_batch,
)
from app.core.database import get_db
from app.core.session import get_current_org
from app.models.database import AuditLog, BankStatement, Organisation, Transaction
from app.models.schemas import (
    AuditLogRead,
    BankStatementDetail,
    BankStatementListResponse,
    BankStatementRead,
    ManualMatchRequest,
    MatchCandidate,
    ReconcileBatchResponse,
)

router = APIRouter(prefix="/api/v1", tags=["reconcile"])

_DATE_WINDOW_DAYS = 7
_AMOUNT_TOLERANCE = Decimal("0.01")


@router.post("/reconcile", response_model=ReconcileBatchResponse)
async def trigger_reconcile(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Run AI reconciliation on all unmatched bank statements."""
    return await reconcile_batch(org.id, db)


@router.get("/bank-statements", response_model=BankStatementListResponse)
async def list_bank_statements(
    match_status: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    page: int = 1,
    page_size: int = 20,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """List bank statements with optional filters and pagination."""

    query = select(BankStatement).where(BankStatement.organisation_id == org.id)

    if match_status:
        query = query.where(BankStatement.match_status == match_status)
    if date_from:
        query = query.where(BankStatement.date >= date_from)
    if date_to:
        query = query.where(BankStatement.date <= date_to)

    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar_one()

    query = (
        query.order_by(BankStatement.date.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(query)
    items = list(result.scalars().all())

    return BankStatementListResponse(
        items=[BankStatementRead.model_validate(s) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/bank-statements/{statement_id}", response_model=BankStatementDetail)
async def get_bank_statement(
    statement_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Get a single bank statement with scored match candidates and audit history."""

    result = await db.execute(
        select(BankStatement).where(
            BankStatement.id == statement_id,
            BankStatement.organisation_id == org.id,
        )
    )
    bs = result.scalar_one_or_none()
    if not bs:
        raise HTTPException(status_code=404, detail="Bank statement not found")

    # Find and score candidate transactions
    tx_result = await db.execute(
        select(Transaction)
        .where(
            Transaction.organisation_id == org.id,
            Transaction.amount.between(
                bs.amount - _AMOUNT_TOLERANCE,
                bs.amount + _AMOUNT_TOLERANCE,
            ),
            Transaction.date.between(
                bs.date - timedelta(days=_DATE_WINDOW_DAYS),
                bs.date + timedelta(days=_DATE_WINDOW_DAYS),
            ),
        )
        .limit(10)
    )
    candidate_txs = list(tx_result.scalars().all())

    candidates = []
    for tx in candidate_txs:
        amount_score = compute_amount_score(tx.amount, bs.amount)
        date_score = compute_date_score(tx.date, bs.date)
        desc_score = compute_description_score(tx.description or "", bs.description or "")
        combined = compute_combined_score(amount_score, date_score, desc_score)
        candidates.append(
            MatchCandidate(
                transaction_id=tx.id,
                description=tx.description or "",
                date=tx.date,
                amount=tx.amount,
                amount_score=round(amount_score, 4),
                date_score=round(date_score, 4),
                description_score=round(desc_score, 4),
                combined_score=round(combined, 4),
            )
        )
    candidates.sort(key=lambda c: c.combined_score, reverse=True)

    # Load audit history
    audit_result = await db.execute(
        select(AuditLog)
        .where(
            AuditLog.entity_type == "bank_statement",
            AuditLog.entity_id == statement_id,
        )
        .order_by(AuditLog.created_at.desc())
    )
    audit_logs = list(audit_result.scalars().all())

    detail = BankStatementDetail.model_validate(bs)
    detail.candidates = candidates
    detail.audit_history = [AuditLogRead.model_validate(a) for a in audit_logs]
    return detail


@router.post("/bank-statements/{statement_id}/confirm", response_model=BankStatementRead)
async def confirm_match(
    statement_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Confirm a suggested match, marking it as reconciled."""

    result = await db.execute(
        select(BankStatement).where(
            BankStatement.id == statement_id,
            BankStatement.organisation_id == org.id,
        )
    )
    bs = result.scalar_one_or_none()
    if not bs:
        raise HTTPException(status_code=404, detail="Bank statement not found")
    if bs.match_status not in ("suggested", "auto_matched", "needs_review"):
        raise HTTPException(status_code=400, detail="Bank statement has no pending match to confirm")
    if not bs.matched_transaction_id:
        raise HTTPException(status_code=400, detail="No matched transaction to confirm")

    old_status = bs.match_status
    bs.match_status = "confirmed"

    # Mark the matched transaction as reconciled
    tx_result = await db.execute(
        select(Transaction).where(Transaction.id == bs.matched_transaction_id)
    )
    tx = tx_result.scalar_one_or_none()
    if tx:
        tx.is_reconciled = True

    audit = AuditLog(
        organisation_id=org.id,
        action="human_confirm_match",
        entity_type="bank_statement",
        entity_id=statement_id,
        old_value={"match_status": old_status},
        new_value={
            "match_status": "confirmed",
            "matched_transaction_id": str(bs.matched_transaction_id),
        },
    )
    db.add(audit)
    await db.commit()
    await db.refresh(bs)
    return BankStatementRead.model_validate(bs)


@router.post("/bank-statements/{statement_id}/unmatch", response_model=BankStatementRead)
async def unmatch_statement(
    statement_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Remove a match and reset the bank statement to unmatched."""

    result = await db.execute(
        select(BankStatement).where(
            BankStatement.id == statement_id,
            BankStatement.organisation_id == org.id,
        )
    )
    bs = result.scalar_one_or_none()
    if not bs:
        raise HTTPException(status_code=404, detail="Bank statement not found")
    if bs.match_status == "unmatched":
        raise HTTPException(status_code=400, detail="Bank statement is already unmatched")

    old_status = bs.match_status
    old_tx_id = bs.matched_transaction_id

    # Un-reconcile the previously matched transaction
    if old_tx_id:
        tx_result = await db.execute(
            select(Transaction).where(Transaction.id == old_tx_id)
        )
        tx = tx_result.scalar_one_or_none()
        if tx:
            tx.is_reconciled = False

    bs.match_status = "unmatched"
    bs.matched_transaction_id = None
    bs.match_confidence = None

    audit = AuditLog(
        organisation_id=org.id,
        action="human_unmatch",
        entity_type="bank_statement",
        entity_id=statement_id,
        old_value={
            "match_status": old_status,
            "matched_transaction_id": str(old_tx_id) if old_tx_id else None,
        },
        new_value={"match_status": "unmatched"},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(bs)
    return BankStatementRead.model_validate(bs)


@router.post("/bank-statements/{statement_id}/match", response_model=BankStatementRead)
async def manual_match(
    statement_id: UUID,
    body: ManualMatchRequest,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Manually match a bank statement to a specific transaction."""

    result = await db.execute(
        select(BankStatement).where(
            BankStatement.id == statement_id,
            BankStatement.organisation_id == org.id,
        )
    )
    bs = result.scalar_one_or_none()
    if not bs:
        raise HTTPException(status_code=404, detail="Bank statement not found")

    tx_result = await db.execute(
        select(Transaction).where(
            Transaction.id == body.transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = tx_result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    old_status = bs.match_status
    old_tx_id = bs.matched_transaction_id

    # Un-reconcile previous match if there was one
    if old_tx_id and old_tx_id != body.transaction_id:
        prev_tx_result = await db.execute(
            select(Transaction).where(Transaction.id == old_tx_id)
        )
        prev_tx = prev_tx_result.scalar_one_or_none()
        if prev_tx:
            prev_tx.is_reconciled = False

    bs.matched_transaction_id = body.transaction_id
    bs.match_status = "confirmed"
    bs.match_confidence = None  # Manual match has no AI confidence score
    tx.is_reconciled = True

    audit = AuditLog(
        organisation_id=org.id,
        action="human_manual_match",
        entity_type="bank_statement",
        entity_id=statement_id,
        old_value={
            "match_status": old_status,
            "matched_transaction_id": str(old_tx_id) if old_tx_id else None,
        },
        new_value={
            "match_status": "confirmed",
            "matched_transaction_id": str(body.transaction_id),
        },
    )
    db.add(audit)
    await db.commit()
    await db.refresh(bs)
    return BankStatementRead.model_validate(bs)

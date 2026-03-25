"""Transaction categorisation and management routes."""
from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.categoriser import categorise_batch
from app.core.database import get_db
from app.core.session import get_current_org
from app.models.database import Account, AuditLog, Organisation, Transaction
from app.models.schemas import (
    AuditLogRead,
    BatchCategoriseResponse,
    TransactionCorrectRequest,
    TransactionDetail,
    TransactionListResponse,
    TransactionRead,
)
from app.services.embedding_service import embed_transaction

router = APIRouter(prefix="/api/v1", tags=["categorise"])


@router.post("/categorise", response_model=BatchCategoriseResponse)
async def trigger_categorise(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Run AI categorisation on all uncategorised transactions."""
    return await categorise_batch(org.id, db)


@router.get("/transactions", response_model=TransactionListResponse)
async def list_transactions(
    status: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """List transactions with optional filters and pagination."""

    query = select(Transaction).where(Transaction.organisation_id == org.id)

    if status:
        query = query.where(Transaction.categorisation_status == status)
    if date_from:
        query = query.where(Transaction.date >= date_from)
    if date_to:
        query = query.where(Transaction.date <= date_to)
    if search:
        query = query.where(Transaction.description.ilike(f"%{search}%"))

    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar_one()

    query = query.order_by(Transaction.date.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = list(result.scalars().all())

    return TransactionListResponse(
        items=[TransactionRead.model_validate(t) for t in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/transactions/{transaction_id}", response_model=TransactionDetail)
async def get_transaction(
    transaction_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Get a single transaction with its full audit history."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    audit_result = await db.execute(
        select(AuditLog)
        .where(
            AuditLog.entity_type == "transaction",
            AuditLog.entity_id == transaction_id,
        )
        .order_by(AuditLog.created_at.desc())
    )
    audit_logs = list(audit_result.scalars().all())

    detail = TransactionDetail.model_validate(tx)
    detail.audit_history = [AuditLogRead.model_validate(a) for a in audit_logs]
    return detail


@router.post("/transactions/{transaction_id}/approve", response_model=TransactionRead)
async def approve_transaction(
    transaction_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Confirm a suggested category."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if tx.categorisation_status not in ("suggested", "auto_categorised"):
        raise HTTPException(status_code=400, detail="Transaction is not in a suggested state")

    old_status = tx.categorisation_status
    tx.categorisation_status = "confirmed"

    audit = AuditLog(
        organisation_id=org.id,
        action="human_approve",
        entity_type="transaction",
        entity_id=transaction_id,
        old_value={"status": old_status},
        new_value={"status": "confirmed", "category": tx.category},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(tx)
    return TransactionRead.model_validate(tx)


@router.post("/transactions/{transaction_id}/correct", response_model=TransactionRead)
async def correct_transaction(
    transaction_id: UUID,
    body: TransactionCorrectRequest,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Apply a human-corrected category and re-embed for future few-shot learning."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Validate the account exists
    account_query = select(Account).where(Account.organisation_id == org.id)
    if body.category_code:
        account_query = account_query.where(Account.code == body.category_code)
    else:
        account_query = account_query.where(Account.name == body.category)
    account_result = await db.execute(account_query)
    account = account_result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=400, detail=f"Account '{body.category}' not found in chart of accounts")

    old_category = tx.category
    old_status = tx.categorisation_status

    tx.category = account.name
    tx.category_confidence = Decimal("1.0000")
    tx.categorisation_status = "confirmed"
    tx.embedding = await embed_transaction(tx)

    audit = AuditLog(
        organisation_id=org.id,
        action="human_correct",
        entity_type="transaction",
        entity_id=transaction_id,
        old_value={"category": old_category, "status": old_status},
        new_value={"category": account.name, "status": "confirmed"},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(tx)
    return TransactionRead.model_validate(tx)


@router.post("/transactions/{transaction_id}/reject", response_model=TransactionRead)
async def reject_transaction(
    transaction_id: UUID,
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Reject a suggested category and reset to uncategorised."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.organisation_id == org.id,
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    old_category = tx.category
    old_status = tx.categorisation_status

    tx.category = None
    tx.category_confidence = None
    tx.categorisation_status = "uncategorised"

    audit = AuditLog(
        organisation_id=org.id,
        action="human_reject",
        entity_type="transaction",
        entity_id=transaction_id,
        old_value={"category": old_category, "status": old_status},
        new_value={"category": None, "status": "uncategorised"},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(tx)
    return TransactionRead.model_validate(tx)

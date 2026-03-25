from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.session import get_current_org
from app.integrations.xero_adapter import XeroAdapter
from app.models.database import Organisation, Account, Transaction, BankStatement
from app.models.schemas import SyncResponse, SyncStatus

router = APIRouter(prefix="/api/v1", tags=["sync"])


@router.post("/sync", response_model=SyncResponse)
async def trigger_sync(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Trigger a full Xero sync (accounts + transactions + bank statements)."""
    adapter = XeroAdapter(org)
    try:
        result = await adapter.full_sync(db)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Xero sync failed: {exc}") from exc
    return result


@router.get("/sync/status", response_model=SyncStatus)
async def sync_status(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Return last sync time and current record counts."""
    accounts_count = await db.scalar(
        select(func.count()).select_from(Account).where(Account.organisation_id == org.id)
    )
    transactions_count = await db.scalar(
        select(func.count()).select_from(Transaction).where(Transaction.organisation_id == org.id)
    )
    statements_count = await db.scalar(
        select(func.count()).select_from(BankStatement).where(BankStatement.organisation_id == org.id)
    )

    return SyncStatus(
        last_sync_at=org.last_sync_at,
        synced_accounts=accounts_count or 0,
        synced_transactions=transactions_count or 0,
        synced_bank_statements=statements_count or 0,
    )

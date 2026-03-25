"""GDPR compliance endpoints.

Implements the three data-subject rights most relevant to this product:

    Article 15 / 20  GET  /api/v1/gdpr/export   Right of access + data portability
    Article 17       DELETE /api/v1/gdpr/erase   Right to erasure

Every route requires a valid JWT session so the operation is scoped strictly
to the authenticated organisation.  Xero OAuth tokens are excluded from the
export because they are credentials, not personal data held about the user.
"""
import uuid
from datetime import datetime, date
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.session import get_current_org
from app.models.database import (
    Account,
    AuditLog,
    BankStatement,
    GeneratedDocument,
    Organisation,
    Transaction,
)

router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])

_TOKEN_FIELDS = {"xero_access_token", "xero_refresh_token"}
_SKIP_FIELDS = {"embedding"}  # binary vector, not useful in a human-readable export


def _coerce(value):
    """Convert non-JSON-serialisable types to plain Python equivalents."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(i) for i in value]
    return value


def _row_to_dict(row, exclude: set | None = None) -> dict:
    skip = (exclude or set()) | _SKIP_FIELDS
    return {
        col: _coerce(getattr(row, col))
        for col in row.__table__.columns.keys()
        if col not in skip
    }


@router.get("/export", summary="Export all organisation data (GDPR Art. 15 / 20)")
async def export_data(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Return all data held for the authenticated organisation as structured JSON.

    Xero OAuth credentials are excluded.  Embeddings (binary vectors) are
    excluded as they are derived data with no human-readable meaning.

    This endpoint satisfies GDPR Article 15 (right of access) and
    Article 20 (right to data portability).
    """
    accounts = (await db.execute(
        select(Account).where(Account.organisation_id == org.id)
    )).scalars().all()

    transactions = (await db.execute(
        select(Transaction).where(Transaction.organisation_id == org.id)
    )).scalars().all()

    statements = (await db.execute(
        select(BankStatement).where(BankStatement.organisation_id == org.id)
    )).scalars().all()

    audit_logs = (await db.execute(
        select(AuditLog).where(AuditLog.organisation_id == org.id)
    )).scalars().all()

    documents = (await db.execute(
        select(GeneratedDocument).where(GeneratedDocument.organisation_id == org.id)
    )).scalars().all()

    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "gdpr_basis": "GDPR Articles 15 and 20 — right of access and data portability",
        "organisation": _row_to_dict(org, exclude=_TOKEN_FIELDS),
        "accounts": [_row_to_dict(r) for r in accounts],
        "transactions": [_row_to_dict(r) for r in transactions],
        "bank_statements": [_row_to_dict(r) for r in statements],
        "audit_logs": [_row_to_dict(r) for r in audit_logs],
        "generated_documents": [_row_to_dict(r) for r in documents],
    }


@router.delete("/erase", summary="Permanently erase all organisation data (GDPR Art. 17)")
async def erase_data(
    org: Organisation = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete all data held for the authenticated organisation.

    Deletion order respects foreign-key constraints:
        1. audit_logs             (FK to org only)
        2. generated_documents    (FK to org only)
        3. bank_statements        (FK to org + nullable FK to transactions)
        4. transactions           (FK to org + nullable FK to accounts)
        5. accounts               (FK to org)
        6. organisation

    This action is irreversible.  The caller's JWT becomes invalid immediately
    because the organisation record no longer exists.

    This endpoint satisfies GDPR Article 17 (right to erasure).
    """
    org_id = org.id

    await db.execute(delete(AuditLog).where(AuditLog.organisation_id == org_id))
    await db.execute(delete(GeneratedDocument).where(GeneratedDocument.organisation_id == org_id))
    await db.execute(delete(BankStatement).where(BankStatement.organisation_id == org_id))
    await db.execute(delete(Transaction).where(Transaction.organisation_id == org_id))
    await db.execute(delete(Account).where(Account.organisation_id == org_id))
    await db.execute(delete(Organisation).where(Organisation.id == org_id))
    await db.commit()

    return {
        "status": "erased",
        "gdpr_basis": "GDPR Article 17 — right to erasure",
        "message": "All data for this organisation has been permanently deleted.",
    }

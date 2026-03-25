import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict


class OrganisationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    xero_tenant_id: str
    xero_token_expires_at: Optional[datetime] = None
    last_sync_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class AccountRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    xero_id: str
    code: Optional[str] = None
    name: str
    type: str
    tax_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    account_id: Optional[uuid.UUID] = None
    xero_id: str
    date: date
    amount: Decimal
    description: str
    reference: Optional[str] = None
    category: Optional[str] = None
    category_confidence: Optional[Decimal] = None
    categorisation_status: str
    is_reconciled: bool
    created_at: datetime
    updated_at: datetime


class TransactionUpdate(BaseModel):
    category: Optional[str] = None
    category_confidence: Optional[Decimal] = None
    categorisation_status: Optional[str] = None
    is_reconciled: Optional[bool] = None


class BankStatementRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    xero_id: str
    date: date
    amount: Decimal
    description: str
    reference: Optional[str] = None
    matched_transaction_id: Optional[uuid.UUID] = None
    match_confidence: Optional[Decimal] = None
    match_status: str
    created_at: datetime
    updated_at: datetime


class AuditLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    action: str
    entity_type: str
    entity_id: uuid.UUID
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None
    ai_model: Optional[str] = None
    ai_confidence: Optional[Decimal] = None
    ai_explanation: Optional[str] = None
    ai_decision_data: Optional[dict] = None
    created_at: datetime


class SyncResponse(BaseModel):
    status: str
    synced_accounts: int = 0
    synced_transactions: int = 0
    synced_bank_statements: int = 0
    message: Optional[str] = None


class CategoriseResponse(BaseModel):
    transaction_id: uuid.UUID
    category: str
    confidence: Decimal
    explanation: str
    status: str


class ReconcileResponse(BaseModel):
    bank_statement_id: uuid.UUID
    matched_transaction_id: Optional[uuid.UUID] = None
    confidence: Optional[Decimal] = None
    status: str
    explanation: Optional[str] = None


class DashboardSummary(BaseModel):
    total_accounts: int
    total_transactions: int
    uncategorised_count: int
    unreconciled_count: int
    last_sync_at: Optional[datetime] = None
    organisation_name: Optional[str] = None


class SyncStatus(BaseModel):
    last_sync_at: Optional[datetime] = None
    synced_accounts: int = 0
    synced_transactions: int = 0
    synced_bank_statements: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


class BatchCategoriseResponse(BaseModel):
    auto_categorised: int = 0
    suggested: int = 0
    needs_review: int = 0
    errors: int = 0
    total_processed: int = 0


class TransactionListResponse(BaseModel):
    items: list[TransactionRead]
    total: int
    page: int
    page_size: int


class TransactionDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    account_id: Optional[uuid.UUID] = None
    xero_id: str
    date: date
    amount: Decimal
    description: str
    reference: Optional[str] = None
    category: Optional[str] = None
    category_confidence: Optional[Decimal] = None
    categorisation_status: str
    is_reconciled: bool
    created_at: datetime
    updated_at: datetime
    audit_history: list[AuditLogRead] = []


class TransactionCorrectRequest(BaseModel):
    category: str
    category_code: Optional[str] = None


class MatchCandidate(BaseModel):
    transaction_id: uuid.UUID
    description: str
    date: date
    amount: Decimal
    amount_score: float
    date_score: float
    description_score: float
    combined_score: float


class BankStatementDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    xero_id: str
    date: date
    amount: Decimal
    description: str
    reference: Optional[str] = None
    matched_transaction_id: Optional[uuid.UUID] = None
    match_confidence: Optional[Decimal] = None
    match_status: str
    created_at: datetime
    updated_at: datetime
    candidates: list[MatchCandidate] = []
    audit_history: list[AuditLogRead] = []


class BankStatementListResponse(BaseModel):
    items: list[BankStatementRead]
    total: int
    page: int
    page_size: int


class ReconcileBatchResponse(BaseModel):
    auto_matched: int = 0
    suggested: int = 0
    needs_review: int = 0
    errors: int = 0
    total_processed: int = 0


class ManualMatchRequest(BaseModel):
    transaction_id: uuid.UUID


class GeneratedDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organisation_id: uuid.UUID
    template: str
    period_start: date
    period_end: date
    ai_model: str
    figures: dict
    generated_at: datetime


class DocumentGenerateRequest(BaseModel):
    template: str
    period_start: date
    period_end: date


class DocumentGenerateResponse(BaseModel):
    document_id: uuid.UUID
    template: str
    period_start: date
    period_end: date
    transaction_count: int
    generated_at: str

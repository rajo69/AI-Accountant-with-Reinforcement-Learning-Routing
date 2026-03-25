import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    String, Text, Boolean, Numeric, Date, DateTime, ForeignKey, Index,
    func, UUID as SA_UUID
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import mapped_column, Mapped, relationship
from pgvector.sqlalchemy import Vector

from app.core.database import Base


class Organisation(Base):
    __tablename__ = "organisations"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    xero_tenant_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    xero_access_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    xero_refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    xero_token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    accounts: Mapped[list["Account"]] = relationship("Account", back_populates="organisation")
    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="organisation")
    bank_statements: Mapped[list["BankStatement"]] = relationship("BankStatement", back_populates="organisation")
    audit_logs: Mapped[list["AuditLog"]] = relationship("AuditLog", back_populates="organisation")
    generated_documents: Mapped[list["GeneratedDocument"]] = relationship("GeneratedDocument", back_populates="organisation")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    xero_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    tax_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organisation: Mapped["Organisation"] = relationship("Organisation", back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=True)
    xero_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    reference: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    category_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    categorisation_status: Mapped[str] = mapped_column(String(50), default="uncategorised", nullable=False)
    is_reconciled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organisation: Mapped["Organisation"] = relationship("Organisation", back_populates="transactions")
    account: Mapped[Optional["Account"]] = relationship("Account", back_populates="transactions")
    bank_statements: Mapped[list["BankStatement"]] = relationship("BankStatement", back_populates="matched_transaction")

    __table_args__ = (
        Index("ix_transactions_date", "date"),
        Index("ix_transactions_categorisation_status", "categorisation_status"),
    )


class BankStatement(Base):
    __tablename__ = "bank_statements"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    xero_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    reference: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    matched_transaction_id: Mapped[Optional[uuid.UUID]] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=True)
    match_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    match_status: Mapped[str] = mapped_column(String(50), default="unmatched", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organisation: Mapped["Organisation"] = relationship("Organisation", back_populates="bank_statements")
    matched_transaction: Mapped[Optional["Transaction"]] = relationship("Transaction", back_populates="bank_statements")

    __table_args__ = (
        Index("ix_bank_statements_match_status", "match_status"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), nullable=False)
    old_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    ai_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ai_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    ai_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_decision_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    organisation: Mapped["Organisation"] = relationship("Organisation", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
    )


class GeneratedDocument(Base):
    __tablename__ = "generated_documents"

    id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(SA_UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    template: Mapped[str] = mapped_column(String(100), nullable=False)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    ai_model: Mapped[str] = mapped_column(String(100), nullable=False)
    figures: Mapped[dict] = mapped_column(JSONB, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    organisation: Mapped["Organisation"] = relationship("Organisation", back_populates="generated_documents")

    __table_args__ = (
        Index("ix_generated_documents_org_id", "organisation_id"),
    )

"""add generated_documents table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "generated_documents",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("organisation_id", sa.UUID(as_uuid=True), sa.ForeignKey("organisations.id"), nullable=False),
        sa.Column("template", sa.String(100), nullable=False),
        sa.Column("period_start", sa.Date, nullable=False),
        sa.Column("period_end", sa.Date, nullable=False),
        sa.Column("ai_model", sa.String(100), nullable=False),
        sa.Column("figures", postgresql.JSONB, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_generated_documents_org_id", "generated_documents", ["organisation_id"])


def downgrade() -> None:
    op.drop_index("ix_generated_documents_org_id", table_name="generated_documents")
    op.drop_table("generated_documents")

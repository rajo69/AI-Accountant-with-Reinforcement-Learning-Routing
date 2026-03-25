"""resize embedding vector to 384

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('transactions', 'embedding')
    op.add_column('transactions', sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=384), nullable=True))


def downgrade() -> None:
    op.drop_column('transactions', 'embedding')
    op.add_column('transactions', sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True))

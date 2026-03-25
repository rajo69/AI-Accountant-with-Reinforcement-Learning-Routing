"""Embedding service using sentence-transformers (all-MiniLM-L6-v2, 384 dims)."""
import asyncio
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Transaction

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed_text_sync(text: str) -> list[float]:
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


async def embed_text(text: str) -> list[float]:
    """Embed a text string into a 384-dim vector. Runs in a thread to avoid blocking."""
    return await asyncio.to_thread(_embed_text_sync, text)


async def embed_transaction(transaction: Transaction) -> list[float]:
    """Create an embedding from transaction fields."""
    parts = [str(transaction.date), transaction.description]
    if transaction.reference:
        parts.append(transaction.reference)
    parts.append(str(transaction.amount))
    text = " ".join(parts)
    return await embed_text(text)


async def find_similar_transactions(
    text: str,
    org_id: UUID,
    db: AsyncSession,
    limit: int = 5,
) -> list[Transaction]:
    """Find categorised transactions similar to the given text via cosine distance."""
    query_vector = await embed_text(text)
    result = await db.execute(
        select(Transaction)
        .where(
            Transaction.organisation_id == org_id,
            Transaction.embedding.isnot(None),
            Transaction.categorisation_status.in_(["auto_categorised", "confirmed"]),
        )
        .order_by(Transaction.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return list(result.scalars().all())


async def backfill_embeddings(org_id: UUID, db: AsyncSession) -> int:
    """Embed all transactions for an org that don't have embeddings yet."""
    result = await db.execute(
        select(Transaction).where(
            Transaction.organisation_id == org_id,
            Transaction.embedding.is_(None),
        )
    )
    transactions = list(result.scalars().all())
    for tx in transactions:
        tx.embedding = await embed_transaction(tx)
    if transactions:
        await db.commit()
    return len(transactions)

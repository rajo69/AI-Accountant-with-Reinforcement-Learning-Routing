"""Tests for Phase 3: transaction categorisation agent and API routes."""
import uuid
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import BatchCategoriseResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transaction(
    org_id: uuid.UUID | None = None,
    status: str = "uncategorised",
    category: str | None = None,
) -> MagicMock:
    tx = MagicMock()
    tx.id = uuid.uuid4()
    tx.organisation_id = org_id or uuid.uuid4()
    tx.xero_id = "xero-tx-001"
    tx.date = date(2026, 1, 15)
    tx.amount = Decimal("250.00")
    tx.description = "OFFICE SUPPLIES LTD"
    tx.reference = "INV-001"
    tx.category = category
    tx.category_confidence = Decimal("0.9200") if category else None
    tx.categorisation_status = status
    tx.is_reconciled = False
    tx.account_id = None
    tx.embedding = None
    tx.created_at = MagicMock()
    tx.updated_at = MagicMock()
    return tx


def _make_org() -> MagicMock:
    org = MagicMock()
    org.id = uuid.uuid4()
    org.name = "Test Ltd"
    org.xero_tenant_id = "tenant-001"
    org.last_sync_at = None
    return org


# ---------------------------------------------------------------------------
# Unit tests — categorise_batch result structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_categorise_batch_returns_correct_schema():
    """categorise_batch should return a BatchCategoriseResponse with correct counts."""
    org_id = uuid.uuid4()
    mock_db = AsyncMock()

    # No uncategorised transactions → all zeros
    empty_result = MagicMock()
    empty_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=empty_result)

    from app.agents.categoriser import categorise_batch

    response = await categorise_batch(org_id, mock_db)
    assert isinstance(response, BatchCategoriseResponse)
    assert response.total_processed == 0
    assert response.auto_categorised == 0
    assert response.suggested == 0
    assert response.needs_review == 0
    assert response.errors == 0


@pytest.mark.asyncio
async def test_categorise_batch_processes_transactions():
    """categorise_batch should invoke the graph for each uncategorised transaction."""
    org_id = uuid.uuid4()
    mock_db = AsyncMock()

    tx1 = _make_transaction(org_id=org_id)
    tx2 = _make_transaction(org_id=org_id)

    txs_result = MagicMock()
    txs_result.scalars.return_value.all.return_value = [tx1, tx2]

    mock_db.execute = AsyncMock(return_value=txs_result)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    fake_graph_result = {"status": "auto_categorised"}

    with patch("app.agents.categoriser.build_categoriser_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=fake_graph_result)
        mock_build.return_value = mock_graph

        from app.agents.categoriser import categorise_batch

        response = await categorise_batch(org_id, mock_db)

    assert response.total_processed == 2
    assert response.auto_categorised == 2
    assert mock_graph.ainvoke.call_count == 2


# ---------------------------------------------------------------------------
# Unit tests — embedding service
# ---------------------------------------------------------------------------

def test_embed_text_returns_list():
    """embed_text should return a list of floats via the sync helper."""
    with patch("app.services.embedding_service._get_model") as mock_get_model:
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_get_model.return_value = mock_model

        from app.services.embedding_service import _embed_text_sync

        result = _embed_text_sync("office supplies")
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Integration tests — FastAPI routes registered
# ---------------------------------------------------------------------------

def test_categorise_routes_registered():
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/categorise" in paths
    assert "/api/v1/transactions" in paths


def test_transactions_id_routes_registered():
    client = TestClient(app)
    response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/api/v1/transactions/{transaction_id}" in paths
    assert "/api/v1/transactions/{transaction_id}/approve" in paths
    assert "/api/v1/transactions/{transaction_id}/correct" in paths
    assert "/api/v1/transactions/{transaction_id}/reject" in paths


# ---------------------------------------------------------------------------
# Integration tests — approve / reject with mocked DB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_transaction_updates_status():
    """Approving a suggested transaction should set status to confirmed."""
    org = _make_org()
    tx = _make_transaction(org_id=org.id, status="suggested", category="Office Supplies")

    mock_db = AsyncMock()

    tx_result = MagicMock()
    tx_result.scalar_one_or_none.return_value = tx

    mock_db.execute = AsyncMock(return_value=tx_result)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    from app.api.v1.categorise import approve_transaction

    result = await approve_transaction(transaction_id=tx.id, org=org, db=mock_db)
    assert tx.categorisation_status == "confirmed"
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_reject_transaction_resets_to_uncategorised():
    """Rejecting a transaction should clear category and reset status."""
    org = _make_org()
    tx = _make_transaction(org_id=org.id, status="suggested", category="Office Supplies")

    mock_db = AsyncMock()

    tx_result = MagicMock()
    tx_result.scalar_one_or_none.return_value = tx

    mock_db.execute = AsyncMock(return_value=tx_result)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    from app.api.v1.categorise import reject_transaction

    await reject_transaction(transaction_id=tx.id, org=org, db=mock_db)
    assert tx.categorisation_status == "uncategorised"
    assert tx.category is None
    assert tx.category_confidence is None
    mock_db.commit.assert_called_once()

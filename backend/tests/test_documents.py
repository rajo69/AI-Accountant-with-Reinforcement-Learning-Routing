"""Tests for document generation endpoints and service helpers."""
import uuid
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.session import get_current_org
from app.main import app
from app.services.document_service import (
    ManagementLetterNarrative,
    _calculate_figures,
    _render_html,
)


# ── Unit tests: financial calculations (no DB, no LLM) ───────────────────────

class _FakeTx:
    def __init__(self, amount, description="desc", cat=None, date_val=date(2026, 1, 15)):
        self.amount = Decimal(str(amount))
        self.description = description
        self.category = cat
        self.date = date_val
        self.reference = None


def test_calculate_figures_basic():
    transactions = [
        _FakeTx(1000.00, cat="Sales"),
        _FakeTx(500.00, cat="Sales"),
        _FakeTx(-200.00, cat="Office Supplies"),
        _FakeTx(-100.00, cat="Travel"),
    ]
    figures = _calculate_figures(transactions, date(2026, 1, 1), date(2026, 1, 31))

    assert figures["total_income"] == pytest.approx(1500.00)
    assert figures["total_expenses_abs"] == pytest.approx(300.00)
    assert figures["net"] == pytest.approx(1200.00)
    assert figures["transaction_count"] == 4
    assert len(figures["top_expense_categories"]) == 2
    assert len(figures["largest_transactions"]) == 4


def test_calculate_figures_empty():
    figures = _calculate_figures([], date(2026, 1, 1), date(2026, 1, 31))
    assert figures["total_income"] == 0.0
    assert figures["total_expenses_abs"] == 0.0
    assert figures["net"] == 0.0
    assert figures["transaction_count"] == 0


def test_calculate_figures_net_loss():
    transactions = [_FakeTx(-500.00, cat="Rent"), _FakeTx(100.00, cat="Sales")]
    figures = _calculate_figures(transactions, date(2026, 1, 1), date(2026, 1, 31))
    assert figures["net"] == pytest.approx(-400.00)


def test_calculate_figures_top_categories_limit():
    """Only top 5 expense categories should be returned."""
    transactions = [_FakeTx(-10.00, cat=f"Cat{i}") for i in range(10)]
    figures = _calculate_figures(transactions, date(2026, 1, 1), date(2026, 1, 31))
    assert len(figures["top_expense_categories"]) == 5


def test_calculate_figures_uncategorised_grouping():
    """Uncategorised transactions should be grouped under 'Uncategorised'."""
    transactions = [_FakeTx(-100.00, cat=None), _FakeTx(-200.00, cat=None)]
    figures = _calculate_figures(transactions, date(2026, 1, 1), date(2026, 1, 31))
    names = [c["name"] for c in figures["top_expense_categories"]]
    assert "Uncategorised" in names


# ── Unit test: HTML rendering (no LLM, no PDF) ───────────────────────────────

def test_render_html_produces_html():
    figures = _calculate_figures(
        [_FakeTx(1000.00, cat="Sales"), _FakeTx(-300.00, cat="Rent")],
        date(2026, 1, 1),
        date(2026, 3, 31),
    )
    narrative = ManagementLetterNarrative(
        executive_summary="Good quarter overall.",
        income_analysis="Income steady.",
        expense_analysis="Rent is the top expense.",
        cash_flow_observations="Cash flow positive.",
        recommendations="Consider reducing office costs.",
    )
    html = _render_html(figures, narrative, "ACME Ltd", date(2026, 1, 1), date(2026, 3, 31))
    assert "<!DOCTYPE html>" in html
    assert "ACME Ltd" in html
    assert "Management Letter" in html
    assert "Good quarter overall." in html
    assert "AI-Assisted Draft" in html


# ── Integration tests: API endpoints (no DB, no LLM, no PDF) ─────────────────

@pytest.mark.asyncio
async def test_list_documents_no_org():
    """Without a connected org, GET /api/v1/documents should return 401."""
    from fastapi import HTTPException

    def raise_401():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_org] = raise_401
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.pop(get_current_org, None)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_generate_document_unsupported_template():
    """Requesting an unsupported template should return 400."""
    fake_org = MagicMock()
    fake_org.id = uuid.uuid4()
    fake_org.name = "Test Org"

    app.dependency_overrides[get_current_org] = lambda: fake_org
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/documents/generate",
                json={"template": "unknown_template", "period_start": "2026-01-01", "period_end": "2026-03-31"},
            )
    finally:
        app.dependency_overrides.pop(get_current_org, None)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_generate_document_invalid_period():
    """period_start after period_end should return 400."""
    fake_org = MagicMock()
    fake_org.id = uuid.uuid4()

    app.dependency_overrides[get_current_org] = lambda: fake_org
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/documents/generate",
                json={"template": "management_letter", "period_start": "2026-03-31", "period_end": "2026-01-01"},
            )
    finally:
        app.dependency_overrides.pop(get_current_org, None)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_generate_document_happy_path():
    """With mocked LLM and WeasyPrint, POST /api/v1/documents/generate should return a PDF."""
    fake_org = MagicMock()
    fake_org.id = uuid.uuid4()
    fake_org.name = "Test Org"

    fake_pdf = b"%PDF-1.4 fake pdf content"

    app.dependency_overrides[get_current_org] = lambda: fake_org
    try:
        with patch(
            "app.api.v1.documents.generate_management_letter",
            new=AsyncMock(
                return_value=(
                    fake_pdf,
                    {
                        "document_id": str(uuid.uuid4()),
                        "template": "management_letter",
                        "period_start": "2026-01-01",
                        "period_end": "2026-03-31",
                        "transaction_count": 42,
                        "generated_at": "2026-03-22T12:00:00",
                    },
                )
            ),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/documents/generate",
                    json={"template": "management_letter", "period_start": "2026-01-01", "period_end": "2026-03-31"},
                )
    finally:
        app.dependency_overrides.pop(get_current_org, None)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == fake_pdf

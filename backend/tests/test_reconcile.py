"""Tests for bank reconciliation agent and API routes."""
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.agents.reconciler import (
    compute_amount_score,
    compute_combined_score,
    compute_date_score,
    compute_description_score,
)
from app.main import app
from app.core.database import get_db
from app.core.session import get_current_org


# ---------------------------------------------------------------------------
# Pure scoring function tests — no DB needed
# ---------------------------------------------------------------------------

class TestComputeAmountScore:
    def test_exact_match(self):
        assert compute_amount_score(Decimal("100.00"), Decimal("100.00")) == 1.0

    def test_exact_match_within_rounding(self):
        # £0.001 difference — still exact
        assert compute_amount_score(Decimal("100.001"), Decimal("100.00")) == 1.0

    def test_within_one_percent(self):
        # £100.50 vs £100.00 — 0.5% difference
        score = compute_amount_score(Decimal("100.50"), Decimal("100.00"))
        assert score == 0.8

    def test_outside_tolerance(self):
        # £110 vs £100 — 10% difference
        assert compute_amount_score(Decimal("110.00"), Decimal("100.00")) == 0.0

    def test_negative_amounts_exact(self):
        # Debit transactions are negative
        assert compute_amount_score(Decimal("-50.00"), Decimal("-50.00")) == 1.0

    def test_negative_amounts_within_percent(self):
        score = compute_amount_score(Decimal("-50.40"), Decimal("-50.00"))
        assert score == 0.8

    def test_zero_amount(self):
        # Zero amount — avoid division by zero
        assert compute_amount_score(Decimal("0.00"), Decimal("0.00")) == 1.0

    def test_zero_base_nonzero_tx(self):
        # bs_amount is 0 but tx differs by more than 0.001 — no match
        assert compute_amount_score(Decimal("0.005"), Decimal("0.00")) == 0.0


class TestComputeDateScore:
    def test_same_day(self):
        d = date(2024, 1, 15)
        assert compute_date_score(d, d) == 1.0

    def test_one_day_apart(self):
        score = compute_date_score(date(2024, 1, 16), date(2024, 1, 15))
        assert abs(score - 0.85) < 0.001  # 1.0 - 0.15

    def test_three_days_apart(self):
        score = compute_date_score(date(2024, 1, 18), date(2024, 1, 15))
        assert abs(score - 0.55) < 0.001  # 1.0 - 3*0.15

    def test_seven_days_apart(self):
        score = compute_date_score(date(2024, 1, 22), date(2024, 1, 15))
        # 1.0 - 7*0.15 = -0.05 → clamped to 0.0
        assert score == 0.0

    def test_minimum_zero(self):
        # Far apart dates — never goes negative
        score = compute_date_score(date(2024, 3, 1), date(2024, 1, 1))
        assert score == 0.0

    def test_order_independent(self):
        # Date score is symmetric
        d1, d2 = date(2024, 1, 10), date(2024, 1, 13)
        assert compute_date_score(d1, d2) == compute_date_score(d2, d1)


class TestComputeDescriptionScore:
    def test_identical(self):
        assert compute_description_score("AMAZON.CO.UK", "AMAZON.CO.UK") == 1.0

    def test_case_insensitive(self):
        # RapidFuzz is case-sensitive by default but token_sort handles reordering
        score = compute_description_score("amazon co uk", "AMAZON CO UK")
        assert score > 0.9

    def test_partial_match(self):
        score = compute_description_score("TESCO METRO LONDON", "TESCO STORES")
        assert 0 < score < 1.0

    def test_completely_different(self):
        score = compute_description_score("AMAZON", "BARCLAYS BANK")
        assert score < 0.5

    def test_empty_strings(self):
        # Empty strings should not raise
        score = compute_description_score("", "")
        assert score == 1.0  # RapidFuzz returns 100 for two empty strings

    def test_one_empty(self):
        score = compute_description_score("AMAZON", "")
        assert score == 0.0


class TestComputeCombinedScore:
    def test_perfect_match(self):
        score = compute_combined_score(1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 0.001

    def test_weights(self):
        # amount=1.0, date=0.0, description=0.0 → 0.5
        assert abs(compute_combined_score(1.0, 0.0, 0.0) - 0.5) < 0.001
        # amount=0.0, date=1.0, description=0.0 → 0.2
        assert abs(compute_combined_score(0.0, 1.0, 0.0) - 0.2) < 0.001
        # amount=0.0, date=0.0, description=1.0 → 0.3
        assert abs(compute_combined_score(0.0, 0.0, 1.0) - 0.3) < 0.001

    def test_typical_good_match(self):
        # Exact amount, same day, 80% description match
        score = compute_combined_score(1.0, 1.0, 0.8)
        assert score > 0.85  # should qualify for auto_match

    def test_below_suggest_threshold(self):
        # Loose match
        score = compute_combined_score(0.0, 0.7, 0.5)
        assert score < 0.6


# ---------------------------------------------------------------------------
# API endpoint tests (mocked DB)
# ---------------------------------------------------------------------------

def make_mock_db():
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def client():
    return TestClient(app)


class TestReconcileEndpoints:
    def test_trigger_reconcile_no_org(self, client):
        # No bearer token → get_current_org raises 401 before any DB call
        response = client.post("/api/v1/reconcile")
        assert response.status_code == 401

    def test_list_bank_statements_no_org(self, client):
        response = client.get("/api/v1/bank-statements")
        assert response.status_code == 401

    def test_get_bank_statement_not_found(self, client):
        from app.models.database import Organisation
        mock_org = MagicMock(spec=Organisation)
        mock_org.id = uuid4()

        mock_db = make_mock_db()
        mock_bs_result = MagicMock()
        mock_bs_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_bs_result

        app.dependency_overrides[get_current_org] = lambda: mock_org
        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            response = client.get(f"/api/v1/bank-statements/{uuid4()}")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.pop(get_current_org, None)
            app.dependency_overrides.pop(get_db, None)

    def test_confirm_match_no_org(self, client):
        response = client.post(f"/api/v1/bank-statements/{uuid4()}/confirm")
        assert response.status_code == 401

    def test_unmatch_no_org(self, client):
        response = client.post(f"/api/v1/bank-statements/{uuid4()}/unmatch")
        assert response.status_code == 401

    def test_manual_match_no_org(self, client):
        response = client.post(
            f"/api/v1/bank-statements/{uuid4()}/match",
            json={"transaction_id": str(uuid4())},
        )
        assert response.status_code == 401

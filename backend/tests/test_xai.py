"""Tests for the XAI fuzzy engine and explainer."""
import pytest
from app.xai.fuzzy_engine import compute_risk_score, _tri_mf


# ── Triangular MF ────────────────────────────────────────────────────────────


def test_tri_mf_peak():
    assert _tri_mf(0.0, 0.5, 1.0, 0.5) == 1.0


def test_tri_mf_below_start():
    assert _tri_mf(0.2, 0.5, 0.8, 0.1) == 0.0


def test_tri_mf_above_end():
    assert _tri_mf(0.2, 0.5, 0.8, 0.9) == 0.0


def test_tri_mf_rising():
    val = _tri_mf(0.0, 0.5, 1.0, 0.25)
    assert 0.0 < val < 1.0


def test_tri_mf_falling():
    val = _tri_mf(0.0, 0.5, 1.0, 0.75)
    assert 0.0 < val < 1.0


# ── Fuzzy risk scoring ───────────────────────────────────────────────────────


def test_low_risk_scenario():
    result = compute_risk_score(
        amount_deviation=0.05,  # very close to average
        vendor_frequency=0.9,   # very frequent vendor
        time_pattern=0.9,       # normal working day
    )
    assert result["risk_label"] == "low"
    assert result["risk_score"] < 0.35
    assert len(result["fired_rules"]) > 0


def test_high_risk_scenario():
    result = compute_risk_score(
        amount_deviation=0.95,  # far from average
        vendor_frequency=0.05,  # very rare vendor
        time_pattern=0.1,       # unusual timing (weekend)
    )
    assert result["risk_label"] == "high"
    assert result["risk_score"] >= 0.65


def test_medium_risk_scenario():
    result = compute_risk_score(
        amount_deviation=0.5,
        vendor_frequency=0.5,
        time_pattern=0.5,
    )
    assert result["risk_label"] == "medium"


def test_result_keys():
    result = compute_risk_score(0.3, 0.6, 0.8)
    assert set(result.keys()) == {"risk_score", "risk_label", "fired_rules", "input_values"}


def test_input_values_stored():
    result = compute_risk_score(0.2, 0.7, 0.9)
    iv = result["input_values"]
    assert iv["amount_deviation"] == pytest.approx(0.2)
    assert iv["vendor_frequency"] == pytest.approx(0.7)
    assert iv["time_pattern"] == pytest.approx(0.9)


def test_risk_score_in_range():
    for ad, vf, tp in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]:
        result = compute_risk_score(ad, vf, tp)
        assert 0.0 <= result["risk_score"] <= 1.0


def test_clamps_inputs_out_of_range():
    result = compute_risk_score(-0.5, 1.5, 2.0)
    assert result["risk_score"] is not None  # should not raise


def test_fired_rules_are_strings():
    result = compute_risk_score(0.8, 0.1, 0.1)
    for rule in result["fired_rules"]:
        assert isinstance(rule, str)
        assert "IF" in rule and "THEN" in rule


# ── Explainer fallback ────────────────────────────────────────────────────────


def test_llm_fallback_explain():
    from app.xai.explainer import _llm_fallback

    tx = {"amount": "123.45", "description": "AMAZON MARKETPLACE"}
    pred = {"reasoning": "Online retailer purchase", "category_name": "Office Supplies"}

    result = _llm_fallback(tx, pred)
    assert result["model_type"] == "llm"
    assert len(result["top_features"]) > 0
    assert result["explanation_text"] == "Online retailer purchase"
    for f in result["top_features"]:
        assert "name" in f and "value" in f and "contribution" in f

"""Simpful fuzzy inference system for transaction risk scoring.

Inputs
------
amount_deviation : 0-1  — how far the amount deviates from the category average
vendor_frequency : 0-1  — how often this vendor appears (0=rare, 1=very frequent)
time_pattern     : 0-1  — how typical the transaction timing is (0=unusual, 1=normal)

Output
------
risk : 0-1  — overall risk level (rendered as low / medium / high label)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Membership-function helpers (triangular) ──────────────────────────────────


def _tri_mf(a: float, b: float, c: float, x: float) -> float:
    """Evaluate a triangular membership function at x."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


# Input membership-function definitions ─────────────────────────────────────

_AMOUNT_MFS = {
    "low":    (0.0, 0.0, 0.4),
    "medium": (0.2, 0.5, 0.8),
    "high":   (0.6, 1.0, 1.0),
}

_VENDOR_MFS = {
    "rare":       (0.0, 0.0, 0.4),
    "occasional": (0.2, 0.5, 0.8),
    "frequent":   (0.6, 1.0, 1.0),
}

_TIME_MFS = {
    "unusual": (0.0, 0.0, 0.6),
    "normal":  (0.4, 1.0, 1.0),
}

# Output membership-function definitions ─────────────────────────────────────

_RISK_MFS = {
    "low":    (0.0, 0.0, 0.4),
    "medium": (0.2, 0.5, 0.8),
    "high":   (0.6, 1.0, 1.0),
}

# Rule base ──────────────────────────────────────────────────────────────────
# Each rule: (antecedents as list of (var, term) pairs, consequent risk term, text)

_RULES: list[tuple[list[tuple[str, str]], str, str]] = [
    (
        [("amount_deviation", "high"), ("vendor_frequency", "rare")],
        "high",
        "IF amount deviation IS high AND vendor frequency IS rare THEN risk IS high",
    ),
    (
        [("amount_deviation", "high"), ("vendor_frequency", "frequent")],
        "medium",
        "IF amount deviation IS high AND vendor frequency IS frequent THEN risk IS medium",
    ),
    (
        [("amount_deviation", "low"), ("vendor_frequency", "frequent")],
        "low",
        "IF amount deviation IS low AND vendor frequency IS frequent THEN risk IS low",
    ),
    (
        [("amount_deviation", "low"), ("vendor_frequency", "rare")],
        "medium",
        "IF amount deviation IS low AND vendor frequency IS rare THEN risk IS medium",
    ),
    (
        [("amount_deviation", "medium")],
        "medium",
        "IF amount deviation IS medium THEN risk IS medium",
    ),
    (
        [("time_pattern", "unusual"), ("vendor_frequency", "rare")],
        "high",
        "IF time pattern IS unusual AND vendor frequency IS rare THEN risk IS high",
    ),
    (
        [("time_pattern", "normal"), ("amount_deviation", "low")],
        "low",
        "IF time pattern IS normal AND amount deviation IS low THEN risk IS low",
    ),
    (
        [("amount_deviation", "high"), ("time_pattern", "unusual")],
        "high",
        "IF amount deviation IS high AND time pattern IS unusual THEN risk IS high",
    ),
]

_ALL_MFS: dict[str, dict] = {
    "amount_deviation": _AMOUNT_MFS,
    "vendor_frequency": _VENDOR_MFS,
    "time_pattern": _TIME_MFS,
}


@dataclass
class RiskResult:
    risk_score: float
    risk_label: str
    fired_rules: list[str]
    input_values: dict[str, float]


def _evaluate_condition(var: str, term: str, inputs: dict[str, float]) -> float:
    """Return the membership degree of an input for a given (var, term) pair."""
    val = inputs.get(var, 0.5)
    params = _ALL_MFS[var][term]
    return _tri_mf(*params, val)


def _defuzzify_centroid(activation_pairs: list[tuple[str, float]]) -> float:
    """Simple centroid defuzzification using representative crisp values per term."""
    centroids = {"low": 0.15, "medium": 0.5, "high": 0.85}
    num = sum(centroids[term] * strength for term, strength in activation_pairs)
    den = sum(strength for _, strength in activation_pairs)
    return num / den if den > 0 else 0.5


def compute_risk_score(
    amount_deviation: float,
    vendor_frequency: float,
    time_pattern: float,
    fired_threshold: float = 0.05,
) -> dict:
    """Run Mamdani-style fuzzy inference and return a RiskResult dict.

    All inputs should be in [0, 1].
    """
    amount_deviation = max(0.0, min(1.0, amount_deviation))
    vendor_frequency = max(0.0, min(1.0, vendor_frequency))
    time_pattern = max(0.0, min(1.0, time_pattern))

    inputs = {
        "amount_deviation": amount_deviation,
        "vendor_frequency": vendor_frequency,
        "time_pattern": time_pattern,
    }

    # Evaluate each rule
    activation_pairs: list[tuple[str, float]] = []
    fired_rule_texts: list[str] = []

    for antecedents, consequent, rule_text in _RULES:
        # Minimum (AND) of all antecedent memberships
        strength = min(
            _evaluate_condition(var, term, inputs) for var, term in antecedents
        )
        if strength >= fired_threshold:
            activation_pairs.append((consequent, strength))
            fired_rule_texts.append(rule_text)

    if not activation_pairs:
        # No rules fired — default to medium risk
        activation_pairs = [("medium", 0.5)]

    risk_score = _defuzzify_centroid(activation_pairs)

    if risk_score < 0.35:
        risk_label = "low"
    elif risk_score < 0.65:
        risk_label = "medium"
    else:
        risk_label = "high"

    return {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "fired_rules": fired_rule_texts,
        "input_values": inputs,
    }


# ── Helpers for computing inputs from transaction data ───────────────────────


async def compute_fuzzy_inputs(
    transaction: dict,
    prediction: dict,
    org_id,
    db,
) -> dict[str, float]:
    """Compute normalised fuzzy input values from a transaction and its DB context."""
    from decimal import Decimal, InvalidOperation
    from datetime import date as date_type
    from sqlalchemy import select, func as sql_func
    from app.models.database import Transaction

    category = prediction.get("category_name", "")

    # --- amount_deviation ---
    try:
        amount = float(transaction.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0.0

    amount_deviation = 0.5  # default
    if category:
        try:
            res = await db.execute(
                select(
                    sql_func.avg(Transaction.amount),
                    sql_func.stddev_pop(Transaction.amount),
                ).where(
                    Transaction.organisation_id == org_id,
                    Transaction.category == category,
                )
            )
            row = res.first()
            if row and row[0] is not None:
                mean_val = float(row[0])
                std_val = float(row[1] or 1.0)
                raw_dev = abs(amount - mean_val) / (std_val + 1.0)
                amount_deviation = min(1.0, raw_dev / 3.0)
        except Exception:
            pass

    # --- vendor_frequency ---
    description = str(transaction.get("description", ""))
    vendor_frequency = 0.3  # default (rare-ish)
    first_word = description.split()[0] if description.strip() else ""
    if len(first_word) >= 3:
        try:
            res = await db.execute(
                select(sql_func.count()).where(
                    Transaction.organisation_id == org_id,
                    Transaction.description.ilike(f"%{first_word}%"),
                )
            )
            count = res.scalar() or 0
            vendor_frequency = min(1.0, count / 20.0)
        except Exception:
            pass

    # --- time_pattern ---
    date_str = str(transaction.get("date", ""))
    time_pattern = 0.7  # default (moderately normal)
    try:
        d = date_type.fromisoformat(date_str)
        # Mon-Fri = normal; weekends = unusual
        time_pattern = 0.3 if d.weekday() >= 5 else 0.9
    except (ValueError, TypeError):
        pass

    return {
        "amount_deviation": round(amount_deviation, 4),
        "vendor_frequency": round(vendor_frequency, 4),
        "time_pattern": round(time_pattern, 4),
    }

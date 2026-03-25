"""
Cost tracker for Anthropic API calls.

Wraps the anthropic client to record token usage per call and compute costs.
Use as a context manager around any eval run to get a full cost report.

Usage:
    tracker = CostTracker(model="claude-haiku-4-5-20251001")
    with tracker:
        response = client.messages.create(...)
        tracker.record(response.usage)
    print(tracker.report())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing table — update when Anthropic changes pricing
# https://www.anthropic.com/pricing
# ---------------------------------------------------------------------------
PRICING = {
    "claude-haiku-4-5-20251001": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
    "claude-sonnet-4-6": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "claude-opus-4-6": {
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
    },
}

DEFAULT_EVAL_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    label: str = ""

    @property
    def cost_usd(self) -> float:
        pricing = PRICING.get(self.model, PRICING[DEFAULT_EVAL_MODEL])
        return (
            self.input_tokens * pricing["input_per_1m"] / 1_000_000
            + self.output_tokens * pricing["output_per_1m"] / 1_000_000
        )


@dataclass
class CostTracker:
    model: str = DEFAULT_EVAL_MODEL
    budget_usd: Optional[float] = None  # set to e.g. 0.10 to hard-stop at $0.10
    calls: list[CallRecord] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic, repr=False)

    def record(self, usage, label: str = "", duration_ms: float = 0.0) -> None:
        """Record usage from an anthropic response.usage object or dict."""
        if hasattr(usage, "input_tokens"):
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
        else:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        rec = CallRecord(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            label=label,
        )
        self.calls.append(rec)

        if self.budget_usd is not None and self.total_cost_usd > self.budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost_usd:.4f} > ${self.budget_usd:.4f}. "
                f"Set a higher budget or use {DEFAULT_EVAL_MODEL} for cheaper testing."
            )

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    def cost_estimate(self, n_transactions: int, avg_input_tokens: int = 800, avg_output_tokens: int = 150) -> float:
        """Estimate cost for N transactions before running."""
        pricing = PRICING.get(self.model, PRICING[DEFAULT_EVAL_MODEL])
        return n_transactions * (
            avg_input_tokens * pricing["input_per_1m"] / 1_000_000
            + avg_output_tokens * pricing["output_per_1m"] / 1_000_000
        )

    def report(self) -> str:
        lines = [
            "=" * 60,
            f"  COST REPORT - model: {self.model}",
            "=" * 60,
            f"  Total API calls:     {self.total_calls}",
            f"  Total input tokens:  {self.total_input_tokens:,}",
            f"  Total output tokens: {self.total_output_tokens:,}",
            f"  Total cost (USD):    ${self.total_cost_usd:.4f}",
        ]
        if self.budget_usd:
            pct = (self.total_cost_usd / self.budget_usd) * 100
            lines.append(f"  Budget used:         {pct:.1f}% of ${self.budget_usd:.2f}")
        lines.append("=" * 60)
        if self.calls:
            lines.append("  Per-call breakdown:")
            for i, c in enumerate(self.calls, 1):
                label = f" [{c.label}]" if c.label else ""
                lines.append(
                    f"    {i:3d}. {c.input_tokens:5d} in / {c.output_tokens:4d} out "
                    f"→ ${c.cost_usd:.5f}{label}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *_):
        elapsed = (time.monotonic() - self._start) * 1000
        # Attach elapsed to last call if any
        if self.calls:
            self.calls[-1].duration_ms = elapsed


class BudgetExceededError(RuntimeError):
    pass


def estimate_before_run(
    n_transactions: int,
    model: str = DEFAULT_EVAL_MODEL,
    avg_input_tokens: int = 800,
    avg_output_tokens: int = 150,
) -> None:
    """Print a cost estimate and ask for confirmation before a real API run."""
    tracker = CostTracker(model=model)
    estimate = tracker.cost_estimate(n_transactions, avg_input_tokens, avg_output_tokens)
    print(f"\n  Estimated cost for {n_transactions} transactions on {model}:")
    print(f"  ${estimate:.4f} USD")
    for name, pricing in PRICING.items():
        cost = n_transactions * (
            avg_input_tokens * pricing["input_per_1m"] / 1_000_000
            + avg_output_tokens * pricing["output_per_1m"] / 1_000_000
        )
        marker = " ← selected" if name == model else ""
        print(f"    {name}: ${cost:.4f}{marker}")
    print()

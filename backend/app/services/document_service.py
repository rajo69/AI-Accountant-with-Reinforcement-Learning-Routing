"""Document generation service: financial figures + LLM narrative + WeasyPrint PDF."""
import asyncio
import logging
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import instructor
from anthropic import AsyncAnthropic
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import AuditLog, GeneratedDocument, Organisation, Transaction

logger = logging.getLogger(__name__)

LLM_MODEL = "claude-sonnet-4-6"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


# ── Structured LLM output ────────────────────────────────────────────────────

class ManagementLetterNarrative(BaseModel):
    executive_summary: str
    income_analysis: str
    expense_analysis: str
    cash_flow_observations: str
    recommendations: str


# ── Financial calculations (pure Python, no LLM) ─────────────────────────────

def _calculate_figures(transactions: list, period_start: date, period_end: date) -> dict:
    income_txs = [t for t in transactions if t.amount > 0]
    expense_txs = [t for t in transactions if t.amount < 0]

    total_income = sum((t.amount for t in income_txs), Decimal("0.00"))
    total_expenses = sum((t.amount for t in expense_txs), Decimal("0.00"))
    net = total_income + total_expenses

    # Top 5 expense categories by absolute spend
    cat_totals: dict[str, Decimal] = {}
    for t in expense_txs:
        cat = t.category or "Uncategorised"
        cat_totals[cat] = cat_totals.get(cat, Decimal("0")) + t.amount

    total_expenses_abs = abs(total_expenses)
    top_categories = []
    for cat_name, cat_total in sorted(cat_totals.items(), key=lambda x: x[1])[:5]:
        pct = (abs(cat_total) / total_expenses_abs * 100) if total_expenses_abs else Decimal("0")
        top_categories.append({
            "name": cat_name,
            "total_abs": float(abs(cat_total)),
            "pct": float(pct),
        })

    # Top 5 largest transactions by absolute value
    largest = [
        {
            "date": str(t.date),
            "description": t.description[:60],
            "amount": float(t.amount),
        }
        for t in sorted(transactions, key=lambda t: abs(t.amount), reverse=True)[:5]
    ]

    return {
        "total_income": float(total_income),
        "total_expenses": float(total_expenses),
        "total_expenses_abs": float(total_expenses_abs),
        "net": float(net),
        "transaction_count": len(transactions),
        "top_expense_categories": top_categories,
        "largest_transactions": largest,
        "period_start": str(period_start),
        "period_end": str(period_end),
    }


# ── LLM narrative generation ──────────────────────────────────────────────────

async def _generate_narrative(figures: dict, org_name: str) -> ManagementLetterNarrative:
    client = instructor.from_anthropic(
        AsyncAnthropic(api_key=settings.anthropic_api_key)
    )

    top_cats = "\n".join(
        f"  - {c['name']}: £{c['total_abs']:,.2f} ({c['pct']:.1f}%)"
        for c in figures["top_expense_categories"]
    ) or "  (no categorised expenses)"

    net_direction = "profit" if figures["net"] >= 0 else "loss"

    prompt = (
        f"You are a professional UK accountant preparing a management letter for {org_name}.\n\n"
        f"Period: {figures['period_start']} to {figures['period_end']}\n"
        f"Total Income: £{figures['total_income']:,.2f}\n"
        f"Total Expenses: £{figures['total_expenses_abs']:,.2f}\n"
        f"Net {net_direction.title()}: £{abs(figures['net']):,.2f}\n"
        f"Transaction Count: {figures['transaction_count']}\n\n"
        f"Top Expense Categories:\n{top_cats}\n\n"
        "Write a concise, professional management letter with 5 sections:\n"
        "1. executive_summary: 2-3 sentences summarising overall financial health\n"
        "2. income_analysis: 2-3 sentences on income patterns and trends\n"
        "3. expense_analysis: 2-3 sentences on spending patterns with specific categories\n"
        "4. cash_flow_observations: 2-3 sentences on cash flow and working capital\n"
        "5. recommendations: 2-3 actionable recommendations for a UK SME\n\n"
        "Use plain, professional English. Do not repeat raw figures — interpret and contextualise them."
    )

    return await client.messages.create(
        model=LLM_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
        response_model=ManagementLetterNarrative,
    )


# ── HTML rendering + PDF ──────────────────────────────────────────────────────

def _fmt_money(value: float) -> str:
    return f"£{abs(value):,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _render_html(
    figures: dict,
    narrative: ManagementLetterNarrative,
    org_name: str,
    period_start: date,
    period_end: date,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template("management_letter.html")
    return template.render(
        figures=figures,
        narrative=narrative,
        fmt_money=_fmt_money,
        fmt_pct=_fmt_pct,
        organisation_name=org_name,
        period_start=period_start.strftime("%d %B %Y"),
        period_end=period_end.strftime("%d %B %Y"),
        generated_at=datetime.now().strftime("%d %B %Y at %H:%M"),
        net_label="Profit" if figures["net"] >= 0 else "Loss",
        net_positive=figures["net"] >= 0,
    )


def _html_to_pdf_sync(html: str) -> bytes:
    """Sync WeasyPrint call — run via asyncio.to_thread."""
    from weasyprint import HTML
    return HTML(string=html).write_pdf()


# ── Public entry point ────────────────────────────────────────────────────────

async def generate_management_letter(
    org_id: uuid.UUID,
    period_start: date,
    period_end: date,
    db: AsyncSession,
) -> tuple[bytes, dict]:
    """Generate a management letter PDF. Returns (pdf_bytes, metadata)."""
    org_result = await db.execute(select(Organisation).where(Organisation.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise ValueError(f"Organisation {org_id} not found")

    # Load transactions for the period
    tx_result = await db.execute(
        select(Transaction).where(
            Transaction.organisation_id == org_id,
            Transaction.date >= period_start,
            Transaction.date <= period_end,
        )
    )
    transactions = list(tx_result.scalars().all())

    # 1. Calculate figures (pure Python)
    figures = _calculate_figures(transactions, period_start, period_end)

    # 2. Generate narrative (LLM)
    narrative = await _generate_narrative(figures, org.name)

    # 3. Render HTML
    html = _render_html(figures, narrative, org.name, period_start, period_end)

    # 4. Convert to PDF (sync library, run in thread)
    pdf_bytes = await asyncio.to_thread(_html_to_pdf_sync, html)

    # 5. Store metadata
    doc_id = uuid.uuid4()
    now = datetime.now()

    doc_record = GeneratedDocument(
        id=doc_id,
        organisation_id=org_id,
        template="management_letter",
        period_start=period_start,
        period_end=period_end,
        ai_model=LLM_MODEL,
        figures=figures,
    )
    db.add(doc_record)

    audit = AuditLog(
        organisation_id=org_id,
        action="generate_document",
        entity_type="generated_document",
        entity_id=doc_id,
        new_value={
            "template": "management_letter",
            "period_start": str(period_start),
            "period_end": str(period_end),
        },
        ai_model=LLM_MODEL,
        ai_explanation="Management letter generated with AI narrative sections",
        ai_decision_data={"transaction_count": figures["transaction_count"]},
    )
    db.add(audit)
    await db.commit()

    metadata = {
        "document_id": str(doc_id),
        "template": "management_letter",
        "period_start": str(period_start),
        "period_end": str(period_end),
        "transaction_count": figures["transaction_count"],
        "generated_at": now.isoformat(),
    }

    return pdf_bytes, metadata

# AI Accountant — Architecture & Design Reference

> This document is the single source of truth for design decisions, architecture
> rationale, and engineering trade-offs. It is intended for README construction,
> onboarding, and future maintainers (including future AI sessions).

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack Decisions](#3-tech-stack-decisions)
4. [Database Design](#4-database-design)
5. [Xero Integration Design](#5-xero-integration-design)
6. [AI Agent Architecture](#6-ai-agent-architecture)
7. [Explainable AI (XAI) Design](#7-explainable-ai-xai-design)
8. [Security Model](#8-security-model)
9. [Cost Architecture](#9-cost-architecture)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Key Invariants (Never Violate)](#11-key-invariants-never-violate)

---

## 1. Product Overview

**AI Accountant** is a Xero-connected assistant for UK accountants. It automates
three tasks that currently consume significant manual time:

| Task | What we do | Manual alternative |
|------|------------|-------------------|
| Transaction categorisation | LLM + pgvector few-shot assigns chart-of-accounts codes with confidence | Accountant reviews each transaction individually |
| Bank reconciliation | Multi-signal fuzzy matching (amount + date + description) | Side-by-side comparison in Xero |
| Management letter drafts | RAG-grounded LLM narrative over computed figures | Writing from scratch each quarter |

Every AI decision is logged with full explainability — input data, output,
confidence score, model version, and a human-readable explanation.

**Target users**: UK accounting firms using Xero, 2–50 staff.
**Monetisation**: SaaS, £49–99/month per firm after beta.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER / API CLIENT                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────┐
│                    Next.js 14 (Vercel)                       │
│  Dashboard · Transactions · Reconciliation · Documents       │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST (JSON)
┌───────────────────────────▼─────────────────────────────────┐
│                  FastAPI (Railway)                           │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ /auth    │  │ /sync    │  │/categorise│  │/reconcile│   │
│  │ /dashboard│  │          │  │/transactions│ │/bank-stmts│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Service Layer                          │ │
│  │  XeroAdapter · EmbeddingService · DocumentService      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │               LangGraph Agents                          │ │
│  │   CategoriserAgent · ReconcilerAgent                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 XAI Engine                              │ │
│  │   SHAP/EBM Explainer · Simpful Fuzzy Risk Scorer       │ │
│  └────────────────────────────────────────────────────────┘ │
└──────┬──────────────────────────┬───────────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  PostgreSQL │          │  Anthropic API   │
│  + pgvector │          │  (Claude)        │
│  (Railway)  │          └─────────────────┘
│             │
│  ┌────────┐ │
│  │ Redis  │ │
│  └────────┘ │
└─────────────┘
                    ┌────────────────┐
                    │   Xero API     │
                    │ (OAuth2 + REST)│
                    └────────────────┘
```

**Data flow for transaction categorisation:**

```
Xero API → sync → transactions table (uncategorised)
                         │
                    CategoriserAgent (LangGraph)
                         │
              ┌──────────┼──────────┐
              │          │          │
         pgvector    Claude API  AuditLog
         (similar    (classify)  (every
          examples)              decision)
              │
         confidence > 0.85 → auto_categorised
         confidence 0.5–0.85 → suggested (human reviews)
         confidence < 0.5 → needs_review
```

---

## 3. Tech Stack Decisions

### 3.1 Python + FastAPI (not Django)

**Decision**: FastAPI for all backend API work.

**Why not Django/DRF**: Django's ORM is synchronous-first. Our entire data layer
uses `async`/`await` (asyncpg for DB, httpx for Xero API, anthropic SDK for LLM).
Django would force us into sync adapters or `sync_to_async` wrappers everywhere,
adding complexity and hurting performance under concurrent AI workloads.

FastAPI gives us: async-native from day one, automatic OpenAPI docs, Pydantic v2
validation on every request/response boundary, and a clean dependency injection
system for DB sessions and auth.

### 3.2 SQLAlchemy 2.0 Async (not raw asyncpg, not Tortoise ORM)

**Decision**: SQLAlchemy 2.0 with `mapped_column` style + asyncpg driver.

**Why not raw asyncpg**: We need the ORM layer for migrations (Alembic), relationship
loading, and audit trail helpers. Raw asyncpg is faster but would require hand-writing
all schema management.

**Why not Tortoise ORM**: SQLAlchemy has pgvector support (`pgvector.sqlalchemy.Vector`),
mature migration tooling, and SQLAlchemy 2.0's async API is now first-class. Tortoise
has less ecosystem support for pgvector.

**Why SQLAlchemy 2.0 `mapped_column` style**: Type-safe column definitions, works
with Python type checkers (mypy/pyright), cleaner than the old `Column()` style.

### 3.3 PostgreSQL 16 + pgvector (not Pinecone, not Weaviate)

**Decision**: Single PostgreSQL instance with pgvector extension for vector storage.

**Why not a dedicated vector DB**: For an early-stage product with <100k transactions,
adding a separate vector database (Pinecone, Weaviate, Qdrant) adds operational
complexity and cost for no benefit. pgvector's performance is excellent up to millions
of vectors. We can migrate to a dedicated vector DB later if needed, but by then we'll
have revenue to justify it.

**Benefit**: One database to back up, one connection pool, one query interface. The
few-shot similarity search (`find_similar_transactions`) is a single SQL query with
a `<->` (L2 distance) or `<=>` (cosine similarity) operator.

### 3.4 Anthropic Claude (not OpenAI GPT-4)

**Decision**: Use Anthropic's Claude API for all LLM calls.

**Why Claude over GPT-4o**:
- Builder preference (user-specified)
- Claude's instruction-following is excellent for structured output via Instructor
- Anthropic's API reliability and latency are competitive
- Claude Haiku is extremely cheap for high-volume categorisation (see Section 9)
- `instructor[anthropic]` provides the same structured output interface regardless of provider

**Model selection strategy**:
- Production categorisation: `claude-sonnet-4-6` (accuracy matters, run once per sync)
- High-volume evals and testing: `claude-haiku-4-5-20251001` (~10x cheaper)
- Document generation: `claude-sonnet-4-6` (narrative quality matters)

### 3.5 LangGraph (not raw LLM chains)

**Decision**: Use LangGraph for the categorisation and reconciliation agents.

**Why LangGraph**: Each agent has multiple nodes (fetch context → classify → validate
→ decide → explain). LangGraph makes the state transitions explicit, debuggable, and
testable in isolation. Each node is a pure function that transforms state — easy to
mock in tests.

**Why not raw LangChain chains**: LangChain chains are harder to test and the control
flow is implicit. LangGraph's explicit graph structure forces good design.

**Why not simple sequential calls**: The validate → decide logic (confidence thresholds,
retry on validation failure) is cleaner as graph edges than as nested if/else.

### 3.6 Instructor + Pydantic v2 (not free-text parsing)

**Decision**: All LLM outputs are structured via `instructor[anthropic]` + Pydantic models.

**Why**: Free-text parsing is brittle. A model that says "I think this is probably
Office Supplies (code 420)" is useless — we need `{"category_code": "420",
"category_name": "Office Supplies", "confidence": 0.91, "reasoning": "..."}` every
time. Instructor uses Anthropic's tool-use feature to guarantee structured output,
with automatic retry on validation failure.

### 3.7 httpx (not requests, not aiohttp)

**Decision**: httpx for all external HTTP calls (Xero API, token refresh).

**Why**: httpx is async-native and has an identical sync API. It supports HTTP/2,
has excellent timeout/retry support, and is the de-facto standard for async Python
HTTP. requests is sync-only. aiohttp has a less ergonomic API.

### 3.8 RapidFuzz (not fuzzywuzzy/thefuzz)

**Decision**: RapidFuzz for description similarity scoring in reconciliation.

**Why**: RapidFuzz is a C-extension reimplementation of fuzzywuzzy, 10–100x faster
with no GPL licence issues. We use `token_sort_ratio` specifically because transaction
descriptions have words in different orders ("TESCO STORES UK" vs "UK STORES TESCO").

### 3.9 WeasyPrint (not Puppeteer, not ReportLab)

**Decision**: WeasyPrint for PDF generation from HTML/CSS templates.

**Why not Puppeteer**: Requires a Node.js/Chromium runtime — too heavy for a Python
backend, and Chromium is a large Docker image.

**Why not ReportLab**: ReportLab requires programmatic layout (x/y coordinates). Our
management letter has complex tables and formatted text — HTML/CSS is a far more
natural authoring format and our designers can edit it.

**Why WeasyPrint**: Pure Python, excellent CSS support, renders standard HTML to PDF
with print-media CSS. The "AI-Assisted Draft" watermark and page numbers are a few
lines of CSS.

---

## 4. Database Design

### 4.1 Schema overview

```
organisations
├── id (UUID PK)
├── name
├── xero_tenant_id (UNIQUE) ← Xero's identifier for the company
├── xero_access_token (encrypted at rest in prod)
├── xero_refresh_token
├── xero_token_expires_at
└── last_sync_at ← updated after every full_sync()

accounts
├── id, organisation_id (FK)
├── xero_id (UNIQUE) ← Xero's AccountID
├── code, name, type, tax_type
└── (this is the chart of accounts)

transactions
├── id, organisation_id (FK), account_id (FK, nullable)
├── xero_id (UNIQUE)
├── date, amount (NUMERIC(12,2)), description, reference
├── category, category_confidence (NUMERIC(5,4))
├── categorisation_status: uncategorised|auto_categorised|suggested|confirmed|rejected
├── is_reconciled (Boolean)
└── embedding (Vector(1536)) ← pgvector, used for few-shot lookup

bank_statements
├── id, organisation_id (FK)
├── xero_id (UNIQUE)
├── date, amount, description, reference
├── matched_transaction_id (FK → transactions, nullable)
├── match_confidence (NUMERIC(5,4))
└── match_status: unmatched|auto_matched|suggested|confirmed

audit_logs
├── id, organisation_id (FK)
├── action, entity_type, entity_id
├── old_value, new_value (JSONB)
├── ai_model, ai_confidence, ai_explanation
└── ai_decision_data (JSONB) ← full XAI payload
```

### 4.2 Money as Decimal — non-negotiable

**All monetary amounts are stored as `NUMERIC(12,2)` in PostgreSQL and `Decimal`
in Python.** `float` is forbidden for financial amounts.

**Why**: IEEE 754 floating point cannot represent 0.1 exactly. `0.1 + 0.2 = 0.30000000000000004`
in float arithmetic. For an accounting tool, a rounding error that causes a penny
discrepancy in a reconciliation match would be a critical bug and a trust-killer.

### 4.3 SPEND transactions are negative

When syncing from Xero, `BankTransaction.Type == "SPEND"` is stored as a negative
amount. `RECEIVE` transactions are positive. This means the sum of all transactions
equals the bank account balance — consistent with double-entry accounting.

### 4.4 Audit log design

Every AI decision writes an `AuditLog` row. The schema is intentionally broad:

- `ai_decision_data` (JSONB): stores the full reasoning payload — similar examples
  used, confidence breakdown, fuzzy rule firings, SHAP values — anything the model
  used to reach its conclusion. This supports future GDPR "right to explanation"
  requests and internal debugging.
- `old_value`/`new_value`: JSON snapshots of the entity before/after. Supports undo.
- `ai_model`: exact model version string (e.g., `claude-sonnet-4-6`). When model
  versions change, we can see which decisions were made by which version.

---

## 5. Xero Integration Design

### 5.1 Why raw httpx instead of the xero-python SDK

We use httpx directly for all Xero API calls rather than the official `xero-python` SDK.

**Reasons**:
1. The SDK wraps `requests` (sync) and requires additional adapters for async contexts
2. Our OAuth2 token management (refresh, expiry detection, storage) needs direct access
   to the raw token response — the SDK abstracts this in a way that's harder to control
3. Direct httpx calls are easier to test (mock the `AsyncClient`) and easier to debug
4. The SDK adds ~10 MB of generated code for 2 API endpoints we actually need

The `xero-python` package remains in `pyproject.toml[xero]` as an optional dep in case
we need it for edge cases later.

### 5.2 Token lifecycle

```
User visits /auth/xero/connect
    → redirect to Xero OAuth2
    → user logs in + authorises
    → Xero redirects to /auth/xero/callback?code=...
    → we POST code to identity.xero.com/connect/token
    → get access_token (30 min), refresh_token (60 days)
    → store in organisations table
    → return success JSON

Every API call:
    _ensure_valid_token() checks expiry with 60s buffer
    If expired → POST to token endpoint with grant_type=refresh_token
    → update organisations table with new tokens
```

### 5.3 Rate limiting

Xero allows 60 requests/minute per app. `_get_with_retry()` detects HTTP 429 and
waits 60 seconds before retrying, up to 3 attempts. The sync methods paginate with
`page=1,2,3...` (Xero pages at 100 items). Large orgs with thousands of transactions
will take several minutes to sync — acceptable for a background operation.

### 5.4 Incremental sync (future)

The current `sync_transactions()` fetches all pages each time. For production with
large data sets, pass `If-Modified-Since` header using `org.last_sync_at`. Xero
honours this header on BankTransactions. This is a Phase 2 enhancement, not in MVP.

---

## 6. AI Agent Architecture

### 6.1 Categorisation agent (LangGraph)

```
State: CategoriserState
  transaction_id, transaction_data
  chart_of_accounts, similar_examples
  prediction, status

Graph:
  fetch_context → classify → validate → decide → explain
      │              │           │          │
      │         Instructor    pydantic   confidence   XAI
      │         + Claude     validation  thresholds  engine
      │
  pgvector
  (5 nearest
  categorised
  transactions)
```

**Few-shot learning via pgvector**: When the accountant corrects a categorisation
(`POST /transactions/{id}/correct`), we re-embed that transaction and it becomes
a training example. The next time a similar transaction comes through, it will be
in the `similar_examples` passed to Claude. No fine-tuning needed — the vector
store *is* the training set.

**Confidence thresholds**:
- `> 0.85`: `auto_categorised` — written to DB, no human needed
- `0.50–0.85`: `suggested` — shown to accountant with Accept/Reject buttons
- `< 0.50`: `needs_review` — flagged, no suggestion shown (confusing low-confidence
  guesses are worse than no guess)

These thresholds are tunable. Initial values are based on LLM categorisation
literature benchmarks for accounting tasks.

### 6.2 Reconciliation agent (LangGraph)

Multi-signal scoring — no LLM needed for matching, only for explanation:

```
Combined score = (amount × 0.5) + (date × 0.2) + (description × 0.3)

amount_score:
  1.0  → exact Decimal match
  0.8  → within 1%
  0.0  → otherwise

date_score:
  1.0  → same day
  -0.15 per day difference
  min 0.0

description_score:
  RapidFuzz.token_sort_ratio(a, b) / 100
```

**Why these weights**: Amount is the strongest signal in UK bank reconciliation —
an exact amount match with a close date is nearly always correct. Date is less
reliable because some Xero transactions are backdated. Description similarity is
useful for high-confidence matches but unreliable alone (descriptions vary by bank).

**LLM only for explanation**: The explanation node uses Claude to write a human-
readable sentence ("Matched to TESCO STORES on 14 Mar: amount matches exactly
(£52.40), same day, description 87% similar"). No LLM is used for the matching
decision itself — it's deterministic and auditable.

### 6.3 Vector embeddings for transactions

- Model: `claude`'s embedding model or `text-embedding-3-small` (1536-dim)
- Input: concatenation of `date + amount + description + reference`
- Storage: `transactions.embedding` (pgvector Vector(1536))
- Query: `ORDER BY embedding <=> $query_embedding LIMIT 5` (cosine similarity)
- Backfill: `embedding_service.backfill_embeddings()` runs on first sync

**Note on embedding model**: Since we're using Claude API, we use Anthropic's
embedding endpoint if available, otherwise fall back to a compatible embedding model.
Embeddings are computed once per transaction and reused — the cost is negligible.

---

## 7. Explainable AI (XAI) Design

### 7.1 Why XAI matters for UK accountants

UK accountants have professional liability for the accounts they sign. An AI tool
that says "category: Office Supplies" with no explanation will not be trusted — and
should not be. If something goes wrong, the accountant needs to show *why* they
accepted an AI suggestion.

Our XAI stack provides three layers of explanation:

| Layer | Technology | What it explains |
|-------|-----------|------------------|
| LLM reasoning | Claude (via Instructor) | Natural language: "This looks like a software subscription based on the ADOBE description and £47.99 amount" |
| Feature importance | SHAP / InterpretML EBM | Which transaction features (amount, vendor frequency, day of week) drove the prediction |
| Risk scoring | Custom Mamdani fuzzy inference | Whether the transaction is anomalous (unusual amount for this category, rare vendor) |

### 7.2 Fuzzy logic risk scoring

The fuzzy risk engine (`xai/fuzzy_engine.py`) is a hand-rolled Mamdani inference
system using triangular membership functions and centroid defuzzification.
It does not use an external fuzzy library; the implementation is self-contained
so the inference logic can be read, audited, and modified without understanding
a third-party API.

Rule base (8 rules, human-readable):

```
IF amount_deviation IS high AND vendor_frequency IS rare THEN risk IS high
IF amount_deviation IS low AND vendor_frequency IS frequent THEN risk IS low
IF time_pattern IS unusual THEN risk IS medium
...
```

**Why fuzzy logic over a neural anomaly detector**: Fuzzy rules are:
1. Human-readable — an accountant can review and challenge the rules
2. Deterministic — the same input always gives the same output
3. Explainable — we can show which rules fired and by how much
4. Cheap — no ML training, no inference cost

A neural anomaly detector would be a black box. For an accounting tool where
professional trust is the product, transparency beats accuracy.

### 7.3 SHAP / EBM feature importance

We use InterpretML's Explainable Boosting Machine (EBM) when enough training data
exists (50+ categorised transactions). EBM is a glass-box model: each feature's
contribution to the prediction is additive and visualisable as a bar chart.

Below 50 samples, we fall back to the LLM's reasoning text as the explanation.

---

## 8. Security Model

### 8.1 Current state (MVP)

- Xero OAuth2 tokens stored in plaintext in PostgreSQL (acceptable for local dev,
  must be encrypted at rest in production via Railway's encryption or application-level
  AES encryption before beta launch)
- No user authentication yet — Phase 8 uses Xero as the SSO provider
- All API endpoints are open (no auth middleware) — acceptable for local dev only
- CORS allows all origins — tighten to `NEXT_PUBLIC_APP_URL` in production

### 8.2 Phase 8 auth model

When Phase 8 ships, auth works like this:
- User visits app → if no session, redirect to `/auth/xero/connect`
- After Xero callback → create JWT stored in httpOnly cookie
- JWT payload: `{ org_id, exp }`
- All API endpoints check JWT via FastAPI middleware
- All DB queries are scoped to `organisation_id` from the JWT

**No separate user table in MVP**: Xero identity *is* the user identity. One Xero
organisation = one tenant. Multi-user within a firm is a post-beta feature.

### 8.3 Data handling

- Xero financial data never leaves our infrastructure (Railway EU region)
- LLM calls send transaction descriptions and amounts to Anthropic API —
  this is disclosed in the privacy policy
- Embeddings are stored only in our PostgreSQL — never sent to external vector DBs
- Audit logs are immutable (no DELETE route on audit_logs table)

---

## 9. Cost Architecture

### 9.1 Claude API pricing (approximate, verify at anthropic.com/pricing)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Use case |
|-------|----------------------|------------------------|----------|
| claude-haiku-4-5-20251001 | ~$0.80 | ~$4.00 | Evals, high-volume testing |
| claude-sonnet-4-6 | ~$3.00 | ~$15.00 | Production categorisation |

### 9.2 Cost per categorisation

A typical categorisation prompt (transaction + chart of accounts + 5 examples):
- Input: ~800 tokens (transaction data + CoA + examples)
- Output: ~150 tokens (JSON prediction + reasoning)

**Cost per transaction on Haiku**: `(800 × $0.80 + 150 × $4.00) / 1,000,000 = $0.00126`
**Cost per transaction on Sonnet**: `(800 × $3.00 + 150 × $15.00) / 1,000,000 = $0.00465`

A typical UK SME has ~200 transactions/month. Full monthly categorisation:
- Haiku: `200 × $0.00126 = $0.25/month`
- Sonnet: `200 × $0.00465 = $0.93/month`

At £49/month pricing, even Sonnet costs <2% of revenue per customer.

### 9.3 Cost optimisation strategy

1. **Use Haiku for evaluation/testing** — all eval runs use `claude-haiku-4-5-20251001`
   by default. Set `EVAL_MODEL=claude-sonnet-4-6` only to test production behaviour.

2. **Response caching** — `evals/response_cache.py` caches API responses to disk
   keyed by a hash of (model + prompt). Re-running evals against unchanged fixtures
   costs $0.

3. **Batch before calling** — the `categorise_batch()` runner processes all
   uncategorised transactions in one pass with `asyncio.Semaphore(5)` concurrency,
   not one API call per user action.

4. **Auto-categorisation reduces volume** — transactions with confidence > 0.85 are
   auto-processed. Over time as the pgvector few-shot store grows, more transactions
   hit this threshold. Cost per new transaction trends toward zero as patterns repeat.

5. **No embedding re-computation** — embeddings are computed once and cached in
   `transactions.embedding`. Only new or corrected transactions trigger re-embedding.

6. **Reconciliation uses no LLM** — the matching algorithm (amount/date/description
   scoring) is purely algorithmic. Claude is only called to generate the human-readable
   explanation after a match is found. Explanation calls are short (~100 output tokens).

### 9.4 Monthly cost estimate at scale

| Customers | Transactions/mo | Categorisation | Reconciliation explanations | Total API cost |
|-----------|----------------|----------------|-----------------------------|----------------|
| 5 (beta) | 1,000 | $0.47 | $0.05 | **~$0.52/mo** |
| 20 | 4,000 | $1.86 | $0.20 | **~$2.06/mo** |
| 100 | 20,000 | $9.30 | $1.00 | **~$10.30/mo** |

Revenue at 100 customers (£49/mo): £4,900 (~$6,200). API cost: ~$10. Margin: >99%.

---

## 10. Deployment Architecture

### 10.1 Local development

```bash
docker-compose up -d          # PostgreSQL 16 + pgvector, Redis
cd backend && alembic upgrade head
uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev    # port 3000
```

### 10.2 Production (Railway + Vercel)

```
GitHub → Railway (auto-deploy on push to main)
       → FastAPI backend (Dockerfile)
       → PostgreSQL with pgvector (Railway managed)
       → Redis (Railway managed)

GitHub → Vercel (auto-deploy on push to main)
       → Next.js frontend
```

**Environment variables** (Railway):
```
DATABASE_URL          ← auto-set by Railway PostgreSQL plugin
REDIS_URL             ← auto-set by Railway Redis plugin
ANTHROPIC_API_KEY     ← set manually
XERO_CLIENT_ID        ← set manually
XERO_CLIENT_SECRET    ← set manually
XERO_REDIRECT_URI     ← https://yourdomain.com/auth/xero/callback
SECRET_KEY            ← generate with: python -c "import secrets; print(secrets.token_hex(32))"
```

### 10.3 Database migrations in production

Migrations run as a startup command in Railway before the server starts:
```
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 10.4 Docker

The backend runs in a Python 3.12 slim container. Key points:
- WeasyPrint requires `libpango` — add `apt-get install -y libpango-1.0-0 libpangoft2-1.0-0`
- pgvector Python package requires `libpq-dev` at build time
- No C compiler needed at runtime, only at build time — use multi-stage build

---

## 11. Key Invariants (Never Violate)

These rules are non-negotiable and must survive refactoring, new contributors,
and future AI sessions:

| # | Rule | Why |
|---|------|-----|
| 1 | `Decimal` for all money | Float rounding causes penny errors in reconciliation |
| 2 | Every AI decision → AuditLog | Legal accountability, GDPR right to explanation |
| 3 | LLM calls only in `services/` or `agents/` | Routes must not call Claude directly |
| 4 | Xero calls only via `XeroAdapter` | Centralised token refresh, rate limiting, retry |
| 5 | Structured LLM output via Instructor | Never parse free text from LLM |
| 6 | Tests before commit | At minimum one happy-path test per endpoint/service |
| 7 | Ship simple | No premature abstraction. Build what's needed now. |
| 8 | `SPEND` transactions are negative | Consistent with double-entry accounting |
| 9 | Embeddings are 1536-dim | Changing dimensionality requires a full re-embedding |
| 10 | Confidence thresholds are config, not hardcode | Will need tuning per customer |

---

*Last updated: 2026-03-22 — Phase 2 complete*

# AI Accountant: Agentic AI for Accounting Workflows

![CI](https://github.com/rajo69/agentic-ai-accounting/actions/workflows/ci.yml/badge.svg)

A production-grade AI assistant for UK accountants built on top of **Xero**. It automates the three most time-consuming manual tasks in a small accounting practice: transaction categorisation, bank reconciliation, and management letter drafting. Every AI decision is transparent, auditable, and correctable by the human in the loop.

Connect your Xero Account and use it : [Click Here](https://agentic-ai-accounting.vercel.app)

---

## Contents

- [The problem](#the-problem)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [How each feature works](#how-each-feature-works)
- [Tech stack](#tech-stack)
- [Database schema](#database-schema)
- [API reference](#api-reference)
- [Evaluation framework](#evaluation-framework)
- [Running locally](#running-locally)
- [Tests](#tests)
- [Cost model](#cost-model)
- [Deployment](#deployment)
- [GDPR compliance](#gdpr-compliance)
- [Research foundations](#research-foundations)
- [Open research questions](#open-research-questions)
- [Engineering challenges](#engineering-challenges-encountered-during-the-build)

---

## The problem

UK accounting firms spend a significant portion of each working week on tasks that are mechanical but error-prone:

- **Transaction categorisation**: Matching hundreds of bank transactions to the right chart-of-accounts code. Every transaction must be correct because errors compound into the final accounts.
- **Bank reconciliation**: Cross-referencing bank statement lines with transaction records one by one, hunting for matching amounts and dates.
- **Management letters**: Writing quarterly commentary from scratch, turning raw figures into narrative that clients can act on.

These are exactly the tasks where AI can save hours, provided the AI is transparent enough that a professional accountant can trust and verify its decisions. An AI tool that says "category: Office Supplies" with no justification is not useful to someone with professional liability for the accounts they sign.

This project is a full-stack implementation of a solution to that problem: an agentic AI system that categorises, reconciles, and writes, but always shows its working.

---

## What it does

| Feature | How it works |
|---|---|
| **Transaction categorisation** | LangGraph agent classifies Xero transactions against the chart of accounts using Claude and pgvector few-shot examples drawn from the firm's own history. High-confidence results are applied automatically; uncertain results are surfaced for human review. |
| **Bank reconciliation** | Algorithmic LangGraph agent matches bank statement lines to transactions by scoring amount, date, and description similarity. Matching is deterministic (no LLM); Claude only writes the human-readable explanation after a match is found. |
| **Management letter generation** | RAG pipeline computes financial figures in pure Python (no LLM), retrieves relevant transaction context via pgvector, then uses Claude and Instructor to write structured narrative sections. WeasyPrint renders the result as a professional A4 PDF. |
| **Explainable AI** | Every AI decision stores feature importances (InterpretML EBM), a custom Mamdani fuzzy logic risk score, and the LLM's own reasoning; all in an immutable audit log. |
| **Xero integration** | Full OAuth2 flow, automatic token refresh, rate-limit backoff, and incremental sync of accounts, transactions, and bank statements. |
| **Few-shot learning from corrections** | When an accountant corrects a categorisation, the transaction is re-embedded and becomes a training example for future predictions. No fine-tuning required. |

---

## Architecture

```mermaid
flowchart TD
    Browser["Browser / API Client"]

    subgraph Vercel["Vercel"]
        Frontend["Next.js 14\nLanding · Dashboard · Transactions · Reconciliation · Documents"]
    end

    subgraph Railway["Railway"]
        subgraph FastAPI["FastAPI"]
            Routes["API Routes\n/auth · /sync · /categorise · /reconcile · /documents"]

            subgraph SvcLayer["Service Layer"]
                XA["XeroAdapter"]
                ES["EmbeddingService"]
                DS["DocumentService"]
            end

            subgraph AgentLayer["LangGraph Agents"]
                CA["CategoriserAgent"]
                RA["ReconcilerAgent"]
            end

            subgraph XAILayer["XAI Engine"]
                EBM["EBM Explainer"]
                Fuzzy["Mamdani Fuzzy Risk Scorer"]
            end
        end

        DB[("PostgreSQL 16\n+ pgvector\nRedis")]
    end

    Claude["Anthropic Claude API"]
    Xero["Xero API\nOAuth2 + REST"]

    Browser -->|HTTPS| Frontend
    Frontend -->|REST / JSON| Routes
    Routes --> SvcLayer
    Routes --> AgentLayer
    Routes --> XAILayer
    FastAPI --> DB
    CA --> Claude
    RA --> Claude
    DS --> Claude
    XA --> Xero
```

---

### Data flow: transaction categorisation

```mermaid
flowchart TD
    XeroAPI["Xero API"]
    TxTable["transactions table\nstatus: uncategorised"]
    Agent["CategoriserAgent\nLangGraph"]
    pgv["pgvector\n5 nearest categorised examples\ncosine similarity"]
    ClaudeAPI["Claude API\nclassify + structured reasoning"]
    Audit["AuditLog\nevery decision\nfull XAI payload stored as JSONB"]
    Auto["auto_categorised\nwritten to DB immediately"]
    Suggested["suggested\nshown with Accept / Reject buttons"]
    Review["needs_review\nflagged for human\nno guess shown"]

    XeroAPI -->|sync| TxTable
    TxTable --> Agent
    Agent <-->|few-shot lookup| pgv
    Agent <-->|CategoryPrediction| ClaudeAPI
    Agent -->|always| Audit
    Agent -->|confidence above 0.85| Auto
    Agent -->|confidence 0.50 to 0.85| Suggested
    Agent -->|confidence below 0.50| Review
```

---

## How each feature works

### 1. Xero data sync

`integrations/xero_adapter.py` handles the entire Xero relationship:

- **OAuth2 flow**: Redirects the user to Xero, receives the auth code, exchanges it for access and refresh tokens, and stores them against the `Organisation` record.
- **Token lifecycle**: Access tokens expire after 30 minutes. `_ensure_valid_token()` detects expiry with a 60-second buffer and silently refreshes before every API call.
- **Three sync methods**: `sync_accounts()` (chart of accounts), `sync_transactions()` (bank transactions), `sync_bank_statements()`. Each upserts on `xero_id` so reruns are idempotent.
- **Rate-limit backoff**: Xero allows 60 requests per minute. `_get_with_retry()` catches HTTP 429 and waits 60 seconds before retrying, up to 3 attempts.
- **Date parsing**: Xero returns dates in two formats: `/Date(1234567890000+0000)/` (Unix ms with offset) and ISO 8601. Both are handled.

The adapter was built with raw `httpx` rather than the official `xero-python` SDK. The SDK wraps synchronous `requests` and requires additional adapters in an async context. Direct httpx is easier to test (mock `AsyncClient`), easier to debug, and gives full control over token management.

---

### 2. Transaction categorisation agent

`agents/categoriser.py`: a five-node LangGraph graph.

#### Graph structure

```mermaid
flowchart LR
    FC["fetch_context"] --> CL["classify"] --> VA["validate"] --> DE["decide"] --> EX["explain"]
```

**Node 1: fetch_context**

Loads the chart of accounts from the database and queries pgvector for the 5 most similar previously categorised transactions using cosine similarity:

```sql
ORDER BY embedding <=> $query_embedding LIMIT 5
```

These become the few-shot examples in the classification prompt.

**Node 2: classify**

Builds a structured prompt containing:
- The transaction to classify (date, amount, description, reference)
- The full chart of accounts (code and name)
- The 5 similar examples with their confirmed categories

Calls Claude via `instructor[anthropic]`, forcing output into:

```python
class CategoryPrediction(BaseModel):
    category_code: str
    category_name: str
    confidence: float      # range 0.0 to 1.0
    reasoning: str         # plain English explanation
```

Claude is never trusted to return parseable free text. Instructor uses Anthropic's tool-use feature to guarantee structured output, with automatic retry on Pydantic validation failure.

**Node 3: validate**

Checks two things:
1. Does `category_code` exist in the chart of accounts?
2. Is `confidence` in the range `[0.0, 1.0]`?

If either check fails, confidence is set to 0 and the reasoning is overwritten with "validation failed". The decision node will then flag this transaction for human review rather than making a bad auto-categorisation.

**Node 4: decide**

Applies confidence thresholds:

| Confidence | Status | What happens |
|---|---|---|
| Above 0.85 | `auto_categorised` | Written to DB immediately; no human needed |
| 0.50 to 0.85 | `suggested` | Shown to accountant with Accept/Reject buttons |
| Below 0.50 | `needs_review` | Flagged; no suggestion shown (a confusing low-confidence guess is worse than no guess) |

An `AuditLog` row is written in all cases.

**Node 5: explain**

Calls the XAI engine (see section 4) to compute feature importances and a fuzzy risk score. Both are appended to the `AuditLog.ai_decision_data` JSONB field.

#### Few-shot learning from corrections

When an accountant corrects a categorisation via `POST /transactions/{id}/correct`, the transaction is re-embedded and the new embedding is stored. The next time a similar transaction arrives, the corrected version appears in `similar_examples`; the vector store acts as the training set. No fine-tuning, no retraining, no separate pipeline.

#### Batch runner

`categorise_batch()` fetches all uncategorised transactions and processes them concurrently with `asyncio.Semaphore(5)`, keeping Claude API costs predictable and preventing runaway parallel requests.

---

### 3. Bank reconciliation agent

`agents/reconciler.py`: a five-node LangGraph graph. No LLM is used for the matching decision; only for the explanation.

#### Graph structure

```mermaid
flowchart LR
    FC["find_candidates"] --> SC["score_candidates"] --> DE["decide"] --> EX["explain"] --> SA["save"]
```

**Node 1: find_candidates**

Queries the transactions table for candidates where:
- Amount is within £0.01 of the bank statement amount (Decimal comparison, no float)
- Date is within 7 days of the bank statement date
- Same `organisation_id`

Returns up to 10 candidates.

**Node 2: score_candidates**

Scores each candidate on three signals:

```
combined_score = (amount_score × 0.5) + (date_score × 0.2) + (description_score × 0.3)

amount_score:
  1.0  <- exact Decimal match
  0.8  <- within 1%
  0.0  <- otherwise

date_score:
  1.0  <- same day
  max(0.0, 1.0 - days_difference × 0.15)

description_score:
  RapidFuzz.token_sort_ratio(a, b) / 100
```

**Why these weights**: Amount is the strongest signal in UK bank reconciliation. An exact match with a close date is nearly always correct. Date is down-weighted because Xero transactions are sometimes backdated. Description similarity is useful for disambiguation but unreliable alone because bank descriptions vary by institution.

**Why `token_sort_ratio`**: Bank descriptions often contain the same words in different order ("TESCO STORES UK" vs "UK STORES TESCO"). `token_sort_ratio` sorts tokens before comparing, making it robust to word-order variation.

**Node 3: decide**

Applies thresholds:

| Score | Status |
|---|---|
| Above 0.9 | `auto_matched` |
| 0.6 to 0.9 | `suggested` |
| Below 0.6 or two candidates within 0.05 of each other | `needs_review` |

**Node 4: explain**

Calls Claude to write one sentence: *"Matched to TESCO STORES on 14 Mar: amount matches exactly (£52.40), same day, description 87% similar."* This is the only LLM call in the reconciliation pipeline.

**Node 5: save**

Updates `BankStatement.matched_transaction_id`, `match_confidence`, and `match_status`. Writes `AuditLog`.

---

### 4. Explainable AI stack

`xai/explainer.py` and `xai/fuzzy_engine.py`

Every categorisation decision receives three layers of explanation, all stored in `AuditLog.ai_decision_data`:

#### Layer 1: LLM reasoning (always present)

Claude's `reasoning` field from the `CategoryPrediction` output. Plain English: *"This is likely a software subscription based on the ADOBE description, the £47.99 recurring amount, and the fact that 3 similar transactions have been categorised as Computer Equipment and Software."*

#### Layer 2: InterpretML EBM feature importances

Once 50 or more labelled transactions exist in the organisation, an Explainable Boosting Machine is trained on five features:

| Feature | How it is computed |
|---|---|
| `amount` | Absolute transaction value |
| `day_of_week` | Integer 0 to 6 |
| `description_length` | Character count of description |
| `vendor_frequency` | Count of transactions with a matching description prefix |
| `category_history_count` | Count of confirmed transactions in the predicted category |

EBM was chosen over a standard boosted tree because its predictions are additive by feature; each feature's contribution to the final score is directly readable and visualisable as a bar chart. It is a glass-box model.

Below 50 samples, the fallback is Layer 1 (the LLM's own reasoning text).

#### Layer 3: Mamdani fuzzy risk score (always present)

A custom Mamdani-style fuzzy inference system with triangular membership functions and centroid defuzzification, implemented from scratch in `fuzzy_engine.py` without an external fuzzy library. The implementation is self-contained so the inference logic can be audited and modified directly.

**Three input variables** (all normalised to the range [0, 1]):

| Variable | What it measures |
|---|---|
| `amount_deviation` | How far the amount deviates from the category average (z-score normalised) |
| `vendor_frequency` | How often this vendor appears in transaction history |
| `time_pattern` | Whether the transaction date is a weekday (0.9) or weekend (0.3) |

**Eight rules**:

```
IF amount_deviation IS high AND vendor_frequency IS rare     -> risk IS high
IF amount_deviation IS high AND vendor_frequency IS frequent -> risk IS medium
IF amount_deviation IS low  AND vendor_frequency IS frequent -> risk IS low
IF amount_deviation IS low  AND vendor_frequency IS rare     -> risk IS medium
IF amount_deviation IS medium                                -> risk IS medium
IF time_pattern IS unusual  AND vendor_frequency IS rare     -> risk IS high
IF time_pattern IS normal   AND amount_deviation IS low      -> risk IS low
IF amount_deviation IS high AND time_pattern IS unusual      -> risk IS high
```

**Why fuzzy logic over a neural anomaly detector**: Fuzzy rules are human-readable, deterministic (same input produces same output always), and transparent. An accountant can review and challenge the rules directly. A neural anomaly detector would be a black box. For an accounting tool where professional trust is the product, interpretability beats marginal accuracy gains.

**Output**: `risk_score` in [0, 1], `risk_label` in {low, medium, high}, and the list of fired rules in plain English; shown directly in the UI so the accountant understands exactly why a transaction was flagged.

---

### 5. Document generation

`services/document_service.py`

Follows a strict separation between computation and language:

**Step 1: Calculate figures (pure Python, no LLM)**
```
total_income, total_expenses, net_profit_or_loss
top_5_expense_categories (by spend)
largest_5_transactions
transaction_count, period_start, period_end
```

**Step 2: Retrieve context (pgvector)**

Semantic search over transactions for the period to find patterns and notable items relevant to the narrative.

**Step 3: Generate narrative (Claude via Instructor)**

The prompt contains all computed figures and vector-retrieved context. Claude writes five structured sections:

```python
class DocumentNarrative(BaseModel):
    executive_summary: str
    income_analysis: str
    expense_analysis: str
    cash_flow_observations: str
    recommendations: str
```

The LLM never touches the numbers directly; it writes around figures that have already been computed and verified in Python.

**Step 4: Render PDF (Jinja2 and WeasyPrint)**

A Jinja2 HTML template is populated with figures and narrative, then rendered to A4 PDF with WeasyPrint. Includes page numbers, professional typography, and an "AI-Assisted Draft" watermark footer. WeasyPrint was chosen over Puppeteer (too heavy for a Python backend) and ReportLab (requires programmatic x/y layout; HTML/CSS is far more natural for structured documents).

---

## Tech stack

### Backend

| Component | Choice | Why |
|---|---|---|
| Framework | FastAPI 0.115 | Async-native from day one; automatic OpenAPI docs; Pydantic v2 on every boundary |
| ORM | SQLAlchemy 2.0 (async) | pgvector support; Alembic migrations; typed `mapped_column` style |
| Database | PostgreSQL 16 + pgvector | Single DB handles both relational and vector queries; no separate vector database needed at this scale |
| Migrations | Alembic | Schema versioning; production startup runs `alembic upgrade head` automatically |
| AI agents | LangGraph | Explicit state machine; each node is a pure function, testable in isolation, state transitions are auditable |
| LLM | Anthropic Claude | Strong instruction-following for structured output; Haiku is extremely cost-efficient for high-volume categorisation |
| Structured output | Instructor + Pydantic v2 | Guarantees typed LLM output via Anthropic tool-use; never parses free text |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim, runs locally, zero API cost; sufficient for semantic similarity on short transaction descriptions |
| XAI | InterpretML (EBM), custom Mamdani fuzzy inference | Glass-box models; outputs are directly interpretable without post-hoc approximation |
| String matching | RapidFuzz | C-extension reimplementation of fuzzywuzzy; 10 to 100x faster, no GPL licence issues |
| PDF | WeasyPrint + Jinja2 | HTML/CSS authoring is natural for structured documents; pure Python, no Chromium dependency |
| HTTP client | httpx (async) | Async-native with identical sync API; HTTP/2 support; standard for modern async Python |
| Auth | JWT + Xero OAuth2 | Xero identity is the login identity in MVP; no separate user table |
| Cache | Redis | Optional; used for session state and future API response caching |

### Frontend

| Component | Choice |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS 3 |
| Components | shadcn/ui |
| Animation | Framer Motion |
| Notifications | Sonner (toast) |
| Icons | lucide-react |

### Infrastructure

| Component | Choice |
|---|---|
| Local dev | Docker Compose (PostgreSQL 16 + pgvector, Redis 7) |
| Backend hosting | Railway (Docker, auto-deploy on push) |
| Frontend hosting | Vercel (Next.js, auto-deploy on push) |
| Container | Python 3.12-slim multi-stage, WeasyPrint system deps pre-installed |

---

## Database schema

Seven tables. All data queries are scoped by `organisation_id`; there is no route that returns data across organisations.

```
organisations
  id (UUID PK)
  name, xero_tenant_id (UNIQUE)
  xero_access_token, xero_refresh_token, xero_token_expires_at
  last_sync_at, created_at, updated_at

accounts                           <- chart of accounts, synced from Xero
  id, organisation_id (FK)
  xero_id (UNIQUE), code, name, type, tax_type

transactions
  id, organisation_id (FK), account_id (FK nullable)
  xero_id (UNIQUE)
  date, amount (NUMERIC 12,2), description, reference
  category, category_confidence (NUMERIC 5,4)
  categorisation_status            <- uncategorised | auto_categorised | suggested | confirmed | rejected
  is_reconciled (BOOLEAN)
  embedding (Vector 384)           <- pgvector, used for few-shot lookup
  [INDEX: date, categorisation_status]

bank_statements
  id, organisation_id (FK)
  xero_id (UNIQUE)
  date, amount, description, reference
  matched_transaction_id (FK nullable)
  match_confidence (NUMERIC 5,4)
  match_status                     <- unmatched | auto_matched | suggested | confirmed
  [INDEX: match_status]

audit_logs                         <- append-only, no update or delete routes
  id, organisation_id (FK)
  action, entity_type, entity_id
  old_value, new_value (JSONB)     <- full entity snapshots before and after
  ai_model, ai_confidence, ai_explanation
  ai_decision_data (JSONB)         <- full XAI payload: features, risk score, fired rules
  created_at
  [INDEX: (entity_type, entity_id)]

generated_documents
  id, organisation_id (FK)
  template, period_start, period_end
  ai_model, figures (JSONB)
  generated_at
```

**Financial amounts are always `NUMERIC(12,2)` / Python `Decimal`.** IEEE 754 float cannot represent 0.1 exactly; a rounding error causing a penny discrepancy in a reconciliation match is a critical bug in an accounting tool. Float is never used for money.

**SPEND transactions are stored as negative amounts.** Consistent with double-entry accounting: the sum of all transactions equals the bank account balance.

**The audit log is append-only.** There is no DELETE or UPDATE route for `audit_logs`. It records the full XAI payload so any decision can be reconstructed, explained, and challenged after the fact; including for GDPR right-to-explanation requests.

---

## API reference

### Auth and data

```
GET  /auth/xero/connect                       Redirect to Xero OAuth2
GET  /auth/xero/callback?code=...             Exchange code for JWT session
GET  /api/v1/auth/me                          Current org (requires JWT)
POST /api/v1/auth/logout                      Clear session

POST /api/v1/sync                             Full Xero sync (accounts + transactions + statements)
GET  /api/v1/sync/status                      Last sync timestamp, row counts
GET  /api/v1/dashboard/summary                Aggregate stats
```

### Categorisation

```
POST /api/v1/categorise                       Batch-categorise all uncategorised transactions
GET  /api/v1/transactions                     List (paginated, filter by status/date/search)
GET  /api/v1/transactions/{id}                Single transaction + audit history
POST /api/v1/transactions/{id}/approve        Accept suggested category
POST /api/v1/transactions/{id}/correct        Override with correct category; re-embeds transaction
POST /api/v1/transactions/{id}/reject         Reject suggestion; reset to uncategorised
GET  /api/v1/transactions/{id}/explanation    Full XAI package
```

### Reconciliation

```
POST /api/v1/reconcile                        Batch-reconcile all unmatched bank statements
GET  /api/v1/bank-statements                  List (paginated, filter by match_status/date)
GET  /api/v1/bank-statements/{id}             Single statement + candidates + scores + audit
POST /api/v1/bank-statements/{id}/confirm     Accept suggested match
POST /api/v1/bank-statements/{id}/unmatch     Remove a match
POST /api/v1/bank-statements/{id}/match       Manually match to a specific transaction_id
```

### Documents

```
POST /api/v1/documents/generate               Generate PDF (body: template, period_start, period_end)
GET  /api/v1/documents                        List previously generated documents
```

### GDPR

```
GET    /api/v1/gdpr/export   Export all organisation data as JSON (Articles 15 and 20)
DELETE /api/v1/gdpr/erase    Permanently delete all organisation data  (Article 17)
```

Both endpoints require a valid JWT. The export excludes Xero OAuth tokens (credentials, not personal data) and raw embedding vectors (binary, not human-readable). The erase endpoint deletes rows in FK-safe order and the caller's session becomes invalid immediately.

### Explanations

```
GET  /api/v1/transactions/{id}/explanation

Response:
{
  "prediction": { "category_code", "category_name", "confidence", "reasoning" },
  "top_features": [{ "name", "value", "contribution" }],
  "model_type": "ebm" or "llm",
  "risk_score": 0.0 to 1.0,
  "risk_label": "low" or "medium" or "high",
  "fired_rules": ["IF amount deviation IS high AND...", ...],
  "input_values": { "amount_deviation", "vendor_frequency", "time_pattern" },
  "audit_trail": [{ "action", "created_at", "ai_model", "ai_confidence" }]
}
```

---

## Evaluation framework

`backend/evals/` contains a dedicated evaluation harness for the categorisation agent, independent of the application test suite.

### Fixture set

`fixtures/transactions.json` contains 50 labelled UK SME bank transactions:
- **31 easy**: Unambiguous (HMRC payments, well-known SaaS, payroll runs)
- **15 medium**: Require context (professional memberships, dual-category spends, inter-company transfers)
- **4 hard**: Industry-specific, balance-sheet vs P&L edge cases, unfamiliar vendors

`fixtures/accounts.json` contains 20 standard UK chart-of-accounts codes.

### Running evals

```bash
cd backend

# Dry run: rule-based mock, zero API cost
python -m evals.eval_runner --mode mock

# Live run with response caching (real API on first run, free on reruns)
python -m evals.eval_runner --mode live --budget 0.05

# Model comparison
python -m evals.eval_runner --mode live --model claude-haiku-4-5-20251001 --budget 0.05
python -m evals.eval_runner --mode live --model claude-sonnet-4-6 --budget 0.50

# Quick smoke test
python -m evals.eval_runner --mode live --limit 10 --budget 0.01
```

### Sample output

```
Overall accuracy: 88.0% (44/50)

By difficulty:
  easy   : 93.5% (29/31)  ████████████████████
  medium : 80.0% (12/15)  ████████████████
  hard   : 75.0%  (3/4)   ███████████████

Confidence calibration:
  high   (>0.85) : acc=95.0% (40 transactions)  <- auto-accept threshold
  medium (0.5 to 0.85): acc=71.4% ( 7 transactions) <- shown to accountant
  low    (<0.5)  : acc=33.3% ( 3 transactions)  <- flagged for review

Per-category F1:
  404 Computer Equipment & Software : F1=0.95
  820 Tax & Statutory Payments      : F1=1.00
  ...
```

### Acceptance criteria

The runner enforces these thresholds and exits with code 1 if any are missed:

| Metric | Minimum | Target | Enforced |
|---|---|---|---|
| Overall accuracy | 80% | 90%+ | ✅ |
| Easy tier accuracy (31 transactions) | 95% | 100% | ✅ |
| Auto-accept accuracy (conf > 0.85) | 90% | 95% | ✅ |
| Cost per transaction | < $0.01 | < $0.005 | — |
| F1 on core categories (HMRC, payroll, software) | 0.90 | 0.95+ | — |

### Cost controls

- `cost_tracker.py` counts tokens and enforces a `--budget` hard limit before the run starts
- `response_cache.py` caches responses to disk keyed by `SHA-256(model + prompt)`. All reruns against unchanged fixtures cost $0
- Default eval model is `claude-haiku-4-5-20251001` (approximately 10x cheaper than Sonnet). Switch to Sonnet only for pre-deployment quality checks

---

## Running locally

**Prerequisites**: Docker Desktop, Python 3.12, Node.js 18+

```bash
# 1. Clone and start the database
git clone <repo>
docker-compose up -d          # PostgreSQL 16 + pgvector, Redis

# 2. Backend
cd backend
cp .env.example .env
# Edit .env: fill in XERO_CLIENT_ID, XERO_CLIENT_SECRET, ANTHROPIC_API_KEY

pip install -e ".[dev,ai,docs]"
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev                   # http://localhost:3000
```

Environment variables (`backend/.env`):

```
DATABASE_URL=postgresql+asyncpg://aiaccountant:localdev123@localhost:5432/aiaccountant
DATABASE_URL_SYNC=postgresql://aiaccountant:localdev123@localhost:5432/aiaccountant
REDIS_URL=redis://localhost:6379
SECRET_KEY=change-me-in-production
XERO_CLIENT_ID=
XERO_CLIENT_SECRET=
XERO_REDIRECT_URI=http://localhost:8000/auth/xero/callback
ANTHROPIC_API_KEY=
FRONTEND_URL=http://localhost:3000
```

---

## Tests

```bash
cd backend

pytest -x -v                          # full test suite
pytest tests/test_health.py           # health endpoint
pytest tests/test_xero_integration.py # OAuth mock, sync methods
pytest tests/test_categorise.py       # categorisation agent (mocked Claude)
pytest tests/test_reconcile.py        # reconciliation scoring + Decimal edge cases
pytest tests/test_documents.py        # document generation
pytest tests/test_xai.py              # fuzzy engine + EBM explainer

cd frontend
npm run lint
npx tsc --noEmit
```

The test suite mocks all external API calls (Xero, Anthropic) so tests run offline without credentials.

---

## Cost model

### Per-operation costs

| Operation | Model | Approx. cost |
|---|---|---|
| Transaction categorisation | Claude Haiku | ~$0.0013 |
| Transaction categorisation | Claude Sonnet | ~$0.0047 |
| Reconciliation explanation | Claude Haiku | ~$0.0002 |
| Management letter | Claude Sonnet | ~$0.011 |
| Embeddings | sentence-transformers (local) | $0.00 |

### At scale

| Customers | Transactions per month | API cost per month |
|---|---|---|
| 5 (beta) | 1,000 | ~$0.52 |
| 20 | 4,000 | ~$2.06 |
| 100 | 20,000 | ~$10.30 |

At 100 customers paying £49 per month, API cost is under 0.2% of revenue.

### Cost optimisation mechanisms

1. **Auto-categorisation reduces volume**: as the pgvector few-shot store grows, more transactions hit the above-0.85 threshold and are categorised without prompting. Cost per new transaction trends toward zero as patterns repeat.
2. **Reconciliation matching uses no LLM**: only the explanation sentence calls Claude.
3. **Eval caching**: `response_cache.py` makes all eval reruns free after the first run.
4. **Concurrency cap**: `asyncio.Semaphore(5)` in `categorise_batch()` prevents runaway parallel API calls.
5. **Embeddings computed once**: stored in the `embedding` column and reused until a correction triggers re-embedding.

---

## Deployment

### Backend to Railway

```
# Railway reads this from Procfile:
web: alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Railway auto-provisions PostgreSQL 16 with pgvector and Redis. Set `ANTHROPIC_API_KEY`, `XERO_CLIENT_ID`, `XERO_CLIENT_SECRET`, `XERO_REDIRECT_URI`, and `SECRET_KEY` in the Railway dashboard. `DATABASE_URL` and `REDIS_URL` are injected automatically.

### Frontend to Vercel

Connect the repository to Vercel, set `NEXT_PUBLIC_API_URL` to the Railway backend URL, and deploy. Auto-deploy on push to `main`.

After deploying, update `XERO_REDIRECT_URI` in Railway and in the Xero Developer Portal to `https://yourdomain.com/auth/xero/callback`.

### Migrations in production

Alembic runs as part of the startup command; the server will not start if there are pending migrations, which prevents schema drift between deployments.

---

## GDPR compliance

The product handles financial data on behalf of UK accounting firms. The following table maps each relevant GDPR data-subject right to the code that fulfils it.

| Right | GDPR Article | How it is implemented |
|---|---|---|
| Right of access | Article 15 | `GET /api/v1/gdpr/export` returns all stored data as structured JSON |
| Right to data portability | Article 20 | Same endpoint; response is machine-readable JSON suitable for import into another system |
| Right to erasure | Article 17 | `DELETE /api/v1/gdpr/erase` deletes all org rows in FK-safe order; session invalidated immediately |
| Right to explanation | Article 22 | Every AI decision stores its full reasoning in `AuditLog.ai_decision_data`; surfaced via `GET /api/v1/transactions/{id}/explanation` |
| Data minimisation | Article 5(1)(c) | Xero OAuth tokens are excluded from exports; embedding vectors are excluded as derived, non-personal data |
| Security of processing | Article 32 | All data routes require Bearer JWT scoped to `organisation_id`; CORS restricts allowed HTTP methods and headers |

**Known gaps before production use**: OAuth tokens are currently stored in plaintext (acceptable for beta; must be AES-encrypted before handling regulated client data). There is no automated data retention policy; old audit logs are retained indefinitely.

---

## Research foundations

The AI and XAI components in this project are grounded in established research. The following references document the academic basis for the key techniques.

**Fuzzy logic inference (XAI risk scoring)**

Mamdani, E.H. and Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), pp.1-13.

The fuzzy risk engine in `xai/fuzzy_engine.py` implements Mamdani-style min-AND inference with centroid defuzzification, as described in this paper. The rule base is intentionally human-readable so domain experts can challenge and revise it without ML expertise. The membership function shapes and rule weights were chosen heuristically based on domain knowledge; formally deriving and empirically validating them against labelled risk outcomes is an open research question that a structured academic collaboration would address.

**Explainability in AI systems**

Hagras, H. (2018). Toward human-understandable, explainable AI. *Computer*, 51(9), pp.28-36.

Provides the theoretical framing for why interpretability must be designed in from the start rather than approximated after the fact. This informed the decision to use glass-box models (EBM, fuzzy logic) rather than post-hoc explanation of black-box models.

**Explainable Boosting Machine (feature importances)**

Lou, Y., Caruana, R. and Gehrke, J. (2012). Intelligible models for classification and regression. *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp.150-158.

EBM is the model class used in `xai/explainer.py` for feature importance once 50 or more labelled transactions are available. Its additive structure means each feature's contribution to the final prediction is directly readable without approximation.

**Shapley value theory (feature attribution)**

Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Provides the theoretical basis for the feature attribution design. EBM was chosen in part because its additive structure gives each feature a directly interpretable contribution that shares the consistency and local accuracy guarantees of Shapley values, without requiring a separate post-hoc attribution step. SHAP is not called directly; the choice of EBM as the glass-box model is informed by this work.

**Retrieval-augmented generation (document generation)**

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S. and Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33.

The management letter pipeline (`services/document_service.py`) follows the RAG pattern: retrieve relevant transaction context via pgvector similarity search, then condition the generative model on that retrieved context rather than relying on parametric memory alone.

---

## Open research questions

The following questions arose from building this system and remain genuinely open in the accounting AI literature. The infrastructure built here makes them tractable to study empirically.

**1. Per-organisation confidence calibration**

The agent uses fixed confidence thresholds (0.85 for auto-accept, 0.50 for suggest). These were chosen from the evaluation fixture set, which represents a generalised UK SME. In practice, different accounting firms have different transaction patterns: a construction firm with irregular project-based payments will have a different confidence distribution from a retail business with high-frequency card transactions. The open question is whether thresholds should be calibrated per organisation, and if so, what method (Platt scaling, isotonic regression, conformal prediction) produces the best-calibrated scores across firm types with limited labelled data.

**2. Explanation utility: do explanations change human decisions?**

The system generates three layers of explanation for every AI decision: LLM reasoning text, EBM feature contributions, and a fuzzy risk score. Whether these explanations actually improve accountant decision quality has not been measured. A controlled evaluation comparing review time, correction rate, and confidence of human decisions with and without the explanation panel would establish whether the added complexity is worthwhile, and which layer (text, feature bars, risk badge) carries the most decision-relevant information.

**3. EBM vs LLM fallback: the crossover point**

Below 50 labelled transactions, the system falls back to the LLM's reasoning text for feature attribution. Above 50, it trains an EBM. The threshold of 50 was chosen conservatively; the actual crossover point at which EBM outperforms the LLM fallback in this domain is unknown. A learning curve analysis varying labelled set size from 10 to 500 samples would identify this crossover and inform the cold-start strategy for new client onboarding.

**4. Active learning for human-in-the-loop efficiency**

Currently, human review is triggered passively: transactions below the confidence threshold are surfaced in arrival order. An active learning strategy would instead select transactions for review that would maximally reduce model uncertainty — for example, those in sparse regions of the embedding space or near the decision boundary. The question is whether intelligent selection reduces the number of human corrections required to reach a target accuracy, and by how much, compared to passive threshold-based review.

---

## Engineering challenges encountered during the build

This section documents non-obvious technical problems that arose during construction and how they were resolved. These are real issues that required investigation, not anticipated design decisions.

---

### Anthropic has no embeddings endpoint

**Problem**: The original design used OpenAI for both LLM classification and embeddings (`text-embedding-3-small`, 1536 dimensions). Midway through Phase 3, the decision was made to switch to Anthropic Claude as the LLM. Anthropic does not provide an embeddings API.

**Resolution**: Decoupled the two concerns entirely. Embeddings are now handled by `sentence-transformers/all-MiniLM-L6-v2` running locally in-process, producing 384-dimensional vectors at zero API cost. The pgvector column was resized from 1536 to 384 via an Alembic migration. `asyncio.to_thread()` offloads the CPU-bound embedding work from the async event loop. The LLM and embedding concerns are now fully independent and can be swapped separately.

---

### LangGraph nodes need database access without polluting graph state

**Problem**: LangGraph nodes are standalone functions that transform a state dict. Each node in the categorisation agent needs an async SQLAlchemy session for database queries. Passing the session through the state dict would pollute it (sessions are not serialisable) and break any future state persistence.

**Resolution**: Used a factory/closure pattern. `build_categoriser_graph(db: AsyncSession)` captures the session in a closure and returns the compiled graph. Each node is a nested function that closes over `db` without it ever appearing in the state dict. The same pattern is used in the reconciliation agent.

---

### Structured LLM output fails silently without Instructor

**Problem**: The first implementation called Claude and parsed the response with string splitting. This worked on simple cases but failed whenever Claude added preamble text ("Sure, here is the classification..."), changed field ordering, or used slightly different key names. The failures were silent: the parser returned a wrong category rather than raising an error.

**Resolution**: Replaced all free-text parsing with `instructor[anthropic]`. Instructor uses Anthropic's tool-use feature to force the model to return data conforming to a Pydantic model schema. If the output fails validation, Instructor retries automatically. The `CategoryPrediction` model is now the contract between the agent and Claude; any deviation raises a validation error rather than silently producing a bad prediction.

---

### Xero OAuth2 scope names changed as a breaking change in March 2026

**Problem**: Xero introduced granular OAuth2 scopes for applications created after 2 March 2026. The original scope list used broad format strings (`accounting.transactions.read`). The application was created on 22 March 2026, so all authorisation attempts returned `unauthorized_client: Invalid scope`.

**Resolution**: Iteratively tested scope name formats against Xero's token endpoint. The colon format (`accounting.transactions:read`) was also incorrect. The working format uses dot notation with specific resource names: `accounting.banktransactions.read`, `accounting.invoices.read`, `accounting.contacts.read`, `accounting.settings.read`. Updated the adapter and documented the breaking change in the decisions log.

---

### Python bytecode cache masking live code changes

**Problem**: After updating the OAuth2 scope names in `xero_adapter.py`, the running server continued serving the old (invalid) scopes despite uvicorn's `--reload` flag being active. The behaviour was reproducible: the corrected scope strings were in the source file, but the authorisation URL still contained the old ones.

**Resolution**: Python's `.pyc` bytecode cache files in `__pycache__/` directories preserved the compiled version of the old source. Uvicorn's file watcher detected the modification but the import system served the cached bytecode. Clearing all `__pycache__` directories and restarting the process resolved it. The lesson: `--reload` watches for file changes to trigger restart, but does not invalidate the bytecode cache independently.

---

### Naive `LIMIT 1` query returns the wrong organisation in a multi-tenant setup

**Problem**: During local testing with both a seed "Demo Company Ltd" organisation and a real connected "Agentic AI Demo" organisation in the database, the `_get_org()` helper used `SELECT ... LIMIT 1` without ordering or filtering. PostgreSQL's heap scan order is non-deterministic; the query returned the seed org (which had no Xero token) roughly half the time, causing every Xero API call to fail with a 400 error.

**Resolution**: Added `WHERE xero_access_token IS NOT NULL` to all `_get_org()` calls across route files. This ensures the query always returns the live connected organisation regardless of row storage order. A longer-term fix (scoping by JWT-derived organisation_id) is part of the Phase 8 auth work.

---

### Xero Demo Company bank statement data is not accessible via the Accounting API

**Problem**: The Xero UK Demo Company shows 29 bank statement lines in the reconciliation UI. The expectation was that `GET /BankTransactions` would return these. It returns zero results, as do `GET /Invoices` and `GET /Payments`.

**Resolution**: The Demo Company's bank statement lines are loaded via Xero's internal bank feed layer, which is separate from the Accounting API layer that third-party apps can access. They are visible in the UI but cannot be retrieved programmatically. The resolution was to create actual `BankTransactions` via the Xero UI (Spend Money entries) and use local seed data for AI pipeline validation rather than expecting the bank feed to be API-accessible.

---

### shadcn v4 installs Tailwind v4 CSS-first config into a Tailwind v3 project

**Problem**: `npx shadcn@latest init` installed shadcn v4, which assumes Tailwind CSS v4's CSS-first configuration (no `tailwind.config.ts`, all configuration in `globals.css` via `@import "tw-animate-css"` and `@import "shadcn/tailwind.css"`). The project was scaffolded with `create-next-app@14`, which generates a Tailwind v3 setup. The two approaches are incompatible and the build failed immediately.

**Resolution**: Rewrote `globals.css` to use standard Tailwind v3 directives (`@tailwind base`, `@tailwind components`, `@tailwind utilities`) with HSL CSS custom properties for all colour tokens. Rewrote `tailwind.config.ts` to map all shadcn colour tokens (`background`, `primary`, `border`, `ring`, etc.) to `hsl(var(--token-name))`. All shadcn components then work as expected against the v3 config.

---

### Next.js frontend initialised its own git repository inside the monorepo

**Problem**: `create-next-app` ran `git init` inside `frontend/`, creating a nested repository. When staged to the outer repo, Git treated `frontend/` as a submodule reference (mode `160000`) rather than tracking its files. The commit showed `frontend` as a single line with no contents; all frontend source was invisible in the repository.

**Resolution**: Removed `frontend/.git`, ran `git rm --cached frontend` to deregister the gitlink entry from the outer index, then re-added all frontend files as regular tracked files with `git add frontend/`. The outer repository now owns all frontend source directly with no submodule relationship.

---

### Decimal and float type mixing in reconciliation scoring

**Problem**: The reconciliation scoring pipeline mixes two numeric types: financial amounts from the database are `Decimal` (required for financial accuracy), while RapidFuzz returns similarity ratios as `float` in the range 0 to 100. Naive combination causes either a `TypeError` or silent precision loss depending on Python's coercion rules.

**Resolution**: Maintained a strict type boundary. All amount comparisons use `Decimal` arithmetic exclusively. The scoring formula converts amounts to `float` only for the final weighted sum (which is a statistical score, not a financial amount and carries no monetary meaning). The rule: financial data stays `Decimal`, ML scores stay `float`, and the two never mix in a financial calculation.

---

### GitHub Actions CI requires a pgvector-enabled PostgreSQL and native WeasyPrint dependencies

**Problem**: Standard GitHub Actions PostgreSQL service images do not include the pgvector extension. WeasyPrint requires system-level C libraries (Cairo, Pango, GLib, font packages) that are not present in a default Ubuntu runner. Both blockers produced CI failures that were not reproducible locally (where Docker and system libraries were already in place).

**Resolution**: Replaced the default `postgres` service image with `pgvector/pgvector:pg16`, which ships with the extension pre-installed. Added an `apt-get install` step to the CI job for WeasyPrint's system dependencies before installing Python packages. Ran `alembic upgrade head` inside the CI job to apply schema migrations before pytest executes. These three steps together make the CI environment match the local development environment structurally.

---

### FastAPI `dependency_overrides` is the correct test mechanism; `unittest.mock.patch` is not

**Problem**: Integration tests were using `patch("app.api.v1.documents._get_org", ...)` to mock authentication. The routes use `Depends(get_current_org)` from a shared session module; there is no `_get_org` attribute on the route module. The tests raised `AttributeError` in CI. A secondary problem: tests expected HTTP 404 for unauthenticated requests, but `get_current_org` raises `401` for all auth failures (no token, bad token, org not in database) before any database lookup that could produce a 404.

**Resolution**: Replaced all `patch(...)` calls with `app.dependency_overrides[get_current_org] = lambda: fake_org`, using `try/finally` to clean up overrides after each test. For "no auth" tests, the correct assertion is `status_code == 401`; no mock_db setup is needed because the 401 fires before the database is touched. For "resource not found" tests (where auth succeeds but the record does not exist), both `get_current_org` and `get_db` must be overridden separately.

---

### Tightening CORS to an explicit origin silently broke all API calls in production

**Problem**: CORS was changed from `allow_origins=["*"]` to `allow_origins=[settings.frontend_url]` as a security improvement. `settings.frontend_url` defaults to `"http://localhost:3000"` when the environment variable is not set. In production the Vercel frontend has a different origin, so the browser's preflight check rejected every API call before it left the browser. The `fetch()` call threw a `TypeError` (no HTTP response, no status code), the dashboard's `catch` block set `error=true`, and the "Xero not connected" screen appeared. OAuth still worked because OAuth uses browser navigation (server-side redirect), not `fetch`, so CORS does not apply. This created a loop: connect Xero, get redirected back to the dashboard, see "not connected", connect again.

**Resolution**: Reverted to `allow_origins=["*"]` with `allow_credentials=False`. Bearer-token authentication does not use cookies, so `allow_credentials=False` is both correct and required when using a wildcard origin. The key diagnostic insight was distinguishing between a CORS failure (a browser-level `TypeError`, no HTTP status visible to application code) and an authentication failure (HTTP 401, handled by the app).

---

## Limitations

These are genuine current limitations, not caveats to dismiss:

**Categorisation accuracy depends on history volume.** The few-shot pgvector lookup improves as the accountant corrects predictions and the vector store grows. A brand-new organisation with zero labelled transactions gets only chart-of-accounts context; accuracy will be lower until approximately 50 confirmed transactions exist. The eval framework's acceptance criteria account for this, but new users should expect a short calibration period.

**Confidence thresholds are not calibrated per client.** The 0.85 and 0.50 thresholds were chosen based on the evaluation fixture set. Different firms have different transaction patterns, and a threshold that works well for a retail client may be too aggressive for a construction firm with irregular payments. Threshold tuning per organisation is planned but not implemented.

**Fuzzy risk scoring uses heuristic input normalisation.** `vendor_frequency` is normalised by dividing raw count by 20 (i.e., 20 or more appearances = "frequent"). This cap is arbitrary and may not fit all organisation sizes. A firm with 5,000 transactions per month will have different frequency distributions than one with 200.

**Xero sync is full-page, not incremental.** `sync_transactions()` fetches all pages on every sync. This is acceptable for SMEs but will be slow for organisations with thousands of transactions. `If-Modified-Since` incremental sync is designed but not yet implemented.

**OAuth tokens are stored in plaintext.** Acceptable for local development and early beta, but must be encrypted at rest (application-level AES or Railway's encryption at rest) before handling production client data.

**Single tenant per Xero organisation.** One Xero firm equals one tenant. Multi-user access within a firm is not implemented. All queries are scoped to `organisation_id` only; user-level permissions within an organisation are post-beta.

**EBM requires 50 or more labelled samples.** Below this threshold, Layer 2 (feature importance) falls back to the LLM's reasoning text. New organisations will rely solely on Layers 1 and 3 (LLM reasoning and fuzzy risk) until sufficient data accumulates.

**PDF rendering is CPU-intensive.** WeasyPrint renders HTML to PDF synchronously. For large management letters it can take 2 to 5 seconds. This is acceptable for an on-demand operation but would block the event loop if called frequently. It should be moved to a background worker before scaling.

**No QuickBooks integration yet.** The XeroAdapter is the only accounting platform integration. The architecture is designed to accommodate additional adapters (the data models are platform-agnostic), but QuickBooks OAuth2 and API mapping is not implemented.

---

## Future work

Items scoped out of the current build, ordered by likely priority:

| Item | Reason not built yet |
|---|---|
| Incremental Xero sync (`If-Modified-Since`) | Not needed at MVP scale; adds complexity |
| Per-organisation confidence threshold tuning | Requires sufficient data to calibrate; post-beta |
| QuickBooks adapter | Second platform; same architecture, different OAuth2 flow |
| Token encryption at rest | Planned for pre-beta; currently plaintext in dev |
| Stripe billing integration | Post-beta; add when users confirm willingness to pay |
| Multi-user within an organisation | Post-beta; user table + RBAC middleware |
| Background PDF rendering | Needed at scale; move WeasyPrint to Celery worker |
| GDPR formal compliance audit | Basic data export and erasure endpoints are implemented; formal ICO registration and DPA review are post-revenue |
| Additional document templates | Tax letters, VAT returns; build based on user requests |
| Fine-tuning on categorisation corrections | Potential accuracy improvement; expensive; evaluate after sufficient data |

---

## Absolute invariants

Rules that must survive refactoring and new contributors:

| Rule | Reason |
|---|---|
| `Decimal` for all money (`NUMERIC(12,2)` in DB) | Float rounding causes penny errors in reconciliation; a critical trust failure |
| Every AI decision writes to `AuditLog` | Professional accountability; GDPR right to explanation |
| LLM calls only in `services/` or `agents/` | Routes must never call Claude directly |
| Xero calls only via `XeroAdapter` | Centralised token refresh, rate limiting, retry logic |
| Structured LLM output via Instructor | Never parse free text from LLM |
| Tests before commit | Minimum: one happy-path test per new endpoint or service method |
| SPEND transactions stored as negative | Consistent with double-entry accounting |
| Confidence thresholds are not hardcoded | Will need tuning per customer; keep them configurable |

---

## Project structure

```
.
├── docker-compose.yml               PostgreSQL 16 + pgvector, Redis
├── docs/
│   └── ARCHITECTURE.md             Full design rationale and tech decisions
└── backend/
    ├── pyproject.toml
    ├── Dockerfile
    ├── Procfile                     Railway deployment command
    ├── alembic/                     4 migrations (schema, embedding resize, documents)
    └── app/
        ├── main.py                  FastAPI app, routers, CORS, lifespan
        ├── core/                    config, database, session/JWT
        ├── models/                  SQLAlchemy models + Pydantic v2 schemas
        ├── integrations/            xero_adapter.py
        ├── services/                embedding_service.py, document_service.py
        ├── agents/                  categoriser.py, reconciler.py (LangGraph)
        ├── xai/                     explainer.py (EBM), fuzzy_engine.py (custom Mamdani)
        ├── api/v1/                  health, auth, sync, dashboard, categorise,
        │                            reconcile, documents, explanations
        ├── templates/               management_letter.html (Jinja2 + WeasyPrint)
        └── tests/ + evals/          test suite + evaluation harness
frontend/
    └── src/
        ├── app/                     page.tsx (landing), dashboard, transactions,
        │                            reconciliation, documents, auth/callback, privacy
        ├── components/              app-shell, sidebar, aurora, explanation-panel,
        │                            gradient-text, fluid-glass-button, shadcn/ui
        └── lib/                     api.ts (fetch wrapper + types), utils.ts
```

---

## Licence

MIT

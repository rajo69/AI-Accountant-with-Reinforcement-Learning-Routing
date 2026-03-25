# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Identity
This is the RL Routing Research Extension of the AI Accountant.
Parent project context is documented below.
New goal: Replace hand-tuned confidence thresholds in CategoriserAgent
with a learned PPO routing policy. See RL_ROUTING_PROJECT.md.

---

# AI ACCOUNTANT — MASTER BUILD PLAN

> **This file is both the CLAUDE.md and the execution plan.**
> Claude Code: read this entire file at session start. Execute the next
> incomplete phase. When you hit a 🛑 STOP marker, pause and wait for the
> user to complete the action before continuing. When you hit a 🔄 RE-ENTRY
> marker, tell the user to exit (/exit) and start a new session.

---

## PROJECT IDENTITY

**Product**: AI-powered Xero assistant for UK accountants
**Features**: Transaction categorisation, bank reconciliation, document generation, explainable AI
**Builder**: Solo founder, full-time
**Goal**: Beta launch with real users in 12 weeks

---

## TECH STACK

- Backend: Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2
- AI/Agents: LangGraph, LangChain, **Anthropic Claude API** (not OpenAI), Instructor (`instructor[anthropic]`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, no API key needed)
- XAI: SHAP, InterpretML (EBM), Simpful (fuzzy logic)
- Database: PostgreSQL 16 + pgvector extension (Docker locally, Railway for prod); Vector columns are **384-dim**
- Cache: Redis
- Frontend: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Framer Motion
- Hosting: Railway (backend + DB + Redis), Vercel (frontend)
- PDF: WeasyPrint + Jinja2
- String matching: RapidFuzz

---

## ABSOLUTE RULES (NEVER VIOLATE)

1. **Decimal for ALL money.** Never use float for financial amounts. Use `Decimal` in Python, `NUMERIC(12,2)` in PostgreSQL.
2. **Audit every AI decision.** Every prediction/classification/match logs: input data, output, confidence, explanation text, model version, timestamp.
3. **LLM calls only in services/ or agents/.** API routes never call Anthropic/LLM directly. Always go through a service layer.
4. **Xero calls only through the adapter.** All Xero API interactions go through `integrations/xero_adapter.py`.
5. **Tests before commit.** At minimum, one happy-path test per new endpoint or service method.
6. **Structured LLM output.** Use Instructor + Pydantic models to force structured responses from LLMs. Never parse free text.
7. **Ship simple.** No over-engineering. No premature abstraction. Build what's needed now.

---

## COMMANDS

```
Start database:       docker-compose up -d db
Start all services:   docker-compose up -d
Start backend:        cd backend && uvicorn app.main:app --reload --port 8000
Run tests:            cd backend && pytest -x -v
Run single test:      cd backend && pytest tests/test_specific.py -v
Run evals (mock):     cd backend && python -m evals.run --mock
Run evals (live):     cd backend && python -m evals.run --cache
Run lint:             cd backend && ruff check .
Start frontend:       cd frontend && npm run dev
Lint frontend:        cd frontend && npm run lint
Type-check frontend:  cd frontend && npx tsc --noEmit
Run migrations:       cd backend && alembic upgrade head
Create migration:     cd backend && alembic revision --autogenerate -m "description"
```

---

## ARCHITECTURE

All phases are complete. Here is the current state of the system.

### Data flow

```
Xero API
  └─► XeroAdapter (integrations/xero_adapter.py)
        └─► sync_accounts / sync_transactions / sync_bank_statements
              └─► PostgreSQL (Organisation, Account, Transaction, BankStatement)

Uncategorised transactions
  └─► CategoriserAgent (agents/categoriser.py) — LangGraph graph
        ├─ fetch_context: chart of accounts + pgvector similar examples
        ├─ classify: Anthropic Claude via Instructor → CategoryPrediction
        ├─ validate: category_code exists, confidence 0–1
        ├─ decide: >0.85 auto_categorised | 0.5–0.85 suggested | <0.5 needs_review
        └─ explain: EmbeddingService + XAI explainer + fuzzy risk engine
              └─► AuditLog (every decision, with ai_decision_data JSONB)

Unmatched bank statements
  └─► ReconcilerAgent (agents/reconciler.py) — LangGraph graph, no LLM
        ├─ find_candidates: amount ±£0.01, date ±5 business days
        ├─ score_candidates: (amount×0.5) + (date×0.2) + (description×0.3)
        ├─ decide: >0.9 auto_match | 0.6–0.9 suggest | <0.6 human review
        ├─ explain: natural language match explanation
        └─► AuditLog

Documents
  └─► DocumentService (services/document_service.py)
        ├─ calculate figures (pure Python, no LLM)
        ├─ pgvector context retrieval
        ├─ Anthropic Claude via Instructor → narrative sections
        └─► WeasyPrint PDF
```

### Key structural patterns

- **Session scoping**: all DB queries filter by `organisation_id` derived from the JWT session
- **Confidence thresholds**: categoriser uses 0.85/0.5; reconciler uses 0.9/0.6 — don't change without updating tests
- **Embeddings**: `EmbeddingService` uses sentence-transformers (384-dim). The `Transaction.embedding` column is `Vector(384)`. The `backfill_embeddings()` method re-embeds when a user corrects a category, so corrections become training examples for future predictions.
- **XAI layers**: (1) LLM reasoning text always present, (2) InterpretML EBM when ≥50 labelled samples, (3) Simpful fuzzy risk score always present. All three stored in `AuditLog.ai_decision_data`.
- **Evals**: `backend/evals/` has a 50-transaction fixture set (31 easy / 15 medium / 4 hard). Acceptance criteria (enforced by runner exit code): 80% overall, 95% easy, 90% auto-accept rate. Run before any change to the categoriser.
- **Xero OAuth tokens**: stored in plaintext for MVP. Phase 8 notes plan for encryption. Don't add complexity until needed.

---

## PROGRESS TRACKING

After completing each phase, update the `## PHASE STATUS` section below by changing `[ ]` to `[x]`. This is how you know where to resume.

### PHASE STATUS

- [x] Phase 1: Project scaffolding (FastAPI + Docker + DB models + migrations)
- [x] Phase 2: Xero OAuth2 and data sync
- [x] Phase 3: Transaction categorisation agent (LangGraph)
- [x] Phase 4: Bank reconciliation agent
- [x] Phase 5: Frontend — dashboard + transaction review UI
- [x] Phase 6: Document generation (RAG + PDF)
- [x] Phase 7: XAI explanations (SHAP + fuzzy logic)
- [x] Phase 8: Landing page + auth + deploy
- [x] Phase 9: Beta launch prep

### DECISIONS LOG

- 2026-03-22: Use Anthropic Claude API instead of OpenAI (user preference). Replaced `openai_api_key` with `anthropic_api_key` in config. Updated `pyproject.toml` ai extras to use `anthropic`, `langchain-anthropic`, `instructor[anthropic]`.
- 2026-03-22: Xero adapter implemented with raw httpx (not the xero-python SDK) for OAuth2 and all API calls. Cleaner and more controllable. xero-python still installed as optional dep.
- 2026-03-22: Added `last_sync_at` column to `organisations` table (migration: a1b2c3d4e5f6). Required for dashboard summary.
- 2026-03-22: Switched embedding model to sentence-transformers `all-MiniLM-L6-v2` (384 dims, no extra API key). Updated Vector dimension from 1536→384 (migration: b2c3d4e5f6a7). Anthropic Claude is used for LLM classification via instructor[anthropic].
- 2026-03-24: Added GitHub Actions CI (`.github/workflows/ci.yml`). Two jobs: `backend` runs pytest with a `pgvector/pgvector:pg16` PostgreSQL service container; `frontend` runs `tsc --noEmit` + `eslint`. Installs WeasyPrint system deps, runs `alembic upgrade head` before tests.
- 2026-03-24: Added GDPR endpoints: `GET /api/v1/gdpr/export` (Art.15/20 data export as JSON) and `DELETE /api/v1/gdpr/erase` (Art.17 right to erasure, deletes rows in FK-safe order). Located in `backend/app/api/v1/gdpr.py`.
- 2026-03-24: Removed `simpful>=2.12.0` from pyproject.toml. `xai/fuzzy_engine.py` is a hand-rolled Mamdani fuzzy inference engine, never used Simpful. README and ARCHITECTURE.md updated to say "custom Mamdani fuzzy inference".
- 2026-03-24: CORS reverted to `allow_origins=["*"]` with `allow_credentials=False`. Previous explicit-origin approach (`settings.frontend_url`) broke production because the Vercel frontend's origin didn't match the Railway default of `http://localhost:3000`. Bearer-token auth does not need `allow_credentials=True`.
- 2026-03-24: Fixed 3 rounds of CI test failures. (1) `test_categorise.py` passed `Depends(...)` objects as values; fixed by passing `org=org` explicitly. (2) `test_documents.py` patched a non-existent `_get_org` helper; fixed with `app.dependency_overrides[get_current_org]`. (3) `test_reconcile.py` expected 404 for unauthenticated routes; `get_current_org` always raises 401, never 404.

### KNOWN ISSUES

- **Test pattern for auth**: All routes use `Depends(get_current_org)`. In tests always use `app.dependency_overrides[get_current_org] = lambda: fake_org`. Never patch a `_get_org` helper (it doesn't exist). `get_current_org` raises **401** for all auth failures (no token / bad token / org not found in DB) — never 404.
- **CORS**: Currently `allow_origins=["*"]`. Tighten to the production Vercel domain once it is stable. Set `FRONTEND_URL` env var on Railway and change `main.py` to `allow_origins=[settings.frontend_url]` with `allow_credentials=False`.
- **OAuth tokens**: Stored in plaintext in the DB. Encrypt before any public beta with real client data.

---
---

## PHASE 1: PROJECT SCAFFOLDING

**Goal**: FastAPI backend running, PostgreSQL with pgvector in Docker, all database models, Alembic migrations, health check passing.

### Step 1.1 — Create docker-compose.yml in project root

```yaml
version: '3.8'

services:
  db:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: aiaccountant
      POSTGRES_PASSWORD: localdev123
      POSTGRES_DB: aiaccountant
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aiaccountant"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### Step 1.2 — Create backend/pyproject.toml

```toml
[project]
name = "ai-accountant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "pgvector>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]
xero = [
    "xero-python>=4.0.0",
]
ai = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "instructor>=1.4.0",
    "openai>=1.40.0",
    "shap>=0.45.0",
    "interpret>=0.6.0",
    "simpful>=2.12.0",
    "rapidfuzz>=3.9.0",
]
docs = [
    "weasyprint>=62.0",
    "jinja2>=3.1.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### Step 1.3 — Create backend file structure

Create these files with the following content:

**backend/.env.example**
```
DATABASE_URL=postgresql+asyncpg://aiaccountant:localdev123@localhost:5432/aiaccountant
DATABASE_URL_SYNC=postgresql://aiaccountant:localdev123@localhost:5432/aiaccountant
REDIS_URL=redis://localhost:6379
SECRET_KEY=change-me-in-production
XERO_CLIENT_ID=
XERO_CLIENT_SECRET=
XERO_REDIRECT_URI=http://localhost:8000/auth/xero/callback
OPENAI_API_KEY=
```

**backend/app/__init__.py** — empty file

**backend/app/core/__init__.py** — empty file

**backend/app/core/config.py**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://aiaccountant:localdev123@localhost:5432/aiaccountant"
    database_url_sync: str = "postgresql://aiaccountant:localdev123@localhost:5432/aiaccountant"
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "change-me-in-production"
    xero_client_id: str = ""
    xero_client_secret: str = ""
    xero_redirect_uri: str = "http://localhost:8000/auth/xero/callback"
    openai_api_key: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
```

**backend/app/core/database.py**
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
```

**backend/app/models/__init__.py** — empty file

**backend/app/models/database.py** — Create SQLAlchemy models:
- Organisation: id (UUID PK), name, xero_tenant_id (unique), xero_access_token (Text nullable), xero_refresh_token (Text nullable), xero_token_expires_at (DateTime nullable), created_at, updated_at
- Account: id (UUID PK), organisation_id (FK), xero_id (String unique), code, name, type, tax_type (nullable), created_at, updated_at
- Transaction: id (UUID PK), organisation_id (FK), account_id (FK nullable), xero_id (String unique), date (Date), amount (Numeric(12,2)), description (Text), reference (String nullable), category (String nullable), category_confidence (Numeric(5,4) nullable), categorisation_status (String default "uncategorised" — values: uncategorised/auto_categorised/suggested/confirmed/rejected), is_reconciled (Boolean default False), embedding (Vector(1536) nullable), created_at, updated_at
- BankStatement: id (UUID PK), organisation_id (FK), xero_id (String unique), date (Date), amount (Numeric(12,2)), description (Text), reference (String nullable), matched_transaction_id (FK nullable), match_confidence (Numeric(5,4) nullable), match_status (String default "unmatched" — values: unmatched/auto_matched/suggested/confirmed), created_at, updated_at
- AuditLog: id (UUID PK), organisation_id (FK), action (String), entity_type (String), entity_id (UUID), old_value (JSONB nullable), new_value (JSONB nullable), ai_model (String nullable), ai_confidence (Numeric(5,4) nullable), ai_explanation (Text nullable), ai_decision_data (JSONB nullable), created_at (server_default=now())

Use `mapped_column` style (SQLAlchemy 2.0). Import Vector from pgvector.sqlalchemy. Add `__tablename__` to each. Add indexes on: Transaction.date, Transaction.categorisation_status, BankStatement.match_status, AuditLog.entity_type + entity_id.

**backend/app/models/schemas.py** — Create Pydantic v2 schemas (BaseModel with `model_config = ConfigDict(from_attributes=True)`) for each model: OrganisationRead, AccountRead, TransactionRead, TransactionUpdate, BankStatementRead, AuditLogRead. Plus: SyncResponse, CategoriseResponse, ReconcileResponse, HealthResponse.

**backend/app/api/__init__.py** — empty file

**backend/app/api/v1/__init__.py** — empty file

**backend/app/api/v1/health.py** — GET /health returning {"status": "healthy", "version": "0.1.0"}

**backend/app/main.py** — FastAPI app with:
- CORS middleware (allow all origins during dev)
- Include health router
- Lifespan handler that creates tables on startup (using Base.metadata.create_all)

**backend/tests/__init__.py** — empty file

**backend/tests/test_health.py** — Test that GET /health returns 200 with status "healthy"

### Step 1.4 — Set up Alembic

Run `alembic init alembic` inside backend/. Configure alembic/env.py to:
- Import Base from app.core.database and all models from app.models.database
- Use settings.database_url_sync for the connection string
- Set target_metadata = Base.metadata

Then generate initial migration and apply it.

### Step 1.5 — Verify everything works

1. Copy .env.example to .env: `cp backend/.env.example backend/.env`
2. Start Docker: `docker-compose up -d`
3. Install deps: `cd backend && pip install -e ".[dev]"`
4. Run migration: `alembic upgrade head`
5. Start server: `uvicorn app.main:app --reload`
6. Test health: `curl http://localhost:8000/health`
7. Run tests: `pytest -x -v`

**All must pass before moving on.**

### Step 1.6 — Commit and update progress

```bash
git add .
git commit -m "feat: backend scaffolding — FastAPI, SQLAlchemy models, Alembic, Docker PostgreSQL"
```

Mark Phase 1 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — Tell user: "Phase 1 is complete. We have a running backend with database. I recommend doing `/exit` and starting a fresh session to keep context clean for Phase 2 (Xero integration). When you restart, tell me to read this file and continue from Phase 2."

---
---

## PHASE 2: XERO OAUTH2 AND DATA SYNC

**Goal**: Connect to Xero, authenticate via OAuth2, pull accounts and transactions into our database.

🛑 **STOP** — Before starting this phase, ask the user:
> "I need your Xero API credentials to build the integration. Have you:
> 1. Created a Xero developer account at https://developer.xero.com/?
> 2. Created an app in My Apps with redirect URI: http://localhost:8000/auth/xero/callback?
> 3. Created a Demo Company (for test data)?
> 4. Added XERO_CLIENT_ID and XERO_CLIENT_SECRET to backend/.env?
>
> Please complete these steps and tell me when ready."

### Step 2.1 — Install Xero dependency

```bash
cd backend && pip install -e ".[xero]"
```

### Step 2.2 — Create the Xero adapter

**backend/app/integrations/__init__.py** — empty file

**backend/app/integrations/xero_adapter.py** — Create a class `XeroAdapter` with:

- `__init__(self, organisation_id, db_session)` — loads org from DB to get tokens
- `get_auth_url()` — class method, returns Xero OAuth2 authorization URL with scopes: `openid profile email accounting.transactions.read accounting.contacts.read accounting.settings.read offline_access`
- `handle_callback(code, db_session)` — exchanges auth code for tokens, creates/updates Organisation in DB, stores encrypted tokens
- `_ensure_valid_token()` — checks if access token is expired, refreshes if needed using refresh token, updates DB
- `sync_accounts()` — calls Xero Accounts API, upserts each account into our Account table, returns count
- `sync_transactions()` — calls Xero BankTransactions API (use `If-Modified-Since` for incremental), upserts into Transaction table, returns count
- `sync_bank_statements()` — calls Xero BankStatements API, upserts into BankStatement table, returns count
- `full_sync()` — runs all three syncs in sequence, returns SyncResponse with counts and timing

Use the `xero-python` SDK. Handle rate limits (Xero allows 60 requests/minute) with retry + backoff. Log all API calls.

### Step 2.3 — Create auth and sync API routes

**backend/app/api/v1/auth.py**:
- `GET /auth/xero/connect` — returns redirect to Xero auth URL
- `GET /auth/xero/callback` — handles OAuth callback, stores tokens, returns success message with org name

**backend/app/api/v1/sync.py**:
- `POST /api/v1/sync` — triggers full Xero sync for the connected organisation, returns SyncResponse
- `GET /api/v1/sync/status` — returns last sync time and counts

**backend/app/api/v1/dashboard.py**:
- `GET /api/v1/dashboard/summary` — returns: total_accounts, total_transactions, uncategorised_count, unreconciled_count, last_sync_at

Register all new routers in main.py.

### Step 2.4 — Write tests

- Test the OAuth URL generation (mock Xero SDK)
- Test the sync methods with mocked Xero API responses
- Test the dashboard summary endpoint

### Step 2.5 — Manual test

🛑 **STOP** — Tell the user:
> "The Xero integration is built. Let's test it live:
> 1. Make sure Docker is running: `docker-compose up -d`
> 2. Start the server: `cd backend && uvicorn app.main:app --reload`
> 3. Open your browser and go to: http://localhost:8000/auth/xero/connect
> 4. You'll be redirected to Xero — log in and authorize the app
> 5. After redirect back, you should see a success message
> 6. Then trigger a sync: `curl -X POST http://localhost:8000/api/v1/sync`
> 7. Check the dashboard: `curl http://localhost:8000/api/v1/dashboard/summary`
>
> Tell me what happens at each step."

### Step 2.6 — Commit

```bash
git add .
git commit -m "feat: Xero OAuth2 integration and full data sync"
```

Mark Phase 2 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 2 is complete. Xero data is syncing. Do `/exit` and start a fresh session for Phase 3 — this is the core AI feature and deserves a clean context window."

---
---

## PHASE 3: TRANSACTION CATEGORISATION AGENT

**Goal**: LangGraph agent that categorises uncategorised transactions using LLM + few-shot examples from pgvector, with confidence scoring and audit logging.

🛑 **STOP** — Ask the user:
> "I need your OpenAI API key for the AI features. Have you:
> 1. Created an OpenAI account at https://platform.openai.com/?
> 2. Added credits (pay-as-you-go, ~£10 is enough to start)?
> 3. Created an API key and added it as OPENAI_API_KEY in backend/.env?
>
> Alternatively, we can use the Anthropic API (Claude) instead of OpenAI.
> Which do you prefer?"

### Step 3.1 — Install AI dependencies

```bash
cd backend && pip install -e ".[ai]"
```

### Step 3.2 — Create embedding service

**backend/app/services/__init__.py** — empty file

**backend/app/services/embedding_service.py**:
- `embed_text(text: str) -> list[float]` — calls OpenAI text-embedding-3-small, returns 1536-dim vector
- `embed_transaction(transaction: Transaction) -> list[float]` — creates text from transaction fields (date, amount, description, reference), embeds it
- `find_similar_transactions(text: str, org_id: UUID, limit: int = 5) -> list[Transaction]` — queries pgvector for nearest neighbours among categorised transactions
- `backfill_embeddings(org_id: UUID)` — embeds all transactions that don't have embeddings yet

### Step 3.3 — Create the categorisation agent

**backend/app/agents/__init__.py** — empty file

**backend/app/agents/categoriser.py**:

Define the LangGraph state:
```python
class CategoriserState(TypedDict):
    transaction_id: str
    transaction_data: dict
    chart_of_accounts: list[dict]
    similar_examples: list[dict]
    prediction: dict | None  # {category_code, category_name, confidence, reasoning}
    status: str  # "pending", "classified", "validated", "decided"
```

Define the graph nodes:
1. **fetch_context** — loads chart of accounts from DB, finds 5 similar categorised transactions via pgvector
2. **classify** — builds prompt with transaction details + chart of accounts + few-shot examples, calls LLM via Instructor forcing output into a Pydantic model: `CategoryPrediction(category_code: str, category_name: str, confidence: float, reasoning: str)`
3. **validate** — checks: does predicted category_code exist in chart of accounts? Is the confidence value between 0 and 1? If validation fails, set confidence to 0 and reasoning to "validation failed"
4. **decide** — if confidence > 0.85: set status to "auto_categorised" and update transaction in DB. If 0.5-0.85: set status to "suggested". If < 0.5: set status to "needs_review". In ALL cases: create AuditLog entry.

Define edges: fetch_context → classify → validate → decide

Create runner function: `async def categorise_batch(org_id: UUID, db: AsyncSession) -> CategoriseResponse` — fetches all uncategorised transactions, runs each through the graph (with asyncio concurrency limit of 5), returns summary counts.

### Step 3.4 — Create categorisation API routes

**backend/app/api/v1/categorise.py**:
- `POST /api/v1/categorise` — triggers batch categorisation, returns {auto_categorised: N, suggested: N, needs_review: N, errors: N}
- `GET /api/v1/transactions` — list all transactions with filters: status, date_from, date_to, search. Paginated.
- `GET /api/v1/transactions/{id}` — single transaction with its audit history
- `POST /api/v1/transactions/{id}/approve` — confirms a suggested category, updates status to "confirmed", logs to audit
- `POST /api/v1/transactions/{id}/correct` — user provides correct category, updates transaction, logs correction to audit, re-embeds the transaction (so it becomes a training example for future predictions)
- `POST /api/v1/transactions/{id}/reject` — marks suggestion as rejected, resets to uncategorised

### Step 3.5 — Write tests

- Test the categorise agent with mocked LLM responses
- Test approve/correct/reject endpoints
- Test that corrections appear as few-shot examples for similar future transactions

### Step 3.6 — Commit

```bash
git add .
git commit -m "feat: LangGraph transaction categorisation agent with pgvector few-shot learning"
```

Mark Phase 3 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 3 is complete. The core AI feature works. `/exit` and restart for Phase 4."

---
---

## PHASE 4: BANK RECONCILIATION AGENT

**Goal**: LangGraph agent that matches bank statement lines to transactions using amount/date/description similarity, with confidence scoring.

### Step 4.1 — Create the reconciliation agent

**backend/app/agents/reconciler.py**:

State:
```python
class ReconcilerState(TypedDict):
    bank_statement_id: str
    bank_statement_data: dict
    candidates: list[dict]  # potential transaction matches with scores
    best_match: dict | None
    match_confidence: float
    explanation: str
    status: str
```

Nodes:
1. **find_candidates** — query transactions table for: amount within £0.01, date within 5 business days of the bank statement date, same organisation. Return up to 10 candidates.
2. **score_candidates** — for each candidate compute:
   - amount_score: 1.0 if exact match (Decimal comparison), 0.8 if within 1%, 0.0 otherwise
   - date_score: 1.0 if same day, subtract 0.15 per day difference, minimum 0.0
   - description_score: RapidFuzz token_sort_ratio / 100
   - combined_score: (amount_score × 0.5) + (date_score × 0.2) + (description_score × 0.3)
3. **decide** — if top score > 0.9: auto_match. If 0.6-0.9: suggest. If < 0.6 or multiple candidates within 0.05 of each other: flag for human review.
4. **explain** — generate natural language: "Matched to [transaction description] on [date]: amount matches exactly (£X), dated Y days apart, description Z% similar."
5. **save** — update BankStatement with matched_transaction_id, match_confidence, match_status. Create AuditLog entry.

Runner: `async def reconcile_batch(org_id: UUID, db: AsyncSession) -> ReconcileResponse`

### Step 4.2 — Create reconciliation API routes

**backend/app/api/v1/reconcile.py**:
- `POST /api/v1/reconcile` — triggers batch reconciliation
- `GET /api/v1/bank-statements` — list with filters: match_status, date range. Paginated.
- `GET /api/v1/bank-statements/{id}` — single statement with match candidates and scores
- `POST /api/v1/bank-statements/{id}/confirm` — confirms a suggested match
- `POST /api/v1/bank-statements/{id}/unmatch` — removes a match
- `POST /api/v1/bank-statements/{id}/match` — manually match to a specific transaction_id

### Step 4.3 — Tests and commit

Test scoring logic extensively — especially Decimal comparisons and edge cases (negative amounts, zero amounts, same-day matches).

```bash
git add .
git commit -m "feat: bank reconciliation agent with fuzzy matching and confidence scoring"
```

Mark Phase 4 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 4 complete. Both AI agents working. `/exit` and restart for Phase 5 — building the frontend."

---
---

## PHASE 5: FRONTEND — DASHBOARD + TRANSACTION REVIEW UI

**Goal**: Next.js web app with dashboard, transaction review table, and reconciliation view.

### Step 5.1 — Scaffold the frontend

Inside the `frontend/` directory, create a Next.js 14 project:

```bash
npx create-next-app@14 . --typescript --tailwind --eslint --app --src-dir --no-import-alias
```

Then add shadcn/ui:
```bash
npx shadcn-ui@latest init
# Choose: New York style, Zinc color, CSS variables: yes
npx shadcn-ui@latest add button card table badge tabs input select dialog toast
```

Create `frontend/src/lib/api.ts` — API client that calls the FastAPI backend at `http://localhost:8000` (configurable via env var). Use fetch with proper error handling. Include functions for every backend endpoint.

### Step 5.2 — Create pages

**Layout** (`src/app/layout.tsx`): Sidebar with navigation links: Dashboard, Transactions, Reconciliation, Documents (coming soon), Settings. Use a clean, professional design. Product name "AI Accountant" in the sidebar header.

**Dashboard** (`src/app/page.tsx`):
- Top row: 4 stat cards — Total Transactions, Uncategorised, Unreconciled, Last Sync Time
- "Sync with Xero" button (calls POST /api/v1/sync, shows loading spinner)
- "Categorise All" button (calls POST /api/v1/categorise, shows results toast)
- "Reconcile All" button (calls POST /api/v1/reconcile, shows results toast)
- If not connected to Xero yet, show a prominent "Connect Xero" button instead

**Transactions** (`src/app/transactions/page.tsx`):
- Filterable table: columns = Date, Description, Amount, Category, Status, Confidence, Actions
- Status badges: colour-coded (green=confirmed, yellow=suggested, red=needs_review, grey=uncategorised)
- For suggested items: "Accept" and "Reject" buttons inline
- Click a row to see full details + AI explanation text + audit history
- Correction modal: if user clicks "Edit Category", show dropdown of chart of accounts to select correct category

**Reconciliation** (`src/app/reconciliation/page.tsx`):
- Two-panel layout
- Left: bank statement lines (filtered by match_status)
- Right: when a statement is selected, show its matched/suggested transaction with confidence score and explanation
- Accept/Reject buttons for suggested matches
- For unmatched items: show a search box to find and manually match a transaction

### Step 5.3 — Test and commit

Verify all pages load and API calls work with the running backend.

```bash
git add .
git commit -m "feat: Next.js frontend with dashboard, transaction review, and reconciliation UI"
```

Mark Phase 5 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 5 complete. Full working app with UI. `/exit` and restart for Phase 6."

---
---

## PHASE 6: DOCUMENT GENERATION (RAG + PDF)

**Goal**: Generate a quarterly management letter from financial data using RAG, render as PDF.

### Step 6.1 — Install doc generation deps

```bash
cd backend && pip install -e ".[docs]"
```

### Step 6.2 — Create the document generation service

**backend/app/services/document_service.py**:

- `generate_management_letter(org_id, period_start, period_end, db)`:
  1. **Calculate figures** (pure Python, NOT LLM): total income, total expenses, net profit/loss, top 5 expense categories, month-over-month change, largest transactions
  2. **Retrieve context** via pgvector: find relevant transactions and patterns for the period
  3. **Generate narrative** via LLM: prompt includes all calculated figures + context. The LLM writes: executive summary, income analysis, expense analysis, cash flow observations, recommendations. Structured output via Instructor into sections.
  4. **Render** via Jinja2 template + WeasyPrint: professional PDF with firm logo placeholder, formatted tables, AI-generated narrative sections, date, page numbers, "AI-Assisted Draft" watermark footer.
  5. Return the PDF as bytes + metadata (generated_at, figures used, model used)

**backend/app/templates/management_letter.html** — Jinja2 HTML template with CSS for print (A4 page, proper margins, professional typography).

### Step 6.3 — API routes

**backend/app/api/v1/documents.py**:
- `POST /api/v1/documents/generate` — body: {template: "management_letter", period_start, period_end}. Returns PDF as streaming response with Content-Type application/pdf.
- `GET /api/v1/documents` — list previously generated documents (store metadata in a new GeneratedDocument table)

### Step 6.4 — Frontend page

**Documents page** (`src/app/documents/page.tsx`):
- Select template (management letter — only one for now)
- Select date range (quarter picker)
- "Generate" button → loading state → display PDF inline (using react-pdf or iframe) + Download button
- List of previously generated documents

### Step 6.5 — Commit

```bash
git add .
git commit -m "feat: management letter generation with RAG and PDF output"
```

Mark Phase 6 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 6 complete. Documents generating. `/exit` and restart for Phase 7 — explainability."

---
---

## PHASE 7: XAI EXPLANATIONS (SHAP + FUZZY LOGIC)

**Goal**: Add SHAP feature importance and fuzzy logic risk scoring to transaction categorisation. Display in UI.

### Step 7.1 — Create the XAI engine

**backend/app/xai/__init__.py** — empty

**backend/app/xai/explainer.py**:
- `explain_categorisation(transaction, prediction, similar_examples)`:
  - Extract features: amount, day_of_week, description_length, vendor_frequency, category_history_count
  - If enough training data (50+ categorised transactions): train an EBM (InterpretML) and get feature importances. Otherwise: use the LLM's reasoning text as the explanation.
  - Return: `{top_features: [{name, value, contribution}], explanation_text: str, model_type: "ebm"|"llm"}`

**backend/app/xai/fuzzy_engine.py**:
- Create a Simpful fuzzy inference system for transaction risk scoring:
  - Input variables:
    - `amount_deviation`: how far the amount is from the average for this category (low/medium/high)
    - `vendor_frequency`: how often this vendor appears (rare/occasional/frequent)
    - `time_pattern`: whether the transaction time is typical (normal/unusual)
  - Output: `risk_level` (low/medium/high) as a float 0-1
  - Rules (must be human-readable):
    - IF amount_deviation IS high AND vendor_frequency IS rare THEN risk IS high
    - IF amount_deviation IS low AND vendor_frequency IS frequent THEN risk IS low
    - IF amount_deviation IS medium THEN risk IS medium
    - (add 5-8 rules total)
  - Return: `{risk_score: float, risk_label: str, fired_rules: [str], input_values: dict}`

### Step 7.2 — Integrate XAI into the categorisation agent

Update `agents/categoriser.py` — after the "decide" node, add an "explain" node that:
1. Calls `explain_categorisation` to get feature importances
2. Calls the fuzzy risk engine to get risk score
3. Stores both in the AuditLog.ai_decision_data field as JSON

### Step 7.3 — Explanation API

**backend/app/api/v1/explanations.py**:
- `GET /api/v1/transactions/{id}/explanation` — returns the full explanation package: prediction, confidence, top features, risk score, risk label, fired rules, explanation text

### Step 7.4 — Explanation UI component

Create a reusable explanation panel component in the frontend:
- Decision + confidence at top (large text)
- Horizontal bar chart showing top 3-5 feature contributions (use a simple div-based bar chart, no heavy charting library needed)
- Risk badge: green (low), amber (medium), red (high)
- Expandable section: "Why this risk level?" showing the fired fuzzy rules as readable text
- Expandable section: "Full audit trail" showing the AuditLog history

Integrate this panel into the transaction detail view and the reconciliation detail view.

### Step 7.5 — Commit

```bash
git add .
git commit -m "feat: XAI explanations with SHAP feature importance and fuzzy logic risk scoring"
```

Mark Phase 7 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 7 complete. Full AI + XAI pipeline working. `/exit` and restart for Phase 8 — shipping it."

---
---

## PHASE 8: LANDING PAGE + AUTH + DEPLOY

**Goal**: Public-facing product with Xero login and hosted on Railway + Vercel.

### Step 8.1 — Authentication via Xero

Update the auth flow so it works as the login mechanism:
- User visits the app → if no session, redirect to Xero OAuth
- After Xero callback, create a session (JWT stored in httpOnly cookie)
- All API endpoints check for valid session
- Session includes organisation_id so all queries are scoped
- Simple middleware — no complex RBAC yet

### Step 8.2 — Landing page

Create a landing page at the root route (unauthenticated):
- Hero section: Clear headline + subtitle explaining the product
- 3 feature blocks: Categorise, Reconcile, Generate — with brief descriptions
- "How it works" section: Connect Xero → AI processes → You review → Done
- "Connect with Xero" CTA button (starts the OAuth flow)
- Simple footer: product name, privacy policy link, contact email

### Step 8.3 — Deploy

**Backend on Railway:**
1. Create a Railway account at https://railway.app/
2. Create a new project
3. Add a PostgreSQL database (Railway has pgvector support)
4. Add a Redis instance
5. Deploy the FastAPI backend from GitHub
6. Set all environment variables (DATABASE_URL is auto-set by Railway)

**Frontend on Vercel:**
1. Create a Vercel account at https://vercel.com/
2. Import the frontend directory from GitHub
3. Set NEXT_PUBLIC_API_URL to the Railway backend URL
4. Deploy

**Domain:**
- Point your domain to Vercel
- Update Xero app redirect URI to https://yourdomain.com/auth/xero/callback

🛑 **STOP** — Tell the user:
> "I've prepared everything for deployment. You now need to:
> 1. Create accounts on Railway and Vercel (if you haven't already)
> 2. Push the code to GitHub
> 3. Connect the repos to Railway and Vercel
> 4. Set environment variables in both platforms
> 5. Update your Xero app redirect URI to the production URL
>
> Walk me through any step you need help with."

### Step 8.4 — Commit

```bash
git add .
git commit -m "feat: landing page, Xero auth, deployment configuration"
```

Mark Phase 8 as `[x]` in PHASE STATUS above.

---

🔄 **RE-ENTRY POINT** — "Phase 8 complete. App is deployed. `/exit` and restart for Phase 9 — finding beta users."

---
---

## PHASE 9: BETA LAUNCH PREP

**Goal**: Polish, basic privacy policy, beta user outreach.

### Step 9.1 — Polish

- Error handling: make sure all API errors return user-friendly messages
- Loading states: every button that triggers an async action shows a spinner
- Empty states: every page handles the "no data yet" case gracefully
- Mobile: basic responsive layout (sidebar collapses on mobile)

### Step 9.2 — Privacy policy

Create a simple privacy policy page explaining:
- What data you collect (Xero accounting data via their API)
- How it's stored (encrypted, Railway EU servers)
- AI processing (data is sent to OpenAI API for categorisation — mention this explicitly)
- User rights (can delete their data, can disconnect Xero)
- Contact email

### Step 9.3 — Beta outreach plan

🛑 **STOP** — Tell the user:
> "The product is ready for beta users. Here's your outreach plan:
>
> **Where to find UK accountants:**
> - AccountingWeb forums (https://www.accountingweb.co.uk/)
> - Reddit r/UKAccountants
> - LinkedIn: search for 'Xero accountant UK' and message directly
> - Xero's own community forums
> - Local accounting networking events
>
> **What to say:**
> 'I've built an AI tool that categorises Xero transactions and explains its
> decisions. Looking for 5 accountants to try it free for 3 months and give
> feedback. Takes 2 minutes to connect your Xero. Interested?'
>
> **Target: 5-10 beta users within 2 weeks.**
>
> After you have beta users, your priority becomes fixing whatever they complain
> about. The roadmap after that depends entirely on their feedback."

Mark Phase 9 as `[x]` in PHASE STATUS above.

---
---

## POST-LAUNCH PRIORITIES (only after beta feedback)

| When | What |
|------|------|
| Users complain about something | Fix it immediately |
| 3+ users say they'd pay | Add Stripe billing (£49-99/month) |
| Users ask for QuickBooks | Build QuickBooks adapter |
| Users want more document types | Add templates based on requests |
| 20+ paying users | Consider Azure migration for scale |
| Revenue covers costs | GDPR formal compliance, Cyber Essentials |

---

## HOW TO RESUME AFTER ANY BREAK

When starting a new Claude Code session at any point:

```
Read the CLAUDE.md file (this file). Look at the PHASE STATUS section
to see what's complete and what's next. Then start executing the next
incomplete phase. If a phase is partially done, look at the git log 
to understand what was last committed and continue from there.
```

That's it. This file is the single source of truth.

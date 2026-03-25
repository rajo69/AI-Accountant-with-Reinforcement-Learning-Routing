from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.sync import router as sync_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.categorise import router as categorise_router
from app.api.v1.reconcile import router as reconcile_router
from app.api.v1.documents import router as documents_router
from app.api.v1.explanations import router as explanations_router
from app.api.v1.gdpr import router as gdpr_router
import app.models.database  # noqa: F401 — ensure models are registered


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="AI Accountant", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(sync_router)
app.include_router(dashboard_router)
app.include_router(categorise_router)
app.include_router(reconcile_router)
app.include_router(documents_router)
app.include_router(explanations_router)
app.include_router(gdpr_router)

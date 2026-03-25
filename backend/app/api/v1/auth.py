from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.session import create_session_token, get_current_org
from app.integrations.xero_adapter import XeroAdapter
from app.models.database import Organisation

router = APIRouter(tags=["auth"])


@router.get("/auth/xero/connect")
async def xero_connect():
    """Redirect the browser to Xero's OAuth2 authorization page."""
    return RedirectResponse(XeroAdapter.get_auth_url())


@router.get("/auth/xero/callback")
async def xero_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Handle the OAuth2 callback, issue a JWT, redirect to the frontend."""
    try:
        org = await XeroAdapter.handle_callback(code, db)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_session_token(org.id, org.name)
    return RedirectResponse(
        url=f"{settings.frontend_url}/auth/callback?token={token}",
        status_code=302,
    )


@router.get("/api/v1/auth/me")
async def me(org: Organisation = Depends(get_current_org)):
    """Return the current authenticated organisation."""
    return {"org_id": str(org.id), "org_name": org.name}


@router.post("/api/v1/auth/logout")
async def logout():
    """Logout — token is client-side only, so just acknowledge."""
    return {"status": "logged_out"}

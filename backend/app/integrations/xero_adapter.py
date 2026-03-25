"""Xero OAuth2 adapter — all Xero API interactions live here."""
import asyncio
import logging
import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.models.database import Organisation, Account, Transaction, BankStatement
from app.models.schemas import SyncResponse

logger = logging.getLogger(__name__)

XERO_AUTH_URL = "https://login.xero.com/identity/connect/authorize"
XERO_TOKEN_URL = "https://identity.xero.com/connect/token"
XERO_CONNECTIONS_URL = "https://api.xero.com/connections"
XERO_API_BASE = "https://api.xero.com/api.xro/2.0"
XERO_SCOPES = (
    "openid profile email offline_access "
    "accounting.banktransactions.read "
    "accounting.banktransactions "
    "accounting.invoices.read "
    "accounting.invoices "
    "accounting.contacts.read "
    "accounting.contacts "
    "accounting.settings.read"
)


def _parse_xero_date(date_str: Optional[str]):
    """Parse Xero's /Date(ms+offset)/ format or ISO 8601 → Python date."""
    if not date_str:
        return None
    match = re.match(r"/Date\((\d+)([+-]\d+)?\)/", date_str)
    if match:
        ms = int(match.group(1))
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
    try:
        return datetime.fromisoformat(date_str.rstrip("Z")).date()
    except ValueError:
        return None


class XeroAdapter:
    def __init__(self, organisation: Organisation):
        self.organisation = organisation

    # ------------------------------------------------------------------
    # OAuth2
    # ------------------------------------------------------------------

    @classmethod
    def get_auth_url(cls) -> str:
        params = {
            "response_type": "code",
            "client_id": settings.xero_client_id,
            "redirect_uri": settings.xero_redirect_uri,
            "scope": XERO_SCOPES,
            "state": "xero_auth",
        }
        return f"{XERO_AUTH_URL}?{urllib.parse.urlencode(params)}"

    @classmethod
    async def handle_callback(cls, code: str, db: AsyncSession) -> "Organisation":
        """Exchange auth code for tokens, upsert Organisation in DB."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                XERO_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": settings.xero_redirect_uri,
                },
                auth=(settings.xero_client_id, settings.xero_client_secret),
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data["access_token"]
        refresh_token = token_data["refresh_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data["expires_in"])

        # Fetch connected tenants
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                XERO_CONNECTIONS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            connections = resp.json()

        if not connections:
            raise ValueError("No Xero organisations found for this account")

        tenant = connections[0]
        tenant_id = tenant["tenantId"]
        tenant_name = tenant.get("tenantName", "Unknown Organisation")

        result = await db.execute(
            select(Organisation).where(Organisation.xero_tenant_id == tenant_id)
        )
        org = result.scalar_one_or_none()

        if org is None:
            org = Organisation(
                name=tenant_name,
                xero_tenant_id=tenant_id,
                xero_access_token=access_token,
                xero_refresh_token=refresh_token,
                xero_token_expires_at=expires_at,
            )
            db.add(org)
        else:
            org.xero_access_token = access_token
            org.xero_refresh_token = refresh_token
            org.xero_token_expires_at = expires_at
            org.name = tenant_name

        await db.commit()
        await db.refresh(org)
        return org

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def _ensure_valid_token(self, db: AsyncSession) -> str:
        """Return a valid access token, refreshing if it's about to expire."""
        org = self.organisation
        now = datetime.now(timezone.utc)

        if (
            org.xero_token_expires_at
            and org.xero_token_expires_at > now + timedelta(seconds=60)
        ):
            return org.xero_access_token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                XERO_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": org.xero_refresh_token,
                },
                auth=(settings.xero_client_id, settings.xero_client_secret),
            )
            resp.raise_for_status()
            token_data = resp.json()

        org.xero_access_token = token_data["access_token"]
        org.xero_refresh_token = token_data.get("refresh_token", org.xero_refresh_token)
        org.xero_token_expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=token_data["expires_in"])
        )
        await db.commit()
        await db.refresh(org)
        return org.xero_access_token

    def _api_headers(self, access_token: str) -> dict:
        return {
            "Authorization": f"Bearer {access_token}",
            "Xero-tenant-id": self.organisation.xero_tenant_id,
            "Accept": "application/json",
        }

    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        params: Optional[dict] = None,
    ) -> dict:
        """GET with rate-limit (429) back-off, up to 3 attempts."""
        for attempt in range(3):
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 429:
                logger.warning("Xero rate limit hit, waiting 60 s (attempt %d)", attempt + 1)
                await asyncio.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Xero API failed after 3 attempts: {url}")

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    async def sync_accounts(self, db: AsyncSession) -> int:
        token = await self._ensure_valid_token(db)
        headers = self._api_headers(token)

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await self._get_with_retry(client, f"{XERO_API_BASE}/Accounts", headers)

        count = 0
        for item in data.get("Accounts", []):
            result = await db.execute(
                select(Account).where(Account.xero_id == item["AccountID"])
            )
            acc = result.scalar_one_or_none()
            if acc is None:
                acc = Account(
                    organisation_id=self.organisation.id,
                    xero_id=item["AccountID"],
                    code=item.get("Code"),
                    name=item["Name"],
                    type=item.get("Type", ""),
                    tax_type=item.get("TaxType"),
                )
                db.add(acc)
            else:
                acc.code = item.get("Code")
                acc.name = item["Name"]
                acc.type = item.get("Type", "")
                acc.tax_type = item.get("TaxType")
            count += 1

        await db.commit()
        logger.info("Synced %d accounts for org %s", count, self.organisation.id)
        return count

    async def sync_transactions(self, db: AsyncSession) -> int:
        token = await self._ensure_valid_token(db)
        headers = self._api_headers(token)

        count = 0
        page = 1

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                data = await self._get_with_retry(
                    client,
                    f"{XERO_API_BASE}/BankTransactions",
                    headers,
                    params={"page": page},
                )
                items = data.get("BankTransactions", [])
                if not items:
                    break

                for item in items:
                    if item.get("Status") == "DELETED":
                        continue

                    xero_id = item["BankTransactionID"]
                    tx_date = _parse_xero_date(item.get("Date"))
                    if tx_date is None:
                        continue

                    # Resolve account
                    bank_account_xero_id = item.get("BankAccount", {}).get("AccountID")
                    account_id = None
                    if bank_account_xero_id:
                        acc_result = await db.execute(
                            select(Account).where(Account.xero_id == bank_account_xero_id)
                        )
                        acc = acc_result.scalar_one_or_none()
                        if acc:
                            account_id = acc.id

                    # Build description
                    line_items = item.get("LineItems", [])
                    description = (
                        line_items[0].get("Description", "")
                        if line_items
                        else ""
                    )
                    if not description:
                        description = (
                            item.get("Contact", {}).get("Name")
                            or item.get("Reference")
                            or "Unknown"
                        )

                    # SPEND transactions are negative
                    amount = Decimal(str(item.get("Total", 0)))
                    if item.get("Type") == "SPEND":
                        amount = -abs(amount)

                    result = await db.execute(
                        select(Transaction).where(Transaction.xero_id == xero_id)
                    )
                    tx = result.scalar_one_or_none()
                    if tx is None:
                        tx = Transaction(
                            organisation_id=self.organisation.id,
                            account_id=account_id,
                            xero_id=xero_id,
                            date=tx_date,
                            amount=amount,
                            description=description,
                            reference=item.get("Reference"),
                        )
                        db.add(tx)
                    else:
                        tx.date = tx_date
                        tx.amount = amount
                        tx.description = description
                        tx.reference = item.get("Reference")
                    count += 1

                await db.commit()

                if len(items) < 100:
                    break
                page += 1

        # Also pull paid invoices and bills as transactions
        page = 1
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                data = await self._get_with_retry(
                    client,
                    f"{XERO_API_BASE}/Invoices",
                    headers,
                    params={"page": page, "Statuses": "PAID,AUTHORISED"},
                )
                items = data.get("Invoices", [])
                if not items:
                    break

                for item in items:
                    xero_id = "inv-" + item["InvoiceID"]
                    tx_date = _parse_xero_date(item.get("Date") or item.get("DateString"))
                    if tx_date is None:
                        continue

                    contact_name = item.get("Contact", {}).get("Name", "Unknown")
                    invoice_number = item.get("InvoiceNumber", "")
                    description = f"{contact_name} — {invoice_number}" if invoice_number else contact_name

                    # ACCPAY = bill (money out), ACCREC = invoice (money in)
                    amount = Decimal(str(item.get("Total", 0)))
                    if item.get("Type") == "ACCPAY":
                        amount = -abs(amount)

                    result = await db.execute(
                        select(Transaction).where(Transaction.xero_id == xero_id)
                    )
                    tx = result.scalar_one_or_none()
                    if tx is None:
                        tx = Transaction(
                            organisation_id=self.organisation.id,
                            xero_id=xero_id,
                            date=tx_date,
                            amount=amount,
                            description=description,
                            reference=invoice_number or None,
                        )
                        db.add(tx)
                    else:
                        tx.date = tx_date
                        tx.amount = amount
                        tx.description = description
                    count += 1

                await db.commit()

                if len(items) < 100:
                    break
                page += 1

        logger.info("Synced %d transactions for org %s", count, self.organisation.id)
        return count

    async def sync_bank_statements(self, db: AsyncSession) -> int:
        """Sync unreconciled bank transactions as statement lines."""
        token = await self._ensure_valid_token(db)
        headers = self._api_headers(token)

        count = 0
        page = 1

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                data = await self._get_with_retry(
                    client,
                    f"{XERO_API_BASE}/BankTransactions",
                    headers,
                    params={"page": page, "where": "IsReconciled=false"},
                )
                items = data.get("BankTransactions", [])
                if not items:
                    break

                for item in items:
                    if item.get("Status") == "DELETED":
                        continue

                    xero_id = f"stmt_{item['BankTransactionID']}"
                    stmt_date = _parse_xero_date(item.get("Date"))
                    if stmt_date is None:
                        continue

                    description = (
                        item.get("Reference")
                        or item.get("Contact", {}).get("Name")
                        or "Unknown"
                    )
                    amount = Decimal(str(item.get("Total", 0)))
                    if item.get("Type") == "SPEND":
                        amount = -abs(amount)

                    result = await db.execute(
                        select(BankStatement).where(BankStatement.xero_id == xero_id)
                    )
                    stmt = result.scalar_one_or_none()
                    if stmt is None:
                        stmt = BankStatement(
                            organisation_id=self.organisation.id,
                            xero_id=xero_id,
                            date=stmt_date,
                            amount=amount,
                            description=description,
                            reference=item.get("Reference"),
                        )
                        db.add(stmt)
                    count += 1

                await db.commit()

                if len(items) < 100:
                    break
                page += 1

        logger.info("Synced %d bank statements for org %s", count, self.organisation.id)
        return count

    async def full_sync(self, db: AsyncSession) -> SyncResponse:
        accounts = await self.sync_accounts(db)
        transactions = await self.sync_transactions(db)
        bank_statements = await self.sync_bank_statements(db)

        self.organisation.last_sync_at = datetime.now(timezone.utc)
        await db.commit()

        return SyncResponse(
            status="ok",
            synced_accounts=accounts,
            synced_transactions=transactions,
            synced_bank_statements=bank_statements,
        )

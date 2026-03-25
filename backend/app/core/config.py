from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql+asyncpg://aiaccountant:localdev123@localhost:5432/aiaccountant"
    database_url_sync: str = "postgresql://aiaccountant:localdev123@localhost:5432/aiaccountant"
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "change-me-in-production"
    xero_client_id: str = ""
    xero_client_secret: str = ""
    xero_redirect_uri: str = "http://localhost:8000/auth/xero/callback"
    anthropic_api_key: str = ""
    frontend_url: str = "http://localhost:3000"


settings = Settings()

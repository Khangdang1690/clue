"""Application configuration management."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields like Cloud SQL metadata
    )

    # App Info
    app_name: str = "iClue API"
    app_version: str = "1.0.0"
    app_description: str = "Auto-ETL System with AI-powered data discovery"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", 10))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", 20))
    db_echo: bool = os.getenv("DB_ECHO", "False").lower() == "true"

    # Clerk Authentication
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")

    # Google Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # CORS
    allowed_origins: list = [
        "http://localhost:3000",  # Local development
        "https://iclue-frontend.vercel.app",  # Production frontend
    ]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

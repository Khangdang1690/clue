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

    # Environment
    env: str = os.getenv("ENV", "development")  # "development" or "production"

    # App Info
    app_name: str = "iClue API"
    app_version: str = "1.0.0"
    app_description: str = "Auto-ETL System with AI-powered data discovery"

    # Server
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8000))  # Cloud Run sets PORT=8080
    debug: bool = True

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env.lower() == "development"

    # Database - Environment-specific URLs
    database_url: str = os.getenv(
        "DATABASE_URL",
        # Default to development database (docker-compose)
        "postgresql://postgres:postgres@localhost:5433/ai_analyst"
    )
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", 10))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", 20))
    db_echo: bool = os.getenv("DB_ECHO", "False").lower() == "true"

    # Clerk Authentication
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")

    # Google Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Google Cloud Storage
    gcs_bucket_name: str = os.getenv("GCS_BUCKET_NAME", "iclue")

    # File Upload Limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))  # Maximum file size in MB

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    # CORS
    allowed_origins: list = [
        "http://localhost:3000",  # Local development
        "https://iclue-frontend.vercel.app",  # Production frontend
    ]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""API dependencies."""

from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from src.database.connection import DatabaseManager
from app.core.config import Settings, get_settings


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.

    Yields:
        Database session
    """
    with DatabaseManager.get_session() as session:
        yield session


def get_current_settings() -> Settings:
    """Get current application settings."""
    return get_settings()

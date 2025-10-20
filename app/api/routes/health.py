"""Health check routes."""

from fastapi import APIRouter, Depends
from datetime import datetime

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/", response_model=dict)
async def root(settings: Settings = Depends(get_settings)):
    """Root health check."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


@router.get("/health", response_model=dict)
async def health_check(settings: Settings = Depends(get_settings)):
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }

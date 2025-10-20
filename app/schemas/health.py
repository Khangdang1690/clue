"""Health check schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    service: str
    version: str
    database: str = "connected"

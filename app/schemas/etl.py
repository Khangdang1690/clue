"""Pydantic schemas for ETL operations."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class ETLUploadRequest(BaseModel):
    """Schema for ETL upload request (not used directly, files come via multipart/form-data)."""
    pass


class ETLProgressUpdate(BaseModel):
    """Schema for SSE progress updates during ETL."""
    step: str  # Current step name (ingestion, semantic_analysis, etc.)
    progress: int  # Progress percentage (0-100)
    message: str  # Human-readable message
    current_step: str  # Step name for display
    status: str  # running, completed, error


class ETLCompleteResponse(BaseModel):
    """Schema for ETL completion response."""
    status: str
    company_id: str
    dataset_ids: Dict[str, str]  # file_id -> dataset_id mapping
    relationships: List[Dict[str, Any]]
    total_datasets: int
    message: str


class ETLErrorResponse(BaseModel):
    """Schema for ETL error response."""
    status: str = "error"
    error: str
    current_step: Optional[str] = None


class DatasetResponse(BaseModel):
    """Schema for dataset response."""
    id: str
    company_id: str
    original_filename: str
    table_name: str
    domain: Optional[str] = None
    status: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    uploaded_at: datetime
    # Business context fields
    department: Optional[str] = None
    description: Optional[str] = None
    dataset_type: Optional[str] = None
    time_period: Optional[str] = None
    entities: Optional[List[str]] = None
    typical_use_cases: Optional[List[str]] = None
    business_context: Optional[dict] = None

    class Config:
        from_attributes = True

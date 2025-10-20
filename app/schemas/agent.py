"""AI Agent schemas."""

from typing import Optional, List, Any
from pydantic import BaseModel


class DiscoveryRequest(BaseModel):
    """Request schema for running data discovery."""
    dataset_id: str
    max_iterations: Optional[int] = 30
    max_insights: Optional[int] = 3
    generate_context: Optional[bool] = True


class DiscoveryResponse(BaseModel):
    """Response schema for data discovery."""
    success: bool
    dataset_id: str
    insights_count: int
    execution_count: int
    viz_data_path: Optional[str] = None
    dashboard_path: Optional[str] = None
    insights: List[dict] = []


class ETLRequest(BaseModel):
    """Request schema for running ETL workflow."""
    company_id: str
    file_paths: List[str]
    company_name: Optional[str] = None


class ETLResponse(BaseModel):
    """Response schema for ETL workflow."""
    success: bool
    company_id: str
    datasets_created: int
    datasets: List[dict] = []
    errors: List[str] = []

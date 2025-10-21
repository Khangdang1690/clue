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


class BusinessDiscoveryRequest(BaseModel):
    """Request schema for running business discovery workflow."""
    user_id: str
    dataset_ids: Optional[List[str]] = None  # If None, analyze all datasets
    analysis_name: Optional[str] = "Business Analysis"


class BusinessDiscoveryResponse(BaseModel):
    """Response schema for business discovery workflow."""
    success: bool
    analysis_id: Optional[str] = None  # ID of the analysis session
    company_id: str
    dataset_count: int
    insights: List[dict] = []
    synthesized_insights: List[dict] = []
    recommendations: List[dict] = []
    analytics_results: Optional[dict] = None
    executive_summary: Optional[str] = None
    dashboard_url: Optional[str] = None
    error: Optional[str] = None

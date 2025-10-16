"""Request and response models for API endpoints."""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# Request Models
class DepartmentRequest(BaseModel):
    """Department information for business context."""
    name: str
    description: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    painpoints: List[str] = Field(default_factory=list)
    perspectives: List[str] = Field(default_factory=list)


class BusinessContextRequest(BaseModel):
    """Business context submission request."""
    company_name: str
    icp: str = Field(description="Ideal Customer Profile")
    mission: str
    current_goal: str
    departments: List[DepartmentRequest] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)


# Response Models
class BusinessContextResponse(BaseModel):
    """Business context submission response."""
    status: str
    message: str
    context_id: str


class FileUploadResponse(BaseModel):
    """File upload response."""
    status: str
    department: str
    files_uploaded: List[str]
    total_size: str


class ChallengeInfo(BaseModel):
    """Challenge information."""
    id: str
    title: str
    priority_score: float
    priority_level: str
    department: Union[str, List[str]]  # Can be single department or multiple
    description: Optional[str] = None


class Phase1Response(BaseModel):
    """Phase 1 execution response."""
    status: str
    challenges_count: int
    top_challenges: List[ChallengeInfo]
    message: Optional[str] = None
    error_message: Optional[str] = None


class VisualizationInfo(BaseModel):
    """Visualization information."""
    id: str
    path: str
    title: str
    type: str


class AnalysisResultResponse(BaseModel):
    """Analysis result for a challenge."""
    challenge_id: str
    challenge_title: str
    key_findings: List[str]
    visualizations: List[VisualizationInfo]
    recommendations: List[str]
    data_sources_used: List[str]


class Phase2Response(BaseModel):
    """Phase 2 execution response."""
    status: str
    challenge_processed: Optional[ChallengeInfo] = None
    analysis_result: Optional[AnalysisResultResponse] = None
    challenges_remaining: int
    message: Optional[str] = None
    error_message: Optional[str] = None


class ReportInfo(BaseModel):
    """Report file information."""
    path: str
    size: str
    pages: Optional[int] = None
    type: str


class ReportsResponse(BaseModel):
    """Reports generation response."""
    status: str
    reports: Dict[str, ReportInfo]
    message: Optional[str] = None
    error_message: Optional[str] = None


class ChallengeStatusResponse(BaseModel):
    """Challenge queue status response."""
    total_challenges: int
    processed: int
    remaining: int
    next_challenge: Optional[ChallengeInfo] = None
    error: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Overall workflow status response."""
    phase: str  # "idle", "phase1", "phase2", "generating_reports", "completed"
    is_running: bool
    current_operation: Optional[str] = None
    progress_percent: Optional[float] = None
    challenges_status: ChallengeStatusResponse


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "log", "progress", "phase_complete", "challenge_complete", "visualization_created", "report_generated", "error"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

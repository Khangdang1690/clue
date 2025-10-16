"""Workflow state definition."""

from typing import TypedDict, List, Dict, Optional
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge
from src.models.analysis_result import AnalysisResult
import pandas as pd


class WorkflowState(TypedDict):
    """State for the ETL to Insights workflow."""

    # Phase 1 state
    business_context: Optional[BusinessContext]
    challenges: List[Challenge]
    current_challenge_index: int

    # Phase 2 state
    current_challenge: Optional[Challenge]
    loaded_data: Dict[str, pd.DataFrame]
    analysis_results: List[AnalysisResult]

    # ETL tracking
    extraction_summary: str
    transformation_summary: str
    load_summary: str

    # Control flow
    phase: str  # "phase1" or "phase2"
    status: str  # "pending", "in_progress", "completed", "error"
    error_message: str

    # Results
    analytical_report_path: Optional[str]
    business_report_path: Optional[str]

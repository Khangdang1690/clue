"""Data models for autonomous discovery system."""

from typing import TypedDict, Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd


class AnsweredQuestion(BaseModel):
    """Represents a fully answered question with evidence."""
    question: str
    answer: str
    question_id: Optional[str] = Field(default=None, description="Question ID (auto-generated if not provided)")

    # Evidence
    supporting_data: Optional[Dict[str, Any]] = Field(default=None, description="Supporting data")
    visualization_path: Optional[str] = Field(default=None, description="Path to visualization")
    supporting_visualizations: List[str] = Field(default_factory=list, description="Paths to viz files")
    supporting_statistics: Dict[str, Any] = Field(default_factory=dict)
    sub_questions_used: List[str] = Field(default_factory=list, description="Sub-question IDs used")

    # Quality
    confidence: float = Field(default=0.8, ge=0, le=1)
    data_quality_score: float = Field(default=0.9, ge=0, le=1, description="Quality of underlying data")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)


class DataProfile(BaseModel):
    """Statistical profile of the dataset."""

    # Basic info
    num_rows: int
    num_columns: int
    memory_usage_mb: float

    # Column types
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    text_columns: List[str] = Field(default_factory=list)

    # Quality metrics
    overall_missing_rate: float = Field(ge=0, le=1)
    columns_with_missing: List[str] = Field(default_factory=list)

    # Statistical summaries
    numeric_summary: Optional[Dict[str, Any]] = Field(default=None)
    categorical_summary: Optional[Dict[str, Any]] = Field(default=None)

    # Patterns detected
    has_temporal_data: bool = Field(default=False)
    temporal_columns: List[str] = Field(default_factory=list)
    outlier_columns: List[str] = Field(default_factory=list)
    high_cardinality_columns: List[str] = Field(default_factory=list)

    # Distribution characteristics
    skewed_columns: List[Tuple[str, float]] = Field(default_factory=list, description="(column, skewness)")


class DiscoveryResult(BaseModel):
    """Final result of discovery process."""

    # Input
    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Discoveries
    data_profile: DataProfile
    answered_questions: List[AnsweredQuestion] = Field(default_factory=list)

    # Key insights (top-level summary)
    key_insights: List[str] = Field(default_factory=list)
    anomalies_detected: List[str] = Field(default_factory=list)

    # Recommendations
    recommended_analyses: List[str] = Field(default_factory=list, description="Further analyses to run")
    data_quality_issues: List[str] = Field(default_factory=list)

    # Paths to outputs
    report_path: Optional[str] = Field(default=None)
    visualization_paths: List[str] = Field(default_factory=list)
    viz_data_path: Optional[str] = Field(default=None, description="Path to visualization data JSON file")


class DiscoveryState(TypedDict):
    """State for LangGraph autonomous discovery workflow."""

    # Input
    raw_data: pd.DataFrame
    dataset_name: str

    # Profiling
    data_profile: Optional[DataProfile]

    # Autonomous exploration fields
    dataset_context: Optional[Dict]  # Context from outer agent (domain, entities, etc.)
    context_summary: Optional[str]  # Summary of context for discovery
    cleaning_report: Optional[Dict]  # Data cleaning report
    exploration_result: Optional[Any]  # Result from autonomous exploration

    # Results
    answered_questions: List[AnsweredQuestion]
    key_insights: List[str]
    discovery_result: Optional[DiscoveryResult]

    # Control flow
    current_phase: str  # "context_generation", "cleaning", "profiling", "exploration", "synthesis"
    status: str  # "pending", "in_progress", "completed", "error"
    error_message: str

    # Configuration (deprecated but kept for compatibility)
    max_questions: int  # Maximum questions to investigate
    max_backtrack_attempts: int  # Max backtracking per question
    confidence_threshold: float  # Minimum confidence to accept answer

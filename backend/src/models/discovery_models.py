"""Data models for context-free discovery system."""

from typing import TypedDict, Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import networkx as nx


class RelationshipEdge(BaseModel):
    """Represents a relationship between two columns/entities."""
    source: str = Field(description="Source column name")
    target: str = Field(description="Target column name")

    # Relationship metrics
    mutual_information: float = Field(default=0.0, description="Mutual information score")
    mic_score: float = Field(default=0.0, description="Maximum Information Coefficient")
    correlation: float = Field(default=0.0, description="Pearson correlation coefficient")
    spearman_correlation: float = Field(default=0.0, description="Spearman correlation")

    # Overall strength (composite score)
    strength: float = Field(ge=0, le=1, description="Overall relationship strength (0-1)")

    # Relationship type
    relationship_type: str = Field(default="correlation", description="Type: correlation, causal, functional")

    # Causality (if applicable)
    is_causal: bool = Field(default=False, description="Whether causal relationship detected")
    causal_direction: Optional[str] = Field(default=None, description="source->target or target->source")


class CommunityCluster(BaseModel):
    """Represents a community/cluster of related entities."""
    cluster_id: int
    members: List[str] = Field(description="Column names in this cluster")
    cohesion_score: float = Field(ge=0, le=1, description="How tightly connected this cluster is")
    description: Optional[str] = Field(default=None, description="LLM-generated cluster description")


class EntityNode(BaseModel):
    """Represents a node in the knowledge graph."""
    name: str = Field(description="Column name")

    # Centrality metrics
    degree_centrality: float = Field(default=0.0, description="Degree centrality")
    betweenness_centrality: float = Field(default=0.0, description="Betweenness centrality")
    pagerank: float = Field(default=0.0, description="PageRank score")

    # Data characteristics
    data_type: str = Field(description="numeric, categorical, datetime, text")
    cardinality: int = Field(description="Number of unique values")
    missing_rate: float = Field(ge=0, le=1, description="Proportion of missing values")

    # Community membership
    community_id: Optional[int] = Field(default=None, description="Which community this node belongs to")

    # Importance score (composite)
    importance_score: float = Field(ge=0, le=1, description="Overall importance (0-1)")


class KnowledgeGraph(BaseModel):
    """Represents the full knowledge graph of the dataset."""
    nodes: List[EntityNode] = Field(default_factory=list)
    edges: List[RelationshipEdge] = Field(default_factory=list)
    communities: List[CommunityCluster] = Field(default_factory=list)

    # Graph-level metrics
    density: float = Field(default=0.0, description="Graph density")
    modularity: float = Field(default=0.0, description="Community modularity score")

    # Store the actual NetworkX graph (not serialized)
    class Config:
        arbitrary_types_allowed = True

    _networkx_graph: Optional[nx.Graph] = None


class InsightQuestion(BaseModel):
    """Represents an insight question to investigate."""
    id: str = Field(description="Unique question ID")
    question: str = Field(description="The question to answer")

    # Priority and relevance
    priority_score: float = Field(ge=0, le=100, description="Priority score (0-100)")

    # What generated this question
    source_type: str = Field(description="centrality, community, anomaly, relationship, temporal")
    entities_involved: List[str] = Field(description="Column names involved in this question")

    # Complexity
    is_complex: bool = Field(default=False, description="Whether this needs decomposition")

    # Status
    status: str = Field(default="pending", description="pending, in_progress, answered, failed")


class SubQuestion(BaseModel):
    """Represents a sub-question in the decomposition tree."""
    id: str
    question: str
    parent_id: Optional[str] = Field(default=None, description="Parent question ID")
    dependencies: List[str] = Field(default_factory=list, description="IDs of sub-questions this depends on")

    # Analysis method
    analysis_method: str = Field(description="aggregation, statistical_test, correlation, filtering, etc.")
    required_columns: List[str] = Field(default_factory=list)

    # Result
    answer: Optional[str] = Field(default=None)
    supporting_data: Optional[Dict[str, Any]] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

    # Status
    status: str = Field(default="pending", description="pending, in_progress, answered, failed")


class QuestionPlan(BaseModel):
    """Execution plan for answering a complex question."""
    root_question_id: str
    root_question: str

    # Decomposition
    is_directly_answerable: bool = Field(description="Can answer without decomposition")
    sub_questions: List[SubQuestion] = Field(default_factory=list)

    # Execution order (topologically sorted)
    execution_order: List[str] = Field(default_factory=list, description="Sub-question IDs in execution order")

    # Progress
    completed_sub_questions: int = Field(default=0)
    total_sub_questions: int = Field(default=0)

    # Final answer
    final_answer: Optional[str] = Field(default=None)
    final_confidence: Optional[float] = Field(default=None)


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
    unanswered_questions: List[InsightQuestion] = Field(default_factory=list)

    # Key insights (top-level summary)
    key_insights: List[str] = Field(default_factory=list)
    anomalies_detected: List[str] = Field(default_factory=list)

    # Recommendations
    recommended_analyses: List[str] = Field(default_factory=list, description="Further analyses to run")
    data_quality_issues: List[str] = Field(default_factory=list)

    # Paths to outputs
    report_path: Optional[str] = Field(default=None)
    visualization_paths: List[str] = Field(default_factory=list)


class DiscoveryState(TypedDict):
    """State for LangGraph discovery workflow."""

    # Input
    raw_data: pd.DataFrame
    dataset_name: str

    # Phase 1: Profiling
    data_profile: Optional[DataProfile]

    # Phase 2: Relationship Detection
    relationship_edges: List[RelationshipEdge]
    mi_matrix: Optional[Any]  # numpy array
    causal_graph_data: Optional[Dict]

    # Phase 3: Graph Construction
    knowledge_graph: Optional[KnowledgeGraph]

    # Phase 4: Question Generation
    insight_questions: List[InsightQuestion]
    current_question: Optional[InsightQuestion]
    question_queue: List[str]  # Question IDs in priority order

    # Phase 5: Question Planning
    current_question_plan: Optional[QuestionPlan]

    # Phase 6: Execution
    answered_questions: List[AnsweredQuestion]
    failed_questions: List[InsightQuestion]
    analysis_cache: Dict[str, Any]  # Cache for intermediate results
    backtrack_count: int  # Track backtracking attempts

    # Phase 7: Synthesis
    key_insights: List[str]
    discovery_result: Optional[DiscoveryResult]

    # Control flow
    current_phase: str  # "profiling", "relationships", "graph", "questions", "planning", "execution", "synthesis"
    status: str  # "pending", "in_progress", "completed", "error"
    error_message: str

    # Configuration
    max_questions: int  # Maximum questions to investigate
    max_backtrack_attempts: int  # Max backtracking per question
    confidence_threshold: float  # Minimum confidence to accept answer

    # New autonomous exploration fields
    dataset_context: Optional[Dict]  # Context from outer agent
    context_summary: Optional[str]  # Summary of context for discovery
    cleaning_report: Optional[Dict]  # Data cleaning report
    exploration_result: Optional[Any]  # Result from autonomous exploration

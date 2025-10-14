"""Analysis result data model."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class VisualizationData(BaseModel):
    """Data for a visualization."""
    viz_type: str = Field(description="Type of visualization (bar, line, scatter, heatmap, etc.)")
    title: str
    data: Dict[str, Any]
    file_path: Optional[str] = None
    description: str


class StatisticalTest(BaseModel):
    """Results from a statistical test."""
    test_name: str
    test_statistic: float
    p_value: float
    significance_level: float = 0.05
    is_significant: bool
    interpretation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Results from analyzing a challenge."""
    challenge_id: str
    challenge_title: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # ETL information
    data_sources_used: List[str] = Field(default_factory=list)
    extraction_summary: str
    transformation_summary: str
    load_summary: str

    # Analysis information
    statistical_tests: List[StatisticalTest] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    correlations: Dict[str, float] = Field(default_factory=dict)

    # Visualizations
    visualizations: List[VisualizationData] = Field(default_factory=list)

    # Insights
    causality_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    def to_summary_string(self) -> str:
        """Convert to summary string."""
        return f"""
Analysis for: {self.challenge_title}
Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Data Sources: {', '.join(self.data_sources_used)}

Key Findings:
{chr(10).join(f'- {finding}' for finding in self.key_findings)}

Statistical Tests Performed: {len(self.statistical_tests)}
Visualizations Generated: {len(self.visualizations)}

Causality Insights:
{chr(10).join(f'- {insight}' for insight in self.causality_insights)}

Recommendations:
{chr(10).join(f'- {rec}' for rec in self.recommendations)}
"""

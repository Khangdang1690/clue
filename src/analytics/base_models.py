"""Base models for advanced analytics."""

from typing import Dict, Any, Optional, List, Tuple, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ConfidenceLevel(str, Enum):
    """Confidence level for analysis results."""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class ValidationReport(BaseModel):
    """Report on statistical validation of analysis."""

    # Overall assessment
    is_valid: bool = Field(description="Whether analysis passes validation")
    confidence_level: ConfidenceLevel = Field(description="Overall confidence")
    confidence_score: float = Field(ge=0, le=1, description="Numeric confidence (0-1)")

    # Sample size validation
    sample_size: int
    min_required_sample: int
    sample_size_adequate: bool

    # Assumption tests (specific to analysis type)
    assumptions_tested: Dict[str, bool] = Field(
        default_factory=dict,
        description="Results of assumption tests (e.g., stationarity, normality)"
    )

    # Warnings and issues
    warnings: List[str] = Field(default_factory=list)
    critical_issues: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_confidence_icon(self) -> str:
        """Get emoji icon for confidence level."""
        icons = {
            ConfidenceLevel.VERY_LOW: "🔴",
            ConfidenceLevel.LOW: "🟡",
            ConfidenceLevel.MEDIUM: "🟠",
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.VERY_HIGH: "✅"
        }
        return icons.get(self.confidence_level, "❓")

    def get_summary(self) -> str:
        """Get human-readable summary of validation."""
        icon = self.get_confidence_icon()
        summary = f"{icon} {self.confidence_level.value} confidence ({self.confidence_score:.0%})"

        if self.critical_issues:
            summary += f"\nCritical issues: {len(self.critical_issues)}"
        if self.warnings:
            summary += f"\nWarnings: {len(self.warnings)}"

        return summary


class AnalysisResult(BaseModel):
    """Base class for all analysis results."""

    # Identification
    analysis_type: str = Field(description="Type of analysis (forecasting, causal, etc.)")
    dataset_name: str

    # Core results (structured data for LLM to interpret)
    results: Dict[str, Any] = Field(
        description="Calculated results (numbers, not interpretations)"
    )

    # Statistical validation
    validation: ValidationReport

    # Metadata for LLM context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Business context for LLM interpretation"
    )

    # Visualization data (for charts)
    visualization_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data for creating visualizations"
    )

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ForecastResult(AnalysisResult):
    """Result from forecasting analysis."""

    analysis_type: Literal["forecasting"] = "forecasting"

    # Override results with specific forecast structure
    results: Dict[str, Any] = Field(
        description="""Forecast results structure:
        {
            'predictions': List[float] - forecasted values,
            'timestamps': List[str] - prediction timestamps,
            'confidence_intervals': List[Tuple[float, float]] - CI bounds,
            'current_value': float - latest actual value,
            'forecast_horizon': int - periods forecasted,
            'model_name': str - model used,
            'accuracy_metrics': {
                'mape': float - Mean Absolute Percentage Error,
                'mae': float - Mean Absolute Error,
                'rmse': float - Root Mean Squared Error
            }
        }"""
    )


class CausalResult(AnalysisResult):
    """Result from causal inference analysis."""

    analysis_type: Literal["causal_inference"] = "causal_inference"

    results: Dict[str, Any] = Field(
        description="""Causal analysis results:
        {
            'relationships': List[{
                'cause': str - cause variable,
                'effect': str - effect variable,
                'p_value': float - statistical significance,
                'f_statistic': float - test statistic,
                'lag': int - optimal lag in periods,
                'is_significant': bool - p < 0.05,
                'strength': str - weak/moderate/strong
            }],
            'granger_test_results': Dict - full Granger test output
        }"""
    )


class VarianceDecompositionResult(AnalysisResult):
    """Result from variance decomposition analysis."""

    analysis_type: Literal["variance_decomposition"] = "variance_decomposition"

    results: Dict[str, Any] = Field(
        description="""Variance decomposition results:
        {
            'feature_contributions': List[{
                'feature': str - feature name,
                'contribution': float - variance explained (0-1),
                'contribution_pct': float - percentage (0-100),
                'is_positive': bool - positive or negative contribution,
                'confidence_interval': Tuple[float, float]
            }],
            'total_variance_explained': float - R-squared,
            'method': str - SHAP/statistical
        }"""
    )


class AnomalyDetectionResult(AnalysisResult):
    """Result from anomaly detection analysis."""

    analysis_type: Literal["anomaly_detection"] = "anomaly_detection"

    results: Dict[str, Any] = Field(
        description="""Anomaly detection results:
        {
            'anomalies': List[{
                'timestamp': str - when anomaly occurred,
                'value': float - anomalous value,
                'expected_value': float - what was expected,
                'deviation': float - how far from expected,
                'severity': str - low/medium/high/critical,
                'anomaly_score': float - statistical score,
                'is_outlier': bool
            }],
            'total_anomalies': int,
            'anomaly_rate': float - proportion of anomalies,
            'detection_method': str - method used
        }"""
    )


class ImpactEstimationResult(AnalysisResult):
    """Result from impact estimation analysis."""

    analysis_type: Literal["impact_estimation"] = "impact_estimation"

    results: Dict[str, Any] = Field(
        description="""Impact estimation results:
        {
            'estimated_impact': float - estimated effect,
            'confidence_interval': Tuple[float, float] - CI for impact,
            'p_value': float - statistical significance,
            'is_significant': bool,
            'effect_size': str - small/medium/large,
            'cohens_d': float - standardized effect size,
            'before_mean': float,
            'after_mean': float,
            'before_std': float,
            'after_std': float,
            'n_before': int,
            'n_after': int
        }"""
    )

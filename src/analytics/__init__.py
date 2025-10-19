"""
Advanced Analytics Layer

This module provides mathematically rigorous analytics backed by proven statistical libraries.
Results are calculated with fixed code, then interpreted by LLMs for business insights.

Modules:
- forecasting: Time series forecasting with confidence intervals
- causal_inference: Statistical causal analysis (Granger causality)
- variance_decomposition: SHAP-based feature attribution
- anomaly_detection: Statistical outlier detection
- impact_estimation: Effect size calculation and significance testing
- validation: Statistical assumption testing and confidence scoring
- synthesizer: LLM-powered business insight generation
"""

from .base_models import (
    AnalysisResult,
    ValidationReport,
    ConfidenceLevel,
    ForecastResult,
    CausalResult,
    VarianceDecompositionResult,
    AnomalyDetectionResult,
    ImpactEstimationResult
)

from .forecasting import TimeSeriesForecaster
from .causal_inference import CausalAnalyzer
from .variance_decomposition import VarianceDecomposer
from .anomaly_detection import AnomalyDetector
from .impact_estimation import ImpactEstimator
from .synthesizer import BusinessInsightSynthesizer
from .validation import StatisticalValidator

__all__ = [
    # Base models
    'AnalysisResult',
    'ValidationReport',
    'ConfidenceLevel',
    'ForecastResult',
    'CausalResult',
    'VarianceDecompositionResult',
    'AnomalyDetectionResult',
    'ImpactEstimationResult',

    # Analyzers
    'TimeSeriesForecaster',
    'CausalAnalyzer',
    'VarianceDecomposer',
    'AnomalyDetector',
    'ImpactEstimator',
    'BusinessInsightSynthesizer',
    'StatisticalValidator'
]

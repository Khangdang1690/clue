"""Causal inference using Granger causality tests."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .base_models import CausalResult
from .validation import StatisticalValidator


class CausalAnalyzer:
    """
    Causal inference using Granger causality and correlation analysis.

    Determines if one time series variable causes another.
    """

    def __init__(
        self,
        max_lag: int = 12,
        significance_level: float = 0.05
    ):
        """
        Initialize causal analyzer.

        Args:
            max_lag: Maximum lag to test (in time periods)
            significance_level: P-value threshold for significance
        """
        self.max_lag = max_lag
        self.alpha = significance_level
        self.validator = StatisticalValidator()

    def analyze_causality(
        self,
        cause: pd.Series,
        effect: pd.Series,
        dataset_name: str = "dataset",
        context: Optional[Dict[str, Any]] = None
    ) -> CausalResult:
        """
        Test if 'cause' Granger-causes 'effect'.

        Args:
            cause: Potential cause variable (time series)
            effect: Potential effect variable (time series)
            dataset_name: Name of dataset
            context: Business context

        Returns:
            CausalResult with Granger test results and validation
        """
        # Validate data
        validation = self.validator.validate_causal_inference(
            cause, effect, self.max_lag
        )

        if not validation.is_valid:
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

        try:
            # Perform Granger causality test
            granger_result = self._granger_causality_test(cause, effect)

            # Determine strength of relationship
            strength = self._determine_strength(granger_result['p_value'])

            # Build relationships list
            relationships = [{
                'cause': context.get('cause_name', 'X') if context else 'X',
                'effect': context.get('effect_name', 'Y') if context else 'Y',
                'p_value': granger_result['p_value'],
                'f_statistic': granger_result['f_statistic'],
                'lag': granger_result['optimal_lag'],
                'is_significant': granger_result['is_significant'],
                'strength': strength
            }]

            # Calculate cross-correlation for additional context
            cross_corr = self._calculate_cross_correlation(cause, effect, granger_result['optimal_lag'])

            results = {
                'relationships': relationships,
                'granger_test_results': granger_result,
                'cross_correlation': cross_corr,
                'interpretation': self._create_interpretation(relationships[0], cross_corr)
            }

            # Create visualization data
            viz_data = self._create_visualization_data(cause, effect, granger_result['optimal_lag'], context)

            return CausalResult(
                dataset_name=dataset_name,
                results=results,
                validation=validation,
                context=context or {},
                visualization_data=viz_data
            )

        except Exception as e:
            validation.critical_issues.append(f"Causal analysis failed: {str(e)}")
            validation.is_valid = False
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

    def analyze_multiple_causes(
        self,
        causes: Dict[str, pd.Series],
        effect: pd.Series,
        dataset_name: str = "dataset",
        context: Optional[Dict[str, Any]] = None
    ) -> CausalResult:
        """
        Test multiple potential causes against one effect.

        Args:
            causes: Dictionary of {name: series} for potential causes
            effect: Effect variable
            dataset_name: Name of dataset
            context: Business context

        Returns:
            CausalResult with all relationships
        """
        relationships = []

        for cause_name, cause_series in causes.items():
            # Test each cause
            result = self.analyze_causality(
                cause=cause_series,
                effect=effect,
                dataset_name=dataset_name,
                context={'cause_name': cause_name, 'effect_name': context.get('effect_name', 'Y') if context else 'Y'}
            )

            if result.validation.is_valid and result.results['relationships']:
                relationships.append(result.results['relationships'][0])

        # Sort by significance (p-value)
        relationships.sort(key=lambda x: x['p_value'])

        # Use validation from first test
        first_cause = list(causes.values())[0]
        validation = self.validator.validate_causal_inference(first_cause, effect, self.max_lag)

        results = {
            'relationships': relationships,
            'significant_causes': [r for r in relationships if r['is_significant']],
            'total_tested': len(causes)
        }

        return CausalResult(
            dataset_name=dataset_name,
            results=results,
            validation=validation,
            context=context or {}
        )

    def _granger_causality_test(
        self,
        cause: pd.Series,
        effect: pd.Series
    ) -> Dict:
        """Perform Granger causality test."""
        from statsmodels.tsa.stattools import grangercausalitytests

        # Prepare data (align series)
        df = pd.DataFrame({
            'effect': effect,
            'cause': cause
        }).dropna()

        # Run Granger test for multiple lags
        max_test_lag = min(self.max_lag, len(df) // 5)  # Ensure enough data

        try:
            test_result = grangercausalitytests(
                df[['effect', 'cause']],
                maxlag=max_test_lag,
                verbose=False
            )

            # Extract results for each lag
            lag_results = []
            for lag in range(1, max_test_lag + 1):
                # Get F-test result (ssr_ftest)
                f_test = test_result[lag][0]['ssr_ftest']
                lag_results.append({
                    'lag': lag,
                    'f_statistic': f_test[0],
                    'p_value': f_test[1]
                })

            # Find optimal lag (minimum p-value)
            optimal = min(lag_results, key=lambda x: x['p_value'])

            return {
                'optimal_lag': optimal['lag'],
                'f_statistic': optimal['f_statistic'],
                'p_value': optimal['p_value'],
                'is_significant': optimal['p_value'] < self.alpha,
                'all_lags': lag_results
            }

        except Exception as e:
            raise ValueError(f"Granger test failed: {e}")

    def _calculate_cross_correlation(
        self,
        cause: pd.Series,
        effect: pd.Series,
        optimal_lag: int
    ) -> Dict:
        """Calculate cross-correlation at optimal lag."""
        # Align series
        df = pd.DataFrame({
            'cause': cause,
            'effect': effect
        }).dropna()

        # Calculate correlation at optimal lag
        if optimal_lag > 0:
            lagged_cause = df['cause'].shift(optimal_lag)
            correlation = lagged_cause.corr(df['effect'])
        else:
            correlation = df['cause'].corr(df['effect'])

        return {
            'correlation_at_optimal_lag': float(correlation),
            'lag': optimal_lag
        }

    def _determine_strength(self, p_value: float) -> str:
        """Determine strength of causal relationship."""
        if p_value >= self.alpha:
            return 'none'
        elif p_value < 0.01:
            return 'strong'
        elif p_value < 0.05:
            return 'moderate'
        else:
            return 'weak'

    def _create_interpretation(self, relationship: Dict, cross_corr: Dict) -> str:
        """Create human-readable interpretation."""
        if not relationship['is_significant']:
            return "No statistically significant causal relationship found"

        cause = relationship['cause']
        effect = relationship['effect']
        lag = relationship['lag']
        strength = relationship['strength']
        corr = cross_corr['correlation_at_optimal_lag']

        direction = "positively" if corr > 0 else "negatively"

        return (
            f"{cause} appears to {direction} cause {effect} with a {strength} "
            f"causal link. The effect appears after {lag} time period(s). "
            f"This relationship is statistically significant (p={relationship['p_value']:.4f})."
        )

    def _create_visualization_data(
        self,
        cause: pd.Series,
        effect: pd.Series,
        optimal_lag: int,
        context: Optional[Dict]
    ) -> Dict:
        """Create visualization data for time series plot."""
        # Align and prepare data
        df = pd.DataFrame({
            'cause': cause,
            'effect': effect
        }).dropna()

        timestamps = (
            df.index.strftime('%Y-%m-%d').tolist()
            if isinstance(df.index, pd.DatetimeIndex)
            else [f"T{i}" for i in range(len(df))]
        )

        cause_name = context.get('cause_name', 'Cause') if context else 'Cause'
        effect_name = context.get('effect_name', 'Effect') if context else 'Effect'

        return {
            'chart_type': 'line',
            'labels': timestamps,
            'x_label': 'Time Period',
            'y_label': 'Value',
            'datasets': [
                {
                    'label': cause_name,
                    'data': df['cause'].tolist(),
                    'yAxisID': 'y-axis-1'
                },
                {
                    'label': effect_name,
                    'data': df['effect'].tolist(),
                    'yAxisID': 'y-axis-2'
                }
            ],
            'annotation': f"Optimal lag: {optimal_lag} period(s)"
        }

    def _create_error_result(
        self,
        dataset_name: str,
        validation: Any,
        context: Dict
    ) -> CausalResult:
        """Create error result."""
        return CausalResult(
            dataset_name=dataset_name,
            results={
                'relationships': [],
                'granger_test_results': {},
                'error': 'Causal analysis failed validation'
            },
            validation=validation,
            context=context
        )

"""Impact estimation and effect size calculation."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_models import ImpactEstimationResult, ConfidenceLevel
from .validation import StatisticalValidator


class ImpactEstimator:
    """
    Impact estimation for interventions, campaigns, or policy changes.

    Calculates effect sizes, confidence intervals, and statistical significance.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        significance_level: float = 0.05
    ):
        """
        Initialize impact estimator.

        Args:
            confidence_level: Confidence level for intervals (0-1)
            significance_level: P-value threshold for significance
        """
        self.confidence_level = confidence_level
        self.alpha = significance_level
        self.validator = StatisticalValidator()

    def estimate_impact(
        self,
        before: pd.Series,
        after: pd.Series,
        dataset_name: str = "dataset",
        intervention_name: str = "intervention",
        context: Optional[Dict[str, Any]] = None
    ) -> ImpactEstimationResult:
        """
        Estimate impact of an intervention using before/after analysis.

        Args:
            before: Data before intervention
            after: Data after intervention
            dataset_name: Name of dataset
            intervention_name: Name of intervention
            context: Business context

        Returns:
            ImpactEstimationResult with impact estimates
        """
        # Basic validation
        if len(before) < 2 or len(after) < 2:
            validation = self._create_invalid_validation(
                "Insufficient data for impact estimation",
                len(before) + len(after)
            )
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

        try:
            # Calculate descriptive statistics
            before_mean = before.mean()
            after_mean = after.mean()
            before_std = before.std()
            after_std = after.std()
            n_before = len(before)
            n_after = len(after)

            # Calculate impact
            estimated_impact = after_mean - before_mean

            # Perform t-test
            t_statistic, p_value = stats.ttest_ind(after, before)

            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(((n_before - 1) * before_std**2 + (n_after - 1) * after_std**2) / (n_before + n_after - 2))
            cohens_d = (after_mean - before_mean) / pooled_std if pooled_std > 0 else 0

            # Calculate confidence interval for the impact
            se_diff = np.sqrt((before_std**2 / n_before) + (after_std**2 / n_after))
            ci_margin = stats.t.ppf((1 + self.confidence_level) / 2, n_before + n_after - 2) * se_diff
            confidence_interval = (
                estimated_impact - ci_margin,
                estimated_impact + ci_margin
            )

            # Determine if significant
            is_significant = p_value < self.alpha

            # Classify effect size
            effect_size = self._classify_effect_size(abs(cohens_d))

            results = {
                'estimated_impact': float(estimated_impact),
                'confidence_interval': (float(confidence_interval[0]), float(confidence_interval[1])),
                'p_value': float(p_value),
                'is_significant': bool(is_significant),
                'effect_size': effect_size,
                'cohens_d': float(cohens_d),
                'before_mean': float(before_mean),
                'after_mean': float(after_mean),
                'before_std': float(before_std),
                'after_std': float(after_std),
                'n_before': int(n_before),
                'n_after': int(n_after),
                'percent_change': float((estimated_impact / before_mean * 100) if before_mean != 0 else 0),
                'intervention_name': intervention_name
            }

            # Create validation report
            validation = self._create_validation_report(
                n_before=n_before,
                n_after=n_after,
                p_value=p_value,
                before=before,
                after=after
            )

            # Create visualization data
            viz_data = self._create_visualization_data(before, after, results)

            return ImpactEstimationResult(
                dataset_name=dataset_name,
                results=results,
                validation=validation,
                context=context or {},
                visualization_data=viz_data
            )

        except Exception as e:
            validation = self._create_invalid_validation(
                f"Impact estimation failed: {str(e)}",
                len(before) + len(after)
            )
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

    def estimate_impact_time_series(
        self,
        data: pd.Series,
        intervention_date: pd.Timestamp,
        dataset_name: str = "dataset",
        intervention_name: str = "intervention",
        context: Optional[Dict[str, Any]] = None
    ) -> ImpactEstimationResult:
        """
        Estimate impact from a time series with a known intervention date.

        Args:
            data: Time series data (with DatetimeIndex)
            intervention_date: Date of intervention
            dataset_name: Name of dataset
            intervention_name: Name of intervention
            context: Business context

        Returns:
            ImpactEstimationResult
        """
        # Split data
        before = data[data.index < intervention_date]
        after = data[data.index >= intervention_date]

        return self.estimate_impact(
            before=before,
            after=after,
            dataset_name=dataset_name,
            intervention_name=intervention_name,
            context=context
        )

    def _classify_effect_size(self, cohens_d: float) -> str:
        """Classify effect size based on Cohen's d."""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def _create_validation_report(
        self,
        n_before: int,
        n_after: int,
        p_value: float,
        before: pd.Series,
        after: pd.Series
    ) -> Any:
        """Create validation report."""
        from .base_models import ValidationReport

        warnings = []
        critical_issues = []
        assumptions = {}

        # Sample size check
        min_required = 30
        sample_adequate = (n_before >= min_required and n_after >= min_required)

        if n_before < min_required:
            warnings.append(f"Before sample size is small: {n_before} (recommended: {min_required})")
        if n_after < min_required:
            warnings.append(f"After sample size is small: {n_after} (recommended: {min_required})")

        # Normality assumption
        if n_before >= 3:
            _, p_before = stats.shapiro(before) if n_before < 5000 else (0, 0.5)
            assumptions['before_normal'] = p_before > 0.05
            if p_before <= 0.05:
                warnings.append("Before data may not be normally distributed")

        if n_after >= 3:
            _, p_after = stats.shapiro(after) if n_after < 5000 else (0, 0.5)
            assumptions['after_normal'] = p_after > 0.05
            if p_after <= 0.05:
                warnings.append("After data may not be normally distributed")

        # Variance equality (Levene's test)
        if n_before >= 3 and n_after >= 3:
            _, p_levene = stats.levene(before, after)
            assumptions['equal_variance'] = p_levene > 0.05
            if p_levene <= 0.05:
                warnings.append("Variances may differ between before and after periods")

        # Calculate confidence score
        confidence_score = 1.0
        confidence_score -= len(critical_issues) * 0.3
        confidence_score -= len(warnings) * 0.1

        # Sample size penalty
        if not sample_adequate:
            sample_ratio = min(n_before, n_after) / min_required
            confidence_score *= sample_ratio

        confidence_score = max(0.0, confidence_score)

        # Determine confidence level
        if confidence_score >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW

        return ValidationReport(
            is_valid=len(critical_issues) == 0,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sample_size=n_before + n_after,
            min_required_sample=min_required * 2,
            sample_size_adequate=sample_adequate,
            assumptions_tested=assumptions,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def _create_invalid_validation(self, error_message: str, sample_size: int) -> Any:
        """Create invalid validation report."""
        from .base_models import ValidationReport

        return ValidationReport(
            is_valid=False,
            confidence_level=ConfidenceLevel.VERY_LOW,
            confidence_score=0.0,
            sample_size=sample_size,
            min_required_sample=30,
            sample_size_adequate=False,
            assumptions_tested={},
            warnings=[],
            critical_issues=[error_message]
        )

    def _create_visualization_data(
        self,
        before: pd.Series,
        after: pd.Series,
        results: Dict
    ) -> Dict:
        """Create visualization data for before/after comparison."""
        return {
            'chart_type': 'bar',
            'labels': ['Before', 'After'],
            'x_label': 'Period',
            'y_label': 'Average Value',
            'datasets': [
                {
                    'label': 'Average Value',
                    'data': [results['before_mean'], results['after_mean']],
                    'backgroundColor': ['#2196f3', '#4caf50']
                }
            ],
            'annotation': f"Impact: {results['estimated_impact']:.2f} ({results['percent_change']:.1f}%)"
        }

    def _create_error_result(
        self,
        dataset_name: str,
        validation: Any,
        context: Dict
    ) -> ImpactEstimationResult:
        """Create error result."""
        return ImpactEstimationResult(
            dataset_name=dataset_name,
            results={
                'estimated_impact': 0.0,
                'confidence_interval': (0.0, 0.0),
                'p_value': 1.0,
                'is_significant': False,
                'effect_size': 'none',
                'cohens_d': 0.0,
                'before_mean': 0.0,
                'after_mean': 0.0,
                'before_std': 0.0,
                'after_std': 0.0,
                'n_before': 0,
                'n_after': 0,
                'error': 'Impact estimation failed validation'
            },
            validation=validation,
            context=context
        )

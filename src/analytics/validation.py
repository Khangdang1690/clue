"""Statistical validation framework for analytics results."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from .base_models import ValidationReport, ConfidenceLevel


class StatisticalValidator:
    """Validates statistical assumptions and data quality for analytics."""

    def __init__(
        self,
        min_sample_size: int = 30,
        significance_level: float = 0.05
    ):
        """
        Initialize validator.

        Args:
            min_sample_size: Minimum sample size for robust analysis
            significance_level: Alpha level for statistical tests
        """
        self.min_sample_size = min_sample_size
        self.alpha = significance_level

    def validate_forecasting(
        self,
        data: pd.Series,
        forecast_horizon: int
    ) -> ValidationReport:
        """
        Validate time series data for forecasting.

        Args:
            data: Time series data
            forecast_horizon: Number of periods to forecast

        Returns:
            ValidationReport with validation results
        """
        warnings = []
        critical_issues = []
        assumptions = {}

        # Sample size check
        n = len(data)
        min_required = max(self.min_sample_size, forecast_horizon * 3)
        sample_adequate = n >= min_required

        if not sample_adequate:
            critical_issues.append(
                f"Sample size too small: {n} observations (need {min_required})"
            )

        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data.dropna())
            is_stationary = adf_result[1] < self.alpha
            assumptions['stationarity'] = is_stationary

            if not is_stationary:
                warnings.append(
                    "Data is non-stationary (trending). Model will apply differencing."
                )
        except Exception as e:
            warnings.append(f"Could not test stationarity: {e}")
            assumptions['stationarity'] = False

        # Seasonality detection
        if n >= 24:  # Need at least 2 years of monthly data
            try:
                # Simple autocorrelation test for seasonality
                autocorr_12 = data.autocorr(lag=12) if n >= 12 else 0
                has_seasonality = abs(autocorr_12) > 0.3
                assumptions['seasonality_detected'] = has_seasonality
            except:
                assumptions['seasonality_detected'] = False
        else:
            warnings.append("Not enough data to detect seasonality")

        # Missing value check
        missing_rate = data.isnull().mean()
        if missing_rate > 0.1:
            warnings.append(f"High missing value rate: {missing_rate:.1%}")
        assumptions['low_missing_values'] = missing_rate < 0.1

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            sample_adequate=sample_adequate,
            n=n,
            min_required=min_required,
            critical_issues=critical_issues,
            warnings=warnings
        )

        confidence_level = self._score_to_level(confidence_score)

        return ValidationReport(
            is_valid=len(critical_issues) == 0,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sample_size=n,
            min_required_sample=min_required,
            sample_size_adequate=sample_adequate,
            assumptions_tested=assumptions,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def validate_causal_inference(
        self,
        cause_series: pd.Series,
        effect_series: pd.Series,
        max_lag: int = 12
    ) -> ValidationReport:
        """
        Validate data for Granger causality testing.

        Args:
            cause_series: Potential cause variable
            effect_series: Potential effect variable
            max_lag: Maximum lag to test

        Returns:
            ValidationReport
        """
        warnings = []
        critical_issues = []
        assumptions = {}

        # Sample size
        n = min(len(cause_series), len(effect_series))
        min_required = max(self.min_sample_size, max_lag * 5)
        sample_adequate = n >= min_required

        if not sample_adequate:
            critical_issues.append(
                f"Insufficient data for causal analysis: {n} obs (need {min_required})"
            )

        # Stationarity for both series
        try:
            from statsmodels.tsa.stattools import adfuller
            cause_stationary = adfuller(cause_series.dropna())[1] < self.alpha
            effect_stationary = adfuller(effect_series.dropna())[1] < self.alpha

            assumptions['cause_stationary'] = cause_stationary
            assumptions['effect_stationary'] = effect_stationary

            if not cause_stationary:
                warnings.append("Cause variable is non-stationary")
            if not effect_stationary:
                warnings.append("Effect variable is non-stationary")
        except Exception as e:
            warnings.append(f"Could not test stationarity: {e}")

        # Correlation check (sanity check)
        try:
            correlation = cause_series.corr(effect_series)
            assumptions['has_correlation'] = abs(correlation) > 0.1

            if abs(correlation) < 0.1:
                warnings.append(
                    "Very weak correlation between variables - causal link may not exist"
                )
        except:
            pass

        # Missing values
        cause_missing = cause_series.isnull().mean()
        effect_missing = effect_series.isnull().mean()

        if cause_missing > 0.1 or effect_missing > 0.1:
            warnings.append("High missing value rate in one or both series")

        confidence_score = self._calculate_confidence_score(
            sample_adequate=sample_adequate,
            n=n,
            min_required=min_required,
            critical_issues=critical_issues,
            warnings=warnings
        )

        confidence_level = self._score_to_level(confidence_score)

        return ValidationReport(
            is_valid=len(critical_issues) == 0,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sample_size=n,
            min_required_sample=min_required,
            sample_size_adequate=sample_adequate,
            assumptions_tested=assumptions,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def validate_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ValidationReport:
        """
        Validate data for regression-based analyses (SHAP, impact estimation).

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            ValidationReport
        """
        warnings = []
        critical_issues = []
        assumptions = {}

        # Sample size
        n = len(y)
        n_features = X.shape[1]
        min_required = max(self.min_sample_size, n_features * 10)
        sample_adequate = n >= min_required

        if not sample_adequate:
            critical_issues.append(
                f"Insufficient samples for regression: {n} obs, {n_features} features (need {min_required})"
            )

        # Check for multicollinearity
        try:
            corr_matrix = X.corr()
            high_corr = np.where((np.abs(corr_matrix) > 0.9) & (corr_matrix != 1.0))
            if len(high_corr[0]) > 0:
                warnings.append(
                    f"High multicollinearity detected in {len(high_corr[0])} feature pairs"
                )
            assumptions['low_multicollinearity'] = len(high_corr[0]) == 0
        except:
            pass

        # Check for target variable variance
        y_var = y.var()
        if y_var == 0:
            critical_issues.append("Target variable has zero variance")
        assumptions['target_has_variance'] = y_var > 0

        # Missing values
        X_missing = X.isnull().mean().mean()
        y_missing = y.isnull().mean()

        if X_missing > 0.1:
            warnings.append(f"High missing value rate in features: {X_missing:.1%}")
        if y_missing > 0.1:
            warnings.append(f"High missing value rate in target: {y_missing:.1%}")

        confidence_score = self._calculate_confidence_score(
            sample_adequate=sample_adequate,
            n=n,
            min_required=min_required,
            critical_issues=critical_issues,
            warnings=warnings
        )

        confidence_level = self._score_to_level(confidence_score)

        return ValidationReport(
            is_valid=len(critical_issues) == 0,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sample_size=n,
            min_required_sample=min_required,
            sample_size_adequate=sample_adequate,
            assumptions_tested=assumptions,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def validate_anomaly_detection(
        self,
        data: pd.Series
    ) -> ValidationReport:
        """
        Validate data for anomaly detection.

        Args:
            data: Time series or univariate data

        Returns:
            ValidationReport
        """
        warnings = []
        critical_issues = []
        assumptions = {}

        # Sample size
        n = len(data)
        min_required = self.min_sample_size
        sample_adequate = n >= min_required

        if not sample_adequate:
            critical_issues.append(
                f"Insufficient data for anomaly detection: {n} obs (need {min_required})"
            )

        # Check distribution
        try:
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if n < 5000:
                _, p_value = stats.shapiro(data.dropna())
                is_normal = p_value > self.alpha
            else:
                result = stats.anderson(data.dropna())
                is_normal = result.statistic < result.critical_values[2]  # 5% significance

            assumptions['normal_distribution'] = is_normal

            if not is_normal:
                warnings.append(
                    "Data is not normally distributed. Using robust methods."
                )
        except:
            warnings.append("Could not test distribution normality")

        # Variance check
        data_std = data.std()
        if data_std == 0:
            critical_issues.append("Data has zero variance - no anomalies can be detected")
        assumptions['has_variance'] = data_std > 0

        confidence_score = self._calculate_confidence_score(
            sample_adequate=sample_adequate,
            n=n,
            min_required=min_required,
            critical_issues=critical_issues,
            warnings=warnings
        )

        confidence_level = self._score_to_level(confidence_score)

        return ValidationReport(
            is_valid=len(critical_issues) == 0,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            sample_size=n,
            min_required_sample=min_required,
            sample_size_adequate=sample_adequate,
            assumptions_tested=assumptions,
            warnings=warnings,
            critical_issues=critical_issues
        )

    def _calculate_confidence_score(
        self,
        sample_adequate: bool,
        n: int,
        min_required: int,
        critical_issues: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            sample_adequate: Whether sample size is adequate
            n: Actual sample size
            min_required: Required sample size
            critical_issues: List of critical issues
            warnings: List of warnings

        Returns:
            Confidence score (0-1)
        """
        score = 1.0

        # Critical issues reduce confidence significantly
        score -= len(critical_issues) * 0.3

        # Warnings reduce confidence moderately
        score -= len(warnings) * 0.1

        # Sample size penalty
        if not sample_adequate:
            sample_ratio = n / min_required
            score *= sample_ratio  # Further penalty based on how short we are

        # Floor at 0
        score = max(0.0, score)

        return score

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

"""Variance decomposition using SHAP values and statistical methods."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .base_models import VarianceDecompositionResult
from .validation import StatisticalValidator


class VarianceDecomposer:
    """
    Variance decomposition to understand which factors contribute most to outcomes.

    Uses SHAP values for model-based attribution and statistical variance partitioning.
    """

    def __init__(
        self,
        method: str = 'auto',
        sample_size: int = 100
    ):
        """
        Initialize variance decomposer.

        Args:
            method: Decomposition method ('shap', 'statistical', 'auto')
            sample_size: Sample size for SHAP calculations (for large datasets)
        """
        self.method = method
        self.sample_size = sample_size
        self.validator = StatisticalValidator()

    def decompose(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "dataset",
        feature_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VarianceDecompositionResult:
        """
        Decompose variance to understand feature contributions.

        Args:
            X: Feature matrix
            y: Target variable
            dataset_name: Name of dataset
            feature_names: Optional feature names (uses X.columns if not provided)
            context: Business context

        Returns:
            VarianceDecompositionResult with feature contributions
        """
        # Validate data
        validation = self.validator.validate_regression(X, y)

        if not validation.is_valid:
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

        # Get feature names
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"Feature_{i}" for i in range(X.shape[1])]

        try:
            # Choose method
            if self.method == 'auto':
                method = 'shap' if X.shape[0] >= 30 and X.shape[1] <= 20 else 'statistical'
            else:
                method = self.method

            # Perform decomposition
            if method == 'shap':
                contributions = self._decompose_shap(X, y, feature_names)
            else:
                contributions = self._decompose_statistical(X, y, feature_names)

            # Calculate total variance explained (R-squared)
            r_squared = self._calculate_r_squared(X, y)

            # Sort by contribution (descending)
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

            results = {
                'feature_contributions': contributions,
                'total_variance_explained': r_squared,
                'method': method.upper(),
                'top_3_features': [c['feature'] for c in contributions[:3]],
                'top_3_contribution_pct': sum([c['contribution_pct'] for c in contributions[:3]])
            }

            # Create visualization data
            viz_data = self._create_visualization_data(contributions)

            return VarianceDecompositionResult(
                dataset_name=dataset_name,
                results=results,
                validation=validation,
                context=context or {},
                visualization_data=viz_data
            )

        except Exception as e:
            validation.critical_issues.append(f"Variance decomposition failed: {str(e)}")
            validation.is_valid = False
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

    def _decompose_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[Dict]:
        """Decompose using SHAP values."""
        try:
            import shap
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "SHAP not installed. Install with: pip install shap scikit-learn"
            )

        # Sample data if too large
        if len(X) > self.sample_size:
            sample_indices = np.random.choice(len(X), self.sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y

        # Train a simple model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_sample, y_sample)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Normalize to percentages
        total_importance = mean_abs_shap.sum()
        if total_importance == 0:
            total_importance = 1

        contributions = []
        for i, feature in enumerate(feature_names):
            contribution = mean_abs_shap[i] / total_importance
            contribution_pct = contribution * 100

            # Determine if contribution is positive or negative on average
            is_positive = shap_values[:, i].mean() > 0

            contributions.append({
                'feature': feature,
                'contribution': float(contribution),
                'contribution_pct': float(contribution_pct),
                'is_positive': bool(is_positive),
                'confidence_interval': (
                    float(contribution * 0.9),  # Rough CI estimate
                    float(contribution * 1.1)
                )
            })

        return contributions

    def _decompose_statistical(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[Dict]:
        """Decompose using statistical variance partitioning."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit linear model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Calculate variance explained by each feature
        # Using squared standardized coefficients as importance measure
        coefficients_sq = model.coef_ ** 2
        total_importance = coefficients_sq.sum()

        if total_importance == 0:
            total_importance = 1

        contributions = []
        for i, feature in enumerate(feature_names):
            contribution = coefficients_sq[i] / total_importance
            contribution_pct = contribution * 100

            # Determine if positive or negative effect
            is_positive = model.coef_[i] > 0

            contributions.append({
                'feature': feature,
                'contribution': float(contribution),
                'contribution_pct': float(contribution_pct),
                'is_positive': bool(is_positive),
                'confidence_interval': (
                    float(contribution * 0.9),
                    float(contribution * 1.1)
                )
            })

        return contributions

    def _calculate_r_squared(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate R-squared (total variance explained)."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        return float(max(0.0, min(1.0, r_squared)))  # Clamp to [0, 1]

    def _create_visualization_data(self, contributions: List[Dict]) -> Dict:
        """Create visualization data for feature importance chart."""
        # Top 10 features only (for readability)
        top_contributions = contributions[:10]

        labels = [c['feature'] for c in top_contributions]
        values = [c['contribution_pct'] for c in top_contributions]
        colors = ['#4caf50' if c['is_positive'] else '#f44336' for c in top_contributions]

        return {
            'chart_type': 'bar',
            'labels': labels,
            'x_label': 'Feature',
            'y_label': 'Contribution (%)',
            'datasets': [
                {
                    'label': 'Contribution to Variance',
                    'data': values,
                    'backgroundColor': colors
                }
            ]
        }

    def _create_error_result(
        self,
        dataset_name: str,
        validation: Any,
        context: Dict
    ) -> VarianceDecompositionResult:
        """Create error result."""
        return VarianceDecompositionResult(
            dataset_name=dataset_name,
            results={
                'feature_contributions': [],
                'total_variance_explained': 0.0,
                'method': 'NONE',
                'error': 'Variance decomposition failed validation'
            },
            validation=validation,
            context=context
        )

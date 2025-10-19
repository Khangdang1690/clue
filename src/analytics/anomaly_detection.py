"""Anomaly detection using statistical methods."""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_models import AnomalyDetectionResult
from .validation import StatisticalValidator


class AnomalyDetector:
    """
    Statistical anomaly detection for time series and univariate data.

    Uses multiple methods: Z-score, IQR, and Isolation Forest.
    """

    def __init__(
        self,
        method: str = 'auto',
        threshold: float = 3.0,
        contamination: float = 0.1
    ):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method ('zscore', 'iqr', 'isolation_forest', 'auto')
            threshold: Z-score threshold for 'zscore' method (default 3.0)
            contamination: Expected proportion of anomalies (for isolation forest)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.validator = StatisticalValidator()

    def detect_anomalies(
        self,
        data: pd.Series,
        dataset_name: str = "dataset",
        context: Optional[Dict[str, Any]] = None
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in time series or univariate data.

        Args:
            data: Data to analyze (time series preferred)
            dataset_name: Name of dataset
            context: Business context

        Returns:
            AnomalyDetectionResult with detected anomalies
        """
        # Validate data
        validation = self.validator.validate_anomaly_detection(data)

        if not validation.is_valid:
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

        try:
            # Choose method
            if self.method == 'auto':
                method = self._choose_method(data)
            else:
                method = self.method

            # Detect anomalies
            if method == 'zscore':
                anomalies = self._detect_zscore(data)
            elif method == 'iqr':
                anomalies = self._detect_iqr(data)
            else:  # isolation_forest
                anomalies = self._detect_isolation_forest(data)

            # Calculate statistics
            total_anomalies = len(anomalies)
            anomaly_rate = total_anomalies / len(data)

            results = {
                'anomalies': anomalies,
                'total_anomalies': total_anomalies,
                'anomaly_rate': float(anomaly_rate),
                'detection_method': method.upper(),
                'severity_breakdown': self._calculate_severity_breakdown(anomalies)
            }

            # Create visualization data
            viz_data = self._create_visualization_data(data, anomalies)

            return AnomalyDetectionResult(
                dataset_name=dataset_name,
                results=results,
                validation=validation,
                context=context or {},
                visualization_data=viz_data
            )

        except Exception as e:
            validation.critical_issues.append(f"Anomaly detection failed: {str(e)}")
            validation.is_valid = False
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

    def _detect_zscore(self, data: pd.Series) -> List[Dict]:
        """Detect anomalies using Z-score method."""
        mean = data.mean()
        std = data.std()

        if std == 0:
            return []

        z_scores = np.abs((data - mean) / std)

        anomalies = []
        for idx, (timestamp, value) in enumerate(data.items()):
            z_score = z_scores.iloc[idx]
            if z_score > self.threshold:
                anomalies.append({
                    'timestamp': str(timestamp) if isinstance(data.index, pd.DatetimeIndex) else f"T{idx}",
                    'value': float(value),
                    'expected_value': float(mean),
                    'deviation': float(value - mean),
                    'severity': self._calculate_severity(z_score),
                    'anomaly_score': float(z_score),
                    'is_outlier': True
                })

        return anomalies

    def _detect_iqr(self, data: pd.Series) -> List[Dict]:
        """Detect anomalies using Interquartile Range (IQR) method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = data.median()

        anomalies = []
        for idx, (timestamp, value) in enumerate(data.items()):
            if value < lower_bound or value > upper_bound:
                # Calculate normalized deviation
                deviation = value - median
                normalized_deviation = abs(deviation) / IQR if IQR > 0 else 0

                anomalies.append({
                    'timestamp': str(timestamp) if isinstance(data.index, pd.DatetimeIndex) else f"T{idx}",
                    'value': float(value),
                    'expected_value': float(median),
                    'deviation': float(deviation),
                    'severity': self._calculate_severity_iqr(value, lower_bound, upper_bound, IQR),
                    'anomaly_score': float(normalized_deviation),
                    'is_outlier': True
                })

        return anomalies

    def _detect_isolation_forest(self, data: pd.Series) -> List[Dict]:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )

        # Reshape data
        X = data.values.reshape(-1, 1)

        # Fit Isolation Forest
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        predictions = clf.fit_predict(X)
        scores = clf.score_samples(X)

        # Get mean for expected value
        mean = data.mean()

        anomalies = []
        for idx, (timestamp, value) in enumerate(data.items()):
            if predictions[idx] == -1:  # Anomaly
                anomaly_score = -scores[idx]  # More negative = more anomalous

                anomalies.append({
                    'timestamp': str(timestamp) if isinstance(data.index, pd.DatetimeIndex) else f"T{idx}",
                    'value': float(value),
                    'expected_value': float(mean),
                    'deviation': float(value - mean),
                    'severity': self._calculate_severity_isolation(anomaly_score),
                    'anomaly_score': float(anomaly_score),
                    'is_outlier': True
                })

        return anomalies

    def _calculate_severity(self, z_score: float) -> str:
        """Calculate severity based on Z-score."""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'

    def _calculate_severity_iqr(
        self,
        value: float,
        lower_bound: float,
        upper_bound: float,
        iqr: float
    ) -> str:
        """Calculate severity based on IQR deviation."""
        if value < lower_bound:
            deviation = (lower_bound - value) / iqr if iqr > 0 else 0
        else:
            deviation = (value - upper_bound) / iqr if iqr > 0 else 0

        if deviation > 3:
            return 'critical'
        elif deviation > 2:
            return 'high'
        elif deviation > 1.5:
            return 'medium'
        else:
            return 'low'

    def _calculate_severity_isolation(self, anomaly_score: float) -> str:
        """Calculate severity for Isolation Forest score."""
        if anomaly_score > 1.5:
            return 'critical'
        elif anomaly_score > 1.0:
            return 'high'
        elif anomaly_score > 0.5:
            return 'medium'
        else:
            return 'low'

    def _calculate_severity_breakdown(self, anomalies: List[Dict]) -> Dict[str, int]:
        """Calculate breakdown of anomalies by severity."""
        breakdown = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for anomaly in anomalies:
            severity = anomaly['severity']
            breakdown[severity] += 1

        return breakdown

    def _choose_method(self, data: pd.Series) -> str:
        """Choose best detection method based on data characteristics."""
        # Check normality
        if len(data) >= 20:
            _, p_value = stats.shapiro(data) if len(data) < 5000 else (0, 0.5)
            is_normal = p_value > 0.05

            if is_normal:
                return 'zscore'

        # IQR is robust for non-normal data
        if len(data) >= 10:
            return 'iqr'

        # Default to IQR
        return 'iqr'

    def _create_visualization_data(
        self,
        data: pd.Series,
        anomalies: List[Dict]
    ) -> Dict:
        """Create visualization data."""
        timestamps = (
            data.index.strftime('%Y-%m-%d').tolist()
            if isinstance(data.index, pd.DatetimeIndex)
            else [f"T{i}" for i in range(len(data))]
        )

        # Create anomaly markers (None for normal points)
        anomaly_timestamps = {a['timestamp'] for a in anomalies}
        anomaly_values = [
            float(value) if timestamps[idx] in anomaly_timestamps else None
            for idx, value in enumerate(data)
        ]

        return {
            'chart_type': 'line',
            'labels': timestamps,
            'x_label': 'Time Period',
            'y_label': 'Value',
            'datasets': [
                {
                    'label': 'Values',
                    'data': data.tolist(),
                    'type': 'line'
                },
                {
                    'label': 'Anomalies',
                    'data': anomaly_values,
                    'type': 'scatter',
                    'backgroundColor': '#f44336',
                    'pointRadius': 8
                }
            ]
        }

    def _create_error_result(
        self,
        dataset_name: str,
        validation: Any,
        context: Dict
    ) -> AnomalyDetectionResult:
        """Create error result."""
        return AnomalyDetectionResult(
            dataset_name=dataset_name,
            results={
                'anomalies': [],
                'total_anomalies': 0,
                'anomaly_rate': 0.0,
                'detection_method': 'NONE',
                'error': 'Anomaly detection failed validation'
            },
            validation=validation,
            context=context
        )

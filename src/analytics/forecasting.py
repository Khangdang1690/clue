"""Time series forecasting with statistical validation."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .base_models import ForecastResult, ConfidenceLevel
from .validation import StatisticalValidator


class TimeSeriesForecaster:
    """
    Time series forecasting using Prophet and ARIMA.

    Provides mathematically validated forecasts with confidence intervals.
    """

    def __init__(
        self,
        method: str = 'auto',
        confidence_level: float = 0.95
    ):
        """
        Initialize forecaster.

        Args:
            method: Forecasting method ('prophet', 'arima', 'auto')
            confidence_level: Confidence level for intervals (0-1)
        """
        self.method = method
        self.confidence_level = confidence_level
        self.validator = StatisticalValidator()

    def forecast(
        self,
        data: pd.Series,
        periods: int,
        dataset_name: str = "dataset",
        freq: str = 'auto',
        context: Optional[Dict[str, Any]] = None
    ) -> ForecastResult:
        """
        Generate forecast with confidence intervals.

        Args:
            data: Time series data (with datetime index preferred)
            periods: Number of periods to forecast
            dataset_name: Name of dataset
            freq: Frequency ('D', 'M', 'Q', 'Y', or 'auto')
            context: Business context for LLM interpretation

        Returns:
            ForecastResult with predictions and validation
        """
        # Validate data
        validation = self.validator.validate_forecasting(data, periods)

        if not validation.is_valid:
            # Return error result
            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

        # Infer frequency if auto
        if freq == 'auto':
            freq = self._infer_frequency(data)

        # Choose method
        if self.method == 'auto':
            method = self._choose_method(data, periods)
        else:
            method = self.method

        # Perform forecast
        try:
            if method == 'prophet':
                forecast_result = self._forecast_prophet(data, periods, freq)
            else:  # arima
                forecast_result = self._forecast_arima(data, periods, freq)

            # Calculate accuracy metrics on historical data
            accuracy_metrics = self._calculate_accuracy_metrics(data, method, periods)

            # Build results dictionary
            results = {
                'predictions': forecast_result['predictions'],
                'timestamps': forecast_result['timestamps'],
                'confidence_intervals': forecast_result['confidence_intervals'],
                'current_value': float(data.iloc[-1]),
                'forecast_horizon': periods,
                'model_name': method.upper(),
                'accuracy_metrics': accuracy_metrics,
                'frequency': freq,
                'growth_rate': self._calculate_growth_rate(
                    current=float(data.iloc[-1]),
                    forecast=forecast_result['predictions'][0] if forecast_result['predictions'] else None
                )
            }

            # Create visualization data
            viz_data = self._create_visualization_data(data, forecast_result, freq)

            return ForecastResult(
                dataset_name=dataset_name,
                results=results,
                validation=validation,
                context=context or {},
                visualization_data=viz_data
            )

        except Exception as e:
            # Add error to validation
            validation.critical_issues.append(f"Forecast failed: {str(e)}")
            validation.is_valid = False
            validation.confidence_level = ConfidenceLevel.VERY_LOW
            validation.confidence_score = 0.0

            return self._create_error_result(
                dataset_name=dataset_name,
                validation=validation,
                context=context or {}
            )

    def _forecast_prophet(
        self,
        data: pd.Series,
        periods: int,
        freq: str
    ) -> Dict:
        """Forecast using Prophet."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet not installed. Install with: pip install prophet"
            )

        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index if isinstance(data.index, pd.DatetimeIndex) else pd.date_range(
                start=datetime.now(),
                periods=len(data),
                freq=freq
            ),
            'y': data.values
        })

        # Initialize and fit Prophet
        model = Prophet(
            interval_width=self.confidence_level,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto'
        )
        model.fit(df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Extract forecast values (only future periods)
        forecast_future = forecast.iloc[-periods:]

        predictions = forecast_future['yhat'].tolist()
        timestamps = forecast_future['ds'].dt.strftime('%Y-%m-%d').tolist()
        confidence_intervals = list(zip(
            forecast_future['yhat_lower'].tolist(),
            forecast_future['yhat_upper'].tolist()
        ))

        return {
            'predictions': predictions,
            'timestamps': timestamps,
            'confidence_intervals': confidence_intervals
        }

    def _forecast_arima(
        self,
        data: pd.Series,
        periods: int,
        freq: str
    ) -> Dict:
        """Forecast using ARIMA."""
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        # Make data stationary if needed
        is_stationary = adfuller(data.dropna())[1] < 0.05
        d = 0 if is_stationary else 1

        # Fit ARIMA model (simple auto params)
        try:
            model = ARIMA(data, order=(1, d, 1))
            fitted = model.fit()

            # Forecast
            forecast = fitted.forecast(steps=periods)
            conf_int = fitted.get_forecast(steps=periods).conf_int(alpha=1-self.confidence_level)

            # Create timestamps
            if isinstance(data.index, pd.DatetimeIndex):
                last_date = data.index[-1]
                timestamps = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=self._freq_to_timedelta_unit(freq)),
                    periods=periods,
                    freq=freq
                ).strftime('%Y-%m-%d').tolist()
            else:
                timestamps = [f"T+{i+1}" for i in range(periods)]

            predictions = forecast.tolist()
            confidence_intervals = [(row[0], row[1]) for _, row in conf_int.iterrows()]

            return {
                'predictions': predictions,
                'timestamps': timestamps,
                'confidence_intervals': confidence_intervals
            }

        except Exception as e:
            raise ValueError(f"ARIMA forecast failed: {e}")

    def _calculate_accuracy_metrics(
        self,
        data: pd.Series,
        method: str,
        periods: int
    ) -> Dict[str, float]:
        """Calculate forecast accuracy on historical data using backtesting."""
        try:
            # Use last 20% of data as test set
            test_size = max(periods, int(len(data) * 0.2))
            if len(data) < test_size + 10:
                return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0}

            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]

            # Refit model and forecast
            if method == 'prophet':
                from prophet import Prophet
                df_train = pd.DataFrame({
                    'ds': train.index if isinstance(train.index, pd.DatetimeIndex) else range(len(train)),
                    'y': train.values
                })
                model = Prophet(interval_width=self.confidence_level)
                model.fit(df_train)
                future = model.make_future_dataframe(periods=test_size)
                forecast = model.predict(future)
                predictions = forecast.iloc[-test_size:]['yhat'].values
            else:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(train, order=(1, 1, 1))
                fitted = model.fit()
                predictions = fitted.forecast(steps=test_size)

            # Calculate metrics
            mape = np.mean(np.abs((test.values - predictions) / test.values)) * 100
            mae = np.mean(np.abs(test.values - predictions))
            rmse = np.sqrt(np.mean((test.values - predictions) ** 2))

            return {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse)
            }

        except:
            return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0}

    def _calculate_growth_rate(
        self,
        current: float,
        forecast: Optional[float]
    ) -> Optional[float]:
        """Calculate growth rate from current to first forecast period."""
        if forecast is None or current == 0:
            return None
        return ((forecast - current) / current) * 100

    def _infer_frequency(self, data: pd.Series) -> str:
        """Infer time series frequency."""
        if isinstance(data.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(data.index)
            if inferred:
                return inferred

        # Default to monthly
        return 'M'

    def _freq_to_timedelta_unit(self, freq: str) -> str:
        """Convert frequency to timedelta unit."""
        mapping = {
            'D': 'D',
            'W': 'W',
            'M': 'M',
            'Q': 'Q',
            'Y': 'Y',
            'H': 'H'
        }
        return mapping.get(freq, 'D')

    def _choose_method(self, data: pd.Series, periods: int) -> str:
        """Choose best forecasting method."""
        # Prophet is better for business data with seasonality
        # Use ARIMA only for very simple/short series
        if len(data) >= 24:
            return 'prophet'
        else:
            return 'arima'

    def _create_visualization_data(
        self,
        historical_data: pd.Series,
        forecast_result: Dict,
        freq: str
    ) -> Dict:
        """Create data for visualization."""
        # Historical data
        historical_timestamps = (
            historical_data.index.strftime('%Y-%m-%d').tolist()
            if isinstance(historical_data.index, pd.DatetimeIndex)
            else [f"T-{len(historical_data)-i}" for i in range(len(historical_data))]
        )

        # Combine historical and forecast
        all_timestamps = historical_timestamps + forecast_result['timestamps']

        # Historical values + forecast
        historical_values = historical_data.tolist()
        forecast_values = forecast_result['predictions']

        return {
            'chart_type': 'line',
            'labels': all_timestamps,
            'x_label': 'Time Period',
            'y_label': 'Value',
            'datasets': [
                {
                    'label': 'Historical',
                    'data': historical_values + [None] * len(forecast_values)
                },
                {
                    'label': 'Forecast',
                    'data': [None] * len(historical_values) + forecast_values
                },
                {
                    'label': 'Lower Bound',
                    'data': [None] * len(historical_values) + [ci[0] for ci in forecast_result['confidence_intervals']],
                    'borderDash': [5, 5]
                },
                {
                    'label': 'Upper Bound',
                    'data': [None] * len(historical_values) + [ci[1] for ci in forecast_result['confidence_intervals']],
                    'borderDash': [5, 5]
                }
            ]
        }

    def _create_error_result(
        self,
        dataset_name: str,
        validation: Any,
        context: Dict
    ) -> ForecastResult:
        """Create error result when forecast fails."""
        return ForecastResult(
            dataset_name=dataset_name,
            results={
                'predictions': [],
                'timestamps': [],
                'confidence_intervals': [],
                'current_value': 0.0,
                'forecast_horizon': 0,
                'model_name': 'NONE',
                'accuracy_metrics': {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0},
                'error': 'Forecast failed validation'
            },
            validation=validation,
            context=context
        )

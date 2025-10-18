"""Data profiler for comprehensive dataset analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.models.discovery_models import DataProfile


class DataProfiler:
    """Profiles datasets to understand their structure and characteristics."""

    def __init__(self, sample_size: int = 100000):
        """
        Initialize data profiler.

        Args:
            sample_size: Maximum rows to use for expensive computations
        """
        self.sample_size = sample_size

    def profile_dataset(self, df: pd.DataFrame) -> DataProfile:
        """
        Generate comprehensive profile of a dataset.

        Args:
            df: DataFrame to profile

        Returns:
            DataProfile object with statistical summary
        """
        print("\n" + "="*60)
        print("=== DATA PROFILING ===")
        print("="*60)

        # Basic info
        num_rows, num_columns = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        print(f"\nDataset size: {num_rows:,} rows × {num_columns} columns")
        print(f"Memory usage: {memory_mb:.2f} MB")

        # Sample for expensive operations
        sample_df = self._get_sample(df)

        # Classify columns by type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        text_cols = self._identify_text_columns(df, categorical_cols)

        # Remove text columns from categorical
        categorical_cols = [c for c in categorical_cols if c not in text_cols]

        print(f"\nColumn types:")
        print(f"  Numeric: {len(numeric_cols)}")
        print(f"  Categorical: {len(categorical_cols)}")
        print(f"  Datetime: {len(datetime_cols)}")
        print(f"  Text: {len(text_cols)}")

        # Missing value analysis
        missing_rate = df.isnull().sum().sum() / (num_rows * num_columns)
        cols_with_missing = df.columns[df.isnull().any()].tolist()

        print(f"\nMissing values:")
        print(f"  Overall missing rate: {missing_rate:.2%}")
        print(f"  Columns with missing: {len(cols_with_missing)}")

        # Statistical summaries
        numeric_summary = None
        if numeric_cols:
            numeric_summary = self._numeric_summary(sample_df[numeric_cols])

        categorical_summary = None
        if categorical_cols:
            categorical_summary = self._categorical_summary(sample_df[categorical_cols])

        # Temporal analysis
        has_temporal = len(datetime_cols) > 0
        temporal_columns = datetime_cols.copy()

        # Outlier detection
        outlier_cols = self._detect_outliers(sample_df, numeric_cols)

        # High cardinality detection
        high_card_cols = self._detect_high_cardinality(df, categorical_cols)

        # Distribution analysis (skewness)
        skewed_cols = self._analyze_skewness(sample_df, numeric_cols)

        print(f"\nPattern detection:")
        print(f"  Temporal columns: {len(temporal_columns)}")
        print(f"  Outlier columns: {len(outlier_cols)}")
        print(f"  High cardinality columns: {len(high_card_cols)}")
        print(f"  Skewed distributions: {len(skewed_cols)}")

        profile = DataProfile(
            num_rows=num_rows,
            num_columns=num_columns,
            memory_usage_mb=memory_mb,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            overall_missing_rate=missing_rate,
            columns_with_missing=cols_with_missing,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            has_temporal_data=has_temporal,
            temporal_columns=temporal_columns,
            outlier_columns=outlier_cols,
            high_cardinality_columns=high_card_cols,
            skewed_columns=skewed_cols
        )

        print("\n[OK] Data profiling completed")
        print("="*60)

        return profile

    def _get_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a representative sample of the data.

        Args:
            df: Full DataFrame

        Returns:
            Sampled DataFrame
        """
        if len(df) <= self.sample_size:
            return df

        # Use stratified sampling if possible
        sample_fraction = self.sample_size / len(df)
        return df.sample(frac=sample_fraction, random_state=42)

    def _identify_text_columns(
        self,
        df: pd.DataFrame,
        candidate_cols: List[str]
    ) -> List[str]:
        """
        Identify text columns (long strings, not categorical).

        Args:
            df: DataFrame
            candidate_cols: Object-type columns to check

        Returns:
            List of text column names
        """
        text_cols = []

        for col in candidate_cols:
            # Sample the column
            sample = df[col].dropna().head(100)

            if len(sample) == 0:
                continue

            # Check average string length
            avg_length = sample.astype(str).str.len().mean()

            # If average length > 50 chars, likely text
            if avg_length > 50:
                text_cols.append(col)

        return text_cols

    def _numeric_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for numeric columns.

        Args:
            df: DataFrame with numeric columns

        Returns:
            Dictionary with summary stats
        """
        summary = {}

        desc = df.describe()

        for col in df.columns:
            summary[col] = {
                'mean': float(desc.loc['mean', col]),
                'std': float(desc.loc['std', col]),
                'min': float(desc.loc['min', col]),
                'max': float(desc.loc['max', col]),
                'q25': float(desc.loc['25%', col]),
                'q50': float(desc.loc['50%', col]),
                'q75': float(desc.loc['75%', col]),
                'unique_count': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum())
            }

        return summary

    def _categorical_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary for categorical columns.

        Args:
            df: DataFrame with categorical columns

        Returns:
            Dictionary with categorical summary
        """
        summary = {}

        for col in df.columns:
            value_counts = df[col].value_counts()

            summary[col] = {
                'unique_count': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum()),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }

        return summary

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[str]:
        """
        Detect columns with significant outliers using IQR method.

        Args:
            df: DataFrame
            numeric_cols: List of numeric column names

        Returns:
            List of columns with outliers
        """
        outlier_cols = []

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 10:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = series[(series < lower_bound) | (series > upper_bound)]

            # If more than 5% outliers, flag it
            if len(outliers) / len(series) > 0.05:
                outlier_cols.append(col)

        return outlier_cols

    def _detect_high_cardinality(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> List[str]:
        """
        Detect high cardinality categorical columns.

        Args:
            df: DataFrame
            categorical_cols: List of categorical column names

        Returns:
            List of high cardinality columns
        """
        high_card_cols = []

        for col in categorical_cols:
            unique_count = df[col].nunique()
            total_count = len(df[col])

            # If unique values > 50% of total, it's high cardinality
            if unique_count / total_count > 0.5 and unique_count > 100:
                high_card_cols.append(col)

        return high_card_cols

    def _analyze_skewness(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyze distribution skewness.

        Args:
            df: DataFrame
            numeric_cols: List of numeric column names

        Returns:
            List of (column, skewness) tuples for skewed columns
        """
        skewed_cols = []

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 10:
                continue

            skewness = float(series.skew())

            # Flag if |skewness| > 1
            if abs(skewness) > 1:
                skewed_cols.append((col, skewness))

        return skewed_cols

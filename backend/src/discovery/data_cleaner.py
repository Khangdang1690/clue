"""Comprehensive data cleaning for discovery workflow."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re


class DataCleaner:
    """Cleans and prepares data for discovery analysis."""

    def __init__(
        self,
        handle_missing: str = 'auto',  # 'auto', 'drop', 'fill'
        handle_outliers: str = 'flag',  # 'flag', 'remove', 'cap', 'none'
        handle_duplicates: str = 'keep_first',  # 'keep_first', 'keep_last', 'remove_all'
        outlier_threshold: float = 3.0,  # IQR multiplier or z-score threshold
        missing_threshold: float = 0.9,  # Drop columns with > 90% missing
        duplicate_subset: Optional[List[str]] = None  # Columns to check for duplicates
    ):
        """
        Initialize data cleaner.

        Args:
            handle_missing: How to handle missing values
            handle_outliers: How to handle outliers
            handle_duplicates: How to handle duplicate rows
            outlier_threshold: Threshold for outlier detection
            missing_threshold: Drop columns with missing rate above this
            duplicate_subset: Columns to check for duplicates (None = all columns)
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.handle_duplicates = handle_duplicates
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.duplicate_subset = duplicate_subset

        self.cleaning_report = {
            'rows_before': 0,
            'rows_after': 0,
            'columns_before': 0,
            'columns_after': 0,
            'duplicates_removed': 0,
            'outliers_flagged': 0,
            'outliers_removed': 0,
            'columns_dropped': [],
            'missing_values_filled': 0,
            'data_types_converted': {},
            'warnings': []
        }

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive data cleaning.

        Args:
            df: DataFrame to clean

        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        print("\n" + "="*60)
        print("=== DATA CLEANING ===")
        print("="*60)

        self.cleaning_report['rows_before'] = len(df)
        self.cleaning_report['columns_before'] = len(df.columns)

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Step 1: Basic validation
        df_clean = self._validate_dataframe(df_clean)

        # Step 2: Detect and convert data types
        df_clean = self._detect_and_convert_types(df_clean)

        # Step 3: Handle duplicates
        df_clean = self._handle_duplicates(df_clean)

        # Step 4: Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Step 5: Handle outliers
        df_clean = self._handle_outliers(df_clean)

        # Step 6: Clean text columns
        df_clean = self._clean_text_columns(df_clean)

        # Step 7: Standardize column names
        df_clean = self._standardize_column_names(df_clean)

        # Step 8: Final validation
        df_clean = self._final_validation(df_clean)

        self.cleaning_report['rows_after'] = len(df_clean)
        self.cleaning_report['columns_after'] = len(df_clean.columns)

        self._print_cleaning_summary()

        return df_clean, self.cleaning_report

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic validation checks."""
        print("\n📋 Step 1: Basic validation...")

        # Check for completely empty DataFrame
        if df.empty:
            self.cleaning_report['warnings'].append("DataFrame is empty")
            return df

        # Check for columns with all null values
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            df = df.drop(columns=all_null_cols)
            self.cleaning_report['columns_dropped'].extend(all_null_cols)
            print(f"  [OK] Dropped {len(all_null_cols)} columns with all null values")

        # Check for columns with single unique value (no variance)
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                single_value_cols.append(col)

        if single_value_cols:
            df = df.drop(columns=single_value_cols)
            self.cleaning_report['columns_dropped'].extend(single_value_cols)
            print(f"  [OK] Dropped {len(single_value_cols)} columns with single value (no variance)")

        print(f"  [OK] Validation complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _detect_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert data types intelligently."""
        print("\n🔍 Step 2: Detecting and converting data types...")

        for col in df.columns:
            original_dtype = df[col].dtype

            # Skip if already numeric or datetime
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            # Try to detect dates
            if self._is_date_column(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.cleaning_report['data_types_converted'][col] = f"{original_dtype} → datetime64"
                    print(f"  [OK] Converted '{col}' to datetime")
                except Exception:
                    pass

            # Try to detect numeric (stored as string)
            elif self._is_numeric_column(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.cleaning_report['data_types_converted'][col] = f"{original_dtype} → numeric"
                    print(f"  [OK] Converted '{col}' to numeric")
                except Exception:
                    pass

            # Try to detect boolean
            elif self._is_boolean_column(df[col]):
                try:
                    df[col] = df[col].map({'true': True, 'false': False, '1': True, '0': False,
                                           'yes': True, 'no': False, 'y': True, 'n': False,
                                           True: True, False: False, 1: True, 0: False})
                    self.cleaning_report['data_types_converted'][col] = f"{original_dtype} → boolean"
                    print(f"  [OK] Converted '{col}' to boolean")
                except Exception:
                    pass

        return df

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column appears to contain dates."""
        # Sample first non-null value
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        # Check for date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',  # 01-15-2024
        ]

        sample_str = sample.astype(str)
        for pattern in date_patterns:
            if sample_str.str.match(pattern).any():
                return True

        return False

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column appears to contain numbers stored as strings."""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        # Try to convert sample to numeric
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            # Check if majority can be converted
            numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
            return numeric_count / len(sample) > 0.8

    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if a column appears to contain boolean values."""
        unique_values = set(series.dropna().astype(str).str.lower().unique())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f'}

        return unique_values.issubset(boolean_values) and len(unique_values) <= 3

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows."""
        print("\n🔄 Step 3: Handling duplicates...")

        duplicates_before = df.duplicated(subset=self.duplicate_subset).sum()

        if duplicates_before == 0:
            print("  [OK] No duplicates found")
            return df

        if self.handle_duplicates == 'keep_first':
            df = df.drop_duplicates(subset=self.duplicate_subset, keep='first')
            print(f"  [OK] Removed {duplicates_before} duplicates (kept first occurrence)")
        elif self.handle_duplicates == 'keep_last':
            df = df.drop_duplicates(subset=self.duplicate_subset, keep='last')
            print(f"  [OK] Removed {duplicates_before} duplicates (kept last occurrence)")
        elif self.handle_duplicates == 'remove_all':
            df = df[~df.duplicated(subset=self.duplicate_subset, keep=False)]
            print(f"  [OK] Removed {duplicates_before} duplicates (removed all occurrences)")

        self.cleaning_report['duplicates_removed'] = duplicates_before

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently."""
        print("\n❓ Step 4: Handling missing values...")

        # Drop columns with > threshold missing
        missing_rates = df.isnull().sum() / len(df)
        cols_to_drop = missing_rates[missing_rates > self.missing_threshold].index.tolist()

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.cleaning_report['columns_dropped'].extend(cols_to_drop)
            print(f"  [OK] Dropped {len(cols_to_drop)} columns with >{self.missing_threshold:.0%} missing values")

        if self.handle_missing == 'auto':
            # Auto strategy: Fill based on data type and cardinality
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric: fill with median (robust to outliers)
                    df[col].fillna(df[col].median(), inplace=True)
                    self.cleaning_report['missing_values_filled'] += missing_count
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                    # Categorical: fill with mode or 'Unknown'
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
                    self.cleaning_report['missing_values_filled'] += missing_count
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Datetime: forward fill (time series assumption)
                    df[col].fillna(method='ffill', inplace=True)
                    self.cleaning_report['missing_values_filled'] += missing_count

            print(f"  [OK] Filled {self.cleaning_report['missing_values_filled']} missing values (auto strategy)")

        elif self.handle_missing == 'drop':
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            print(f"  [OK] Dropped {rows_dropped} rows with missing values")

        # If still have missing values, keep them (for 'fill' or partial 'auto')
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"  ℹ️  {remaining_missing} missing values remain (will be handled during analysis)")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        print("\n📊 Step 5: Handling outliers...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("  [OK] No numeric columns to check for outliers")
            return df

        outliers_detected = 0

        for col in numeric_cols:
            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_outliers = outlier_mask.sum()

            if col_outliers > 0:
                outliers_detected += col_outliers

                if self.handle_outliers == 'flag':
                    # Create flag column
                    df[f'{col}_is_outlier'] = outlier_mask
                elif self.handle_outliers == 'remove':
                    # Remove outlier rows
                    df = df[~outlier_mask]
                elif self.handle_outliers == 'cap':
                    # Cap at bounds (Winsorization)
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound

        if self.handle_outliers == 'flag':
            self.cleaning_report['outliers_flagged'] = outliers_detected
            print(f"  [OK] Flagged {outliers_detected} outliers (created _is_outlier columns)")
        elif self.handle_outliers == 'remove':
            self.cleaning_report['outliers_removed'] = outliers_detected
            print(f"  [OK] Removed {outliers_detected} outlier rows")
        elif self.handle_outliers == 'cap':
            self.cleaning_report['outliers_flagged'] = outliers_detected
            print(f"  [OK] Capped {outliers_detected} outliers at IQR bounds")
        else:
            print(f"  [OK] Detected {outliers_detected} outliers (no action taken)")

        return df

    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text/categorical columns."""
        print("\n🔤 Step 6: Cleaning text columns...")

        text_cols = df.select_dtypes(include=['object']).columns
        cleaned_count = 0

        for col in text_cols:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Remove leading/trailing special characters
            df[col] = df[col].str.replace(r'^[^\w\s]+|[^\w\s]+$', '', regex=True)

            # Standardize case for low-cardinality columns (likely categorical)
            if df[col].nunique() < 50:
                # Keep original case, just standardize whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            cleaned_count += 1

        print(f"  [OK] Cleaned {cleaned_count} text columns (whitespace, special chars)")

        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        print("\n📝 Step 7: Standardizing column names...")

        original_names = df.columns.tolist()
        new_names = []

        for col in df.columns:
            # Convert to lowercase
            new_name = col.lower()

            # Replace spaces and special chars with underscore
            new_name = re.sub(r'[^\w\s]', '_', new_name)
            new_name = re.sub(r'\s+', '_', new_name)

            # Remove multiple underscores
            new_name = re.sub(r'_+', '_', new_name)

            # Remove leading/trailing underscores
            new_name = new_name.strip('_')

            # Ensure uniqueness
            if new_name in new_names:
                suffix = 1
                while f"{new_name}_{suffix}" in new_names:
                    suffix += 1
                new_name = f"{new_name}_{suffix}"

            new_names.append(new_name)

        df.columns = new_names

        renamed_count = sum(1 for old, new in zip(original_names, new_names) if old != new)
        print(f"  [OK] Standardized {renamed_count} column names (lowercase, underscores)")

        return df

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and consistency checks."""
        print("\n[OK] Step 8: Final validation...")

        # Ensure we still have data
        if len(df) == 0:
            self.cleaning_report['warnings'].append("All rows were removed during cleaning")
            print("  [WARN]  WARNING: All rows were removed during cleaning")
            return df

        if len(df.columns) == 0:
            self.cleaning_report['warnings'].append("All columns were removed during cleaning")
            print("  [WARN]  WARNING: All columns were removed during cleaning")
            return df

        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  [OK] Final dataset: {len(df):,} rows × {len(df.columns)} columns ({memory_mb:.1f} MB)")

        return df

    def _print_cleaning_summary(self):
        """Print cleaning summary report."""
        print("\n" + "="*60)
        print("=== CLEANING SUMMARY ===")
        print("="*60)

        print(f"\n📊 Dataset size:")
        print(f"  Before: {self.cleaning_report['rows_before']:,} rows × {self.cleaning_report['columns_before']} columns")
        print(f"  After:  {self.cleaning_report['rows_after']:,} rows × {self.cleaning_report['columns_after']} columns")

        rows_removed = self.cleaning_report['rows_before'] - self.cleaning_report['rows_after']
        if rows_removed > 0:
            pct = rows_removed / self.cleaning_report['rows_before'] * 100
            print(f"  Removed: {rows_removed:,} rows ({pct:.1f}%)")

        cols_removed = len(self.cleaning_report['columns_dropped'])
        if cols_removed > 0:
            print(f"\n📉 Columns dropped ({cols_removed}):")
            for col in self.cleaning_report['columns_dropped'][:10]:
                print(f"  - {col}")
            if cols_removed > 10:
                print(f"  ... and {cols_removed - 10} more")

        if self.cleaning_report['data_types_converted']:
            print(f"\n🔄 Data types converted ({len(self.cleaning_report['data_types_converted'])}):")
            for col, conversion in list(self.cleaning_report['data_types_converted'].items())[:5]:
                print(f"  - {col}: {conversion}")
            if len(self.cleaning_report['data_types_converted']) > 5:
                remaining = len(self.cleaning_report['data_types_converted']) - 5
                print(f"  ... and {remaining} more")

        if self.cleaning_report['duplicates_removed'] > 0:
            print(f"\n🔄 Duplicates removed: {self.cleaning_report['duplicates_removed']:,}")

        if self.cleaning_report['missing_values_filled'] > 0:
            print(f"\n❓ Missing values filled: {self.cleaning_report['missing_values_filled']:,}")

        if self.cleaning_report['outliers_flagged'] > 0:
            print(f"\n📊 Outliers flagged: {self.cleaning_report['outliers_flagged']:,}")

        if self.cleaning_report['outliers_removed'] > 0:
            print(f"\n📊 Outliers removed: {self.cleaning_report['outliers_removed']:,}")

        if self.cleaning_report['warnings']:
            print(f"\n[WARN]  Warnings:")
            for warning in self.cleaning_report['warnings']:
                print(f"  - {warning}")

        print("\n" + "="*60)
        print("[OK] Data cleaning complete!")
        print("="*60 + "\n")

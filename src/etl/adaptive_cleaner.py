"""Context-aware data cleaning using relationship information."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class AdaptiveCleaner:
    """Cleans data with awareness of relationships and business context."""

    def __init__(self):
        pass

    def clean_dataset(
        self,
        df: pd.DataFrame,
        semantic_metadata: Dict,
        relationships: List[Dict],
        dataset_id: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean dataset with context awareness.

        Args:
            df: Raw DataFrame
            semantic_metadata: Semantic analysis result
            relationships: List of relationships this dataset participates in
            dataset_id: Current dataset ID

        Returns:
            (cleaned_df, cleaning_report)
        """
        table_name = semantic_metadata.get('table_name', 'unknown')
        print(f"\n[CLEANING] {table_name}")

        cleaned_df = df.copy()
        report = {
            'original_rows': len(df),
            'original_columns': len(df.columns),
            'table_name': table_name,
            'steps': []
        }

        # 1. Type conversion (guided by semantic metadata)
        cleaned_df, type_report = self._convert_types(cleaned_df, semantic_metadata)
        report['steps'].append(type_report)

        # 2. Handle missing values (preserve FK relationships)
        cleaned_df, missing_report = self._handle_missing_values(
            cleaned_df, semantic_metadata, relationships, dataset_id
        )
        report['steps'].append(missing_report)

        # 3. Remove duplicates
        cleaned_df, dup_report = self._remove_duplicates(cleaned_df, semantic_metadata)
        report['steps'].append(dup_report)

        # 4. Standardize FK columns (ensure type compatibility)
        cleaned_df, fk_report = self._standardize_foreign_keys(
            cleaned_df, relationships, dataset_id
        )
        report['steps'].append(fk_report)

        # 5. Handle outliers (flag, don't remove)
        cleaned_df, outlier_report = self._handle_outliers(
            cleaned_df, semantic_metadata
        )
        report['steps'].append(outlier_report)

        # 6. Standardize formats (dates, currencies, text)
        cleaned_df, format_report = self._standardize_formats(
            cleaned_df, semantic_metadata
        )
        report['steps'].append(format_report)

        # 7. Data validation
        cleaned_df, validation_report = self._validate_data(
            cleaned_df, semantic_metadata
        )
        report['steps'].append(validation_report)

        report['final_rows'] = len(cleaned_df)
        report['final_columns'] = len(cleaned_df.columns)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        report['columns_added'] = len(cleaned_df.columns) - report['original_columns']

        print(f"[OK] Cleaned: {report['original_rows']:,} -> {report['final_rows']:,} rows")

        return cleaned_df, report

    def _convert_types(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Convert data types based on semantic understanding."""
        conversions = []
        errors = []

        column_semantics = metadata.get('column_semantics', {})

        for col in df.columns:
            if col not in column_semantics:
                continue

            sem_type = column_semantics[col].get('semantic_type')
            current_dtype = df[col].dtype

            try:
                # Date conversion
                if sem_type == 'date':
                    if current_dtype != 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        conversions.append(f"{col}: {current_dtype} -> datetime")

                # Numeric conversion for measures
                elif sem_type == 'measure':
                    if current_dtype == 'object':
                        # Remove currency symbols, commas, percentages
                        df[col] = (df[col].astype(str)
                                  .str.replace('$', '', regex=False)
                                  .str.replace(',', '', regex=False)
                                  .str.replace('%', '', regex=False)
                                  .str.strip())
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        conversions.append(f"{col}: object -> numeric")

                # Key columns should be strings for consistency
                elif sem_type == 'key':
                    if current_dtype != 'object':
                        df[col] = df[col].astype(str)
                        conversions.append(f"{col}: {current_dtype} -> string")

            except Exception as e:
                errors.append(f"Failed to convert {col}: {str(e)}")

        return df, {
            'step': 'type_conversion',
            'conversions': conversions,
            'errors': errors,
            'columns_converted': len(conversions)
        }

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        relationships: List[Dict],
        dataset_id: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with FK awareness."""

        # Identify FK columns
        fk_columns = set()
        for rel in relationships:
            if rel['from_dataset_id'] == dataset_id:
                fk_columns.add(rel['from_column'])
            elif rel['to_dataset_id'] == dataset_id:
                fk_columns.add(rel['to_column'])

        rows_before = len(df)
        columns_filled = []
        columns_dropped = []

        # Strategy: Drop rows with missing FK values (to preserve referential integrity)
        if fk_columns:
            existing_fk_cols = [col for col in fk_columns if col in df.columns]
            if existing_fk_cols:
                df = df.dropna(subset=existing_fk_cols)

        # For other columns, use context-aware strategies
        column_semantics = metadata.get('column_semantics', {})

        for col in df.columns:
            if col in fk_columns:
                continue

            if df[col].isnull().any():
                null_pct = df[col].isnull().mean() * 100
                sem_type = column_semantics.get(col, {}).get('semantic_type')

                # If more than 90% null, consider dropping the column
                if null_pct > 90:
                    df = df.drop(columns=[col])
                    columns_dropped.append(col)
                    continue

                # Fill based on semantic type
                if sem_type == 'measure':
                    # Fill with 0 for measures (conservative)
                    df[col].fillna(0, inplace=True)
                    columns_filled.append(f"{col} (filled with 0)")
                elif sem_type == 'dimension':
                    # Fill with 'Unknown'
                    df[col].fillna('Unknown', inplace=True)
                    columns_filled.append(f"{col} (filled with 'Unknown')")
                elif sem_type == 'date':
                    # Forward fill for dates
                    df[col].fillna(method='ffill', inplace=True)
                    columns_filled.append(f"{col} (forward filled)")

        rows_after = len(df)

        return df, {
            'step': 'missing_values',
            'rows_removed': rows_before - rows_after,
            'fk_columns_protected': list(fk_columns),
            'columns_filled': columns_filled,
            'columns_dropped': columns_dropped
        }

    def _remove_duplicates(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows."""
        rows_before = len(df)

        # Identify PK columns
        column_semantics = metadata.get('column_semantics', {})
        pk_columns = [
            col for col, meta in column_semantics.items()
            if meta.get('is_primary_key', False) and col in df.columns
        ]

        # Also consider columns marked as 'key'
        key_columns = [
            col for col, meta in column_semantics.items()
            if meta.get('semantic_type') == 'key' and col in df.columns
        ]

        # Combine PK and key columns
        dedup_columns = list(set(pk_columns + key_columns))

        if dedup_columns:
            # Remove duplicates based on key columns
            df = df.drop_duplicates(subset=dedup_columns, keep='first')
        else:
            # Remove exact duplicates
            df = df.drop_duplicates(keep='first')

        rows_after = len(df)

        return df, {
            'step': 'remove_duplicates',
            'rows_removed': rows_before - rows_after,
            'dedup_columns': dedup_columns if dedup_columns else 'all columns'
        }

    def _standardize_foreign_keys(
        self,
        df: pd.DataFrame,
        relationships: List[Dict],
        dataset_id: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Ensure FK columns have compatible types."""
        standardized = []

        for rel in relationships:
            col = None
            if rel['from_dataset_id'] == dataset_id:
                col = rel['from_column']
            elif rel['to_dataset_id'] == dataset_id:
                col = rel['to_column']

            if col and col in df.columns:
                # Ensure consistent type (convert to string for safety)
                original_dtype = df[col].dtype
                df[col] = df[col].astype(str)

                # Remove any trailing spaces
                df[col] = df[col].str.strip()

                standardized.append(f"{col} ({original_dtype} -> string)")

        return df, {
            'step': 'standardize_foreign_keys',
            'columns_standardized': standardized
        }

    def _handle_outliers(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers (flag, don't remove)."""
        column_semantics = metadata.get('column_semantics', {})
        flagged_columns = []

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                sem_type = column_semantics.get(col, {}).get('semantic_type')

                if sem_type == 'measure':
                    # Use IQR method to detect outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    # 3x IQR for extreme outliers
                    lower = Q1 - 3 * IQR
                    upper = Q3 + 3 * IQR

                    outliers = (df[col] < lower) | (df[col] > upper)

                    if outliers.any():
                        outlier_col = f'{col}_is_outlier'
                        df[outlier_col] = outliers
                        flagged_columns.append(col)

        return df, {
            'step': 'handle_outliers',
            'flagged_columns': flagged_columns,
            'strategy': 'flag_not_remove',
            'outlier_columns_added': len(flagged_columns)
        }

    def _standardize_formats(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize date, currency, and text formats."""
        standardized = []

        column_semantics = metadata.get('column_semantics', {})

        for col in df.columns:
            sem_type = column_semantics.get(col, {}).get('semantic_type')

            try:
                # Text standardization
                if sem_type in ['text', 'dimension'] or df[col].dtype == 'object':
                    # Strip whitespace, standardize case for dimensions
                    df[col] = df[col].astype(str).str.strip()

                    # For dimensions, also standardize case
                    if sem_type == 'dimension':
                        # Check if mostly uppercase or lowercase
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            upper_count = sum(s.isupper() for s in sample)
                            lower_count = sum(s.islower() for s in sample)

                            if upper_count > lower_count * 2:
                                df[col] = df[col].str.upper()
                                standardized.append(f"{col} (uppercase)")
                            elif lower_count > upper_count * 2:
                                df[col] = df[col].str.lower()
                                standardized.append(f"{col} (lowercase)")

                # Date formatting
                elif sem_type == 'date' and df[col].dtype == 'datetime64[ns]':
                    # Ensure consistent timezone (if needed)
                    standardized.append(f"{col} (date)")

            except Exception as e:
                pass  # Skip if standardization fails

        return df, {
            'step': 'standardize_formats',
            'columns_standardized': standardized
        }

    def _validate_data(self, df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Final data validation and quality checks."""
        issues = []
        warnings = []

        # Check for remaining nulls in critical columns
        column_semantics = metadata.get('column_semantics', {})
        for col, meta in column_semantics.items():
            if col not in df.columns:
                continue

            if meta.get('is_primary_key') or meta.get('is_foreign_key'):
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"{col} has {null_count} null values (key column)")

        # Check for negative values in measure columns that shouldn't be negative
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                sem_type = column_semantics.get(col, {}).get('semantic_type')
                business_meaning = column_semantics.get(col, {}).get('business_meaning', '').lower()

                if sem_type == 'measure':
                    # Check for negative values in columns that shouldn't have them
                    if any(word in business_meaning for word in ['price', 'cost', 'revenue', 'quantity', 'count']):
                        negative_count = (df[col] < 0).sum()
                        if negative_count > 0:
                            warnings.append(f"{col} has {negative_count} negative values")

        # Add validation flag column
        df['_data_quality_flag'] = 'clean'
        if issues:
            df['_data_quality_flag'] = 'has_issues'

        return df, {
            'step': 'data_validation',
            'issues': issues,
            'warnings': warnings,
            'validation_passed': len(issues) == 0
        }
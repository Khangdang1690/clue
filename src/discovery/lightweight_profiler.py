"""Lightweight table profiler that leverages existing ETL context for multi-table analysis."""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


class LightweightTableProfiler:
    """
    Statistical profiler that uses existing ETL context instead of making new LLM calls.

    Designed for multi-table discovery to avoid burning API quota on redundant analysis.
    """

    def __init__(self):
        """Initialize the lightweight profiler."""
        pass

    def profile_for_cross_table(
        self,
        df: pd.DataFrame,
        etl_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive profile using ETL context + statistical analysis.

        Args:
            df: The DataFrame to profile
            etl_context: Context from ETL semantic analysis (already has LLM insights)

        Returns:
            Profile optimized for cross-table analysis
        """
        profile = {
            # 1. Reuse ALL the expensive ETL context (no new LLM calls!)
            'semantic_context': {
                'domain': etl_context.get('domain', 'Unknown'),
                'description': etl_context.get('description', ''),
                'entities': etl_context.get('entities', []),
                'dataset_type': etl_context.get('dataset_type'),
                'time_period': etl_context.get('time_period'),
                'typical_use_cases': etl_context.get('typical_use_cases', []),
                'business_context': etl_context.get('business_context', {}),
                'department': etl_context.get('department')
            },

            # 2. Statistical fingerprint for validation
            'statistics': self._compute_statistics(df),

            # 3. Join readiness (what columns can connect to other tables)
            'join_readiness': self._analyze_join_readiness(df, etl_context),

            # 4. Aggregation potential (what metrics can be rolled up)
            'aggregation_potential': self._analyze_aggregation_potential(df, etl_context),

            # 5. Temporal alignment (for time-based cross-table analysis)
            'temporal_info': self._analyze_temporal_characteristics(df, etl_context),

            # 6. Data quality summary
            'quality_metrics': self._compute_quality_metrics(df)
        }

        return profile

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics without LLM calls."""
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'high_cardinality_columns': []
        }

        for col in df.columns:
            col_stats = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': df[col].isna().sum(),
                'null_pct': df[col].isna().sum() / len(df) if len(df) > 0 else 0,
                'unique_count': df[col].nunique(),
                'cardinality': df[col].nunique() / len(df) if len(df) > 0 else 0
            }

            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                # Skip boolean columns for numeric statistics
                col_stats.update({
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'has_outliers': self._detect_outliers(df[col])
                })
                stats['numeric_columns'].append(col_stats)
            elif pd.api.types.is_bool_dtype(df[col]):
                # Handle boolean columns separately
                col_stats.update({
                    'true_count': df[col].sum(),
                    'false_count': (~df[col]).sum(),
                    'true_pct': df[col].mean() if len(df) > 0 else 0
                })
                stats['categorical_columns'].append(col_stats)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats.update({
                    'min_date': str(df[col].min()) if not df[col].isna().all() else None,
                    'max_date': str(df[col].max()) if not df[col].isna().all() else None,
                    'date_range_days': (df[col].max() - df[col].min()).days if not df[col].isna().all() else None
                })
                stats['datetime_columns'].append(col_stats)

            else:
                # Categorical column
                col_stats.update({
                    'top_values': df[col].value_counts().head(5).to_dict() if len(df) > 0 else {}
                })
                stats['categorical_columns'].append(col_stats)

            # Track high cardinality columns (potential join keys)
            if col_stats['cardinality'] > 0.5 and col_stats['unique_count'] > 10:
                stats['high_cardinality_columns'].append(col)

        return stats

    def _analyze_join_readiness(self, df: pd.DataFrame, etl_context: Dict) -> Dict[str, Any]:
        """Analyze which columns are ready for joining with other tables."""
        column_semantics = etl_context.get('column_semantics', {})

        join_info = {
            'primary_keys': [],
            'foreign_keys': [],
            'join_candidates': [],
            'common_dimensions': []
        }

        for col in df.columns:
            col_meta = column_semantics.get(col, {})

            # Use ETL's semantic understanding
            if col_meta.get('is_primary_key'):
                join_info['primary_keys'].append(col)

            if col_meta.get('is_foreign_key'):
                join_info['foreign_keys'].append({
                    'column': col,
                    'potential_relationships': col_meta.get('potential_relationships', [])
                })

            # Identify join candidates by name patterns
            if col.lower().endswith(('_id', '_code', '_key', 'id')):
                if col not in join_info['primary_keys'] and col not in [fk['column'] for fk in join_info['foreign_keys']]:
                    join_info['join_candidates'].append(col)

            # Common dimensions for cross-table analysis
            if col.lower() in ['date', 'region', 'country', 'state', 'city', 'category',
                               'type', 'status', 'department', 'product_category', 'customer_segment']:
                join_info['common_dimensions'].append(col)

        # Add cardinality info for join planning
        for col in join_info['primary_keys'] + join_info['join_candidates']:
            if col in df.columns:
                join_info[f'{col}_cardinality'] = df[col].nunique()

        return join_info

    def _analyze_aggregation_potential(self, df: pd.DataFrame, etl_context: Dict) -> Dict[str, Any]:
        """Identify metrics that can be aggregated across tables."""
        column_semantics = etl_context.get('column_semantics', {})

        aggregation_info = {
            'measures': [],  # Columns that can be summed/averaged
            'dimensions': [],  # Columns to group by
            'time_dimensions': [],  # Temporal grouping options
            'suggested_aggregations': []
        }

        for col in df.columns:
            col_meta = column_semantics.get(col, {})
            semantic_type = col_meta.get('semantic_type', '')

            if semantic_type == 'measure' or (
                pd.api.types.is_numeric_dtype(df[col]) and
                not col.lower().endswith(('_id', 'id', '_key', '_code'))
            ):
                aggregation_info['measures'].append({
                    'column': col,
                    'business_meaning': col_meta.get('business_meaning', ''),
                    'aggregation_functions': ['sum', 'mean', 'min', 'max', 'count']
                })

            elif semantic_type == 'dimension' or (
                df[col].nunique() < len(df) * 0.5 and  # Low cardinality
                df[col].nunique() > 1 and df[col].nunique() <= 100  # Reasonable for grouping
            ):
                aggregation_info['dimensions'].append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'business_meaning': col_meta.get('business_meaning', '')
                })

            elif semantic_type == 'date' or pd.api.types.is_datetime64_any_dtype(df[col]):
                aggregation_info['time_dimensions'].append({
                    'column': col,
                    'granularities': ['day', 'week', 'month', 'quarter', 'year']
                })

        # Suggest cross-table aggregations based on domain
        domain = etl_context.get('domain', 'Unknown')
        if domain == 'Sales' and aggregation_info['measures']:
            aggregation_info['suggested_aggregations'].append(
                "Revenue by product category across regions"
            )
        elif domain == 'Marketing' and aggregation_info['measures']:
            aggregation_info['suggested_aggregations'].append(
                "Campaign performance metrics by channel"
            )

        return aggregation_info

    def _analyze_temporal_characteristics(self, df: pd.DataFrame, etl_context: Dict) -> Dict[str, Any]:
        """Analyze temporal aspects for time-based cross-table analysis."""
        temporal_info = {
            'has_time_dimension': False,
            'time_columns': [],
            'time_range': None,
            'granularity': None,
            'is_time_series': False
        }

        # Check for datetime columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Also check columns that might be dates stored as strings
        column_semantics = etl_context.get('column_semantics', {})
        for col, meta in column_semantics.items():
            if meta.get('semantic_type') == 'date' and col not in date_cols:
                # Try to parse as date
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass

        if date_cols:
            temporal_info['has_time_dimension'] = True
            temporal_info['time_columns'] = date_cols

            # Analyze primary time column
            primary_date = date_cols[0]
            if not df[primary_date].isna().all():
                temporal_info['time_range'] = {
                    'start': str(df[primary_date].min()),
                    'end': str(df[primary_date].max()),
                    'days_covered': (df[primary_date].max() - df[primary_date].min()).days
                }

                # Detect granularity
                if len(df) > 1:
                    date_diffs = df[primary_date].sort_values().diff().dropna()
                    if not date_diffs.empty:
                        mode_diff = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else date_diffs.median()

                        if mode_diff.days == 1:
                            temporal_info['granularity'] = 'daily'
                        elif mode_diff.days == 7:
                            temporal_info['granularity'] = 'weekly'
                        elif 28 <= mode_diff.days <= 31:
                            temporal_info['granularity'] = 'monthly'
                        elif 90 <= mode_diff.days <= 92:
                            temporal_info['granularity'] = 'quarterly'
                        elif 365 <= mode_diff.days <= 366:
                            temporal_info['granularity'] = 'yearly'

                # Check if it's a time series (regular intervals)
                if temporal_info['granularity'] and len(df) >= 12:
                    temporal_info['is_time_series'] = True

        # Add time period from ETL context
        temporal_info['etl_time_period'] = etl_context.get('time_period')

        return temporal_info

    def _compute_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute data quality metrics for cross-table validation."""
        total_cells = len(df) * len(df.columns)
        total_nulls = df.isna().sum().sum()

        quality = {
            'completeness': 1 - (total_nulls / total_cells) if total_cells > 0 else 0,
            'row_duplicates': df.duplicated().sum(),
            'duplicate_pct': df.duplicated().sum() / len(df) if len(df) > 0 else 0,
            'columns_with_nulls': df.columns[df.isna().any()].tolist(),
            'columns_all_null': df.columns[df.isna().all()].tolist(),
            'columns_no_variance': []  # Columns with single value
        }

        # Find columns with no variance (single value)
        for col in df.columns:
            if df[col].nunique() == 1:
                quality['columns_no_variance'].append(col)

        # Statistical confidence based on sample size
        sample_size = len(df)
        if sample_size < 30:
            quality['statistical_confidence'] = 'very_low'
            quality['confidence_warning'] = 'Sample size too small for reliable statistics'
        elif sample_size < 100:
            quality['statistical_confidence'] = 'low'
            quality['confidence_warning'] = 'Limited sample size may affect accuracy'
        elif sample_size < 1000:
            quality['statistical_confidence'] = 'medium'
        else:
            quality['statistical_confidence'] = 'high'

        return quality

    def _detect_outliers(self, series: pd.Series) -> bool:
        """Simple outlier detection using IQR method."""
        if series.isna().all() or len(series.dropna()) < 4:
            return False

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return outliers > 0

    def generate_cross_table_context(
        self,
        profiles: Dict[str, Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Generate context for cross-table LLM analysis.

        Args:
            profiles: Dict of dataset_id -> profile
            relationships: List of detected relationships

        Returns:
            Context string optimized for cross-table analysis
        """
        context_parts = []

        # 1. Overview
        context_parts.append("CROSS-TABLE ANALYSIS CONTEXT\n" + "="*50)
        context_parts.append(f"Datasets: {len(profiles)}")
        context_parts.append(f"Relationships: {len(relationships)}")

        # 2. Each dataset's key info (from ETL + stats)
        context_parts.append("\n\nDATASET SUMMARIES:")
        for dataset_id, profile in profiles.items():
            semantic = profile['semantic_context']
            stats = profile['statistics']

            context_parts.append(f"\n{dataset_id}:")
            context_parts.append(f"  - Description: {semantic['description']}")
            context_parts.append(f"  - Domain: {semantic['domain']}")
            context_parts.append(f"  - Type: {semantic.get('dataset_type', 'Unknown')}")
            context_parts.append(f"  - Rows: {stats['row_count']:,}")
            context_parts.append(f"  - Entities: {', '.join(semantic['entities'])}")

            # Key metrics for cross-table analysis
            if profile['aggregation_potential']['measures']:
                measures = [m['column'] for m in profile['aggregation_potential']['measures'][:3]]
                context_parts.append(f"  - Key Metrics: {', '.join(measures)}")

            if profile['temporal_info']['time_range']:
                context_parts.append(f"  - Time Period: {profile['temporal_info']['time_range']['start']} to {profile['temporal_info']['time_range']['end']}")

        # 3. Relationships
        if relationships:
            context_parts.append("\n\nRELATIONSHIPS:")
            for rel in relationships:
                context_parts.append(f"  - {rel['from_column']} → {rel['to_column']} (confidence: {rel['confidence']:.2f})")

        # 4. Cross-table opportunities
        context_parts.append("\n\nCROSS-TABLE ANALYSIS OPPORTUNITIES:")

        # Find common dimensions across tables
        all_dimensions = set()
        for profile in profiles.values():
            all_dimensions.update(profile['join_readiness']['common_dimensions'])

        if all_dimensions:
            context_parts.append(f"  - Common dimensions: {', '.join(list(all_dimensions)[:5])}")

        # Find time alignment
        time_aligned = all(
            profile['temporal_info']['has_time_dimension']
            for profile in profiles.values()
        )
        if time_aligned:
            context_parts.append("  - All datasets have temporal dimensions (time-based analysis possible)")

        # Suggest analysis based on domains
        domains = set(p['semantic_context']['domain'] for p in profiles.values())
        if 'Sales' in domains and 'Marketing' in domains:
            context_parts.append("  - Sales + Marketing: Analyze campaign ROI and customer acquisition")
        elif 'Sales' in domains and 'Product' in domains:
            context_parts.append("  - Sales + Product: Product performance and inventory optimization")

        return "\n".join(context_parts)
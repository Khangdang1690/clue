"""Multi-table discovery workflow with autonomous cross-table analysis."""

from typing import List, Dict, Optional, Any
import pandas as pd
from langgraph.graph import StateGraph, END
from src.models.discovery_models import DiscoveryResult
from src.models.etl_models import MultiTableDiscoveryState
from src.graph.discovery_workflow import DiscoveryWorkflow
from src.discovery.lightweight_profiler import LightweightTableProfiler
from src.database.connection import DatabaseManager
from src.database.repository import (
    DatasetRepository, RelationshipRepository, AnalysisSessionRepository,
    SimilarityRepository
)
from src.discovery.autonomous_explorer import AutonomousExplorer
from src.utils.embedding_service import get_embedding_service
# Advanced Analytics modules
from src.analytics import (
    TimeSeriesForecaster, AnomalyDetector, CausalAnalyzer,
    VarianceDecomposer, ImpactEstimator, BusinessInsightSynthesizer
)
from src.analytics.llm_test_selector import StatisticalTestSelector
import os
import time
from datetime import datetime


class MultiTableDiscovery:
    """
    Two-phase discovery for multiple related tables:

    Modes:
    - 'full': Full single-table discovery + cross-table analysis (high API usage)
    - 'cross_table_focus': Lightweight profiling + deep cross-table analysis (optimized for API quota)
    """

    def __init__(self, mode: str = 'cross_table_focus'):
        """
        Initialize multi-table discovery.

        Args:
            mode: Discovery mode - 'full' or 'cross_table_focus'
                  'cross_table_focus' uses lightweight profiling to save API quota
        """
        self.mode = mode
        print(f"[INIT] Multi-table discovery mode: {mode}")

        if mode == 'full':
            # Traditional approach with full single-table discovery
            self.single_table_workflow = DiscoveryWorkflow(
                max_iterations=15,
                max_insights=3,
                generate_context=True,
                skip_visualizations=True
            )
        else:
            # Lightweight approach that reuses ETL context
            self.lightweight_profiler = LightweightTableProfiler()
            self.single_table_workflow = None  # Not used in lightweight mode

        # Cross-table explorer gets more iterations in lightweight mode
        if mode == 'cross_table_focus':
            # More API budget for cross-table since we saved on single-table
            self.cross_table_explorer = AutonomousExplorer(
                max_iterations=15,  # Reduced for free tier safety
                max_insights=3  # Reduced for free tier safety
            )
        else:
            self.cross_table_explorer = AutonomousExplorer(
                max_iterations=25,
                max_insights=3
            )

        self.embedding_service = get_embedding_service()
        self.synthesizer = BusinessInsightSynthesizer()

        # Build workflow based on mode
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build multi-table discovery workflow based on mode."""
        workflow = StateGraph(MultiTableDiscoveryState)

        # Common nodes for both modes
        workflow.add_node("load_datasets", self._load_datasets_node)
        workflow.add_node("suggest_related", self._suggest_related_node)
        workflow.add_node("prepare_cross_table", self._prepare_cross_table_node)
        workflow.add_node("cross_table_analysis", self._cross_table_analysis_node)
        workflow.add_node("cross_table_advanced_analytics", self._cross_table_advanced_analytics_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Mode-specific nodes
        if self.mode == 'full':
            # Full discovery with expensive single-table analysis
            workflow.add_node("single_table_analysis", self._single_table_analysis_node)
            workflow.add_node("single_table_advanced_analytics", self._single_table_advanced_analytics_node)

            # Full workflow
            workflow.set_entry_point("load_datasets")
            workflow.add_edge("load_datasets", "suggest_related")
            workflow.add_edge("suggest_related", "single_table_analysis")
            workflow.add_edge("single_table_analysis", "single_table_advanced_analytics")
            workflow.add_edge("single_table_advanced_analytics", "prepare_cross_table")
        else:
            # Lightweight mode with statistical profiling only
            workflow.add_node("lightweight_profiling", self._lightweight_profiling_node)
            # Add advanced analytics even in lightweight mode for causal detection
            workflow.add_node("single_table_advanced_analytics", self._single_table_advanced_analytics_node)

            # Lightweight workflow
            workflow.set_entry_point("load_datasets")
            workflow.add_edge("load_datasets", "suggest_related")
            workflow.add_edge("suggest_related", "lightweight_profiling")
            workflow.add_edge("lightweight_profiling", "single_table_advanced_analytics")
            workflow.add_edge("single_table_advanced_analytics", "prepare_cross_table")

        # Common flow for both modes
        workflow.add_edge("prepare_cross_table", "cross_table_analysis")
        workflow.add_edge("cross_table_analysis", "cross_table_advanced_analytics")
        workflow.add_edge("cross_table_advanced_analytics", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def run_discovery(
        self,
        company_id: str,
        dataset_ids: List[str],
        analysis_name: Optional[str] = None
    ) -> Dict:
        """
        Run multi-table discovery.

        Args:
            company_id: Company ID
            dataset_ids: List of dataset IDs to analyze
            analysis_name: Optional name for the analysis session

        Returns:
            Discovery results
        """
        if not analysis_name:
            analysis_name = f"Multi-table Analysis {datetime.now().strftime('%Y-%m-%d')}"

        print("\n" + "="*80)
        print("[MULTI-TABLE DISCOVERY]")
        print("="*80)
        print(f"Company ID: {company_id}")
        print(f"Datasets: {len(dataset_ids)}")
        print(f"Analysis: {analysis_name}")

        initial_state: MultiTableDiscoveryState = {
            'company_id': company_id,
            'dataset_ids': dataset_ids,
            'analysis_name': analysis_name,
            'datasets': {},
            'metadata': {},
            'relationships': [],
            'single_table_results': {},
            'single_table_advanced_insights': {},  # Advanced analytics per table
            'joined_dataframe': None,
            'cross_table_insights': [],
            'cross_table_advanced_insights': [],  # Advanced analytics across tables
            'suggested_datasets': [],
            'unified_report_path': None,
            'analysis_session_id': None,
            'status': 'running',
            'error_message': '',
            'current_phase': 'initialization'
        }

        # Run workflow
        try:
            final_state = self.graph.invoke(initial_state)

            if final_state['status'] == 'completed':
                print("\n" + "="*80)
                print("[SUCCESS] MULTI-TABLE DISCOVERY COMPLETED")
                print("="*80)
                print(f"Single-table insights: {sum(len(r.answered_questions) for r in final_state['single_table_results'].values())}")
                print(f"Cross-table insights: {len(final_state.get('cross_table_insights', []))}")
                print(f"Report: {final_state.get('unified_report_path', 'Not generated')}")

                return {
                    'single_table_results': final_state['single_table_results'],
                    'cross_table_insights': final_state.get('cross_table_insights', []),
                    'unified_report_path': final_state.get('unified_report_path'),
                    'analysis_session_id': final_state.get('analysis_session_id')
                }
            else:
                raise Exception(final_state.get('error_message', 'Discovery failed'))

        except Exception as e:
            print(f"\n[ERROR] Multi-table discovery failed: {e}")
            raise

    # Node implementations

    def _load_datasets_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Load datasets from database."""
        state['current_phase'] = 'loading_data'
        print("\n[PHASE 1] LOADING DATASETS")

        try:
            with DatabaseManager.get_session() as session:
                for dataset_id in state['dataset_ids']:
                    # Load DataFrame
                    df = DatasetRepository.load_dataframe(session, dataset_id)
                    dataset = DatasetRepository.get_dataset_by_id(session, dataset_id)

                    if dataset:
                        state['datasets'][dataset_id] = df
                        # Load full unified context from ETL
                        state['metadata'][dataset_id] = {
                            'table_name': dataset.table_name,
                            'domain': dataset.domain,
                            'description': dataset.description,
                            'entities': dataset.entities,
                            'row_count': dataset.row_count,
                            'column_count': dataset.column_count,
                            # New unified context fields from ETL
                            'dataset_type': dataset.dataset_type,
                            'time_period': dataset.time_period,
                            'typical_use_cases': dataset.typical_use_cases,
                            'business_context': dataset.business_context,
                            'department': dataset.department
                        }

                        print(f"  Loaded {dataset.table_name}: {dataset.row_count:,} rows")

                # Load relationships and extract data before session closes
                print(f"  Looking for relationships for datasets: {state['dataset_ids'][:2]}...")  # Show first 2
                relationships = RelationshipRepository.get_relationships_for_datasets(
                    session,
                    state['dataset_ids'],
                    min_confidence=0.8
                )
                print(f"  Raw relationships from DB: {len(relationships)}")

                # Convert relationship objects to dicts to avoid detached instance errors
                state['relationships'] = []
                for rel in relationships:
                    rel_dict = {
                        'from_dataset_id': rel.from_dataset_id,
                        'to_dataset_id': rel.to_dataset_id,
                        'from_column': rel.from_column,
                        'to_column': rel.to_column,
                        'relationship_type': rel.relationship_type,
                        'confidence': rel.confidence,
                        'join_strategy': rel.join_strategy if hasattr(rel, 'join_strategy') else 'inner'
                    }
                    state['relationships'].append(rel_dict)
                    # Debug: show first relationship
                    if len(state['relationships']) == 1:
                        print(f"    Example: {rel.from_column} -> {rel.to_column} (conf: {rel.confidence:.2f})")

                print(f"  Found {len(state['relationships'])} relationships")

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Failed to load datasets: {e}"

        # Add delay before next phase to avoid rate limits (only in full mode)
        if self.mode == 'full':
            print(f"\nWaiting 60s before similarity search (rate limit protection)...")
            time.sleep(60)
        else:
            print(f"\nWaiting 15s - lightweight mode (reduced API calls)")
            time.sleep(15)

        return state

    def _suggest_related_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Suggest related datasets using embedding similarity."""
        state['current_phase'] = 'suggesting_related'
        print("\n[SIMILARITY] SUGGESTING RELATED DATASETS")

        try:
            with DatabaseManager.get_session() as session:
                suggested_ids = set()

                for dataset_id in state['dataset_ids']:
                    dataset = DatasetRepository.get_dataset_by_id(session, dataset_id)

                    if dataset and dataset.description_embedding is not None:
                        # Convert numpy array to list if necessary
                        import numpy as np
                        embedding = dataset.description_embedding
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()

                        # Find similar datasets
                        similar = SimilarityRepository.find_similar_datasets(
                            session,
                            embedding,
                            limit=3,
                            threshold=0.75
                        )

                        for sim_dataset in similar:
                            if sim_dataset['id'] not in state['dataset_ids']:
                                suggested_ids.add(sim_dataset['id'])
                                print(f"  Suggested: {sim_dataset['table_name']} "
                                      f"(similarity: {sim_dataset['similarity']:.2f})")

                state['suggested_datasets'] = list(suggested_ids)

        except Exception as e:
            print(f"[WARN] Failed to suggest related datasets: {e}")
            state['suggested_datasets'] = []

        # Add delay before next phase to avoid rate limits (reduced in lightweight mode)
        if self.mode == 'full':
            print(f"\nWaiting 60s before single-table analysis (rate limit protection)...")
            time.sleep(60)
        else:
            print(f"\nWaiting 20s - lightweight mode")
            time.sleep(20)

        return state

    def _lightweight_profiling_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """
        Lightweight profiling node that uses ETL context instead of running full discovery.
        This dramatically reduces API usage for multi-table analysis.
        """
        state['current_phase'] = 'lightweight_profiling'
        print("\n[PHASE 2] LIGHTWEIGHT PROFILING (API-Optimized)")
        print("  Using existing ETL context - no additional LLM calls needed!")

        try:
            # Initialize storage for profiles
            state['table_profiles'] = {}

            dataset_count = len(state['datasets'])
            for idx, (dataset_id, df) in enumerate(state['datasets'].items(), 1):
                etl_metadata = state['metadata'][dataset_id]
                print(f"\n  Profiling {etl_metadata['table_name']} ({idx}/{dataset_count})...")

                # Use lightweight profiler with ETL context
                # Ensure we have the minimum required context
                if not etl_metadata.get('column_semantics'):
                    etl_metadata['column_semantics'] = {}

                profile = self.lightweight_profiler.profile_for_cross_table(df, etl_metadata)
                state['table_profiles'][dataset_id] = profile

                # Print summary
                stats = profile['statistics']
                print(f"    ✓ {stats['row_count']:,} rows × {stats['column_count']} columns")
                print(f"    ✓ Domain: {profile['semantic_context']['domain']}")
                print(f"    ✓ Type: {profile['semantic_context']['dataset_type']}")

                # Show key metrics found
                measures = profile['aggregation_potential']['measures']
                if measures:
                    measure_names = [m['column'] for m in measures[:3]]
                    print(f"    ✓ Key metrics: {', '.join(measure_names)}")

                # Show join readiness
                join_info = profile['join_readiness']
                if join_info['primary_keys']:
                    print(f"    ✓ Primary keys: {', '.join(join_info['primary_keys'])}")
                if join_info['foreign_keys']:
                    print(f"    ✓ Foreign keys: {len(join_info['foreign_keys'])}")

                # Data quality warning if needed
                quality = profile['quality_metrics']
                if quality['statistical_confidence'] in ['very_low', 'low']:
                    print(f"    ⚠️ {quality['confidence_warning']}")

            # Generate cross-table context for later use
            if state['table_profiles']:
                cross_table_context = self.lightweight_profiler.generate_cross_table_context(
                    state['table_profiles'],
                    state['relationships']
                )
                state['cross_table_context'] = cross_table_context
                print(f"\n  Generated cross-table analysis context")

            print(f"\n[SUCCESS] Profiled {len(state['table_profiles'])} datasets using ETL context")
            print(f"  API calls saved: ~{dataset_count * 15} (compared to full discovery)")

            # Store minimal results for compatibility with report generation
            # Create mock DiscoveryResult objects with basic info
            from src.models.discovery_models import DiscoveryResult, AnsweredQuestion, DataProfile

            state['single_table_results'] = {}
            for dataset_id, profile in state['table_profiles'].items():
                stats = profile['statistics']
                quality = profile['quality_metrics']

                # Create DataProfile from lightweight profiling stats
                data_profile = DataProfile(
                    num_rows=stats['row_count'],
                    num_columns=stats['column_count'],
                    memory_usage_mb=stats.get('memory_usage_mb', 0),
                    numeric_columns=[c['name'] for c in stats.get('numeric_columns', [])],
                    categorical_columns=[c['name'] for c in stats.get('categorical_columns', [])],
                    datetime_columns=[c['name'] for c in stats.get('datetime_columns', [])],
                    text_columns=[],  # Not tracked in lightweight mode
                    overall_missing_rate=1 - quality['completeness'],
                    columns_with_missing=quality['columns_with_nulls'],
                    has_temporal_data=profile['temporal_info']['has_time_dimension'],
                    temporal_columns=profile['temporal_info']['time_columns'],
                    high_cardinality_columns=stats.get('high_cardinality_columns', [])
                )

                # Create a minimal DiscoveryResult for report compatibility
                answered_questions = []

                # Add a summary insight about the dataset
                summary_q = AnsweredQuestion(
                    question=f"What does this {profile['semantic_context']['domain']} dataset contain?",
                    answer=f"### Dataset Overview\n\n{profile['semantic_context']['description']}\n\n"
                           f"**Rows:** {profile['statistics']['row_count']:,}\n"
                           f"**Time Period:** {profile['semantic_context'].get('time_period', 'Not specified')}\n"
                           f"**Key Entities:** {', '.join(profile['semantic_context']['entities'][:3]) if profile['semantic_context']['entities'] else 'None identified'}"
                )
                answered_questions.append(summary_q)

                # Add quality assessment
                quality_q = AnsweredQuestion(
                    question="Data quality assessment",
                    answer=f"### Data Quality\n\n"
                           f"**Completeness:** {quality['completeness']:.1%}\n"
                           f"**Duplicates:** {quality['duplicate_pct']:.1%}\n"
                           f"**Statistical Confidence:** {quality['statistical_confidence'].upper()}"
                )
                answered_questions.append(quality_q)

                state['single_table_results'][dataset_id] = DiscoveryResult(
                    dataset_name=state['metadata'][dataset_id]['table_name'],
                    data_profile=data_profile,  # Now providing the required field
                    answered_questions=answered_questions
                )

        except Exception as e:
            print(f"[ERROR] Lightweight profiling failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to empty results
            state['table_profiles'] = {}
            state['single_table_results'] = {}

        # No delay needed - we didn't use any API calls!
        print("\n  ✅ No API rate limiting needed - proceeding immediately")

        return state

    def _single_table_analysis_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Analyze each table individually."""
        state['current_phase'] = 'single_table_analysis'
        print("\n[PHASE 2] SINGLE-TABLE ANALYSIS")

        try:
            dataset_count = len(state['datasets'])
            for idx, (dataset_id, df) in enumerate(state['datasets'].items(), 1):
                metadata = state['metadata'][dataset_id]
                print(f"\n  Analyzing {metadata['table_name']} ({idx}/{dataset_count})...")

                # Run single-table discovery with unified context from ETL
                result = self.single_table_workflow.run_discovery(
                    df,
                    metadata['table_name'],
                    context=metadata  # Pass full unified context from ETL
                )

                state['single_table_results'][dataset_id] = result

                print(f"    Found {len(result.answered_questions)} insights")

                # Add delay between datasets to avoid rate limits (Gemini free tier: 10 req/min)
                if idx < dataset_count:  # Don't delay after last dataset
                    delay_seconds = 60  # Wait 60 seconds between datasets
                    print(f"  Waiting {delay_seconds}s before next dataset (rate limit protection)...")
                    time.sleep(delay_seconds)

        except Exception as e:
            print(f"[WARN] Single-table analysis failed: {e}")

        # Add delay before next phase to avoid rate limits
        print(f"\nWaiting 60s before advanced analytics phase (rate limit protection)...")
        time.sleep(60)

        return state

    def _single_table_advanced_analytics_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Run LLM-selected advanced analytics on each table individually."""
        state['current_phase'] = 'single_table_advanced_analytics'
        print("\n[ADVANCED] SINGLE-TABLE ADVANCED ANALYTICS (LLM-Directed)")

        # Initialize insights storage
        if 'single_table_advanced_insights' not in state:
            state['single_table_advanced_insights'] = {}

        # Initialize test selector
        test_selector = StatisticalTestSelector()

        try:
            for dataset_id, df in state['datasets'].items():
                metadata = state['metadata'][dataset_id]
                table_name = metadata['table_name']

                print(f"\n  Analyzing {table_name}...")
                print(f"    Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

                # Initialize insights list for this dataset
                state['single_table_advanced_insights'][dataset_id] = []

                # Get LLM recommendations for which tests to run
                print("    🤖 Asking LLM to select optimal statistical tests...")

                recommended_tests = test_selector.select_tests(
                    df=df,
                    domain=metadata.get('domain', 'Unknown'),
                    dataset_context={
                        'table_name': table_name,
                        'description': metadata.get('description', ''),
                        'key_metrics': metadata.get('key_metrics', [])
                    },
                    max_tests=3,
                    mode='single_table'
                )

                if not recommended_tests:
                    print("    [SKIP] No suitable statistical tests for this dataset")
                    continue

                print(f"    Selected {len(recommended_tests)} tests based on data characteristics")

                # Execute each recommended test
                for test_rec in recommended_tests:
                    test_key = test_rec['test_key']
                    print(f"    Running: {test_rec['test_info']['name']}")
                    print(f"      Rationale: {test_rec['rationale'][:100]}...")

                    try:
                        if test_key == 'time_series_forecast':
                            self._run_forecast_test(df, table_name, metadata, state, dataset_id, test_rec)

                        elif test_key == 'anomaly_detection':
                            self._run_anomaly_test(df, table_name, metadata, state, dataset_id, test_rec)

                        elif test_key == 'trend_analysis':
                            self._run_trend_analysis(df, table_name, metadata, state, dataset_id, test_rec)

                        elif test_key == 'correlation_analysis':
                            self._run_correlation_analysis(df, table_name, metadata, state, dataset_id, test_rec)

                        elif test_key == 'segmentation_analysis':
                            self._run_segmentation_analysis(df, table_name, metadata, state, dataset_id, test_rec)

                        elif test_key == 'causal_analysis':
                            self._run_single_table_causal_analysis(df, table_name, metadata, state, dataset_id, test_rec)

                        else:
                            print(f"      [SKIP] Test {test_key} not yet implemented for single table")

                    except Exception as e:
                        print(f"      [ERROR] Test {test_key} failed: {str(e)[:100]}")

                total_insights = len(state['single_table_advanced_insights'][dataset_id])
                print(f"    Advanced insights generated: {total_insights}")

                # Show what high-value tests were skipped (for transparency)
                print("    Tests not selected:")
                skipped_explanation = test_selector.explain_skipped_tests(df, metadata.get('domain', 'Unknown'))
                for line in skipped_explanation.split('\n')[:3]:  # Show top 3 skipped
                    if line.strip():
                        print(f"      {line}")

        except Exception as e:
            print(f"[ERROR] Single-table advanced analytics failed: {e}")
            import traceback
            traceback.print_exc()

        # Add delay before next phase to avoid rate limits
        print(f"\nWaiting 60s before cross-table preparation (rate limit protection)...")
        time.sleep(60)

        return state

    def _run_forecast_test(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute time series forecasting test."""
        # Find time series columns
        time_columns = self._detect_time_series_columns(df)

        if not time_columns:
            # Try to create time series by aggregating
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns

            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                metric_col = test_rec.get('specific_columns', {}).get('target', numeric_cols[0])

                # Aggregate by date
                ts_data = df.groupby(date_col)[metric_col].sum().reset_index()
                ts_data = ts_data.set_index(date_col)
                time_columns = [{'column': metric_col, 'series': ts_data[metric_col]}]

        if time_columns:
            col_info = time_columns[0]  # Use first/best time series
            col_name = col_info['column']
            series = col_info['series']

            forecaster = TimeSeriesForecaster(method='auto')
            forecast_result = forecaster.forecast(
                data=series,
                periods=6,
                dataset_name=table_name,
                context={
                    'metric_name': col_name,
                    'table': table_name,
                    'domain': metadata.get('domain', 'Unknown')
                }
            )

            insight_text = self.synthesizer.synthesize_forecast(forecast_result)

            state['single_table_advanced_insights'][dataset_id].append({
                'type': 'forecast',
                'title': f"{col_name.title()} Forecast",
                'insight': insight_text,
                'confidence': forecast_result.validation.confidence_level.value
            })

            print(f"      ✓ Forecast generated for {col_name}")

    def _run_anomaly_test(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute anomaly detection test."""
        # Use specified columns or find best numeric columns
        if 'specific_columns' in test_rec and 'features' in test_rec['specific_columns']:
            target_cols = test_rec['specific_columns']['features']
        else:
            # Select numeric columns with highest variance (most interesting for anomaly detection)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                # Prioritize columns by variance (normalized by mean to avoid scale bias)
                col_variance = {}
                for col in numeric_cols:
                    if df[col].std() > 0:  # Has variation
                        # Coefficient of variation (CV) = std / mean
                        col_variance[col] = df[col].std() / (abs(df[col].mean()) + 1e-10)

                # Sort by variance and take top 2
                sorted_cols = sorted(col_variance.items(), key=lambda x: x[1], reverse=True)
                target_cols = [col for col, _ in sorted_cols[:2]] if sorted_cols else numeric_cols[:2]
            else:
                target_cols = []

        for col_name in target_cols:
            if col_name in df.columns and df[col_name].nunique() > 1:
                detector = AnomalyDetector(method='auto')
                anomaly_result = detector.detect_anomalies(
                    data=df[col_name],
                    dataset_name=table_name,
                    context={
                        'metric_name': col_name,
                        'table': table_name
                    }
                )

                if anomaly_result.results['total_anomalies'] > 0:
                    insight_text = self.synthesizer.synthesize_anomalies(anomaly_result)

                    state['single_table_advanced_insights'][dataset_id].append({
                        'type': 'anomaly',
                        'title': f"Unusual Patterns in {col_name.title()}",
                        'insight': insight_text,
                        'confidence': anomaly_result.validation.confidence_level.value
                    })

                    print(f"      ✓ {anomaly_result.results['total_anomalies']} anomalies in {col_name}")

    def _run_trend_analysis(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute trend analysis for time series data."""
        date_cols = df.select_dtypes(include=['datetime64']).columns

        if len(date_cols) > 0:
            date_col = date_cols[0]

            # Find metric to analyze
            if 'specific_columns' in test_rec and 'target' in test_rec['specific_columns']:
                metric_col = test_rec['specific_columns']['target']
            else:
                numeric_cols = df.select_dtypes(include=['number']).columns
                metric_col = numeric_cols[0] if len(numeric_cols) > 0 else None

            if metric_col and metric_col in df.columns:
                # Aggregate by time period
                df_sorted = df.sort_values(by=date_col)

                # Simple trend analysis using linear regression
                from scipy import stats
                import numpy as np

                # Convert dates to numeric for regression
                df_sorted['date_numeric'] = pd.to_datetime(df_sorted[date_col]).astype(int) / 10**9

                # Remove NaN values
                clean_data = df_sorted[['date_numeric', metric_col]].dropna()

                if len(clean_data) >= 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        clean_data['date_numeric'], clean_data[metric_col]
                    )

                    # Determine trend direction and significance
                    if p_value < 0.05:
                        trend = "increasing" if slope > 0 else "decreasing"
                        trend_strength = "strong" if abs(r_value) > 0.7 else "moderate"

                        insight = (f"**{metric_col.title()} shows a {trend_strength} {trend} trend** "
                                 f"(correlation: {abs(r_value):.2f}). ")

                        if slope > 0:
                            insight += f"The metric is growing over time, indicating positive momentum."
                        else:
                            insight += f"The metric is declining over time, which may require attention."

                        state['single_table_advanced_insights'][dataset_id].append({
                            'type': 'trend',
                            'title': f"{metric_col.title()} Trend Analysis",
                            'insight': insight,
                            'confidence': 'high' if abs(r_value) > 0.7 else 'medium'
                        })

                        print(f"      ✓ Trend analysis completed for {metric_col}")

    def _run_correlation_analysis(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute correlation analysis between variables."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find strong correlations (excluding diagonal)
            strong_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_corr.append({
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlation': corr_val
                        })

            if strong_corr:
                # Report top correlations
                strong_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
                top_corr = strong_corr[0]

                direction = "positive" if top_corr['correlation'] > 0 else "negative"
                insight = (f"**Strong {direction} correlation ({top_corr['correlation']:.2f}) between "
                         f"{top_corr['var1']} and {top_corr['var2']}**. ")

                if top_corr['correlation'] > 0:
                    insight += f"These metrics move together, suggesting a related business dynamic."
                else:
                    insight += f"These metrics move in opposite directions, indicating a trade-off relationship."

                state['single_table_advanced_insights'][dataset_id].append({
                    'type': 'correlation',
                    'title': f"Key Correlation: {top_corr['var1']} ↔ {top_corr['var2']}",
                    'insight': insight,
                    'confidence': 'high'
                })

                print(f"      ✓ Found {len(strong_corr)} strong correlations")

    def _run_segmentation_analysis(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute segmentation analysis (placeholder for now)."""
        print(f"      [INFO] Segmentation analysis not yet fully implemented")
        # This would use clustering algorithms to find natural groups in the data

    def _run_single_table_causal_analysis(self, df, table_name, metadata, state, dataset_id, test_rec):
        """Execute causal analysis for single table data."""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(date_cols) > 0 and len(numeric_cols) >= 2:
            date_col = date_cols[0]
            n_unique_dates = df[date_col].nunique()

            if n_unique_dates >= 20:  # Need enough time points
                # Select column pairs based on statistical properties, not domain keywords
                # We want pairs with high correlation (potential causal relationship)

                if len(numeric_cols) >= 2:
                    # Compute pairwise correlations
                    import pandas as pd
                    corr_pairs = []
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            try:
                                corr = df[[col1, col2]].corr().iloc[0, 1]
                                if abs(corr) > 0.3:  # Moderate correlation threshold
                                    corr_pairs.append((col1, col2, abs(corr)))
                            except:
                                pass

                    # Sort by correlation strength and take top pair
                    corr_pairs.sort(key=lambda x: x[2], reverse=True)

                    if corr_pairs:
                        # Test the most correlated pair
                        cause_col, effect_col, _ = corr_pairs[0]
                    else:
                        # No correlations found, just test first two numeric columns
                        cause_col = numeric_cols[0]
                        effect_col = numeric_cols[1]
                else:
                    cause_col = None
                    effect_col = None

                # If we found good candidates, test them
                if cause_col and effect_col:

                    # Aggregate to daily time series if not already
                    ts_data = df.groupby(date_col).agg({
                        cause_col: 'sum',
                        effect_col: 'sum'
                    }).ffill().fillna(0)

                    if len(ts_data) >= 20:
                        # Run causal analysis with various lags
                        analyzer = CausalAnalyzer(max_lag=min(10, len(ts_data) // 5))
                        causal_result = analyzer.analyze_causality(
                            cause=ts_data[cause_col],
                            effect=ts_data[effect_col],
                            dataset_name=table_name,
                            context={'cause_name': cause_col, 'effect_name': effect_col}
                        )

                        relationships = causal_result.results.get('relationships', [])
                        if relationships and relationships[0].get('is_significant'):
                            insight_text = self.synthesizer.synthesize_causal(causal_result)
                            optimal_lag = relationships[0].get('optimal_lag', 0)

                            state['single_table_advanced_insights'][dataset_id].append({
                                'type': 'causal',
                                'title': f"Causal Relationship: {cause_col} → {effect_col}",
                                'insight': insight_text,
                                'optimal_lag': optimal_lag,
                                'confidence': causal_result.validation.confidence_level.value
                            })
                            print(f"      ✓ Found causal relationship: {cause_col} → {effect_col} (lag: {optimal_lag} periods)")
                        else:
                            print(f"      No significant causal relationship found between {cause_col} and {effect_col}")
                else:
                    print(f"      No suitable column pairs found for causal analysis")

    def _run_cross_table_causal_analysis(self, joined_df, state, test_rec):
        """Execute causal analysis for cross-table data."""
        date_cols = joined_df.select_dtypes(include=['datetime64']).columns
        numeric_cols = joined_df.select_dtypes(include=['number']).columns.tolist()

        if len(date_cols) > 0 and len(numeric_cols) >= 2:
            date_col = date_cols[0]
            n_unique_dates = joined_df[date_col].nunique()

            if n_unique_dates >= 20:
                # Get specific columns if provided
                if 'specific_columns' in test_rec:
                    cause_col = test_rec['specific_columns'].get('features', [numeric_cols[0]])[0]
                    effect_col = test_rec['specific_columns'].get('target', numeric_cols[1])
                else:
                    # Find meaningful pairs
                    effect_candidates = [col for col in numeric_cols if any(
                        term in col.lower() for term in ['revenue', 'sales', 'profit', 'value']
                    )]
                    cause_candidates = [col for col in numeric_cols if any(
                        term in col.lower() for term in ['cost', 'price', 'quantity']
                    )]

                    if effect_candidates and cause_candidates:
                        cause_col = cause_candidates[0]
                        effect_col = effect_candidates[0]
                    else:
                        return

                # Aggregate to time series
                ts_data = joined_df.groupby(date_col).agg({
                    cause_col: 'sum',
                    effect_col: 'sum'
                }).ffill().fillna(0)

                if len(ts_data) >= 20:
                    analyzer = CausalAnalyzer(max_lag=min(5, len(ts_data) // 5))
                    causal_result = analyzer.analyze_causality(
                        cause=ts_data[cause_col],
                        effect=ts_data[effect_col],
                        dataset_name="Cross-Table Analysis",
                        context={'cause_name': cause_col, 'effect_name': effect_col}
                    )

                    relationships = causal_result.results.get('relationships', [])
                    if relationships and relationships[0].get('is_significant'):
                        insight_text = self.synthesizer.synthesize_causal(causal_result)
                        state['cross_table_advanced_insights'].append({
                            'type': 'causal',
                            'title': f"Causal Link: {cause_col} → {effect_col}",
                            'insight': insight_text,
                            'confidence': causal_result.validation.confidence_level.value
                        })
                        print(f"    ✓ Causal relationship: {cause_col} → {effect_col}")

    def _run_cross_table_variance_decomposition(self, joined_df, state, test_rec):
        """Execute variance decomposition for cross-table data."""
        numeric_cols = joined_df.select_dtypes(include=['number']).columns.tolist()

        # Find target variable
        if 'specific_columns' in test_rec and 'target' in test_rec['specific_columns']:
            target_col = test_rec['specific_columns']['target']
        else:
            target_candidates = [col for col in numeric_cols if any(
                term in col.lower() for term in ['revenue', 'sales', 'profit', 'value']
            )]
            if not target_candidates:
                return
            target_col = target_candidates[0]

        # Get feature columns
        feature_cols = [col for col in numeric_cols if col != target_col][:5]

        if len(feature_cols) >= 3 and target_col in joined_df.columns:
            X = joined_df[feature_cols].fillna(0)
            y = joined_df[target_col].fillna(0)

            if y.nunique() > 1 and X.shape[0] >= 30:
                decomposer = VarianceDecomposer(method='statistical')
                variance_result = decomposer.decompose(
                    X=X, y=y,
                    dataset_name="Cross-Table Analysis",
                    context={'outcome': target_col}
                )

                insight_text = self.synthesizer.synthesize_variance_decomposition(variance_result)
                state['cross_table_advanced_insights'].append({
                    'type': 'variance',
                    'title': f"Key Drivers of {target_col.title()}",
                    'insight': insight_text,
                    'confidence': variance_result.validation.confidence_level.value
                })
                print(f"    ✓ Variance decomposition for {target_col}")

    def _run_cross_table_correlation_analysis(self, joined_df, state, test_rec):
        """Execute correlation analysis for cross-table data."""
        numeric_cols = joined_df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            # Focus on cross-table correlations (columns from different tables)
            table_prefixes = set()
            for col in numeric_cols:
                if '_' in col:
                    prefix = col.split('_')[0]
                    table_prefixes.add(prefix)

            if len(table_prefixes) >= 2:
                # Find correlations between columns from different tables
                corr_matrix = joined_df[numeric_cols].corr()
                cross_table_corr = []

                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                        # Check if columns are from different tables
                        prefix1 = col1.split('_')[0] if '_' in col1 else 'main'
                        prefix2 = col2.split('_')[0] if '_' in col2 else 'main'

                        if prefix1 != prefix2:
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                cross_table_corr.append({
                                    'col1': col1, 'col2': col2,
                                    'correlation': corr_val
                                })

                if cross_table_corr:
                    top_corr = max(cross_table_corr, key=lambda x: abs(x['correlation']))
                    direction = "positive" if top_corr['correlation'] > 0 else "negative"

                    insight = (f"**Strong {direction} cross-table correlation ({top_corr['correlation']:.2f}) "
                             f"between {top_corr['col1']} and {top_corr['col2']}**. ")

                    if top_corr['correlation'] > 0:
                        insight += "These metrics from different tables move together, suggesting integrated business dynamics."
                    else:
                        insight += "These metrics from different tables move oppositely, indicating a trade-off across systems."

                    state['cross_table_advanced_insights'].append({
                        'type': 'correlation',
                        'title': f"Cross-Table Link: {top_corr['col1']} ↔ {top_corr['col2']}",
                        'insight': insight,
                        'confidence': 'high'
                    })
                    print(f"    ✓ Found {len(cross_table_corr)} cross-table correlations")

    def _run_cross_table_impact_analysis(self, joined_df, state, test_rec):
        """Execute impact analysis for cross-table data (placeholder)."""
        print("    [INFO] Cross-table impact analysis not yet implemented")

    def _run_cross_table_trend_analysis(self, joined_df, state, test_rec):
        """Execute trend analysis for cross-table data."""
        date_cols = joined_df.select_dtypes(include=['datetime64']).columns
        numeric_cols = joined_df.select_dtypes(include=['number']).columns

        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]

            # Analyze trends for key metrics from different tables
            key_metrics = []
            for col in numeric_cols:
                if any(term in col.lower() for term in ['revenue', 'sales', 'profit', 'cost']):
                    key_metrics.append(col)

            if key_metrics:
                from scipy import stats
                trends = []

                for metric in key_metrics[:3]:  # Limit to 3 metrics
                    df_sorted = joined_df.sort_values(by=date_col)
                    df_sorted['date_numeric'] = pd.to_datetime(df_sorted[date_col]).astype(int) / 10**9
                    clean_data = df_sorted[['date_numeric', metric]].dropna()

                    if len(clean_data) >= 10:
                        slope, _, r_value, p_value, _ = stats.linregress(
                            clean_data['date_numeric'], clean_data[metric]
                        )

                        if p_value < 0.05:
                            trends.append({
                                'metric': metric,
                                'slope': slope,
                                'correlation': abs(r_value),
                                'direction': 'increasing' if slope > 0 else 'decreasing'
                            })

                if trends:
                    # Report the most significant trend
                    best_trend = max(trends, key=lambda x: x['correlation'])

                    insight = (f"**{best_trend['metric'].title()} shows {best_trend['direction']} trend** "
                             f"across the integrated dataset (correlation: {best_trend['correlation']:.2f}). ")

                    if best_trend['direction'] == 'increasing':
                        insight += "This positive trajectory indicates growth across related business areas."
                    else:
                        insight += "This declining trend requires attention across multiple business units."

                    state['cross_table_advanced_insights'].append({
                        'type': 'trend',
                        'title': f"Cross-Table Trend: {best_trend['metric'].title()}",
                        'insight': insight,
                        'confidence': 'high' if best_trend['correlation'] > 0.7 else 'medium'
                    })
                    print(f"    ✓ Identified {len(trends)} significant trends")

    def _detect_time_series_columns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect columns that could be time series (have datetime index or enough numeric data)."""
        time_series = []

        # Check if DataFrame has datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            # Check numeric columns
            for col in df.select_dtypes(include=['number']).columns:
                if df[col].notna().sum() >= 12:  # At least 12 observations
                    time_series.append({
                        'column': col,
                        'series': df[col]
                    })

        return time_series

    def _prepare_cross_table_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Prepare data for cross-table analysis."""
        state['current_phase'] = 'preparing_cross_table'
        print("\n[PHASE 3] PREPARING CROSS-TABLE DATA")

        try:
            if len(state['relationships']) > 0:
                # Create joined DataFrame using relationships
                joined_df = self._create_joined_dataframe(
                    state['datasets'],
                    state['relationships'],
                    state['metadata']
                )
                state['joined_dataframe'] = joined_df
                print(f"  Created joined DataFrame: {len(joined_df):,} rows × {len(joined_df.columns)} columns")
            else:
                # No relationships - analyze tables side by side
                print("  No relationships found - will analyze tables independently")
                # Concatenate all dataframes with table name prefix
                dfs = []
                for dataset_id, df in state['datasets'].items():
                    table_name = state['metadata'][dataset_id]['table_name']
                    # Prefix columns with table name
                    df_copy = df.copy()
                    df_copy.columns = [f"{table_name}_{col}" for col in df_copy.columns]
                    dfs.append(df_copy)

                if dfs:
                    state['joined_dataframe'] = pd.concat(dfs, axis=1)
                    print(f"  Created side-by-side DataFrame: {len(state['joined_dataframe']):,} rows")

        except Exception as e:
            print(f"[WARN] Failed to prepare cross-table data: {e}")
            state['joined_dataframe'] = None

        # Add delay before next phase to avoid rate limits (reduced in lightweight mode)
        if self.mode == 'cross_table_focus':
            print(f"\nWaiting 60s before cross-table analysis (rate limit protection)")
            time.sleep(60)
        else:
            print(f"\nWaiting 60s before cross-table analysis (rate limit protection)...")
            time.sleep(60)

        return state

    def _cross_table_analysis_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Analyze cross-table relationships and insights."""
        state['current_phase'] = 'cross_table_analysis'
        print("\n[PHASE 4] CROSS-TABLE ANALYSIS")

        if state['joined_dataframe'] is None or state['joined_dataframe'].empty:
            print("  [SKIP] No joined data available for cross-table analysis")
            state['cross_table_insights'] = []
            return state

        try:
            # Prepare context for cross-table exploration
            exploration_context = {
                'domain': 'Multi-Domain',
                'dataset_type': 'Integrated',
                'tables': [meta['table_name'] for meta in state['metadata'].values()],
                'relationships': [
                    f"{r['from_column']} -> {r['to_column']}" for r in state['relationships']
                ] if state['relationships'] else [],
                'domains': list(set(meta.get('domain', 'Unknown') for meta in state['metadata'].values())),
                'description': f"Cross-table analysis of {len(state['metadata'])} related datasets"
            }

            print("  Running autonomous cross-table exploration...")

            # Use AutonomousExplorer for cross-table insights
            exploration_result = self.cross_table_explorer.explore(
                state['joined_dataframe'],
                "cross_table_analysis",
                exploration_context
            )

            # Convert insights to standard format
            cross_table_insights = []
            for insight in exploration_result.insights:
                cross_table_insights.append({
                    'question': insight.question,
                    'finding': insight.finding,
                    'confidence': insight.confidence,
                    'supporting_data': insight.supporting_data,
                    'code_used': insight.code_used,
                    'business_impact': insight.business_impact,
                    'visualization_paths': insight.visualization_paths
                })

            state['cross_table_insights'] = cross_table_insights

            print(f"  Found {len(cross_table_insights)} cross-table insights")

        except Exception as e:
            print(f"[ERROR] Cross-table analysis failed: {e}")
            state['cross_table_insights'] = []
            import traceback
            traceback.print_exc()

        # Add delay before next phase to avoid rate limits (reduced in lightweight mode)
        if self.mode == 'cross_table_focus':
            print(f"\nWaiting 25s before advanced analytics (rate limit protection)")
            time.sleep(25)
        else:
            print(f"\nWaiting 60s before cross-table advanced analytics (rate limit protection)...")
            time.sleep(60)

        return state

    def _cross_table_advanced_analytics_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Run LLM-selected advanced analytics across related tables."""
        state['current_phase'] = 'cross_table_advanced_analytics'
        print("\n[ADVANCED] CROSS-TABLE ADVANCED ANALYTICS (LLM-Directed)")

        # Initialize insights storage
        if 'cross_table_advanced_insights' not in state:
            state['cross_table_advanced_insights'] = []

        # Skip if no joined data
        if state['joined_dataframe'] is None or state['joined_dataframe'].empty:
            print("  [SKIP] No joined data available")
            return state

        joined_df = state['joined_dataframe']

        try:
            print(f"  Analyzing joined data: {joined_df.shape[0]} rows × {joined_df.shape[1]} columns")

            # Initialize test selector
            test_selector = StatisticalTestSelector()

            # Get LLM recommendations for cross-table tests
            print("  🤖 Asking LLM to select optimal cross-table statistical tests...")

            # Determine domain from combined metadata
            domains = [meta.get('domain', 'Unknown') for meta in state['metadata'].values()]
            combined_domain = ', '.join(set(domains))

            recommended_tests = test_selector.select_tests(
                df=joined_df,
                domain=combined_domain,
                dataset_context={
                    'tables': [meta['table_name'] for meta in state['metadata'].values()],
                    'relationships': len(state.get('relationships', [])),
                    'joined_data': True,
                    'description': 'Cross-table analysis of related datasets'
                },
                max_tests=4,  # Allow more tests for cross-table
                mode='cross_table'
            )

            if not recommended_tests:
                print("  [SKIP] No suitable cross-table tests for this dataset")
                return state

            print(f"  Selected {len(recommended_tests)} tests based on data characteristics")

            # Execute each recommended test
            for test_rec in recommended_tests:
                test_key = test_rec['test_key']
                print(f"  Running: {test_rec['test_info']['name']}")
                print(f"    Rationale: {test_rec['rationale'][:100]}...")

                try:
                    if test_key == 'causal_analysis':
                        self._run_cross_table_causal_analysis(joined_df, state, test_rec)

                    elif test_key == 'variance_decomposition':
                        self._run_cross_table_variance_decomposition(joined_df, state, test_rec)

                    elif test_key == 'correlation_analysis':
                        self._run_cross_table_correlation_analysis(joined_df, state, test_rec)

                    elif test_key == 'impact_analysis':
                        self._run_cross_table_impact_analysis(joined_df, state, test_rec)

                    elif test_key == 'trend_analysis':
                        self._run_cross_table_trend_analysis(joined_df, state, test_rec)

                    else:
                        print(f"    [SKIP] Test {test_key} not yet implemented for cross-table")

                except Exception as e:
                    print(f"    [ERROR] Test {test_key} failed: {str(e)[:100]}")

            # Show what high-value tests were skipped
            print("  Tests not selected:")
            skipped_explanation = test_selector.explain_skipped_tests(joined_df, combined_domain)
            for line in skipped_explanation.split('\n')[:3]:
                if line.strip():
                    print(f"    {line}")


            total_insights = len(state['cross_table_advanced_insights'])
            print(f"  Cross-table advanced insights generated: {total_insights}")

        except Exception as e:
            print(f"[ERROR] Cross-table advanced analytics failed: {e}")
            import traceback
            traceback.print_exc()

        # Add delay before next phase to avoid rate limits (reduced in lightweight mode)
        if self.mode == 'cross_table_focus':
            print(f"\nWaiting 15s before report generation")
            time.sleep(15)
        else:
            print(f"\nWaiting 60s before report generation (rate limit protection)...")
            time.sleep(60)

        return state

    def _generate_report_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Generate unified report and save session."""
        state['current_phase'] = 'reporting'
        print("\n[PHASE 5] GENERATING REPORT")

        try:
            # Create analysis session in database
            with DatabaseManager.get_session() as session:
                analysis_session = AnalysisSessionRepository.create_session(
                    session,
                    state['company_id'],
                    state['analysis_name'],
                    state['dataset_ids'],
                    description=f"Multi-table discovery with {len(state['dataset_ids'])} datasets"
                )

                state['analysis_session_id'] = analysis_session.id

                # Generate report path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = os.path.join(
                    "data", "outputs", "multi_table",
                    f"{state['analysis_name'].replace(' ', '_')}_{timestamp}"
                )
                os.makedirs(report_dir, exist_ok=True)

                # Create summary report
                report_path = os.path.join(report_dir, "unified_report.md")
                self._generate_markdown_report(state, report_path)

                state['unified_report_path'] = report_path

                # Update analysis session
                total_insights = (
                    sum(len(r.answered_questions) for r in state['single_table_results'].values()) +
                    len(state.get('cross_table_insights', []))
                )

                AnalysisSessionRepository.update_session_status(
                    session,
                    analysis_session.id,
                    status='completed',
                    insights_generated=total_insights,
                    report_path=report_path
                )

            state['status'] = 'completed'
            print(f"  Report saved: {report_path}")

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Report generation failed: {e}"

        return state

    def _create_joined_dataframe(
        self,
        datasets: Dict[str, pd.DataFrame],
        relationships: List,
        metadata: Dict
    ) -> pd.DataFrame:
        """Create a joined DataFrame from multiple tables."""

        if not relationships:
            return pd.DataFrame()

        # Find the best base table (the one that appears most in relationships)
        relationship_counts = {}
        for rel in relationships:
            relationship_counts[rel['from_dataset_id']] = relationship_counts.get(rel['from_dataset_id'], 0) + 1
            relationship_counts[rel['to_dataset_id']] = relationship_counts.get(rel['to_dataset_id'], 0) + 1

        # Sort by count and pick the most connected table as base
        dataset_ids = list(datasets.keys())
        if relationship_counts:
            # Choose the dataset that appears most in relationships
            base_dataset_id = max(relationship_counts, key=relationship_counts.get)
            # Ensure it's in our datasets
            if base_dataset_id not in datasets:
                base_dataset_id = dataset_ids[0]
        else:
            # Fallback to the largest table as base (likely the fact table)
            base_dataset_id = max(dataset_ids, key=lambda x: len(datasets[x]))

        base_df = datasets[base_dataset_id].copy()
        base_table = metadata[base_dataset_id]['table_name']

        print(f"  Using {base_table} as base table ({len(base_df)} rows)")

        # Track which datasets have been joined
        joined_ids = {base_dataset_id}

        # Iteratively join others using relationships
        for rel in relationships:
            from_id = rel['from_dataset_id']
            to_id = rel['to_dataset_id']

            # Determine which dataset to join
            if from_id in joined_ids and to_id not in joined_ids:
                # Join to_id dataset
                right_df = datasets[to_id].copy()
                right_table = metadata[to_id]['table_name']

                print(f"  Joining {right_table} ({len(right_df)} rows) on {rel['from_column']} = {rel['to_column']}")

                # Rename columns to avoid conflicts (except join key)
                right_df = right_df.rename(columns={
                    col: f"{right_table}_{col}" if col != rel['to_column'] else col
                    for col in right_df.columns
                })

                base_df = base_df.merge(
                    right_df,
                    left_on=rel['from_column'],
                    right_on=rel['to_column'] if rel['to_column'] in right_df.columns else f"{right_table}_{rel['to_column']}",
                    how=rel.get('join_strategy', 'inner'),  # Default to inner join
                    suffixes=('', f'_{right_table}')
                )

                print(f"    Result: {len(base_df)} rows after join")

                joined_ids.add(to_id)

            elif to_id in joined_ids and from_id not in joined_ids:
                # Join from_id dataset
                right_df = datasets[from_id].copy()
                right_table = metadata[from_id]['table_name']

                print(f"  Joining {right_table} ({len(right_df)} rows) on {rel['to_column']} = {rel['from_column']}")

                # Rename columns to avoid conflicts
                right_df = right_df.rename(columns={
                    col: f"{right_table}_{col}" if col != rel['from_column'] else col
                    for col in right_df.columns
                })

                base_df = base_df.merge(
                    right_df,
                    left_on=rel['to_column'],
                    right_on=rel['from_column'] if rel['from_column'] in right_df.columns else f"{right_table}_{rel['from_column']}",
                    how=rel.get('join_strategy', 'inner'),  # Default to inner join
                    suffixes=('', f'_{right_table}')
                )

                print(f"    Result: {len(base_df)} rows after join")

                joined_ids.add(from_id)

        print(f"  Final joined DataFrame: {len(base_df)} rows × {len(base_df.columns)} columns")
        print(f"  Joined tables: {', '.join([metadata[did]['table_name'] for did in joined_ids])}")

        return base_df

    def _generate_markdown_report(self, state: MultiTableDiscoveryState, report_path: str):
        """Generate a business-focused report of the analysis."""

        lines = []
        lines.append(f"# Business Intelligence Report")
        lines.append(f"\n**Date:** {datetime.now().strftime('%B %d, %Y')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("This report analyzes your business data to identify growth opportunities, operational risks, and actionable recommendations.")
        lines.append("")

        # 1. BUSINESS PERFORMANCE
        lines.append("## 1. Business Performance")
        lines.append("")

        # Extract key metrics from advanced insights
        revenue_trends = []
        growth_patterns = []
        risk_indicators = []

        # Look for trends and growth
        for insights in state.get('single_table_advanced_insights', {}).values():
            for insight in insights:
                if insight.get('type') == 'trend' and 'increasing' in insight.get('insight', '').lower():
                    growth_patterns.append(insight)
                elif insight.get('type') == 'trend' and 'decreasing' in insight.get('insight', '').lower():
                    risk_indicators.append(insight)
                elif insight.get('type') == 'anomaly':
                    risk_indicators.append(insight)

        if growth_patterns:
            lines.append("### ✅ Positive Trends")
            for pattern in growth_patterns[:3]:
                # Extract the business-relevant part of the insight
                insight_text = pattern['insight'].replace('**', '')
                # Simplify the language
                if 'increasing trend' in insight_text.lower():
                    metric = pattern['title'].replace('Trend Analysis', '').strip()
                    lines.append(f"• {metric} is growing consistently")
                else:
                    lines.append(f"• {pattern['title'].replace('Analysis', '').strip()}")
            lines.append("")

        if risk_indicators:
            lines.append("### ⚠️ Areas Requiring Attention")
            for risk in risk_indicators[:3]:
                if 'anomaly' in risk.get('type', ''):
                    metric = risk['title'].replace('Unusual Patterns in', '').replace('Anomaly in', '').strip()
                    lines.append(f"• Unusual activity detected in {metric}")
                elif 'decreasing' in risk.get('insight', '').lower():
                    metric = risk['title'].replace('Trend Analysis', '').strip()
                    lines.append(f"• {metric} is declining")
                else:
                    lines.append(f"• {risk['title']}")
            lines.append("")

        # 2. KEY DRIVERS
        lines.append("## 2. What's Driving Your Business")
        lines.append("")

        # Look for causal relationships and correlations in both single-table and cross-table insights
        causal_insights = []

        # Collect from single-table insights
        for dataset_id, insights in state.get('single_table_advanced_insights', {}).items():
            for insight in insights:
                if insight.get('type') in ['causal', 'correlation', 'variance']:
                    causal_insights.append(insight)

        # Add from cross-table insights
        causal_insights.extend([i for i in state.get('cross_table_advanced_insights', [])
                               if i.get('type') in ['causal', 'correlation', 'variance']])

        if causal_insights:
            for insight in causal_insights[:3]:
                if insight['type'] == 'causal':
                    # Extract cause and effect from title
                    if '→' in insight['title']:
                        title = insight['title'].replace('Causal Link:', '').replace('Causal Relationship:', '').strip()
                        parts = title.split('→')
                        cause = parts[0].strip()
                        effect = parts[1].strip() if len(parts) > 1 else ''
                        lag = insight.get('optimal_lag', None)
                        if lag and lag > 0:
                            lines.append(f"• **{cause}** drives **{effect}** with a {lag}-day delay")
                        else:
                            lines.append(f"• **{cause}** directly impacts **{effect}**")
                elif insight['type'] == 'variance':
                    title = insight['title'].replace('Key Drivers of', 'Factors affecting').strip()
                    lines.append(f"• {title}")
                elif insight['type'] == 'correlation':
                    if 'positive' in insight.get('insight', '').lower():
                        lines.append(f"• {insight['title'].replace('Cross-Table Link:', '').replace('↔', 'and').strip()} move together")
                    else:
                        lines.append(f"• Trade-off between {insight['title'].replace('Cross-Table Link:', '').replace('↔', 'and').strip()}")
        else:
            lines.append("• Further analysis needed to identify key business drivers")
        lines.append("")

        # 3. OPPORTUNITIES & RISKS
        lines.append("## 3. Opportunities & Risks")
        lines.append("")

        # Top opportunities from insights
        lines.append("### 💡 Opportunities")
        opportunities = []
        for dataset_id, result in state['single_table_results'].items():
            for q in result.answered_questions[:2]:
                if any(term in q.answer.lower() for term in ['opportunity', 'potential', 'growth', 'increase']):
                    opportunities.append(q)

        if opportunities:
            for opp in opportunities[:3]:
                # Extract actionable insight
                lines.append(f"• {opp.question}")
        else:
            lines.append("• Analyze customer segments for expansion opportunities")
            lines.append("• Review product performance for optimization")
        lines.append("")

        lines.append("### 🔴 Risks")
        risks = []
        for dataset_id, result in state['single_table_results'].items():
            for q in result.answered_questions:
                if any(term in q.answer.lower() for term in ['risk', 'decline', 'concern', 'warning']):
                    risks.append(q)

        if risks:
            for risk in risks[:3]:
                lines.append(f"• {risk.question}")
        else:
            # Look for anomalies as risks
            anomaly_count = sum(1 for insights in state.get('single_table_advanced_insights', {}).values()
                              for i in insights if i.get('type') == 'anomaly')
            if anomaly_count > 0:
                lines.append(f"• {anomaly_count} unusual patterns detected requiring investigation")
            else:
                lines.append("• Continue monitoring for unusual patterns")
        lines.append("")

        # 4. RECOMMENDATIONS
        lines.append("## 4. Recommended Actions")
        lines.append("")

        # Generate recommendations based on insights
        has_recommendations = False

        # Based on causal insights
        for insight in state.get('cross_table_advanced_insights', [])[:3]:
            if insight.get('type') == 'causal':
                if 'marketing' in insight['title'].lower() and 'revenue' in insight['title'].lower():
                    lines.append("• **Increase marketing investment** - Analysis shows direct impact on revenue")
                    has_recommendations = True
                elif 'price' in insight['title'].lower():
                    lines.append("• **Review pricing strategy** - Price changes significantly affect sales volume")
                    has_recommendations = True
                elif 'customer' in insight['title'].lower():
                    lines.append("• **Enhance customer engagement** - Customer activity drives business growth")
                    has_recommendations = True

        # Based on trends
        if growth_patterns:
            lines.append("• **Capitalize on growth momentum** - Continue investing in growing areas")
            has_recommendations = True

        if risk_indicators and len(risk_indicators) > 2:
            lines.append("• **Investigate unusual patterns** - Multiple anomalies require attention")
            has_recommendations = True

        if not has_recommendations:
            lines.append("• Review detailed insights below for specific action items")
            lines.append("• Set up monitoring for key business metrics")
            lines.append("• Schedule regular data reviews to track progress")

        lines.append("")

        # 5. DATA OVERVIEW
        lines.append("## 5. Data Analyzed")
        lines.append("")

        for dataset_id, meta in state['metadata'].items():
            table_name = meta['table_name'].replace('_', ' ').title()
            row_count = meta.get('row_count', 0)
            lines.append(f"• **{table_name}**: {row_count:,} records")

        lines.append("")

        # Add date range if available
        date_info = []
        for dataset_id, meta in state['metadata'].items():
            if meta.get('temporal_coverage'):
                date_info.append(meta['temporal_coverage'])

        if date_info:
            lines.append(f"**Time Period:** {date_info[0]}")
            lines.append("")

        # End of report
        lines.append("\n---")
        lines.append("\n_This report was generated automatically from your business data._")
        lines.append(f"_For questions or detailed analysis, please contact your data team._")
        lines.append("")

        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

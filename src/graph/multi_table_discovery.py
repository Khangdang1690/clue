"""Multi-table discovery workflow with autonomous cross-table analysis."""

from typing import List, Dict, Optional, Any
import pandas as pd
from langgraph.graph import StateGraph, END
from src.models.discovery_models import DiscoveryResult
from src.models.etl_models import MultiTableDiscoveryState
from src.graph.discovery_workflow import DiscoveryWorkflow
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
import os
from datetime import datetime


class MultiTableDiscovery:
    """
    Two-phase discovery for multiple related tables:
    1. Analyze each table individually
    2. Analyze cross-table relationships and insights
    3. Use similarity search to suggest related datasets
    """

    def __init__(self):
        self.single_table_workflow = DiscoveryWorkflow()
        # Use more iterations for cross-table analysis
        self.cross_table_explorer = AutonomousExplorer(
            max_iterations=25,
            max_insights=15
        )
        self.embedding_service = get_embedding_service()

        # Advanced Analytics: Business Insight Synthesizer (converts stats to plain English)
        self.synthesizer = BusinessInsightSynthesizer()

        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build multi-table discovery workflow."""
        workflow = StateGraph(MultiTableDiscoveryState)

        # Add nodes
        workflow.add_node("load_datasets", self._load_datasets_node)
        workflow.add_node("suggest_related", self._suggest_related_node)
        workflow.add_node("single_table_analysis", self._single_table_analysis_node)
        workflow.add_node("single_table_advanced_analytics", self._single_table_advanced_analytics_node)
        workflow.add_node("prepare_cross_table", self._prepare_cross_table_node)
        workflow.add_node("cross_table_analysis", self._cross_table_analysis_node)
        workflow.add_node("cross_table_advanced_analytics", self._cross_table_advanced_analytics_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define flow
        workflow.set_entry_point("load_datasets")
        workflow.add_edge("load_datasets", "suggest_related")
        workflow.add_edge("suggest_related", "single_table_analysis")
        workflow.add_edge("single_table_analysis", "single_table_advanced_analytics")
        workflow.add_edge("single_table_advanced_analytics", "prepare_cross_table")
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
                        state['metadata'][dataset_id] = {
                            'table_name': dataset.table_name,
                            'domain': dataset.domain,
                            'description': dataset.description,
                            'entities': dataset.entities,
                            'row_count': dataset.row_count,
                            'column_count': dataset.column_count
                        }

                        print(f"  Loaded {dataset.table_name}: {dataset.row_count:,} rows")

                # Load relationships
                state['relationships'] = RelationshipRepository.get_relationships_for_datasets(
                    session,
                    state['dataset_ids'],
                    min_confidence=0.8
                )

                print(f"  Found {len(state['relationships'])} relationships")

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Failed to load datasets: {e}"

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

                    if dataset and dataset.description_embedding:
                        # Find similar datasets
                        similar = SimilarityRepository.find_similar_datasets(
                            session,
                            dataset.description_embedding,
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

        return state

    def _single_table_analysis_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Analyze each table individually."""
        state['current_phase'] = 'single_table_analysis'
        print("\n[PHASE 2] SINGLE-TABLE ANALYSIS")

        try:
            for dataset_id, df in state['datasets'].items():
                metadata = state['metadata'][dataset_id]
                print(f"\n  Analyzing {metadata['table_name']}...")

                # Run single-table discovery
                result = self.single_table_workflow.run_discovery(
                    df,
                    metadata['table_name']
                )

                state['single_table_results'][dataset_id] = result

                print(f"    Found {len(result.answered_questions)} insights")

        except Exception as e:
            print(f"[WARN] Single-table analysis failed: {e}")

        return state

    def _single_table_advanced_analytics_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Run advanced analytics on each table individually."""
        state['current_phase'] = 'single_table_advanced_analytics'
        print("\n[ADVANCED] SINGLE-TABLE ADVANCED ANALYTICS")

        # Initialize insights storage
        if 'single_table_advanced_insights' not in state:
            state['single_table_advanced_insights'] = {}

        try:
            for dataset_id, df in state['datasets'].items():
                metadata = state['metadata'][dataset_id]
                table_name = metadata['table_name']

                print(f"\n  Analyzing {table_name}...")

                # Initialize insights list for this dataset
                state['single_table_advanced_insights'][dataset_id] = []

                # 1. Forecasting - detect time series columns
                time_columns = self._detect_time_series_columns(df)

                if time_columns:
                    print(f"    Found {len(time_columns)} time series column(s)")

                    for col_info in time_columns[:2]:  # Limit to top 2 time series
                        col_name = col_info['column']
                        series = col_info['series']

                        try:
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

                            # Generate business insight (NO statistics shown to user)
                            insight_text = self.synthesizer.synthesize_forecast(forecast_result)

                            state['single_table_advanced_insights'][dataset_id].append({
                                'type': 'forecast',
                                'title': f"{col_name.title()} Forecast",
                                'insight': insight_text,
                                'confidence': forecast_result.validation.confidence_level.value
                            })

                            print(f"      -> Forecast generated for {col_name}")

                        except Exception as e:
                            print(f"      [WARN] Forecasting failed for {col_name}: {e}")

                # 2. Anomaly Detection - run on numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                if numeric_cols:
                    print(f"    Found {len(numeric_cols)} numeric column(s)")

                    for col_name in numeric_cols[:3]:  # Limit to top 3 numeric columns
                        try:
                            # Skip if all values are the same
                            if df[col_name].nunique() <= 1:
                                continue

                            detector = AnomalyDetector(method='auto')
                            anomaly_result = detector.detect_anomalies(
                                data=df[col_name],
                                dataset_name=table_name,
                                context={
                                    'metric_name': col_name,
                                    'table': table_name
                                }
                            )

                            # Only report if anomalies found
                            if anomaly_result.results['total_anomalies'] > 0:
                                insight_text = self.synthesizer.synthesize_anomalies(anomaly_result)

                                state['single_table_advanced_insights'][dataset_id].append({
                                    'type': 'anomaly',
                                    'title': f"Unusual Patterns in {col_name.title()}",
                                    'insight': insight_text,
                                    'confidence': anomaly_result.validation.confidence_level.value
                                })

                                print(f"      -> {anomaly_result.results['total_anomalies']} anomalies detected in {col_name}")

                        except Exception as e:
                            print(f"      [WARN] Anomaly detection failed for {col_name}: {e}")

                total_insights = len(state['single_table_advanced_insights'][dataset_id])
                print(f"    Advanced insights generated: {total_insights}")

        except Exception as e:
            print(f"[ERROR] Single-table advanced analytics failed: {e}")
            import traceback
            traceback.print_exc()

        return state

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
                    f"{r.from_column} -> {r.to_column}" for r in state['relationships']
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

        return state

    def _cross_table_advanced_analytics_node(self, state: MultiTableDiscoveryState) -> MultiTableDiscoveryState:
        """Run advanced analytics across related tables."""
        state['current_phase'] = 'cross_table_advanced_analytics'
        print("\n[ADVANCED] CROSS-TABLE ADVANCED ANALYTICS")

        # Initialize insights storage
        if 'cross_table_advanced_insights' not in state:
            state['cross_table_advanced_insights'] = []

        # Skip if no joined data
        if state['joined_dataframe'] is None or state['joined_dataframe'].empty:
            print("  [SKIP] No joined data available")
            return state

        joined_df = state['joined_dataframe']

        try:
            # Get numeric columns for analysis
            numeric_cols = joined_df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) < 2:
                print("  [SKIP] Not enough numeric columns for cross-table analytics")
                return state

            # 1. Causal Inference - test potential cause-effect relationships
            print(f"  Testing causal relationships between {len(numeric_cols)} numeric columns...")

            # Test top pairs of columns (limit to avoid too many tests)
            for i in range(min(2, len(numeric_cols) - 1)):
                cause_col = numeric_cols[i]
                effect_col = numeric_cols[i + 1]

                try:
                    # Skip if too few observations
                    if joined_df[cause_col].notna().sum() < 30 or joined_df[effect_col].notna().sum() < 30:
                        continue

                    analyzer = CausalAnalyzer(max_lag=5)
                    causal_result = analyzer.analyze_causality(
                        cause=joined_df[cause_col],
                        effect=joined_df[effect_col],
                        dataset_name="Cross-Table Analysis",
                        context={
                            'cause_name': cause_col,
                            'effect_name': effect_col
                        }
                    )

                    # Only report if relationship is significant
                    relationships = causal_result.results.get('relationships', [])
                    if relationships and relationships[0].get('is_significant'):
                        insight_text = self.synthesizer.synthesize_causal(causal_result)

                        state['cross_table_advanced_insights'].append({
                            'type': 'causal',
                            'title': f"Causal Link: {cause_col} -> {effect_col}",
                            'insight': insight_text,
                            'confidence': causal_result.validation.confidence_level.value
                        })

                        print(f"    -> Causal relationship found: {cause_col} -> {effect_col}")

                except Exception as e:
                    print(f"    [WARN] Causal analysis failed for {cause_col} -> {effect_col}: {e}")

            # 2. Variance Decomposition - if we can identify a target variable
            # Look for common target variable names
            target_candidates = [col for col in numeric_cols if any(
                keyword in col.lower() for keyword in ['revenue', 'sales', 'profit', 'churn', 'value']
            )]

            if target_candidates and len(numeric_cols) >= 4:
                target_col = target_candidates[0]
                feature_cols = [col for col in numeric_cols if col != target_col][:5]  # Limit to 5 features

                try:
                    X = joined_df[feature_cols].fillna(0)
                    y = joined_df[target_col].fillna(0)

                    # Skip if not enough variance
                    if y.nunique() > 1 and X.shape[0] >= 30:
                        decomposer = VarianceDecomposer(method='statistical')
                        variance_result = decomposer.decompose(
                            X=X,
                            y=y,
                            dataset_name="Cross-Table Analysis",
                            context={'outcome': target_col}
                        )

                        insight_text = self.synthesizer.synthesize_variance_decomposition(variance_result)

                        state['cross_table_advanced_insights'].append({
                            'type': 'variance',
                            'title': f"Drivers of {target_col.title()}",
                            'insight': insight_text,
                            'confidence': variance_result.validation.confidence_level.value
                        })

                        print(f"    -> Variance decomposition completed for {target_col}")

                except Exception as e:
                    print(f"    [WARN] Variance decomposition failed: {e}")

            total_insights = len(state['cross_table_advanced_insights'])
            print(f"  Cross-table advanced insights generated: {total_insights}")

        except Exception as e:
            print(f"[ERROR] Cross-table advanced analytics failed: {e}")
            import traceback
            traceback.print_exc()

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

        # Start with the first dataset
        dataset_ids = list(datasets.keys())
        base_df = datasets[dataset_ids[0]].copy()
        base_table = metadata[dataset_ids[0]]['table_name']

        # Track which datasets have been joined
        joined_ids = {dataset_ids[0]}

        # Iteratively join others using relationships
        for rel in relationships:
            from_id = rel.from_dataset_id
            to_id = rel.to_dataset_id

            # Determine which dataset to join
            if from_id in joined_ids and to_id not in joined_ids:
                # Join to_id dataset
                right_df = datasets[to_id].copy()
                right_table = metadata[to_id]['table_name']

                # Rename columns to avoid conflicts (except join key)
                right_df = right_df.rename(columns={
                    col: f"{right_table}_{col}" if col != rel.to_column else col
                    for col in right_df.columns
                })

                base_df = base_df.merge(
                    right_df,
                    left_on=rel.from_column,
                    right_on=rel.to_column if rel.to_column in right_df.columns else f"{right_table}_{rel.to_column}",
                    how=rel.join_strategy,
                    suffixes=('', f'_{right_table}')
                )

                joined_ids.add(to_id)

            elif to_id in joined_ids and from_id not in joined_ids:
                # Join from_id dataset
                right_df = datasets[from_id].copy()
                right_table = metadata[from_id]['table_name']

                # Rename columns to avoid conflicts
                right_df = right_df.rename(columns={
                    col: f"{right_table}_{col}" if col != rel.from_column else col
                    for col in right_df.columns
                })

                base_df = base_df.merge(
                    right_df,
                    left_on=rel.to_column,
                    right_on=rel.from_column if rel.from_column in right_df.columns else f"{right_table}_{rel.from_column}",
                    how=rel.join_strategy,
                    suffixes=('', f'_{right_table}')
                )

                joined_ids.add(from_id)

        return base_df

    def _generate_markdown_report(self, state: MultiTableDiscoveryState, report_path: str):
        """Generate a markdown report of the analysis."""

        lines = []
        lines.append(f"# Multi-Table Discovery Report")
        lines.append(f"\n**Analysis:** {state['analysis_name']}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Datasets:** {len(state['dataset_ids'])}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        # Count insights
        total_insights = sum(len(result.answered_questions) for result in state['single_table_results'].values())
        cross_insights = len(state.get('cross_table_insights', []))

        lines.append(f"**Key Metrics:**")
        lines.append(f"- {len(state['dataset_ids'])} datasets analyzed")
        lines.append(f"- {total_insights} single-table insights discovered")
        lines.append(f"- {cross_insights} cross-table patterns identified")
        lines.append(f"- {len(state.get('relationships', []))} relationships detected")
        lines.append("")

        # Top 3 insights across all datasets
        lines.append("**Top Findings:**")
        all_insights = []
        for dataset_id, result in state['single_table_results'].items():
            table_name = state['metadata'][dataset_id]['table_name']
            for q in result.answered_questions[:2]:  # Top 2 from each dataset
                all_insights.append((table_name, q.question, q.confidence))

        # Sort by confidence and take top 3
        all_insights.sort(key=lambda x: x[2], reverse=True)
        for i, (table, question, conf) in enumerate(all_insights[:3], 1):
            lines.append(f"{i}. **{table}**: {question}")

        lines.append("")

        # Dataset summary
        lines.append("## Datasets Analyzed")
        for dataset_id, meta in state['metadata'].items():
            lines.append(f"- **{meta['table_name']}** ({meta['domain']} domain)")
            lines.append(f"  - {meta.get('row_count', 0):,} rows × {meta.get('column_count', 0)} columns")
            if meta.get('description'):
                lines.append(f"  - {meta['description']}")

        # Data Quality Assessment
        lines.append("\n## Data Quality Assessment")
        lines.append("")

        for dataset_id, meta in state['metadata'].items():
            table_name = meta['table_name']
            row_count = meta.get('row_count', 0)

            lines.append(f"### {table_name}")

            # Sample size assessment
            if row_count < 30:
                confidence = "VERY LOW"
                icon = "🔴"
            elif row_count < 100:
                confidence = "LOW"
                icon = "🟡"
            elif row_count < 1000:
                confidence = "MEDIUM"
                icon = "🟢"
            else:
                confidence = "HIGH"
                icon = "🟢"

            lines.append(f"- **Sample Size**: {row_count:,} rows - Statistical confidence: {icon} {confidence}")

            # Temporal coverage (if available in metadata)
            if row_count > 0:
                lines.append(f"- **Schema**: {meta.get('column_count', 0)} columns validated")
                lines.append(f"- **Domain Classification**: {meta.get('domain', 'Unknown')}")

        lines.append("")

        # Relationships
        if state['relationships']:
            lines.append("\n## Detected Relationships")
            for rel in state['relationships']:
                lines.append(f"- {rel.from_column} → {rel.to_column} "
                            f"({rel.relationship_type}, confidence: {rel.confidence:.2f})")

        # Single-table insights
        lines.append("\n## Single-Table Insights")
        for dataset_id, result in state['single_table_results'].items():
            table_name = state['metadata'][dataset_id]['table_name']
            lines.append(f"\n### {table_name}")

            # Add sample size warning if dataset is small
            row_count = state['metadata'][dataset_id].get('row_count', 0)
            if row_count > 0 and row_count < 100:
                lines.append(f"\n> ⚠️ **Sample Size Warning**: Based on only {row_count:,} rows. Statistical confidence may be LIMITED.")

            for q in result.answered_questions[:5]:  # Top 5 insights
                # Don't duplicate the question header - answer already contains "### Insight N: [Title]"
                lines.append(f"{q.answer}")

            # Add advanced analytics insights for this table
            if dataset_id in state.get('single_table_advanced_insights', {}):
                advanced_insights = state['single_table_advanced_insights'][dataset_id]
                if advanced_insights:
                    lines.append(f"\n#### Advanced Analytics\n")

                    for adv_insight in advanced_insights:
                        lines.append(f"**{adv_insight['title']}**\n")
                        lines.append(adv_insight['insight'])
                        lines.append("")

        # Cross-table insights
        if state['cross_table_insights']:
            lines.append("\n## Cross-Table Insights")
            for insight in state['cross_table_insights']:
                lines.append(f"- **{insight['question']}**")
                lines.append(f"  - {insight['finding']}")
                lines.append(f"  - Confidence: {insight['confidence']:.0%}")

        # Cross-table advanced analytics insights
        if state.get('cross_table_advanced_insights'):
            lines.append("\n## Cross-Table Advanced Analytics")
            lines.append("")

            for adv_insight in state['cross_table_advanced_insights']:
                lines.append(f"### {adv_insight['title']}\n")
                lines.append(adv_insight['insight'])
                lines.append("")

        # Suggested datasets
        if state['suggested_datasets']:
            lines.append("\n## Suggested Related Datasets")
            lines.append("Consider including these similar datasets in future analyses:")
            for dataset_id in state['suggested_datasets']:
                lines.append(f"- Dataset ID: {dataset_id}")

        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
"""Business-focused discovery workflow using dynamic LLM exploration."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import json
import pandas as pd
import asyncio
from langgraph.graph import StateGraph, END
from src.database.connection import DatabaseManager
from src.database.repository import DatasetRepository, CompanyRepository
from src.analytics.dynamic_explorer import DynamicDataExplorer
from src.discovery.multi_table_executor import MultiTableCodeExecutor
from src.discovery.visualization_data_store import VisualizationDataStore
from src.discovery.plotly_dashboard_generator import PlotlyDashboardGenerator
from src.analytics.anomaly_detection import AnomalyDetector
from src.analytics.forecasting import TimeSeriesForecaster
from src.analytics.causal_inference import CausalAnalyzer
from src.analytics.variance_decomposition import VarianceDecomposer
import google.generativeai as genai

# Import message service for real-time streaming
try:
    from app.services.analysis_message_service import AnalysisMessageService
except ImportError:
    # Fallback if not available
    AnalysisMessageService = None

# Import storage service for GCS uploads
try:
    from app.services.storage_service import StorageService
except ImportError:
    # Fallback if not available (e.g., during testing)
    StorageService = None


class BusinessDiscoveryState(Dict):
    """State for business discovery workflow."""
    company_id: str
    dataset_ids: List[str]
    analysis_id: Optional[str]  # Unique ID for this analysis session
    datasets: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]
    business_context: Dict[str, Any]
    exploration_results: Dict[str, Any]
    analytics_results: Dict[str, Any]  # Results from advanced analytics
    insights: List[Dict[str, Any]]
    synthesized_insights: List[Dict[str, Any]]  # LLM-synthesized narratives
    recommendations: List[Dict[str, Any]]
    executive_summary: str
    report_path: Optional[str]
    viz_data_path: Optional[str]  # Path to viz_data JSON
    dashboard_path: Optional[str]  # Path to Plotly dashboard HTML
    status: str
    error: Optional[str]


class BusinessDiscoveryWorkflow:
    """Workflow for business-focused data discovery."""

    def __init__(self, event_loop=None):
        """Initialize the business discovery workflow.

        Args:
            event_loop: Optional event loop for async operations (message streaming)
        """
        self.event_loop = event_loop  # Store event loop for message streaming
        self.explorer = DynamicDataExplorer()

        # Use the BusinessAnalystLLM for proper code generation
        from src.analytics.business_analyst_llm import BusinessAnalystLLM
        self.business_analyst = BusinessAnalystLLM()

        # Configure Gemini for business analysis
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key, transport='rest')  # Use REST to avoid gRPC ALTS warnings
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

        # Initialize visualization components
        self.viz_data_store = None  # Will be initialized per run
        self.dashboard_generator = PlotlyDashboardGenerator()

        # Initialize analytics components
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = TimeSeriesForecaster()
        self.causal_analyzer = CausalAnalyzer()
        self.variance_decomposer = VarianceDecomposer()

        # Build workflow
        self.graph = self._build_workflow()

    def _emit_message(self, analysis_id: Optional[str], message: str, message_type: str = 'info'):
        """Emit a narrative message for streaming to clients."""
        if not analysis_id or not AnalysisMessageService:
            return

        try:
            print(f"[WORKFLOW] Streaming message: {message[:80]}")

            # Use the stored event loop (FastAPI's main loop)
            if self.event_loop:
                # Schedule the coroutine in the main event loop (don't wait for result)
                future = asyncio.run_coroutine_threadsafe(
                    AnalysisMessageService.emit_message(analysis_id, message, message_type),
                    self.event_loop
                )
                # Wait for the message to actually be sent before continuing
                import time
                time.sleep(0.1)  # 100ms delay to ensure message is flushed

                # Optional: wait for the future to complete (with timeout)
                try:
                    future.result(timeout=1.0)  # Wait max 1 second
                    print(f"[WORKFLOW] Message sent successfully")
                except Exception as e:
                    print(f"[WORKFLOW] Message send failed: {e}")
            else:
                # Fallback: create temporary loop (shouldn't happen in production)
                print("[WORKFLOW WARNING] No event loop provided, using fallback")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        AnalysisMessageService.emit_message(analysis_id, message, message_type)
                    )
                finally:
                    loop.close()

            print(f"[WORKFLOW] Message sent successfully")
        except Exception as e:
            print(f"[WORKFLOW ERROR] Failed to emit message: {e}")
            import traceback
            traceback.print_exc()

    def _build_workflow(self) -> StateGraph:
        """Build the business discovery workflow."""

        workflow = StateGraph(BusinessDiscoveryState)

        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("understand_business", self._understand_business_node)
        workflow.add_node("explore_dynamically", self._explore_dynamically_node)
        workflow.add_node("run_analytics", self._run_advanced_analytics_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("synthesize_insights", self._synthesize_insights_node)
        workflow.add_node("create_recommendations", self._create_recommendations_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define flow
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "understand_business")
        workflow.add_edge("understand_business", "explore_dynamically")
        workflow.add_edge("explore_dynamically", "run_analytics")
        workflow.add_edge("run_analytics", "generate_insights")
        workflow.add_edge("generate_insights", "synthesize_insights")
        workflow.add_edge("synthesize_insights", "create_recommendations")
        workflow.add_edge("create_recommendations", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def run_discovery(self,
                     company_id: str,
                     dataset_ids: List[str],
                     analysis_name: str = "Business Analysis",
                     analysis_id: str = None) -> BusinessDiscoveryState:
        """Run the business discovery workflow."""

        print("\n" + "="*80)
        print("BUSINESS DISCOVERY WORKFLOW")
        print("="*80)
        print(f"Company ID: {company_id}")
        print(f"Datasets: {len(dataset_ids)}")
        print(f"Analysis: {analysis_name}")
        if analysis_id:
            print(f"Analysis ID: {analysis_id}")

        initial_state = BusinessDiscoveryState(
            company_id=company_id,
            dataset_ids=dataset_ids,
            analysis_id=analysis_id,  # NEW: Pass analysis_id through state
            datasets={},
            metadata={},
            business_context={},
            exploration_results={},
            insights=[],
            recommendations=[],
            executive_summary="",
            report_path=None,
            status="running",
            error=None
        )

        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            print(f"[ERROR] Workflow failed: {e}")
            initial_state['status'] = 'error'
            initial_state['error'] = str(e)
            return initial_state

    def _load_data_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Load selected datasets into memory for analysis.

        If state['dataset_ids'] is provided and non-empty, only those datasets are
        loaded. Otherwise, all company datasets are loaded (legacy behavior).
        """

        print("\n[STEP 1] Loading Data from Database")
        print("-" * 40)

        # Stream message: starting
        self._emit_message(
            state.get('analysis_id'),
            'Loading datasets from database...',
            'thinking'
        )

        try:
            selected_ids = state.get('dataset_ids') or []

            if selected_ids:
                # Load only selected datasets
                from src.database.repository import DatasetRepository
                from src.database.connection import DatabaseManager

                self.explorer.datasets = {}
                self.explorer.metadata = {}

                with DatabaseManager.get_session() as session:
                    for ds_id in selected_ids:
                        ds = DatasetRepository.get_dataset_by_id(session, ds_id)
                        if not ds:
                            continue
                        try:
                            df = DatasetRepository.load_dataframe(session, ds_id)
                            if df is not None:
                                self.explorer.datasets[ds.table_name] = df
                                self.explorer.metadata[ds.table_name] = {
                                    'domain': ds.domain,
                                    'description': ds.description,
                                    'entities': ds.entities,
                                    'row_count': ds.row_count,
                                    'column_count': ds.column_count
                                }
                        except Exception as e:
                            print(f"  [WARN] Failed to load dataset {ds_id}: {e}")

                datasets = self.explorer.datasets
                state['datasets'] = datasets
                state['metadata'] = self.explorer.metadata
            else:
                # Fallback: load all company datasets
                datasets = self.explorer.load_datasets(state['company_id'])
                state['datasets'] = datasets
                state['metadata'] = self.explorer.metadata

            for name, df in datasets.items():
                print(f"  [OK] {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

            # Stream message: found datasets
            total_rows = sum(df.shape[0] for df in datasets.values())
            self._emit_message(
                state.get('analysis_id'),
                f'Found {len(datasets)} dataset(s) with {total_rows:,} total rows.',
                'info'
            )

            # Create analysis directory early so viz_data.json is stored there
            company_id = state['company_id']
            analysis_id = state.get('analysis_id')
            analysis_dir = os.path.join("data", "outputs", "analyses", company_id, analysis_id)
            os.makedirs(analysis_dir, exist_ok=True)

            # Derive company name from DB (fallback to ID) and sanitize for filenames
            from src.database.connection import DatabaseManager as _DBM
            from src.database.repository import CompanyRepository as _CR
            company_name_val = None
            try:
                with _DBM.get_session() as _session:
                    _company = _CR.get_company_by_id(_session, company_id)
                    company_name_val = _company.name if _company else None
            except Exception:
                company_name_val = None

            if not company_name_val:
                company_name_val = company_id

            company_slug = ''.join(
                ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in company_name_val.lower().replace(' ', '_')
            )

            # Compute a simple domain label from selected datasets' metadata
            meta_values = state.get('metadata', {}) or {}
            domain_values = [m.get('domain') for m in meta_values.values() if isinstance(m, dict) and m.get('domain')]
            unique_domains = list({d for d in domain_values if d})
            if len(unique_domains) == 0:
                domain_label = 'Unknown'
            elif len(unique_domains) == 1:
                domain_label = unique_domains[0]
            else:
                domain_label = 'Multiple'

            # Initialize visualization data store for this analysis
            self.viz_data_store = VisualizationDataStore(
                dataset_name=f"business_discovery_{company_slug}",
                dataset_context={
                    'company_id': state['company_id'],
                    'company_name': company_name_val,
                    'datasets': list(datasets.keys()),
                    'domain': domain_label,
                    'dataset_type': 'Business Discovery'
                },
                output_dir=analysis_dir
            )
            state['viz_data_path'] = str(self.viz_data_store.json_path)

            # Load relationships
            with DatabaseManager.get_session() as session:
                from src.database.repository import RelationshipRepository

                relationships = RelationshipRepository.get_relationships_for_datasets(
                    session,
                    state['dataset_ids'],
                    min_confidence=0.7
                )

                state['metadata']['relationships'] = [
                    {
                        'from': rel.from_column,
                        'to': rel.to_column,
                        'type': rel.relationship_type,
                        'confidence': rel.confidence
                    }
                    for rel in relationships
                ]

                print(f"\n  Found {len(relationships)} relationships between tables")

                # Stream message: relationships found
                if len(relationships) > 0:
                    self._emit_message(
                        state.get('analysis_id'),
                        f'Discovered {len(relationships)} relationships between tables.',
                        'insight'
                    )

        except Exception as e:
            state['error'] = f"Failed to load data: {e}"
            state['status'] = 'error'
            # Stream error message
            self._emit_message(
                state.get('analysis_id'),
                f'Error loading data: {str(e)}',
                'error'
            )

        return state

    def _understand_business_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Understand the business context from the data."""

        print("\n[STEP 2] Understanding Business Context")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Analyzing business context and understanding your data domain...',
            'thinking'
        )

        if not self.model:
            print("  [WARN] No LLM configured - using basic analysis")
            state['business_context'] = self._basic_business_understanding(state)
            return state

        try:
            # Build context for LLM
            context = self._build_business_context(state)

            prompt = f"""Analyze this data and identify:
1. What domain or type of analysis does this data represent?
2. What appear to be the key drivers or important factors?
3. What are potential areas requiring deeper investigation?
4. What analytical questions should we explore?

DATA CONTEXT:
{context}

IMPORTANT: For key_questions, provide simple analytical questions based ONLY on the actual data structure shown above.
Do NOT assume any specific business domain (like sales, customers, products, etc.).

Good examples (adapt to actual data):
- "Which groups/categories show the highest values?"
- "What are the trends over time in the key metrics?"
- "Are there any unusual patterns or outliers?"

Bad examples (avoid assuming column names):
- "What drives 'churn_risk' in the customer_profiles table?"
- "How does marketing_spend correlate with revenue?"

Provide a structured analysis in JSON format with keys:
business_type, key_drivers, challenges, key_questions"""

            response = self.model.generate_content(prompt)
            text = response.text

            # Extract JSON
            if '```json' in text:
                json_str = text.split('```json')[1].split('```')[0].strip()
            else:
                json_str = text

            try:
                business_context = json.loads(json_str)
            except:
                # Fallback if JSON parsing fails - use generic context
                business_context = {
                    'business_type': f'Data Analysis ({len(state["datasets"])} datasets)',
                    'key_drivers': ['Data patterns', 'Significant trends'],
                    'challenges': ['Requires deeper analysis'],
                    'key_questions': self.explorer.generate_business_questions()
                }

            state['business_context'] = business_context

            print(f"  Business Type: {business_context.get('business_type', 'Unknown')}")
            # Display key drivers (could be revenue_drivers, key_drivers, or other variants from LLM)
            drivers = business_context.get('key_drivers') or business_context.get('revenue_drivers') or business_context.get('drivers', [])
            if drivers:
                print(f"  Key Drivers: {', '.join(drivers)}")

            # Emit progress: completed
            self._emit_message(
                state.get('analysis_id'),
                f'Business context understood: {business_context.get("business_type", "Data Analysis")}',
                'info'
            )

        except Exception as e:
            print(f"  [ERROR] Business understanding failed: {e}")
            state['business_context'] = self._basic_business_understanding(state)
            # Still emit completed even if fallback was used
            self._emit_message(
                state.get('analysis_id'),
                f'Business context understood: {business_context.get("business_type", "Data Analysis")}',
                'info'
            )

        return state

    def _explore_dynamically_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Let the LLM explore the data dynamically."""

        print("\n[STEP 3] Dynamic Data Exploration")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Exploring data patterns and relationships...',
            'thinking'
        )

        # Always run basic exploration first
        state['exploration_results'] = self._basic_exploration(state)

        if not self.model:
            print("  [WARN] No LLM configured - using basic exploration only")
            return state

        try:
            # Get key questions to explore - use basic ones if context generation failed
            questions = state['business_context'].get('key_questions', [])
            if not questions:
                questions = [
                    "Who are the top revenue generating customers?",
                    "What products have the highest profit margins?",
                    "Which customers are at highest risk of churning?",
                    "What is the marketing ROI trend?"
                ]
            questions = questions[:4]  # Limit to 4 questions

            print(f"  Exploring {len(questions)} business questions...")

            # Let LLM explore each question
            all_executions = []
            all_insights = []

            for i, question in enumerate(questions, 1):
                print(f"\n  Question {i}: {question[:80]}...")

                # Stream message for current question
                self._emit_message(
                    state.get('analysis_id'),
                    f'Analyzing: {question[:100]}...',
                    'thinking'
                )

                # Note: BusinessAnalystLLM handles code generation with dynamic examples
                # The hardcoded exploration_prompt was removed because BusinessAnalystLLM generates better prompts

                try:
                    # Use the BusinessAnalystLLM instead of raw model
                    print(f"    [DEBUG] Using BusinessAnalystLLM for: {question}")
                    result = self.business_analyst.analyze_business_question(
                        question=question,
                        datasets=state['datasets'],
                        max_attempts=2,
                        viz_data_store=self.viz_data_store
                    )

                    if result['success']:
                        print(f"    [DEBUG] BusinessAnalystLLM succeeded")
                        code_blocks = [result['code']]

                        # Store the successful result with correct keys
                        all_insights.append({
                            'question': question,
                            'insight': result.get('output', ''),  # Changed from 'finding' to 'insight'
                            'code': result['code'],
                            'evidence': result.get('output', '')  # Changed from 'output' to 'evidence'
                        })
                        continue  # Skip the rest of the loop since we have success
                    else:
                        print(f"    [ERROR] BusinessAnalystLLM failed: {result.get('error')}")
                        code_blocks = []

                    if not code_blocks:
                        # Try to extract any code-like content
                        print(f"    [WARN] No code blocks found in LLM response")
                        continue

                    # Create executor with all datasets
                    executor = MultiTableCodeExecutor(state['datasets'])

                    for code in code_blocks:
                        execution = executor.execute(code)
                        all_executions.append(execution)

                        if execution.error:
                            print(f"    [ERROR] Code execution failed: {execution.error[:100]}")
                            # Log the failing code for debugging
                            if execution.error and 'name' in execution.error and 'not defined' in execution.error:
                                print(f"    [DEBUG] Failed code snippet:")
                                code_lines = code.split('\n')[:10]  # Show first 10 lines
                                for line in code_lines:
                                    if line.strip():
                                        print(f"      {line[:80]}")
                        else:
                            print(f"    [OK] Executed analysis code successfully")

                            # Extract insights from the output
                            if execution.stdout:
                                # Look for FINDING: or specific numbers in output
                                lines = execution.stdout.strip().split('\n')
                                for line in lines:
                                    if 'FINDING:' in line or '$' in line or any(char.isdigit() for char in line):
                                        all_insights.append({
                                            'question': question,
                                            'insight': line.strip(),
                                            'code': code,
                                            'evidence': execution.stdout
                                        })
                                        break

                except Exception as e:
                    print(f"    [ERROR] LLM exploration failed: {str(e)[:100]}")

            state['exploration_results'] = {
                'executions': all_executions,
                'raw_insights': all_insights
            }

            print(f"\n  Completed {len(all_executions)} analyses")
            print(f"  Found {len(all_insights)} potential insights")

            # Stream completion message
            self._emit_message(
                state.get('analysis_id'),
                f'Exploration complete. Identified {len(all_insights)} data patterns for analysis.',
                'success'
            )

        except Exception as e:
            import traceback
            print(f"  [ERROR] Dynamic exploration failed: {e}")
            print(f"  [DEBUG] Traceback: {traceback.format_exc()[:500]}")
            state['exploration_results'] = {}
            # Emit progress: completed with error
            self._emit_message(
                state.get('analysis_id'),
                f'Exploration complete. Continuing with analysis...',
                'success'
            )

        return state

    def _run_advanced_analytics_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Run advanced analytics modules on the datasets."""

        print("\n[STEP 4] Running Advanced Analytics")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Running advanced analytics: anomaly detection, forecasting, causal analysis...',
            'thinking'
        )

        analytics_results = {}

        try:
            datasets = state['datasets']

            # 1. Anomaly Detection
            print("\n  Running Anomaly Detection...")
            anomaly_detector = AnomalyDetector()
            anomaly_results = []

            for table_name, df in datasets.items():
                # Find numeric columns for anomaly detection
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                for col in numeric_cols:
                    if len(df[col].dropna()) > 10:  # Need enough data points
                        try:
                            result = anomaly_detector.detect_anomalies(
                                df[col],
                                dataset_name=f"{table_name}.{col}"
                            )
                            if result and result.results and result.results.get('anomalies'):
                                anomalies = result.results['anomalies']
                                if len(anomalies) > 0:
                                    anomaly_results.append({
                                        'table': table_name,
                                        'column': col,
                                        'num_anomalies': len(anomalies),
                                        'anomaly_rate': result.results.get('anomaly_rate', 0),
                                        'detection_method': result.results.get('detection_method', 'unknown'),
                                        'anomalies_sample': anomalies[:5]  # First 5 anomalies
                                    })
                                    print(f"    [OK] Found {len(anomalies)} anomalies in {table_name}.{col}")
                        except Exception as e:
                            print(f"    [WARN] Anomaly detection failed for {table_name}.{col}: {str(e)[:50]}")

            analytics_results['anomalies'] = anomaly_results

            # 2. Time Series Forecasting
            print("\n  Running Time Series Analysis...")
            forecaster = TimeSeriesForecaster()
            forecast_results = []

            for table_name, df in datasets.items():
                # Look for date/time columns
                date_cols = df.select_dtypes(include=['datetime64', 'object']).columns

                for date_col in date_cols:
                    # Try to parse as datetime if string
                    try:
                        if df[date_col].dtype == 'object':
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

                        if df[date_col].dtype != 'object' and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                            # Found a date column, look for metrics to forecast
                            numeric_cols = df.select_dtypes(include=['number']).columns

                            for metric_col in numeric_cols:
                                if len(df[metric_col].dropna()) > 20:  # Need enough history
                                    try:
                                        # Create clean data without NaN values
                                        clean_df = df[[date_col, metric_col]].dropna()

                                        # Aggregate by date if needed
                                        if len(clean_df) > 0:
                                            ts_data = clean_df.groupby(clean_df[date_col].dt.date)[metric_col].sum()
                                        else:
                                            continue

                                        if len(ts_data) > 10:
                                            # Pass as Series with datetime index
                                            result = forecaster.forecast(
                                                data=ts_data,
                                                periods=7,
                                                dataset_name=f"{table_name}.{metric_col}"
                                            )
                                            if result and result.results and result.results.get('predictions'):
                                                forecast_results.append({
                                                    'table': table_name,
                                                    'metric': metric_col,
                                                    'date_column': date_col,
                                                    'forecast_periods': result.results.get('forecast_horizon', 7),
                                                    'forecast_values': result.results['predictions'],
                                                    'timestamps': result.results.get('timestamps', []),
                                                    'model_name': result.results.get('model_name', 'unknown')
                                                })
                                                print(f"    [OK] Generated 7-day forecast for {table_name}.{metric_col}")
                                    except Exception as e:
                                        print(f"    [WARN] Forecasting failed for {table_name}.{metric_col}: {str(e)[:50]}")
                    except:
                        continue

            analytics_results['forecasts'] = forecast_results

            # 3. Causal Analysis
            print("\n  Running Causal Analysis...")
            causal_analyzer = CausalAnalyzer()
            causal_results = []

            # Look for potential cause-effect relationships
            for table_name, df in datasets.items():
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                if len(numeric_cols) >= 2:
                    # Test relationships between numeric columns
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            if len(df[col1].dropna()) > 10 and len(df[col2].dropna()) > 10:
                                try:
                                    # Use analyze_causality method
                                    result = causal_analyzer.analyze_causality(
                                        cause=df[col1],
                                        effect=df[col2],
                                        dataset_name=f"{table_name}"
                                    )
                                    if result and result.results and result.results.get('relationships'):
                                        # Check if any significant relationships found
                                        for rel in result.results['relationships']:
                                            if rel.get('is_significant', False):
                                                causal_results.append({
                                                    'table': table_name,
                                                    'cause': col1,
                                                    'effect': col2,
                                                    'p_value': rel.get('p_value', 1.0),
                                                    'strength': rel.get('strength', 'unknown'),
                                                    'is_significant': True
                                                })
                                                print(f"    [OK] Found causal relationship: {col1} -> {col2} (strength={rel.get('strength', 'unknown')})")
                                                break
                                except Exception as e:
                                    continue

            analytics_results['causal_relationships'] = causal_results

            # 4. Variance Decomposition (for metrics with categories)
            print("\n  Running Variance Decomposition...")
            variance_decomposer = VarianceDecomposer()
            variance_results = []

            for table_name, df in datasets.items():
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                for metric in numeric_cols:
                    for category in categorical_cols:
                        if df[category].nunique() < 20 and len(df[metric].dropna()) > 10:  # Reasonable number of categories
                            try:
                                # Create X (features) and y (target)
                                X = pd.get_dummies(df[[category]], drop_first=True)
                                y = df[metric]

                                # Remove NaN values
                                mask = ~y.isna()
                                X_clean = X[mask]
                                y_clean = y[mask]

                                if len(y_clean) > 10:
                                    result = variance_decomposer.decompose(
                                        X=X_clean,
                                        y=y_clean,
                                        dataset_name=f"{table_name}",
                                        feature_names=[category]
                                    )
                                    if result and result.results and result.results.get('feature_contributions'):
                                        total_variance = result.results.get('total_variance_explained', 0)
                                        if total_variance > 0:
                                            variance_results.append({
                                                'table': table_name,
                                                'metric': metric,
                                                'factor': category,
                                                'variance_explained': float(total_variance),
                                                'contributions': result.results['feature_contributions'][:5]  # Top 5 contributions
                                            })
                                            print(f"    [OK] {category} explains {total_variance:.1%} of {metric} variance")
                            except Exception as e:
                                continue

            analytics_results['variance_decomposition'] = variance_results

            # Store results in state
            state['analytics_results'] = analytics_results

            # Summary
            print(f"\n  Analytics Summary:")
            print(f"    - Anomalies detected: {len(anomaly_results)}")
            print(f"    - Forecasts generated: {len(forecast_results)}")
            print(f"    - Causal relationships: {len(causal_results)}")
            print(f"    - Variance analyses: {len(variance_results)}")

            # Emit progress: completed
            self._emit_message(
                state.get('analysis_id'),
                f'Advanced analytics complete. Detected {len(anomaly_results)} anomalies, generated {len(forecast_results)} forecasts.',
                'success'
            )

        except Exception as e:
            import traceback
            print(f"  [ERROR] Advanced analytics failed: {e}")
            print(f"  [DEBUG] Traceback: {traceback.format_exc()[:500]}")
            state['analytics_results'] = {}
            # Emit progress: completed with errors
            self._emit_message(
                state.get('analysis_id'),
                f'Advanced analytics complete. Detected {len(anomaly_results)} anomalies, generated {len(forecast_results)} forecasts.',
                'success'
            )

        return state

    def _generate_insights_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Generate business insights from exploration and analytics results."""

        print("\n[STEP 5] Generating Business Insights")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Discovering key insights and trends in your data...',
            'insight'
        )

        raw_insights = state['exploration_results'].get('raw_insights', [])
        analytics_results = state.get('analytics_results', {})

        # Process ONLY exploration insights as true "business insights"
        business_insights = []
        for raw in raw_insights:
            insight = self._convert_to_business_insight(raw)
            if insight:
                business_insights.append(insight)

        # Count analytics results separately (these are technical findings, not insights)
        num_anomalies = len([a for a in analytics_results.get('anomalies', []) if a.get('num_anomalies', 0) > 0])
        num_forecasts = len([f for f in analytics_results.get('forecasts', []) if f.get('forecast_values')])
        num_relationships = len([c for c in analytics_results.get('causal_relationships', []) if c.get('is_significant')])
        num_drivers = len([v for v in analytics_results.get('variance_decomposition', []) if v.get('variance_explained', 0) > 0.2])

        if business_insights:
            print(f"  [OK] Generated {len(business_insights)} business insights")
            for insight in business_insights:
                print(f"    - {insight['title']}")
        else:
            print("  No insights discovered")

        print(f"  [OK] Analytics results: {num_anomalies} anomalies, {num_forecasts} forecasts, {num_relationships} relationships, {num_drivers} drivers")

        # Stream completion message - separate insights from analytics
        analytics_summary = []
        if num_anomalies > 0:
            analytics_summary.append(f"{num_anomalies} anomalies")
        if num_forecasts > 0:
            analytics_summary.append(f"{num_forecasts} forecasts")
        if num_relationships > 0:
            analytics_summary.append(f"{num_relationships} causal relationships")
        if num_drivers > 0:
            analytics_summary.append(f"{num_drivers} key drivers")

        analytics_text = ", ".join(analytics_summary) if analytics_summary else "no additional analytics"

        self._emit_message(
            state.get('analysis_id'),
            f'Generated {len(business_insights)} business insights with {analytics_text}.',
            'success'
        )

        state['insights'] = business_insights
        return state

    def _synthesize_insights_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Synthesize insights into meaningful business narratives using LLM."""

        print("\n[STEP 6] Synthesizing Business Narratives")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Synthesizing insights into actionable business narratives...',
            'thinking'
        )

        insights = state['insights']
        analytics = state.get('analytics_results', {})
        business_context = state['business_context']

        if not insights:
            print("  No insights to synthesize")
            state['synthesized_insights'] = []
            return state

        # Prepare insights summary for LLM
        insights_summary = self._prepare_insights_for_synthesis(insights, analytics)

        # Use LLM to synthesize narratives
        synthesis_prompt = f"""You are a senior business analyst reviewing analysis results for a {business_context.get('business_type', 'company')}.

RAW INSIGHTS AND DATA:
{insights_summary}

YOUR TASK:
Read all the insights and data above. Create 3-5 HIGH-VALUE BUSINESS NARRATIVES that:
1. CONNECT multiple data points into a coherent story
2. Identify ROOT CAUSES and RELATIONSHIPS between different findings
3. Provide ACTIONABLE BUSINESS IMPLICATIONS
4. Focus on what matters most to business outcomes

DO NOT just repeat the raw stats. Instead, SYNTHESIZE them into insights that answer "SO WHAT?" and "WHY DOES THIS MATTER?"

EXAMPLE OF GOOD SYNTHESIS:
Instead of:
- "307 customers have high churn risk"
- "High churn customers are in Tech and Retail"
- "High churn customers have low engagement"

Write:
"CRITICAL CHURN RISK PATTERN: 61% of our customer base (307 companies) shows high churn risk, concentrated heavily in Tech (31%) and Retail (21%) industries. These at-risk customers exhibit significantly lower engagement scores (18 vs 71 for healthy customers) and lower lifetime value ($17K vs $95K). This pattern suggests our product-market fit is weakening specifically in Tech and Retail sectors, likely due to insufficient industry-specific engagement strategies. Immediate action needed to prevent ~$5.2M revenue loss."

For each narrative, provide:
1. TITLE: Concise, action-oriented title
2. NARRATIVE: The synthesized story (2-4 sentences) connecting multiple data points
3. BUSINESS_IMPACT: What this means for the business (revenue, growth, risk)
4. PRIORITY: High/Medium/Low based on urgency and impact

Return your response as a JSON array:
```json
[
  {{
    "title": "...",
    "narrative": "...",
    "business_impact": "...",
    "priority": "High/Medium/Low",
    "connected_insights": ["insight_title_1", "insight_title_2"]
  }}
]
```
"""

        try:
            response = self.business_analyst.model.generate_content(synthesis_prompt)
            synthesized = self._parse_synthesis_response(response.text)

            if synthesized:
                print(f"  [OK] Created {len(synthesized)} synthesized narratives")
                for syn in synthesized:
                    print(f"    [{syn.get('priority', 'Medium')}] {syn['title']}")
            else:
                print("  [WARN] LLM synthesis failed, using raw insights")
                synthesized = insights  # Fallback to raw insights

            state['synthesized_insights'] = synthesized

        except Exception as e:
            print(f"  [ERROR] Synthesis failed: {str(e)[:100]}")
            print("  Using raw insights as fallback")
            state['synthesized_insights'] = insights

        # Stream completion message
        self._emit_message(
            state.get('analysis_id'),
            f'Created {len(state["synthesized_insights"])} business narratives.',
            'success'
        )

        return state

    def _prepare_insights_for_synthesis(self, insights: List[Dict], analytics: Dict) -> str:
        """Prepare insights summary for LLM synthesis."""

        summary_parts = []

        # Group insights by category
        exploration_insights = [i for i in insights if i.get('type') == 'discovery']
        analytics_insights = [i for i in insights if i.get('type') == 'analytics']

        if exploration_insights:
            summary_parts.append("EXPLORATION FINDINGS:")
            for insight in exploration_insights:
                summary_parts.append(f"- {insight['title']}: {insight['finding'][:200]}")

        if analytics_insights:
            summary_parts.append("\nANALYTICS FINDINGS:")
            for insight in analytics_insights:
                summary_parts.append(f"- {insight['title']}: {insight['finding'][:200]}")

        if analytics:
            summary_parts.append(f"\nANALYTICS SUMMARY:")
            summary_parts.append(f"- Anomalies detected: {len(analytics.get('anomalies', []))}")
            summary_parts.append(f"- Causal relationships: {len(analytics.get('causal_relationships', []))}")
            summary_parts.append(f"- Variance analyses: {len(analytics.get('variance_decomposition', []))}")

        return "\n".join(summary_parts)

    def _parse_synthesis_response(self, response_text: str) -> List[Dict]:
        """Parse LLM synthesis response."""
        import re
        import json

        # Try to extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try to parse as raw JSON
        try:
            return json.loads(response_text)
        except:
            return []

    def _create_recommendations_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Create actionable recommendations from synthesized insights."""

        print("\n[STEP 7] Creating Recommendations")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Creating actionable business recommendations...',
            'recommendation'
        )

        # Use synthesized insights if available, otherwise fall back to raw insights
        insights = state.get('synthesized_insights', state.get('insights', []))
        if not insights:
            print("  No insights to base recommendations on")
            return state

        recommendations = []

        for insight in insights:
            recs = self._generate_recommendations(insight, state)
            recommendations.extend(recs)

        # Prioritize recommendations
        recommendations.sort(key=lambda x: x.get('impact', 0), reverse=True)

        state['recommendations'] = recommendations[:10]  # Top 10

        for rec in state['recommendations'][:5]:
            print(f"  • {rec['action']}")
            if 'impact' in rec:
                print(f"    Impact: {rec['impact']}")

        # Stream completion message
        self._emit_message(
            state.get('analysis_id'),
            f'Created {len(state["recommendations"])} actionable recommendations.',
            'success'
        )

        return state

    def _generate_report_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Generate the final business report."""

        print("\n[STEP 8] Generating Business Report")
        print("-" * 40)

        # Emit progress: starting
        self._emit_message(
            state.get('analysis_id'),
            'Generating comprehensive report and interactive dashboard...',
            'narrative'
        )

        # Get company_id and analysis_id from state
        company_id = state.get('company_id')
        analysis_id = state.get('analysis_id', f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Create company-specific directory structure
        # Format: data/outputs/analyses/{company_id}/{analysis_id}/
        analysis_dir = os.path.join("data", "outputs", "analyses", company_id, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)

        # Generate executive summary
        state['executive_summary'] = self._generate_executive_summary(state)

        # Write report to company-specific directory
        report_filename = "report.md"
        report_full_path = os.path.join(analysis_dir, report_filename)
        self._write_business_report(state, report_full_path)

        # Store RELATIVE path (for database storage)
        report_relative_path = os.path.join("analyses", company_id, analysis_id, report_filename)
        state['report_path'] = report_relative_path

        state['status'] = 'completed'

        print(f"  [OK] Report saved: {report_full_path}")
        print(f"  [OK] Relative path: {report_relative_path}")

        # Generate Plotly dashboard if viz_data was created
        dashboard_full_path = None
        if state.get('viz_data_path') and self.viz_data_store:
            try:
                # Specify output path in company-specific directory
                dashboard_filename = "dashboard.html"
                dashboard_full_path = os.path.join(analysis_dir, dashboard_filename)

                dashboard_path = self.dashboard_generator.generate_dashboard(
                    viz_data_json_path=state['viz_data_path'],
                    output_path=dashboard_full_path
                )

                # Store RELATIVE path (for database storage)
                dashboard_relative_path = os.path.join("analyses", company_id, analysis_id, dashboard_filename)
                state['dashboard_path'] = dashboard_relative_path

                print(f"  [OK] Dashboard generated: {dashboard_full_path}")
                print(f"  [OK] Relative path: {dashboard_relative_path}")
            except Exception as e:
                print(f"  [WARN] Dashboard generation failed: {e}")

        # Upload all outputs to storage (GCS in production, local in development)
        if StorageService:
            try:
                storage_service = StorageService()
                storage_type = "Google Cloud Storage" if storage_service.use_gcs else "local storage"
                print(f"\n  [STORAGE] Uploading analysis outputs to {storage_type}...")

                # Get local file paths
                viz_data_full_path = state.get('viz_data_path')  # Already has full path

                # Upload files to storage
                storage_paths = storage_service.upload_analysis_outputs(
                    company_id=company_id,
                    analysis_id=analysis_id,
                    report_local_path=report_full_path,
                    dashboard_local_path=dashboard_full_path,
                    viz_data_local_path=viz_data_full_path
                )

                # Update state with storage paths (overwrite local paths)
                if 'report_path' in storage_paths:
                    state['report_path'] = storage_paths['report_path']
                if 'dashboard_path' in storage_paths:
                    state['dashboard_path'] = storage_paths['dashboard_path']
                if 'viz_data_path' in storage_paths:
                    state['viz_data_path'] = storage_paths['viz_data_path']

                print(f"  [STORAGE] Upload complete to {storage_type}")
            except Exception as e:
                print(f"  [WARN] Storage upload failed (using local paths): {e}")
                # Keep local paths in state if storage upload fails

        # Emit progress: completed
        self._emit_message(
                state.get('analysis_id'),
                'Report and dashboard generation complete. Your analysis is ready!',
                'success'
            )

        return state

    # Helper methods

    def _build_business_context(self, state: BusinessDiscoveryState) -> str:
        """Build context about the business from the data."""

        context = []
        for name, df in state['datasets'].items():
            meta = state['metadata'].get(name, {})
            context.append(f"""
Table: {name}
- Rows: {df.shape[0]:,}
- Columns: {list(df.columns)[:10]}
- Domain: {meta.get('domain', 'Unknown')}
- Sample values: {df.head(2).to_dict()}""")

        return '\n'.join(context)

    def _basic_business_understanding(self, state: BusinessDiscoveryState) -> Dict:
        """Basic business understanding without LLM - domain-agnostic."""

        # Analyze data structure without assuming domain
        dataset_count = len(state['datasets'])
        total_rows = sum(len(df) for df in state['datasets'].values())

        # Identify data characteristics
        has_numeric_data = any(
            len(df.select_dtypes(include=['number']).columns) > 0
            for df in state['datasets'].values()
        )
        has_time_data = any(
            len(df.select_dtypes(include=['datetime64']).columns) > 0
            for df in state['datasets'].values()
        )

        return {
            'business_type': f'Data Analysis ({dataset_count} datasets, {total_rows} total rows)',
            'data_characteristics': {
                'has_numeric_data': has_numeric_data,
                'has_time_series': has_time_data
            },
            'challenges': ['Requires deeper analysis to understand domain'],
            'key_questions': self.explorer.generate_business_questions()
        }

    def _basic_exploration(self, state: BusinessDiscoveryState) -> Dict:
        """Basic exploration without LLM."""

        executions = []

        # Try some basic analyses
        for name, df in state['datasets'].items():
            # Get basic stats
            code = f"""
# Basic analysis of {name}
print(f"Dataset: {name}")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")

# Numeric summary
numeric_cols = {name}.select_dtypes(include='number').columns
if len(numeric_cols) > 0:
    print(f"\\nNumeric summary:")
    print({name}[numeric_cols].describe())
"""
            execution = self.explorer.execute_code(code)
            executions.append(execution)

        return {'executions': executions, 'raw_insights': []}

    def _get_dataset_summary(self, state: BusinessDiscoveryState) -> str:
        """Get a summary of available datasets."""

        summary = []
        for name, df in state['datasets'].items():
            summary.append(f"- {name}: {df.shape}, columns: {list(df.columns)[:5]}...")

        return '\n'.join(summary)

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text."""
        import re
        blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        return blocks

    def _extract_insight_from_output(self, output: str, question: str) -> Optional[str]:
        """Extract insight from execution output."""

        # Look for patterns that indicate insights
        if any(keyword in output.lower() for keyword in ['top', 'highest', 'lowest', 'average', 'total']):
            # Extract the key finding
            lines = output.strip().split('\n')
            for line in lines:
                if any(char.isdigit() for char in line):  # Has numbers
                    return line.strip()

        return None

    def _convert_to_business_insight(self, raw_insight: Dict) -> Dict:
        """Convert raw insight to business insight."""

        return {
            'title': raw_insight['question'],
            'finding': raw_insight['insight'],
            'evidence': raw_insight['evidence'],
            'type': 'discovery',
            'confidence': 0.8
        }

    def _generate_recommendations(self, insight: Dict, state: BusinessDiscoveryState) -> List[Dict]:
        """Generate recommendations from an insight."""

        recommendations = []

        # Get text content from either synthesized insights or raw insights
        narrative = insight.get('narrative', '')
        finding = insight.get('finding', '')
        business_impact = insight.get('business_impact', '')
        priority = insight.get('priority', 'Medium')

        # Combine all text for pattern matching
        text = f"{narrative} {finding} {business_impact}".lower()
        title = insight.get('title', 'Unknown')

        # If no text content, skip
        if not text.strip():
            return recommendations

        # Pattern: High value/top performers
        if ('high' in text and 'value' in text) or ('top' in text and 'perform' in text) or 'enterprise' in text:
            recommendations.append({
                'action': f'Focus on high-value segments identified: {title}',
                'rationale': narrative or finding or 'Target high-value customer segments',
                'impact': priority,
                'urgency': 'Medium'
            })

        # Pattern: Decline or negative trend
        if any(word in text for word in ['decline', 'decrease', 'drop', 'falling', 'churn', 'loss']):
            recommendations.append({
                'action': f'Address decline: {title}',
                'rationale': narrative or finding or 'Investigate and reverse negative trends',
                'impact': 'High',
                'urgency': 'High'
            })

        # Pattern: Opportunity or growth
        if any(word in text for word in ['opportunit', 'grow', 'increas', 'expand', 'potential']):
            recommendations.append({
                'action': f'Capitalize on opportunity: {title}',
                'rationale': narrative or finding or 'Develop growth strategy',
                'impact': priority,
                'urgency': 'Medium'
            })

        # Pattern: Risk, anomaly, or data quality
        if any(word in text for word in ['risk', 'anomal', 'unusual', 'concern', 'data quality', 'integrity']):
            recommendations.append({
                'action': f'Investigate and mitigate: {title}',
                'rationale': narrative or finding or 'Monitor and address identified risks',
                'impact': 'High',
                'urgency': 'High'
            })

        # Pattern: Pricing and profitability
        if any(word in text for word in ['pric', 'profit', 'margin', 'discount', 'revenue']):
            recommendations.append({
                'action': f'Optimize pricing strategy: {title}',
                'rationale': narrative or finding or 'Review and optimize pricing and profitability',
                'impact': priority,
                'urgency': 'Medium'
            })

        # Pattern: Marketing and customer acquisition
        if any(word in text for word in ['marketing', 'traffic', 'customer acquisition', 'spend']):
            recommendations.append({
                'action': f'Optimize marketing investments: {title}',
                'rationale': narrative or finding or 'Refine marketing and customer acquisition strategy',
                'impact': priority,
                'urgency': 'Medium'
            })

        # Pattern: Customer segmentation
        if any(word in text for word in ['segment', 'industry', 'region', 'category', 'group']):
            recommendations.append({
                'action': f'Tailor strategy by segment: {title}',
                'rationale': narrative or finding or 'Develop segment-specific strategies',
                'impact': priority,
                'urgency': 'Low'
            })

        return recommendations

    def _generate_executive_summary(self, state: BusinessDiscoveryState) -> str:
        """Generate executive summary from synthesized insights."""

        summary = ["# Executive Summary\n"]

        # Business context
        context = state['business_context']
        summary.append(f"**Business Type:** {context.get('business_type', 'Unknown')}\n")
        summary.append(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n")

        # Use synthesized insights if available
        insights = state.get('synthesized_insights', state.get('insights', []))

        # Key narratives (synthesized insights)
        if insights:
            summary.append("\n## Key Business Narratives\n")
            for insight in insights[:5]:
                # Check if this is a synthesized insight with narrative
                if 'narrative' in insight:
                    priority = insight.get('priority', 'Medium')
                    summary.append(f"**[{priority}] {insight['title']}**\n")
                    summary.append(f"{insight['narrative']}\n")
                else:
                    # Fall back to raw insight format
                    summary.append(f"• {insight.get('finding', insight.get('title', 'N/A'))}\n")

        # Top recommendations
        if state['recommendations']:
            summary.append("\n## Priority Actions\n")
            for rec in state['recommendations'][:3]:
                summary.append(f"• {rec['action']}\n")

        return '\n'.join(summary)

    def _write_business_report(self, state: BusinessDiscoveryState, report_path: str):
        """Write the business report with synthesized narratives."""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(state['executive_summary'])

            f.write("\n\n## Detailed Analysis\n")

            # Use synthesized insights if available
            insights = state.get('synthesized_insights', state.get('insights', []))

            # Business Narratives section (synthesized insights)
            f.write("\n### Business Narratives\n")
            for i, insight in enumerate(insights, 1):
                if 'narrative' in insight:
                    # Synthesized insight format
                    priority = insight.get('priority', 'Medium')
                    f.write(f"\n{i}. **[{priority}] {insight['title']}**\n\n")
                    f.write(f"{insight['narrative']}\n\n")
                    if 'business_impact' in insight:
                        f.write(f"**Business Impact:** {insight['business_impact']}\n")
                else:
                    # Fall back to raw insight format
                    f.write(f"\n{i}. **{insight['title']}**\n")
                    f.write(f"   - {insight.get('finding', insight.get('description', 'N/A'))}\n")

            # Recommendations section
            f.write("\n### Recommendations\n\n")
            if state.get('recommendations'):
                for i, rec in enumerate(state['recommendations'], 1):
                    f.write(f"{i}. **{rec['action']}**\n")
                    if 'rationale' in rec and rec['rationale']:
                        # Truncate long rationales
                        rationale = rec['rationale'][:300] + '...' if len(rec['rationale']) > 300 else rec['rationale']
                        f.write(f"   - **Rationale**: {rationale}\n")
                    f.write(f"   - **Impact**: {rec.get('impact', 'Medium')}\n")
                    f.write(f"   - **Urgency**: {rec.get('urgency', 'Medium')}\n\n")
            else:
                f.write("*No specific recommendations generated. Review the business narratives above for strategic guidance.*\n")

            # Advanced Analytics section
            analytics_results = state.get('analytics_results', {})
            if analytics_results:
                f.write("\n## Advanced Analytics\n")
                f.write("\n*Powered by statistical analysis and machine learning models*\n\n")

                # Anomaly Detection Results
                anomalies = analytics_results.get('anomalies', [])
                if anomalies:
                    f.write("\n### Anomalies Detected\n")
                    f.write(f"\nFound {len(anomalies)} unusual patterns in your data:\n\n")
                    f.write("| Dataset | Column | Count | Anomaly Rate | Detection Method | Sample Values |\n")
                    f.write("|---------|--------|-------|--------------|------------------|---------------|\n")
                    for anomaly in anomalies[:20]:  # Limit to top 20
                        dataset = anomaly.get('table', 'N/A')
                        column = anomaly.get('column', 'N/A')
                        count = anomaly.get('num_anomalies', 'N/A')
                        rate = anomaly.get('anomaly_rate', 0)
                        method = anomaly.get('detection_method', 'N/A')

                        # Extract sample anomalous values
                        sample = anomaly.get('anomalies_sample', [])
                        if sample:
                            sample_values = ', '.join([f"{a.get('value', 'N/A'):.2f}" if isinstance(a.get('value'), (int, float)) else str(a.get('value', 'N/A')) for a in sample[:3]])
                        else:
                            sample_values = 'N/A'

                        rate_pct = f"{rate*100:.1f}%" if isinstance(rate, (int, float)) else str(rate)
                        f.write(f"| {dataset} | {column} | {count} | {rate_pct} | {method} | {sample_values} |\n")

                # Time Series Forecasts
                forecasts = analytics_results.get('forecasts', [])
                if forecasts:
                    f.write("\n### 7-Day Forecasts\n")
                    f.write(f"\nGenerated {len(forecasts)} forecasts for key metrics:\n\n")
                    for forecast in forecasts[:10]:  # Limit to 10 forecasts
                        table = forecast.get('table', 'N/A')
                        metric = forecast.get('metric', 'N/A')
                        model = forecast.get('model_name', 'N/A')

                        f.write(f"\n**{table}.{metric}** (Model: {model})\n\n")
                        f.write("| Day | Forecast Value |\n")
                        f.write("|-----|----------------|\n")

                        # forecast_values is a dict or array from the predictions
                        values = forecast.get('forecast_values', {})
                        if isinstance(values, dict):
                            predictions = values.get('yhat', []) if 'yhat' in values else []
                        elif isinstance(values, list):
                            predictions = values
                        else:
                            predictions = []

                        for i, val in enumerate(predictions[:7], 1):  # 7 days
                            if isinstance(val, (int, float)):
                                f.write(f"| Day {i} | {val:.2f} |\n")
                            else:
                                f.write(f"| Day {i} | {val} |\n")

                # Causal Relationships
                causal_relationships = analytics_results.get('causal_relationships', [])
                if causal_relationships:
                    f.write("\n### Causal Relationships\n")
                    f.write(f"\nIdentified {len(causal_relationships)} cause-effect relationships:\n\n")
                    f.write("| Cause | Effect | Strength | P-Value | Dataset |\n")
                    f.write("|-------|--------|----------|---------|----------|\n")
                    for rel in causal_relationships[:15]:  # Top 15
                        cause = rel.get('cause', 'N/A')
                        effect = rel.get('effect', 'N/A')
                        strength = rel.get('strength', 'N/A')
                        p_value = rel.get('p_value', 'N/A')
                        table = rel.get('table', 'N/A')

                        # Format p-value
                        if isinstance(p_value, (int, float)):
                            p_str = f"{p_value:.4f}"
                        else:
                            p_str = str(p_value)

                        f.write(f"| {cause} | {effect} | {strength} | {p_str} | {table} |\n")

                # Variance Decomposition
                variance_decomposition = analytics_results.get('variance_decomposition', [])
                if variance_decomposition:
                    f.write("\n### Variance Analysis\n")
                    f.write("\nWhat drives the variation in your key metrics:\n\n")
                    for decomp in variance_decomposition:
                        metric = decomp.get('metric', 'N/A')
                        f.write(f"\n**{metric}**\n\n")
                        f.write("| Component | Contribution | Percentage |\n")
                        f.write("|-----------|--------------|------------|\n")
                        components = decomp.get('components', [])
                        for comp in components:
                            name = comp.get('name', 'N/A')
                            contribution = comp.get('contribution', 'N/A')
                            percentage = comp.get('percentage', 'N/A')
                            f.write(f"| {name} | {contribution} | {percentage} |\n")

            # Data analyzed section
            f.write("\n## Data Analyzed\n")
            for name, df in state['datasets'].items():
                f.write(f"- **{name}**: {df.shape[0]:,} records\n")

            f.write("\n---\n")
            f.write("*This report was generated using dynamic business analysis.*\n")
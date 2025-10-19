"""Business-focused discovery workflow using dynamic LLM exploration."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import json
import pandas as pd
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


class BusinessDiscoveryState(Dict):
    """State for business discovery workflow."""
    company_id: str
    dataset_ids: List[str]
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

    def __init__(self):
        """Initialize the business discovery workflow."""
        self.explorer = DynamicDataExplorer()

        # Use the BusinessAnalystLLM for proper code generation
        from src.analytics.business_analyst_llm import BusinessAnalystLLM
        self.business_analyst = BusinessAnalystLLM()

        # Configure Gemini for business analysis
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
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
                     analysis_name: str = "Business Analysis") -> BusinessDiscoveryState:
        """Run the business discovery workflow."""

        print("\n" + "="*80)
        print("BUSINESS DISCOVERY WORKFLOW")
        print("="*80)
        print(f"Company ID: {company_id}")
        print(f"Datasets: {len(dataset_ids)}")
        print(f"Analysis: {analysis_name}")

        initial_state = BusinessDiscoveryState(
            company_id=company_id,
            dataset_ids=dataset_ids,
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
        """Load all datasets into memory for analysis."""

        print("\n[STEP 1] Loading Data from Database")
        print("-" * 40)

        try:
            # Load all datasets
            datasets = self.explorer.load_datasets(state['company_id'])
            state['datasets'] = datasets
            state['metadata'] = self.explorer.metadata

            for name, df in datasets.items():
                print(f"  [OK] {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

            # Initialize visualization data store for this analysis
            company_name = state.get('company_name', 'unknown_company')
            self.viz_data_store = VisualizationDataStore(
                dataset_name=f"business_discovery_{company_name}",
                dataset_context={
                    'company_id': state['company_id'],
                    'datasets': list(datasets.keys()),
                    'analysis_type': 'business_discovery'
                }
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

        except Exception as e:
            state['error'] = f"Failed to load data: {e}"
            state['status'] = 'error'

        return state

    def _understand_business_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Understand the business context from the data."""

        print("\n[STEP 2] Understanding Business Context")
        print("-" * 40)

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

        except Exception as e:
            print(f"  [ERROR] Business understanding failed: {e}")
            state['business_context'] = self._basic_business_understanding(state)

        return state

    def _explore_dynamically_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Let the LLM explore the data dynamically."""

        print("\n[STEP 3] Dynamic Data Exploration")
        print("-" * 40)

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

        except Exception as e:
            import traceback
            print(f"  [ERROR] Dynamic exploration failed: {e}")
            print(f"  [DEBUG] Traceback: {traceback.format_exc()[:500]}")
            state['exploration_results'] = {}

        return state

    def _run_advanced_analytics_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Run advanced analytics modules on the datasets."""

        print("\n[STEP 4] Running Advanced Analytics")
        print("-" * 40)

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

        except Exception as e:
            import traceback
            print(f"  [ERROR] Advanced analytics failed: {e}")
            print(f"  [DEBUG] Traceback: {traceback.format_exc()[:500]}")
            state['analytics_results'] = {}

        return state

    def _generate_insights_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Generate business insights from exploration and analytics results."""

        print("\n[STEP 5] Generating Business Insights")
        print("-" * 40)

        raw_insights = state['exploration_results'].get('raw_insights', [])
        analytics_results = state.get('analytics_results', {})

        # Combine insights from exploration and analytics
        all_insights = []

        # Process exploration insights
        for raw in raw_insights:
            insight = self._convert_to_business_insight(raw)
            if insight:
                all_insights.append(insight)

        # Process analytics insights
        if analytics_results:
            # Anomaly insights
            for anomaly in analytics_results.get('anomalies', []):
                if anomaly['num_anomalies'] > 0:
                    finding = f"Found {anomaly['num_anomalies']} unusual values in {anomaly['table']}.{anomaly['column']} that may indicate data quality issues or exceptional business events"
                    all_insights.append({
                        'title': f"Anomalies Detected in {anomaly['table']}.{anomaly['column']}",
                        'finding': finding,
                        'description': finding,
                        'evidence': anomaly,
                        'category': 'anomaly',
                        'impact': 'medium',
                        'type': 'analytics'
                    })

            # Forecast insights
            for forecast in analytics_results.get('forecasts', []):
                if forecast.get('forecast_values'):
                    finding = f"Projected trends for {forecast['metric']} based on historical patterns in {forecast['table']}"
                    all_insights.append({
                        'title': f"7-Day Forecast for {forecast['metric']}",
                        'finding': finding,
                        'description': finding,
                        'evidence': forecast,
                        'category': 'forecast',
                        'impact': 'high',
                        'type': 'analytics'
                    })

            # Causal insights
            for causal in analytics_results.get('causal_relationships', []):
                if causal.get('is_significant', False):
                    finding = f"Statistical analysis reveals a {causal.get('strength', 'significant')} causal relationship between {causal['cause']} and {causal['effect']} (p-value: {causal.get('p_value', 0):.4f})"
                    all_insights.append({
                        'title': f"Significant relationship: {causal['cause']} -> {causal['effect']}",
                        'finding': finding,
                        'description': finding,
                        'evidence': causal,
                        'category': 'relationship',
                        'impact': 'high' if causal.get('strength') == 'strong' else 'medium',
                        'type': 'analytics'
                    })

            # Variance insights
            for variance in analytics_results.get('variance_decomposition', []):
                if variance['variance_explained'] > 0.2:  # Explains >20% of variance
                    finding = f"{variance['factor']} explains {variance['variance_explained']:.1%} of the variance in {variance['metric']}"
                    all_insights.append({
                        'title': f"{variance['factor']} drives {variance['metric']} variation",
                        'finding': finding,
                        'description': finding,
                        'evidence': variance,
                        'category': 'driver',
                        'impact': 'high' if variance['variance_explained'] > 0.5 else 'medium',
                        'type': 'analytics'
                    })

        if all_insights:
            print(f"  [OK] Generated {len(all_insights)} insights")
            for insight in all_insights:
                print(f"    - {insight['title']}")
        else:
            print("  No insights discovered")

        state['insights'] = all_insights
        return state

    def _synthesize_insights_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Synthesize insights into meaningful business narratives using LLM."""

        print("\n[STEP 6] Synthesizing Business Narratives")
        print("-" * 40)

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

        return state

    def _generate_report_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Generate the final business report."""

        print("\n[STEP 8] Generating Business Report")
        print("-" * 40)

        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(
            "data", "outputs", "business_discovery",
            f"analysis_{timestamp}"
        )
        os.makedirs(report_dir, exist_ok=True)

        # Generate executive summary
        state['executive_summary'] = self._generate_executive_summary(state)

        # Write report
        report_path = os.path.join(report_dir, "business_report.md")
        self._write_business_report(state, report_path)

        state['report_path'] = report_path
        state['status'] = 'completed'

        print(f"  [OK] Report saved: {report_path}")

        # Generate Plotly dashboard if viz_data was created
        if state.get('viz_data_path') and self.viz_data_store:
            try:
                dashboard_path = self.dashboard_generator.generate_dashboard(
                    viz_data_json_path=state['viz_data_path'],
                    output_path=None  # Will auto-generate path
                )
                state['dashboard_path'] = dashboard_path
                print(f"  [OK] Dashboard generated: {dashboard_path}")
            except Exception as e:
                print(f"  [WARN] Dashboard generation failed: {e}")

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

        # Generic recommendations based on analytical patterns, not domain keywords
        finding = insight.get('finding', '').lower()

        # Pattern: High value/top performers
        if ('high' in finding or 'top' in finding) and ('value' in finding or 'perform' in finding):
            recommendations.append({
                'action': 'Focus efforts on maintaining and expanding top performers identified',
                'rationale': insight['finding'],
                'impact': 'High',
                'urgency': 'Medium'
            })

        # Pattern: Decline or negative trend
        if 'decline' in finding or 'decrease' in finding or 'drop' in finding:
            recommendations.append({
                'action': 'Investigate root causes of decline and develop corrective action plan',
                'rationale': insight['finding'],
                'impact': 'High',
                'urgency': 'High'
            })

        # Pattern: Opportunity or growth
        if 'opportunit' in finding or 'grow' in finding or 'increas' in finding:
            recommendations.append({
                'action': 'Develop strategy to capitalize on growth opportunity identified',
                'rationale': insight['finding'],
                'impact': 'Medium',
                'urgency': 'Medium'
            })

        # Pattern: Risk or anomaly
        if 'risk' in finding or 'anomal' in finding or 'unusual' in finding:
            recommendations.append({
                'action': 'Monitor and mitigate identified risks through targeted interventions',
                'rationale': insight['finding'],
                'impact': 'High',
                'urgency': 'High'
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

        with open(report_path, 'w') as f:
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
            f.write("\n### Recommendations\n")
            for i, rec in enumerate(state['recommendations'], 1):
                f.write(f"\n{i}. **{rec['action']}**\n")
                if 'rationale' in rec:
                    f.write(f"   - Rationale: {rec['rationale']}\n")
                f.write(f"   - Impact: {rec.get('impact', 'Unknown')}\n")

            # Data analyzed section
            f.write("\n## Data Analyzed\n")
            for name, df in state['datasets'].items():
                f.write(f"- **{name}**: {df.shape[0]:,} records\n")

            f.write("\n---\n")
            f.write("*This report was generated using dynamic business analysis.*\n")
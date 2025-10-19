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
import google.generativeai as genai


class BusinessDiscoveryState(Dict):
    """State for business discovery workflow."""
    company_id: str
    dataset_ids: List[str]
    datasets: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]
    business_context: Dict[str, Any]
    exploration_results: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    executive_summary: str
    report_path: Optional[str]
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
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None

        # Build workflow
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the business discovery workflow."""

        workflow = StateGraph(BusinessDiscoveryState)

        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("understand_business", self._understand_business_node)
        workflow.add_node("explore_dynamically", self._explore_dynamically_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("create_recommendations", self._create_recommendations_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define flow
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "understand_business")
        workflow.add_edge("understand_business", "explore_dynamically")
        workflow.add_edge("explore_dynamically", "generate_insights")
        workflow.add_edge("generate_insights", "create_recommendations")
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

            prompt = f"""Analyze this business data and identify:
1. What type of business is this?
2. What are the key revenue drivers?
3. What are the main business challenges?
4. What questions should we explore?

DATA CONTEXT:
{context}

IMPORTANT: For key_questions, provide simple business questions WITHOUT column names or technical details.
Good examples:
- "Which customers generate the most revenue?"
- "What products have the highest profit margins?"
- "Which customers are at risk of leaving?"

Bad examples (avoid these):
- "What drives 'churn_risk' in the customer_profiles table?"
- "How does marketing_spend correlate with revenue?"

Provide a structured analysis in JSON format with keys:
business_type, revenue_drivers, challenges, key_questions"""

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
                # Fallback if JSON parsing fails
                business_context = {
                    'business_type': 'Sales/Retail',
                    'revenue_drivers': ['Product sales', 'Customer volume'],
                    'challenges': ['Customer retention', 'Margin optimization'],
                    'key_questions': self.explorer.generate_business_questions()
                }

            state['business_context'] = business_context

            print(f"  Business Type: {business_context.get('business_type', 'Unknown')}")
            print(f"  Revenue Drivers: {', '.join(business_context.get('revenue_drivers', []))}")

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

                # Build exploration prompt with EXPLICIT instructions
                exploration_prompt = f"""Answer this business question by analyzing the data: {question}

AVAILABLE DATASETS (use these exact names as variables):
{self._get_dataset_summary(state)}

IMPORTANT RULES:
1. Write complete, self-contained Python code
2. Each dataset is already loaded as a DataFrame with its exact name
3. Do NOT reference undefined variables
4. Always check if a dataset exists before using it
5. Print clear findings with numbers

CORRECT EXAMPLE:
```python
# Check for high-value customers with churn risk
if 'customer_profiles' in locals() and 'sales_transactions' in locals():
    # Calculate revenue by customer
    customer_revenue = sales_transactions.groupby('customer_id')['net_amount'].sum().reset_index()
    customer_revenue.columns = ['customer_id', 'total_revenue']

    # Merge with profiles
    merged_data = customer_profiles.merge(customer_revenue, on='customer_id', how='left')

    # Find high-risk valuable customers - ALWAYS use df['column_name'] syntax!
    high_risk_customers = merged_data[(merged_data['churn_risk'] > 0.7) & (merged_data['total_revenue'] > 10000)]

    if len(high_risk_customers) > 0:
        print(f"FINDING: {{len(high_risk_customers)}} high-value customers at risk")
        print(f"Revenue at risk: ${{high_risk_customers['total_revenue'].sum():,.2f}}")
else:
    print("Required datasets not available")
```

Write code to answer: {question}"""

                try:
                    # Use the BusinessAnalystLLM instead of raw model
                    print(f"    [DEBUG] Using BusinessAnalystLLM for: {question}")
                    result = self.business_analyst.analyze_business_question(
                        question=question,
                        datasets=state['datasets'],
                        max_attempts=2
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

    def _generate_insights_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Generate business insights from exploration results."""

        print("\n[STEP 4] Generating Business Insights")
        print("-" * 40)

        raw_insights = state['exploration_results'].get('raw_insights', [])

        if not raw_insights:
            print("  No insights discovered")
            return state

        # Convert raw insights to business insights
        business_insights = []

        for raw in raw_insights:
            insight = self._convert_to_business_insight(raw)
            if insight:
                business_insights.append(insight)
                print(f"  [OK] {insight['title']}")

        state['insights'] = business_insights
        print(f"\n  Generated {len(business_insights)} business insights")

        return state

    def _create_recommendations_node(self, state: BusinessDiscoveryState) -> BusinessDiscoveryState:
        """Create actionable recommendations from insights."""

        print("\n[STEP 5] Creating Recommendations")
        print("-" * 40)

        insights = state['insights']
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

        print("\n[STEP 6] Generating Business Report")
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
        """Basic business understanding without LLM."""

        # Analyze what we have
        has_sales = any('sales' in name or 'transaction' in name for name in state['datasets'])
        has_customers = any('customer' in name for name in state['datasets'])
        has_products = any('product' in name for name in state['datasets'])

        return {
            'business_type': 'Sales/Commerce',
            'revenue_drivers': ['Sales transactions'] if has_sales else [],
            'challenges': ['Unknown'],
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

        # Basic recommendation based on insight type
        finding = insight.get('finding', '').lower()

        if 'customer' in finding and ('high' in finding or 'top' in finding):
            recommendations.append({
                'action': 'Focus retention efforts on high-value customers identified',
                'rationale': insight['finding'],
                'impact': 'High',
                'urgency': 'Medium'
            })

        if 'revenue' in finding and ('decline' in finding or 'decrease' in finding):
            recommendations.append({
                'action': 'Investigate and address revenue decline immediately',
                'rationale': insight['finding'],
                'impact': 'High',
                'urgency': 'High'
            })

        if 'product' in finding and 'margin' in finding:
            recommendations.append({
                'action': 'Optimize product mix to focus on high-margin items',
                'rationale': insight['finding'],
                'impact': 'Medium',
                'urgency': 'Low'
            })

        return recommendations

    def _generate_executive_summary(self, state: BusinessDiscoveryState) -> str:
        """Generate executive summary."""

        summary = ["# Executive Summary\n"]

        # Business context
        context = state['business_context']
        summary.append(f"**Business Type:** {context.get('business_type', 'Unknown')}\n")
        summary.append(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n")

        # Key findings
        if state['insights']:
            summary.append("\n## Key Findings\n")
            for insight in state['insights'][:5]:
                summary.append(f"• {insight['finding']}")

        # Top recommendations
        if state['recommendations']:
            summary.append("\n## Priority Actions\n")
            for rec in state['recommendations'][:3]:
                summary.append(f"• {rec['action']}")

        return '\n'.join(summary)

    def _write_business_report(self, state: BusinessDiscoveryState, report_path: str):
        """Write the business report."""

        with open(report_path, 'w') as f:
            f.write(state['executive_summary'])

            f.write("\n\n## Detailed Analysis\n")

            # Insights section
            f.write("\n### Business Insights\n")
            for i, insight in enumerate(state['insights'], 1):
                f.write(f"\n{i}. **{insight['title']}**\n")
                f.write(f"   - Finding: {insight['finding']}\n")

            # Recommendations section
            f.write("\n### Recommendations\n")
            for i, rec in enumerate(state['recommendations'], 1):
                f.write(f"\n{i}. **{rec['action']}**\n")
                f.write(f"   - Rationale: {rec['rationale']}\n")
                f.write(f"   - Impact: {rec.get('impact', 'Unknown')}\n")

            # Data analyzed section
            f.write("\n## Data Analyzed\n")
            for name, df in state['datasets'].items():
                f.write(f"- **{name}**: {df.shape[0]:,} records\n")

            f.write("\n---\n")
            f.write("*This report was generated using dynamic business analysis.*\n")
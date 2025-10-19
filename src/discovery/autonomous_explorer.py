"""Autonomous data exploration agent using LLM and code generation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.discovery.python_tool import PythonExecutorTool, DataSummaryTool, DataProfileTool
from src.discovery.visualization_data_store import VisualizationDataStore
from src.utils.llm_client import get_llm
import os


@dataclass
class ExplorationInsight:
    """A single insight discovered during exploration."""
    question: str
    finding: str
    supporting_data: Dict[str, Any]
    code_used: str
    confidence: float
    business_impact: str
    visualization_paths: List[str] = field(default_factory=list)  # Paths to generated visualizations
    visualization_type: Optional[str] = None  # Type of visualization (line, bar, scatter, etc.)


@dataclass
class ExplorationResult:
    """Result of autonomous exploration."""
    dataset_name: str
    insights: List[ExplorationInsight] = field(default_factory=list)
    total_executions: int = 0
    exploration_summary: str = ""
    dataset_context: Optional[Dict] = None
    viz_directory: Optional[str] = None  # Directory where visualizations are saved
    viz_data_path: Optional[str] = None  # Path to viz_data JSON file for Plotly dashboards


class AutonomousExplorer:
    """
    Autonomous agent that explores datasets by generating and executing Python code.

    Unlike the old fixed-analysis approach, this agent:
    1. Understands the business context
    2. Generates Python code dynamically
    3. Executes code to explore data
    4. Interprets results
    5. Iterates to find deeper insights
    """

    def __init__(
        self,
        max_iterations: int = 20,
        max_insights: int = 3,
        temperature: float = 0.3
    ):
        """
        Initialize autonomous explorer.

        Args:
            max_iterations: Maximum agent iterations (increased to allow more thorough analysis)
            max_insights: Target number of high-quality insights (reduced to encourage depth over breadth)
            temperature: LLM temperature for code generation
        """
        self.max_iterations = max_iterations
        self.max_insights = max_insights  # This is a target, not a strict limit
        self.temperature = temperature

        # Initialize LLM for agent using get_llm
        # Use gemini-2.5-flash with lower temperature for stability
        self.llm = get_llm(
            temperature=0.1,  # Very low temperature for more deterministic behavior
            model="gemini-2.5-flash",
            max_retries=3  # More retries on API failures
        )

    def explore(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        dataset_context: Optional[Dict] = None,
        skip_visualizations: bool = False  # NEW: Skip viz generation for multi-table analysis
    ) -> ExplorationResult:
        """
        Autonomously explore dataset and generate insights.

        Args:
            df: DataFrame to explore
            dataset_name: Name of dataset
            dataset_context: Context from outer agent (domain, entities, etc.)
            skip_visualizations: If True, skip visualization generation (for multi-table analysis)

        Returns:
            ExplorationResult with discovered insights
        """
        print("\n" + "="*80)
        print("[AI] AUTONOMOUS EXPLORATION")
        print("="*80)
        print(f"Dataset: {dataset_name}")
        print(f"Size: {len(df):,} rows × {len(df.columns)} columns")

        if dataset_context:
            print(f"Domain: {dataset_context.get('domain', 'Unknown')}")
            print(f"Dataset Type: {dataset_context.get('dataset_type', 'Unknown')}")

        if skip_visualizations:
            print("⚡ Skipping visualizations (multi-table mode)")
            viz_dir = None
            viz_data_store = None
        else:
            # Create visualization output directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_dir = os.path.join("data", "outputs", "discovery", "visualizations", f"{dataset_name}_{timestamp}")
            os.makedirs(viz_dir, exist_ok=True)
            print(f"Visualization directory: {viz_dir}")

            # Create visualization data store for incremental JSON writing
            viz_data_store = VisualizationDataStore(
                dataset_name=dataset_name,
                dataset_context=dataset_context
            )
            print(f"Viz data JSON: {viz_data_store.get_json_path()}")

        # Create tools with visualization support (or None if skipped)
        python_tool = PythonExecutorTool(df=df, viz_output_dir=viz_dir, viz_data_store=viz_data_store)
        summary_tool = DataSummaryTool(df=df)
        profile_tool = DataProfileTool(df=df)

        tools = [python_tool, summary_tool, profile_tool]

        # Create ReAct agent
        agent = self._create_agent(tools, dataset_context, skip_visualizations)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="force",  # Force agent to return output even if max_iterations hit
            max_execution_time=600  # 10 minute timeout
        )

        # Create exploration goal
        goal = self._create_exploration_goal(dataset_name, dataset_context, df, skip_visualizations)

        print("\n[GOAL] Exploration Goal:")
        print(goal)
        print("\n" + "="*80)

        # Run agent
        try:
            result = agent_executor.invoke({"input": goal})

            # Extract insights from agent output
            insights = self._extract_insights(
                result,
                python_tool,
                dataset_context,
                viz_dir
            )

            # Finalize visualization data store (only if visualizations were generated)
            if viz_data_store is not None:
                viz_data_store.finalize()

            # Create exploration result
            exploration_result = ExplorationResult(
                dataset_name=dataset_name,
                insights=insights,
                total_executions=len(python_tool.execution_history),
                exploration_summary=result.get("output", ""),
                dataset_context=dataset_context,
                viz_directory=viz_dir,
                viz_data_path=viz_data_store.get_json_path() if viz_data_store else None
            )

            print("\n" + "="*80)
            print("[SUCCESS] EXPLORATION COMPLETE")
            print("="*80)
            print(f"Insights found: {len(insights)}")
            print(f"Code executions: {exploration_result.total_executions}")
            print(f"Viz data JSON: {exploration_result.viz_data_path}")
            print("="*80)

            return exploration_result

        except Exception as e:
            print(f"\n[ERROR] Exploration failed: {e}")
            import traceback
            traceback.print_exc()

            # Return partial result
            return ExplorationResult(
                dataset_name=dataset_name,
                insights=[],
                total_executions=0,
                exploration_summary=f"Exploration failed: {str(e)}",
                dataset_context=dataset_context
            )

    def _create_agent(self, tools: List, dataset_context: Optional[Dict], skip_visualizations: bool = False) -> Any:
        """Create ReAct agent with appropriate prompt."""

        # Build context description
        context_desc = ""
        if dataset_context:
            context_desc = f"""
Dataset Context:
- Domain: {dataset_context.get('domain', 'Unknown')}
- Type: {dataset_context.get('dataset_type', 'Unknown')}
- Key Entities: {', '.join(dataset_context.get('key_entities', []))}
- Business Terms: {dataset_context.get('business_terms', {})}
"""

        template = """You are an expert data analyst exploring a dataset to find valuable business insights.

{context_desc}

Your goal is to inspect the raw data, clean it as needed, then explore it to find meaningful insights.

IMPORTANT WORKFLOW:
1. INSPECT THE DATA FIRST:
   - Use get_data_summary to see columns, types, missing values
   - Use get_data_profile to understand distributions
   - Check for data quality issues (NaN, duplicates, wrong types, outliers)

2. CLEAN THE DATA AS NEEDED:
   - You receive RAW, UNCLEANED data - you must clean it yourself
   - Handle missing values (drop, fill, or filter based on business context)
   - Fix data types (convert strings to numbers, dates, etc.)
   - Remove or handle duplicates based on what makes sense
   - Handle outliers appropriately for the analysis
   - Clean text (strip whitespace, standardize formats)
   - The df modifications persist across executions (df = df.dropna() will persist)

3. EXPLORE AND FIND INSIGHTS:
   - Generate Python code to explore patterns, trends, and relationships
   - Focus on business-relevant insights (growth, trends, comparisons, anomalies)
   - Calculate metrics that matter (YoY growth, seasonality, correlations, segments)
   - Compare groups, time periods, or categories when relevant
   - Look for surprising patterns or outliers

   COMPLEX AGGREGATIONS - Use pandas .agg() for powerful multi-metric analysis:
   ```python
   # Example 1: Multiple metrics per group (CAGR analysis)
   company_metrics = df.groupby('company').agg(
       total_revenue=('revenue', 'sum'),
       avg_revenue=('revenue', 'mean'),
       max_revenue=('revenue', 'max'),
       min_revenue=('revenue', 'min'),
       num_periods=('revenue', 'count')
   )
   # Then calculate CAGR: ((end_value / start_value)**(1/years) - 1) * 100

   # Example 2: Year-over-year analysis with multiple metrics
   yearly = df.groupby('year').agg({{
       'revenue': ['sum', 'mean', 'count'],
       'profit': 'sum',
       'customers': 'sum'
   }})

   # Example 3: Pivot table for cross-tabulation
   revenue_by_year_company = pd.pivot_table(
       df, values='revenue', index='company',
       columns='year', aggfunc='sum', fill_value=0
   )

   # Example 4: Multi-level grouping (industry + year)
   segment_analysis = df.groupby(['industry', 'year']).agg({{
       'revenue': 'sum',
       'profit_margin': 'mean'
   }}).reset_index()
   ```

4. VALIDATE INSIGHTS WITH CODE (CRITICAL):
   - An insight is ONLY an insight if validated by code execution and data
   - You MUST execute code to calculate the numbers BEFORE claiming an insight
   - NEVER state an insight without running code first
   - Process for each insight:
     a) Execute MULTIPLE code blocks to thoroughly analyze the data
     b) Review the numbers/results from each execution
     c) ONLY THEN write "### Insight N: [Title]" with the validated finding
   - IMPORTANT: One insight typically requires 3-5+ code executions:
     * First execution: Explore and understand the pattern
     * Second execution: Calculate specific metrics and numbers
     * Third+ execution: Validate with different angles or time periods
   - If code shows the insight is wrong, don't write it - try a different analysis

5. QUALITY OVER QUANTITY:
   - Find as many valuable insights as you can, up to a maximum of {max_insights}
   - Stop when you've exhausted meaningful insights, even if under the maximum
   - Don't be afraid to make mistakes during exploration - if code has errors, debug it and try again
   - When you're confident you've found all meaningful insights, provide a Final Answer summarizing them

{visualization_guidelines}

INSIGHT VALIDATION EXAMPLES:

Good (Multiple Code Executions Per Insight):
Thought: I should analyze revenue growth trends
Action: python_executor
Action Input:  # EXECUTION 1: Explore the data
df['year'] = pd.to_datetime(df['date']).dt.year
yearly_revenue = df.groupby('year')['revenue'].sum()
print(yearly_revenue)
result = yearly_revenue.to_dict()
Observation: [OK] {{2020: 100M, 2021: 145M, 2022: 153M, 2023: 161M}}

Thought: Good! Now let me calculate YoY growth rates
Action: python_executor
Action Input:  # EXECUTION 2: Calculate growth metrics
yoy_growth = yearly_revenue.pct_change() * 100
print("YoY Growth Rates:")
print(yoy_growth)
result = yoy_growth.to_dict()
Observation: [OK] {{2021: 45.0%, 2022: 5.5%, 2023: 5.2%}}

Thought: Excellent! Growth decelerated sharply. Let me create a line chart
Action: python_executor
Action Input:  # EXECUTION 3: Create trend visualization
import matplotlib.pyplot as plt
years = ['2020', '2021', '2022', '2023']
revenue = [100, 145, 153, 161]
plt.figure(figsize=(10, 6))
plt.plot(years, revenue, marker='o')
plt.title('Revenue Trend 2020-2023')
plt.xlabel('Year')
plt.ylabel('Revenue (USD Millions)')
plt.savefig(f'{{VIZ_DIR}}/insight_1_revenue_trend.png', bbox_inches='tight', dpi=100)
plt.close()
result = {{'chart_type': 'line', 'labels': years, 'x_label': 'Year', 'y_label': 'Revenue (USD Millions)', 'datasets': [{{'label': 'Revenue', 'data': revenue}}]}}
Observation: [OK] Chart saved

Thought: Now let me create a bar chart showing growth rates
Action: python_executor
Action Input:  # EXECUTION 4: Create growth rate visualization
plt.figure(figsize=(10, 6))
periods = ['2020-21', '2021-22', '2022-23']
growth = [45.0, 5.5, 5.2]
plt.bar(periods, growth, color=['green', 'orange', 'red'])
plt.title('YoY Growth Deceleration')
plt.xlabel('Period')
plt.ylabel('Growth Rate (%)')
plt.savefig(f'{{VIZ_DIR}}/insight_1_growth_rates.png', bbox_inches='tight', dpi=100)
plt.close()
result = {{'chart_type': 'bar', 'labels': periods, 'x_label': 'Period', 'y_label': 'Growth Rate (%)', 'datasets': [{{'label': 'YoY Growth', 'data': growth}}]}}
Observation: [OK] Chart saved

Thought: Perfect! I've validated the insight with 4 code executions and 2 visualizations
### Insight 1: Revenue Growth Decelerated Sharply After 2021
Revenue grew 45% YoY in 2021 but slowed to only 5.5% in 2022 and 5.2% in 2023...

Bad (No Code Validation):
Thought: I should write about revenue growth
### Insight 1: Revenue is Growing
Revenue appears to be growing year over year...
[WRONG - No code execution, no numbers, no validation!]

Available tools:
{tools}

CODE DEBUGGING:
- If your code execution returns an error, READ THE ERROR MESSAGE carefully
- Debug by fixing the code and trying again - you can retry multiple times
- Common issues: missing imports, wrong column names, data type mismatches, NaN values
- Use try-except blocks for robustness
- Check data types before operations (df.dtypes)
- Handle missing values explicitly (dropna, fillna)

DATA CLEANING & QUALITY VALIDATION:
- After initial cleaning, you may still encounter NaN values or data quality issues
- You can clean data as needed during exploration
- Common cleaning operations:
  * df.dropna() - remove rows with missing values
  * df.fillna(value) - fill missing values
  * df[df['column'] > 0] - filter out invalid values
  * pd.to_numeric(df['column'], errors='coerce') - convert to numeric, invalid values become NaN
- CRITICAL: Validate data quality before making claims
  * Check for duplicate records that could inflate sums
  * Verify aggregations make sense (e.g., sum of quarters shouldn't be 10x higher than annual)
  * Look for data extraction errors or inconsistent units
  * If you find massive discrepancies (>100% differences), investigate the root cause
  * Don't report data quality issues as "insights" unless they're genuinely surprising
  * Example: If quarterly sum >> annual total, this likely indicates duplicate records or data errors, not a real finding

Use the following format:

Question: the input question you must answer
Thought: think about what to explore next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
... IMPORTANT: Execute code FIRST, review results, THEN write insight
... If code has errors, debug and retry
... Each insight needs code execution to validate the numbers
... Continue until you've found all meaningful insights (up to {max_insights})
Thought: I have validated N insights through code execution. Time to summarize.
Final Answer: A comprehensive summary with all insights, each formatted as "### Insight N: [Title]" followed by the full explanation with:
- The specific numbers from code execution
- Business context explaining what it means
- Why it matters
- Supporting visualization (if created)

CRITICAL RULES:
1. You MUST execute code to validate EVERY insight
2. You MUST provide a Final Answer after generating insights
3. Do not state insights without code-validated numbers

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

        # Conditionally add visualization guidelines
        if skip_visualizations:
            # Multi-table mode - NO visualizations
            visualization_guidelines = """
NOTE: Visualizations are DISABLED for this analysis (multi-table mode).
- DO NOT attempt to create any charts or plots
- DO NOT use VIZ_DIR (it is not available)
- Focus ONLY on data analysis, calculations, and numerical insights
- Save analysis results in the 'result' variable for each execution
"""
        else:
            # Single-table mode - visualizations enabled
            visualization_guidelines = ""  # Handled by goal prompt

        prompt = PromptTemplate.from_template(template)
        prompt = prompt.partial(
            context_desc=context_desc,
            max_insights=self.max_insights,
            visualization_guidelines=visualization_guidelines
        )

        return create_react_agent(self.llm, tools, prompt)

    def _create_exploration_goal(
        self,
        dataset_name: str,
        dataset_context: Optional[Dict],
        df: pd.DataFrame,
        skip_visualizations: bool = False
    ) -> str:
        """Create exploration goal based on context."""

        goal_parts = [
            f"Explore the {dataset_name} dataset to find {self.max_insights} high-quality, well-validated business insights."
        ]

        if dataset_context:
            domain = dataset_context.get('domain', '')
            dataset_type = dataset_context.get('dataset_type', '')

            if domain and dataset_type:
                goal_parts.append(
                    f"This is a {dataset_type} dataset in the {domain} domain."
                )

            # Add suggested questions
            suggested_questions = dataset_context.get('suggested_questions', [])
            if suggested_questions:
                goal_parts.append("\nConsider exploring:")
                for q in suggested_questions[:5]:
                    goal_parts.append(f"  - {q}")

        goal_parts.append("\nFocus on:")
        goal_parts.append("  1. Trends over time (growth rates, seasonality)")
        goal_parts.append("  2. Comparisons between groups/categories")
        goal_parts.append("  3. Key correlations and relationships")
        goal_parts.append("  4. Anomalies or surprising patterns")
        goal_parts.append("  5. Segment analysis (high/low performers)")

        goal_parts.append("\nFor each insight:")
        goal_parts.append("  - Execute 3-5+ code blocks to thoroughly validate")

        # Only include visualization instructions if not skipped
        if not skip_visualizations:
            goal_parts.append("  - Create 2+ visualizations from different angles")
            goal_parts.append("  - Include proper axis labels (x_label, y_label) in viz data")

        goal_parts.append("  - Provide clear business explanation with specific numbers")
        goal_parts.append("  - Explain why it matters to the business")

        return "\n".join(goal_parts)

    def _extract_insights(
        self,
        agent_result: Dict,
        python_tool: PythonExecutorTool,
        dataset_context: Optional[Dict],
        viz_dir: str
    ) -> List[ExplorationInsight]:
        """
        Extract insights from agent execution and link them to code executions.

        Each insight should be validated by code execution(s).
        """

        insights = []

        # Get agent output
        final_answer = agent_result.get("output", "")
        intermediate_steps = agent_result.get("intermediate_steps", [])

        # Get all code executions for linking
        code_executions = python_tool.execution_history
        print(f"\n[DEBUG] Total code executions during exploration: {len(code_executions)}")

        # Parse final answer for insights formatted as "### Insight X:"
        import re

        # Debug: Print first 500 chars to see format
        print(f"\n[DEBUG] First 500 chars of final_answer:\n{final_answer[:500]}\n")

        insight_pattern = r'###\s*Insight\s*\d+:.*?(?=###\s*Insight\s*\d+:|$)'
        insight_matches = re.findall(insight_pattern, final_answer, re.DOTALL | re.IGNORECASE)

        print(f"[DEBUG] Found {len(insight_matches)} insight matches in final answer")

        for idx, match in enumerate(insight_matches):
            # Clean up the insight text
            insight_text = match.strip()

            # Extract title (first line)
            lines = insight_text.split('\n')
            title = lines[0].replace('###', '').strip() if lines else "Insight"

            # Get full insight text (limit to reasonable length)
            if len(insight_text) > 50:  # Valid insight
                # Collect all code that was executed (for now, all code as we can't perfectly map)
                # In a more sophisticated version, we could parse the intermediate_steps
                # to find which code executions led to this specific insight
                all_code = "\n\n".join([
                    f"# Execution {i+1}\n{exec['code']}"
                    for i, exec in enumerate(code_executions)
                ])

                # Extract any numbers/data mentioned in the insight for supporting_data
                supporting_data = {
                    "total_code_executions": len(code_executions),
                    "insight_text": insight_text[:500]
                }

                insights.append(ExplorationInsight(
                    question=title,
                    finding=insight_text[:2000],  # Limit length
                    supporting_data=supporting_data,
                    code_used=all_code[:5000] if all_code else "No code executions found",  # Limit code length
                    confidence=0.85,
                    business_impact="High"
                ))
                print(f"[DEBUG] Added insight {idx+1}: {title[:50]}... (validated by {len(code_executions)} code executions)")

        # If no formatted insights found, try simple parsing
        if len(insights) == 0:
            print("[WARN] No properly formatted insights found. Trying fallback parsing...")

            # Collect all code executions
            all_code = "\n\n".join([
                f"# Execution {i+1}\n{exec['code']}"
                for i, exec in enumerate(code_executions)
            ])

            lines = final_answer.split('\n')
            current_insight = []

            for line in lines:
                line = line.strip()
                if line and (line.startswith('Insight') or line.startswith('-') or line.startswith('•') or (line and line[0].isdigit() and '.' in line[:3])):
                    if current_insight:
                        insight_text = ' '.join(current_insight)
                        if len(insight_text) > 50:
                            insights.append(ExplorationInsight(
                                question="Autonomous exploration",
                                finding=insight_text[:2000],
                                supporting_data={"total_code_executions": len(code_executions)},
                                code_used=all_code[:5000] if all_code else "No code executions",
                                confidence=0.7,  # Lower confidence for unparsed insights
                                business_impact="Medium"
                            ))
                    current_insight = [line]
                elif current_insight:
                    current_insight.append(line)

            # Process last insight
            if current_insight:
                insight_text = ' '.join(current_insight)
                if len(insight_text) > 50:
                    insights.append(ExplorationInsight(
                        question="Autonomous exploration",
                        finding=insight_text[:2000],
                        supporting_data={"total_code_executions": len(code_executions)},
                        code_used=all_code[:5000] if all_code else "No code executions",
                        confidence=0.7,
                        business_impact="Medium"
                    ))

        print(f"\n[DEBUG] Extracted {len(insights)} insights from agent output")

        # Scan for generated visualization files and attach to insights
        viz_files = self._scan_visualization_files(viz_dir)
        if viz_files:
            print(f"[DEBUG] Found {len(viz_files)} visualization files")
            # Attach visualizations to insights based on filename patterns
            for viz_file in viz_files:
                filename = os.path.basename(viz_file)
                # Try to extract insight number from filename (e.g., insight_1_revenue.png)
                import re
                match = re.search(r'insight[_\s]*(\d+)', filename, re.IGNORECASE)
                if match:
                    insight_num = int(match.group(1)) - 1  # 0-indexed
                    if 0 <= insight_num < len(insights):
                        insights[insight_num].visualization_paths.append(viz_file)
                        print(f"[DEBUG] Attached {filename} to insight {insight_num + 1}")

        return insights[:self.max_insights]

    def _scan_visualization_files(self, viz_dir: str) -> List[str]:
        """
        Scan visualization directory for generated image files.

        Args:
            viz_dir: Directory to scan (can be None if visualizations skipped)

        Returns:
            List of absolute paths to visualization files
        """
        viz_files = []
        if viz_dir and os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.html')):
                    viz_files.append(os.path.join(viz_dir, file))
        return sorted(viz_files)


if __name__ == "__main__":
    print("AutonomousExplorer module - ready for use")
    print("Import: from src.discovery.autonomous_explorer import AutonomousExplorer")

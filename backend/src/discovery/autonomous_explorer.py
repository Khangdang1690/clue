"""Autonomous data exploration agent using LLM and code generation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.discovery.python_tool import PythonExecutorTool, DataSummaryTool, DataProfileTool
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
        max_iterations: int = 15,
        max_insights: int = 10,
        temperature: float = 0.3
    ):
        """
        Initialize autonomous explorer.

        Args:
            max_iterations: Maximum agent iterations
            max_insights: Target number of insights to find
            temperature: LLM temperature for code generation
        """
        self.max_iterations = max_iterations
        self.max_insights = max_insights
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
        dataset_context: Optional[Dict] = None
    ) -> ExplorationResult:
        """
        Autonomously explore dataset and generate insights.

        Args:
            df: DataFrame to explore
            dataset_name: Name of dataset
            dataset_context: Context from outer agent (domain, entities, etc.)

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

        # Create visualization output directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = os.path.join("data", "outputs", "discovery", "visualizations", f"{dataset_name}_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization directory: {viz_dir}")

        # Create tools with visualization support
        python_tool = PythonExecutorTool(df=df, viz_output_dir=viz_dir)
        summary_tool = DataSummaryTool(df=df)
        profile_tool = DataProfileTool(df=df)

        tools = [python_tool, summary_tool, profile_tool]

        # Create ReAct agent
        agent = self._create_agent(tools, dataset_context)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="generate",  # Force agent to generate output even if max_iterations hit
            max_execution_time=600  # 10 minute timeout
        )

        # Create exploration goal
        goal = self._create_exploration_goal(dataset_name, dataset_context, df)

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

            # Create exploration result
            exploration_result = ExplorationResult(
                dataset_name=dataset_name,
                insights=insights,
                total_executions=len(python_tool.execution_history),
                exploration_summary=result.get("output", ""),
                dataset_context=dataset_context,
                viz_directory=viz_dir
            )

            print("\n" + "="*80)
            print("[SUCCESS] EXPLORATION COMPLETE")
            print("="*80)
            print(f"Insights found: {len(insights)}")
            print(f"Code executions: {exploration_result.total_executions}")
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

    def _create_agent(self, tools: List, dataset_context: Optional[Dict]) -> Any:
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

Your goal is to explore the data by writing and executing Python code, then interpreting the results to find meaningful insights.

IMPORTANT GUIDELINES:
1. Start by getting data summary and profile to understand the dataset
2. Generate Python code to explore patterns, trends, and relationships
3. Focus on business-relevant insights (growth, trends, comparisons, anomalies)
4. Calculate metrics that matter (YoY growth, seasonality, correlations, segments)
5. Compare groups, time periods, or categories when relevant
6. Look for surprising patterns or outliers
7. Provide business explanations for technical findings
8. After generating each insight, write it in the format: "Insight N: [Title]" followed by the explanation
9. STOP after you've generated 10 insights and provide a Final Answer summarizing all insights

VISUALIZATION GUIDELINES:
- Create visualizations to support your insights using matplotlib or seaborn
- The variable VIZ_DIR is already available - DO NOT import os or create directories
- Save visualizations using: plt.savefig(f'{{VIZ_DIR}}/insight_N_name.png', bbox_inches='tight', dpi=100)
- Choose appropriate chart types:
  * Line charts: For trends over time (revenue growth, seasonal patterns)
  * Bar charts: For comparisons between categories (department performance, segment analysis)
  * Scatter plots: For correlations between variables
  * Heatmaps: For correlation matrices or cross-tabulations
- Always close plots after saving: plt.close()
- IMPORTANT: Save the visualization data for interactive dashboards
  After creating your chart, store the data in result dict:
  ```python
  result = {{
      'chart_type': 'bar',  # or 'line', 'scatter', 'pie', etc.
      'labels': ['Q1', 'Q2', 'Q3', 'Q4'],  # X-axis labels
      'datasets': [
          {{'label': 'Revenue', 'data': [100, 120, 150, 180]}}
      ]
  }}
  ```
- Example visualization code:
  ```python
  import matplotlib.pyplot as plt
  # VIZ_DIR is already available, no need to import os
  # Prepare data
  labels = ['Q1', 'Q2', 'Q3', 'Q4']
  revenue = [100, 120, 150, 180]

  # Create matplotlib chart
  plt.figure(figsize=(10, 6))
  plt.plot(labels, revenue)
  plt.title('Revenue Growth Over Time')
  plt.xlabel('Quarter')
  plt.ylabel('Revenue')
  plt.savefig(f'{{VIZ_DIR}}/insight_1_revenue_growth.png', bbox_inches='tight', dpi=100)
  plt.close()

  # Store data for interactive dashboard
  result = {{
      'chart_type': 'line',
      'labels': labels,
      'datasets': [{{'label': 'Revenue', 'data': revenue}}]
  }}
  ```

Example good insights:
- "Revenue grew 45% YoY in 2021 but slowed to 5% in 2022 due to market saturation"
- "Q4 consistently accounts for 35% of annual revenue, indicating strong holiday seasonality"
- "Premium customers have 3x higher LTV but represent only 15% of base"

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what to explore next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation until you have generated 10 insights)
Thought: I have generated 10 insights. Time to summarize.
Final Answer: A comprehensive summary with all 10 insights, each formatted as "### Insight N: [Title]" followed by the full explanation with business context, supporting numbers, and why it matters.

CRITICAL: You MUST provide a Final Answer after generating insights. Do not stop without a Final Answer.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

        prompt = PromptTemplate.from_template(template)
        prompt = prompt.partial(context_desc=context_desc)

        return create_react_agent(self.llm, tools, prompt)

    def _create_exploration_goal(
        self,
        dataset_name: str,
        dataset_context: Optional[Dict],
        df: pd.DataFrame
    ) -> str:
        """Create exploration goal based on context."""

        goal_parts = [
            f"Explore the {dataset_name} dataset to find {self.max_insights} valuable business insights."
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

        goal_parts.append("\nFor each insight, provide:")
        goal_parts.append("  - Clear business explanation")
        goal_parts.append("  - Supporting numbers/percentages")
        goal_parts.append("  - Why it matters")

        return "\n".join(goal_parts)

    def _extract_insights(
        self,
        agent_result: Dict,
        python_tool: PythonExecutorTool,
        dataset_context: Optional[Dict],
        viz_dir: str
    ) -> List[ExplorationInsight]:
        """Extract insights from agent execution."""

        insights = []

        # Get agent output
        final_answer = agent_result.get("output", "")
        intermediate_steps = agent_result.get("intermediate_steps", [])

        # Parse final answer for insights formatted as "### Insight X:"
        import re

        # Debug: Print first 500 chars to see format
        print(f"\n[DEBUG] First 500 chars of final_answer:\n{final_answer[:500]}\n")

        insight_pattern = r'###\s*Insight\s*\d+:.*?(?=###\s*Insight\s*\d+:|$)'
        insight_matches = re.findall(insight_pattern, final_answer, re.DOTALL | re.IGNORECASE)

        print(f"[DEBUG] Found {len(insight_matches)} matches with regex pattern")

        for match in insight_matches:
            # Clean up the insight text
            insight_text = match.strip()

            # Extract title (first line)
            lines = insight_text.split('\n')
            title = lines[0].replace('###', '').strip() if lines else "Insight"

            # Get full insight text (limit to reasonable length)
            if len(insight_text) > 50:  # Valid insight
                insights.append(ExplorationInsight(
                    question=title,
                    finding=insight_text[:2000],  # Limit length
                    supporting_data={},
                    code_used="",
                    confidence=0.85,
                    business_impact="High"
                ))
                print(f"[DEBUG] Added insight: {title[:50]}...")

        # If no formatted insights found, try simple parsing
        if len(insights) == 0:
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
                                supporting_data={},
                                code_used="",
                                confidence=0.8,
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
                        supporting_data={},
                        code_used="",
                        confidence=0.8,
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
            viz_dir: Directory to scan

        Returns:
            List of absolute paths to visualization files
        """
        viz_files = []
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.html')):
                    viz_files.append(os.path.join(viz_dir, file))
        return sorted(viz_files)


if __name__ == "__main__":
    print("AutonomousExplorer module - ready for use")
    print("Import: from src.discovery.autonomous_explorer import AutonomousExplorer")

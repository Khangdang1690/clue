"""Dynamic data explorer that allows LLMs to write and execute queries."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import ast
import traceback
from io import StringIO
import contextlib
import sys
from datetime import datetime, timedelta
import re
from sqlalchemy import text
from src.database.connection import DatabaseManager
from src.database.repository import DatasetRepository
import google.generativeai as genai
import os


class DynamicDataExplorer:
    """Allows LLM to dynamically explore data through code execution."""

    def __init__(self):
        """Initialize the dynamic explorer."""
        self.execution_history = []
        self.insights_discovered = []
        self.datasets = {}
        self.metadata = {}

        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def load_datasets(self, company_id: str) -> Dict[str, pd.DataFrame]:
        """Load all datasets for a company into memory."""
        with DatabaseManager.get_session() as session:
            from src.database.repository import DatasetRepository, CompanyRepository

            # Get company
            company = CompanyRepository.get_company_by_id(session, company_id)
            if not company:
                return {}

            # Get all datasets
            datasets = DatasetRepository.get_datasets_by_company(session, company_id)

            for ds in datasets:
                # Load DataFrame
                df = DatasetRepository.load_dataframe(session, ds.id)
                if df is not None:
                    self.datasets[ds.table_name] = df
                    self.metadata[ds.table_name] = {
                        'domain': ds.domain,
                        'description': ds.description,
                        'entities': ds.entities,
                        'row_count': ds.row_count,
                        'column_count': ds.column_count
                    }

        return self.datasets

    def execute_code(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python/pandas code and return results."""

        # Create execution environment
        exec_env = {
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            **self.datasets  # Make all datasets available as variables
        }

        # Add any additional context
        if context:
            exec_env.update(context)

        # Capture output
        output_buffer = StringIO()
        error = None
        result = None

        try:
            # Redirect stdout to capture prints
            with contextlib.redirect_stdout(output_buffer):
                # Execute the code
                exec(code, exec_env)

                # Try to get the last expression's value
                # This allows us to return DataFrame results
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.unparse(tree.body[-1].value)
                    result = eval(last_expr, exec_env)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Get captured output
        output = output_buffer.getvalue()

        # Format result for display
        display_result = self._format_result(result)

        execution = {
            'code': code,
            'output': output,
            'result': display_result,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }

        self.execution_history.append(execution)

        return execution

    def _format_result(self, result: Any) -> str:
        """Format execution result for display."""
        if result is None:
            return None

        if isinstance(result, pd.DataFrame):
            if len(result) > 20:
                return f"DataFrame: {result.shape}\n{result.head(10).to_string()}\n...\n{result.tail(5).to_string()}"
            else:
                return f"DataFrame: {result.shape}\n{result.to_string()}"

        elif isinstance(result, pd.Series):
            if len(result) > 20:
                return f"Series: {len(result)} items\n{result.head(10).to_string()}\n..."
            else:
                return f"Series: {len(result)} items\n{result.to_string()}"

        elif isinstance(result, (list, tuple)):
            if len(result) > 10:
                return f"{type(result).__name__}: {len(result)} items\n{result[:10]}..."
            else:
                return str(result)

        elif isinstance(result, dict):
            if len(result) > 10:
                items = list(result.items())[:10]
                return f"Dict: {len(result)} items\n{dict(items)}..."
            else:
                return json.dumps(result, indent=2, default=str)

        elif isinstance(result, (int, float, str, bool)):
            return str(result)

        else:
            return f"{type(result).__name__}: {str(result)[:500]}"

    def generate_business_questions(self) -> List[str]:
        """Generate business questions based on actual data structure without domain assumptions."""

        questions = []

        if not self.datasets:
            return ["What insights can we derive from the available data?"]

        # Analyze data capabilities across all datasets
        has_numeric_data = False
        has_time_data = False
        has_categorical_data = False
        multi_table = len(self.datasets) > 1

        for dataset_name, df in self.datasets.items():
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            time_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if numeric_cols:
                has_numeric_data = True
            if time_cols:
                has_time_data = True
            if cat_cols:
                has_categorical_data = True

        # Generate generic analytical questions based on data capabilities
        if has_numeric_data and has_categorical_data:
            questions.extend([
                "What are the key patterns and segments in the data?",
                "Which groups show the highest and lowest performance?",
                "What factors drive the largest differences in outcomes?"
            ])

        if has_time_data and has_numeric_data:
            questions.extend([
                "What are the trends and patterns over time?",
                "Are there any seasonal or cyclical patterns?",
                "Which time periods show unusual activity?"
            ])

        if has_numeric_data:
            questions.extend([
                "What is the distribution of key metrics?",
                "Are there any significant outliers or anomalies?",
                "What are the strongest correlations in the data?"
            ])

        if multi_table:
            questions.extend([
                "How do different datasets relate to each other?",
                "What insights emerge from cross-dataset analysis?"
            ])

        # If no specific capabilities detected, return generic questions
        if not questions:
            dataset_names = list(self.datasets.keys())
            questions.extend([
                f"What are the key characteristics of the {dataset_names[0]} dataset?",
                "What patterns and insights exist in the available data?"
            ])

        return questions

    def explore_with_llm(self,
                        objective: str = "Find actionable business insights",
                        max_iterations: int = 10) -> Dict[str, Any]:
        """Let the LLM explore the data iteratively."""

        if not self.model:
            return {"error": "Gemini API not configured"}

        # Build context about available data
        data_context = self._build_data_context()

        exploration_prompt = f"""You are a business analyst exploring a company's data.

AVAILABLE DATASETS:
{data_context}

OBJECTIVE: {objective}

You can write Python code to analyze the data. All datasets are available as DataFrames with their table names.
The datasets variable names match the table names shown above. You can access them directly.

Example pattern (adapt to actual dataset and column names):
```python
# If you have a dataset called 'data_table' with columns 'category' and 'value'
grouped = data_table.groupby('category')['value'].sum().sort_values(ascending=False)
print(f"Top 10 {category} by {value}:")
print(grouped.head(10))
```

Based on the data available, generate Python code to explore and analyze.
Focus on discovering:
1. Key patterns and trends in the data
2. Significant segments or groupings
3. Unusual patterns or outliers
4. Relationships between variables
5. Quantifying business impact

Return your code in ```python``` blocks. After each execution, I'll show you the results and you can explore further.

Start by exploring the structure and relationships between the datasets."""

        messages = [exploration_prompt]

        for iteration in range(max_iterations):
            try:
                # Get LLM response
                response = self.model.generate_content('\n'.join(messages))
                llm_response = response.text

                # Extract code blocks
                code_blocks = re.findall(r'```python\n(.*?)\n```', llm_response, re.DOTALL)

                if not code_blocks:
                    # No code to execute, might be done
                    if "FINAL INSIGHTS" in llm_response or iteration > max_iterations/2:
                        break
                    messages.append(f"Please provide Python code to explore the data.")
                    continue

                # Execute each code block
                for code in code_blocks:
                    execution = self.execute_code(code)

                    # Build feedback for LLM
                    feedback = f"\nExecution Result:\n"
                    if execution['error']:
                        feedback += f"ERROR: {execution['error']}\n"
                    else:
                        if execution['output']:
                            feedback += f"Output:\n{execution['output']}\n"
                        if execution['result']:
                            feedback += f"Result:\n{execution['result']}\n"

                    messages.append(feedback)

                # Check if LLM found insights
                if "INSIGHT:" in llm_response or "FINDING:" in llm_response:
                    # Extract insights
                    insights = re.findall(r'(?:INSIGHT|FINDING):\s*(.*?)(?:\n|$)', llm_response)
                    self.insights_discovered.extend(insights)

            except Exception as e:
                print(f"LLM exploration error: {e}")
                break

        return {
            'execution_history': self.execution_history,
            'insights': self.insights_discovered,
            'iterations': iteration + 1
        }

    def _build_data_context(self) -> str:
        """Build context about available datasets for the LLM."""

        context = []
        for name, df in self.datasets.items():
            meta = self.metadata.get(name, {})
            context.append(f"""
{name}:
  - Shape: {df.shape[0]} rows × {df.shape[1]} columns
  - Columns: {list(df.columns)}
  - Domain: {meta.get('domain', 'Unknown')}
  - Description: {meta.get('description', 'No description')[:100]}
  - Data types: {df.dtypes.to_dict()}
  - Sample: {df.head(2).to_dict()}
""")

        return '\n'.join(context)

    def generate_executive_summary(self) -> str:
        """Generate an executive summary of findings."""

        if not self.insights_discovered:
            return "No significant insights discovered yet."

        summary = ["## Executive Summary\n"]
        summary.append(f"After analyzing {len(self.datasets)} datasets with {self.execution_history} analytical queries:\n")

        # Group insights by category
        revenue_insights = [i for i in self.insights_discovered if 'revenue' in i.lower() or 'sales' in i.lower()]
        customer_insights = [i for i in self.insights_discovered if 'customer' in i.lower() or 'churn' in i.lower()]
        product_insights = [i for i in self.insights_discovered if 'product' in i.lower() or 'margin' in i.lower()]

        if revenue_insights:
            summary.append("\n### Revenue & Growth")
            for insight in revenue_insights[:3]:
                summary.append(f"• {insight}")

        if customer_insights:
            summary.append("\n### Customer Analysis")
            for insight in customer_insights[:3]:
                summary.append(f"• {insight}")

        if product_insights:
            summary.append("\n### Product Performance")
            for insight in product_insights[:3]:
                summary.append(f"• {insight}")

        return '\n'.join(summary)
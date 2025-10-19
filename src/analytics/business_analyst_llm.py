"""Business analyst LLM that writes and executes analysis code dynamically."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
import google.generativeai as genai
import os
from src.analytics.dynamic_explorer import DynamicDataExplorer


class BusinessAnalystLLM:
    """LLM that acts as a business analyst, writing code to answer questions."""

    def __init__(self):
        """Initialize the business analyst."""
        self.explorer = DynamicDataExplorer()

        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            # Use the correct model name
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

        self.analysis_history = []

        # Track global execution counter (like autonomous_explorer's PythonExecutorTool)
        self.execution_counter = 0

    def analyze_business_question(self,
                                 question: str,
                                 datasets: Dict[str, pd.DataFrame],
                                 max_attempts: int = 3,
                                 viz_data_store=None) -> Dict[str, Any]:
        """Analyze a business question by writing and executing code.

        Args:
            question: Business question to analyze
            datasets: Dictionary of dataframes
            max_attempts: Maximum attempts to generate working code
            viz_data_store: Optional VisualizationDataStore for saving viz data
        """

        if not self.model:
            return {"error": "No LLM configured"}

        # Build dataset context
        dataset_info = self._build_dataset_info(datasets)

        # Create the prompt for code generation
        prompt = self._create_analysis_prompt(question, dataset_info)

        previous_codes = []
        previous_errors = []

        # Try to generate and execute working code
        for attempt in range(max_attempts):
            try:
                print(f"\n[LLM] Attempt {attempt + 1}/{max_attempts} for: {question[:50]}...")

                # Generate code
                response = self.model.generate_content(prompt)
                code = self._extract_code(response.text)

                if not code:
                    # If no code block found, try to extract it differently
                    code = self._extract_code_fallback(response.text)

                if not code:
                    print("[LLM] No code found in response, retrying...")
                    prompt = self._create_analysis_prompt(question, dataset_info)
                    prompt += "\n\nPLEASE PROVIDE CODE IN A ```python CODE BLOCK```"
                    continue

                print(f"[LLM] Generated {len(code.split(chr(10)))} lines of code")

                # Execute the code using multi-table executor
                from src.discovery.multi_table_executor import MultiTableCodeExecutor
                executor = MultiTableCodeExecutor(datasets)
                execution = executor.execute(code)

                # Increment global execution counter (like PythonExecutorTool)
                self.execution_counter += 1
                execution_number = self.execution_counter

                # Check if execution returned chart data for visualization
                if execution.success and execution.result and viz_data_store:
                    # If result is a dict with chart_type, save it (like PythonExecutorTool does)
                    if isinstance(execution.result, dict) and 'chart_type' in execution.result:
                        try:
                            viz_data_store.add_visualization(
                                execution_number=execution_number,  # Use global execution counter
                                chart_data=execution.result,
                                title=execution.result.get('title', f"Execution {execution_number}"),
                                description=f"Visualization for: {question}",
                                question=question  # Group multiple vizs for same question
                            )
                            print(f"[VIZ] Saved chart {execution_number}: {execution.result.get('chart_type', 'unknown')}")
                        except Exception as e:
                            print(f"[VIZ] Failed to save visualization: {str(e)[:50]}")

                if not execution.success:
                    # Store failed attempt
                    previous_codes.append(code)
                    previous_errors.append(execution.error)

                    # Create detailed error feedback
                    error_feedback = f"""
The code failed with this error:
{execution.error}

COMMON ISSUES TO FIX:
1. Undefined variables - make sure to define ALL variables before use
2. Missing data checks - always check if datasets exist with: if 'dataset_name' in locals():
3. Incorrect column names - use the exact column names from the dataset info

Please write CORRECTED code that:
- Defines all variables before using them
- Checks for dataset existence
- Handles missing data gracefully
"""
                    # Update prompt for retry
                    prompt = self._create_analysis_prompt(question, dataset_info)
                    prompt += error_feedback
                    print(f"[LLM] Execution failed: {execution.error.split(chr(10))[0]}")
                    continue

                # Success! Extract insights
                print("[LLM] Code executed successfully!")
                insights = self._extract_insights(execution.stdout, question)

                result = {
                    'question': question,
                    'code': code,
                    'output': execution.stdout,
                    'insights': insights,
                    'success': True,
                    'attempts': attempt + 1
                }

                # Visualization data is now saved inline during execution (see above)
                # No need to parse text output

                self.analysis_history.append(result)
                return result

            except Exception as e:
                print(f"[LLM] Attempt {attempt + 1} failed with exception: {e}")

        # All attempts failed
        return {
            'question': question,
            'error': 'Failed to generate working code after all attempts',
            'failed_codes': previous_codes,
            'errors': previous_errors,
            'success': False
        }

    def _create_analysis_prompt(self, question: str, dataset_info: str) -> str:
        """Create a prompt for the LLM to generate analysis code."""

        return f"""You are a business analyst. Write Python code to answer this question:

{question}

AVAILABLE DATA:
{dataset_info}

CRITICAL RULES - MUST FOLLOW:
1. Each dataset is ALREADY available as a DataFrame variable - DO NOT recreate them!
2. NEVER create mock data or redefine the datasets - they already exist!
3. ALWAYS check if a dataset exists before using it: if 'dataset_name' in locals():
4. NEVER use a variable you haven't defined yet! Always create variables BEFORE using them
5. NEVER write: high_risk = df[at_risk > 0.5] if 'at_risk' doesn't exist
6. ALWAYS write: high_risk = df[df['column_name'] > 0.5] using the actual column name
7. Print specific findings with actual numbers from the REAL data

VISUALIZATION RULES:
- To create a chart for Plotly dashboards, return a dictionary with chart data
- Use this format: result = {{'chart_type': 'bar', 'data': [...values...], 'labels': [...labels...], 'title': '...'}}
- Chart types: 'bar', 'line', 'pie', 'scatter'
- Example:
  ```python
  # After calculating top_products
  result = {{
      'chart_type': 'bar',
      'data': top_products['revenue'].tolist(),
      'labels': top_products['product_name'].tolist(),
      'title': 'Top 10 Products by Revenue'
  }}
  ```
- The result variable will be automatically saved as visualization data

WORKING EXAMPLE (DO NOT CREATE MOCK DATA):
```python
import pandas as pd
import numpy as np

# DO NOT create any test/mock data - the datasets already exist!
# Check and analyze the REAL customer data
if 'customer_profiles' in locals() and 'sales_transactions' in locals():
    # First calculate revenue per customer
    customer_revenue = sales_transactions.groupby('customer_id')['net_amount'].sum().reset_index()
    customer_revenue.columns = ['customer_id', 'total_revenue']

    # Then merge with profiles (DEFINING merged_data variable)
    merged_data = customer_profiles.merge(customer_revenue, on='customer_id', how='left')

    # Now we can use merged_data (it's been defined above)
    high_value = merged_data['total_revenue'].quantile(0.8)
    top_customers = merged_data[merged_data['total_revenue'] >= high_value]

    print(f"FINDING: {{len(top_customers)}} customers are in top 20% by revenue")
    print(f"FINDING: They generate ${{top_customers['total_revenue'].sum():,.2f}} total")
elif 'sales_transactions' in locals():
    # Fallback if only sales data available
    total_rev = sales_transactions['net_amount'].sum()
    print(f"FINDING: Total revenue is ${{total_rev:,.2f}}")
else:
    print("Required datasets not available")
```

Write code for: {question}

REMEMBER:
- EVERY variable must be defined before use
- Use if/elif/else to handle different data availability
- Never assume a variable exists - always define it first
"""

    def _build_dataset_info(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Build information about available datasets."""

        info = []
        for name, df in datasets.items():
            # Get column info with types
            columns_info = []
            for col in df.columns[:10]:  # Show first 10 columns
                dtype = str(df[col].dtype)
                columns_info.append(f"{col} ({dtype})")

            info.append(f"""
{name}: {df.shape[0]} rows × {df.shape[1]} columns
  Columns: {', '.join(columns_info)}
  Sample values: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}""")

        return '\n'.join(info)

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response."""

        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Try with just ```
        code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        return None

    def _extract_code_fallback(self, text: str) -> Optional[str]:
        """Try to extract code even without proper markdown."""

        # Look for import statements as start
        if 'import pandas' in text or 'import numpy' in text:
            # Find where code likely starts and ends
            lines = text.split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                if 'import' in line or (in_code and line.strip()):
                    in_code = True
                    code_lines.append(line)
                elif in_code and not line.strip() and len(code_lines) > 5:
                    # Probably end of code
                    break

            if code_lines:
                return '\n'.join(code_lines)

        return None

    def _extract_insights(self, output: str, question: str) -> List[str]:
        """Extract insights from execution output."""

        insights = []

        if not output:
            return insights

        lines = output.strip().split('\n')
        for line in lines:
            # Look for explicit findings
            if 'FINDING:' in line:
                insights.append(line.replace('FINDING:', '').strip())
            # Look for lines with specific numbers
            elif '$' in line and any(char.isdigit() for char in line):
                insights.append(line.strip())
            elif '%' in line and any(char.isdigit() for char in line):
                insights.append(line.strip())

        return insights

    def _save_viz_data(self, viz_data_store, question: str, output: str, insights: List[str]):
        """Save visualization data for dashboard generation."""
        import re
        import json

        # Parse the output for numeric findings
        viz_entries = []

        # Look for patterns like "Top 5 customers" or "revenue by X"
        lines = output.split('\n')
        current_data = None

        for line in lines:
            # Detect headers like "Top 5" or "By industry"
            if re.search(r'(Top \d+|by \w+|average|total|breakdown)', line, re.IGNORECASE):
                if current_data:
                    viz_entries.append(current_data)
                current_data = {
                    'title': line.strip(),
                    'data': [],
                    'viz_type': 'bar'  # Default viz type
                }
            # Detect data lines with numbers
            elif current_data and ('$' in line or '%' in line):
                # Try to extract label and value
                parts = re.split(r'[:|-]', line)
                if len(parts) >= 2:
                    label = parts[0].strip()
                    value_str = parts[-1].strip()
                    # Extract numeric value
                    value_match = re.search(r'[\d,]+\.?\d*', value_str.replace(',', ''))
                    if value_match:
                        value = float(value_match.group())
                        current_data['data'].append({'label': label, 'value': value})

        if current_data and current_data['data']:
            viz_entries.append(current_data)

        # Save each visualization entry
        for i, viz_entry in enumerate(viz_entries):
            if viz_entry['data']:
                viz_data_store.add_visualization(
                    insight_id=f"{question[:30]}_{i}",
                    viz_type=viz_entry['viz_type'],
                    data=viz_entry['data'],
                    title=viz_entry.get('title', question),
                    description=f"Analysis for: {question}"
                )

    def explore_business_data(self,
                             datasets: Dict[str, pd.DataFrame],
                             focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explore business data with specific focus areas."""

        if not focus_areas:
            focus_areas = [
                "revenue and profitability",
                "customer behavior and retention",
                "product performance",
                "operational efficiency"
            ]

        all_results = []

        for area in focus_areas:
            print(f"\n📊 Analyzing: {area}")
            print("-" * 40)

            # Generate specific questions for this area
            questions = self._generate_area_questions(area, datasets)

            for question in questions[:2]:  # Limit to 2 questions per area
                print(f"\n❓ {question}")

                result = self.analyze_business_question(question, datasets)

                if result.get('success'):
                    print("✅ Analysis successful")
                    for insight in result.get('insights', []):
                        print(f"  → {insight}")
                else:
                    print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

                all_results.append(result)

        return {
            'analyses': all_results,
            'successful': sum(1 for r in all_results if r.get('success')),
            'failed': sum(1 for r in all_results if not r.get('success'))
        }

    def _generate_area_questions(self, area: str, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate specific questions for a focus area based on available data."""

        questions = []

        if "revenue" in area.lower():
            if 'sales_transactions' in datasets:
                questions.append("What are the top 10 revenue-generating customers and what percentage of total revenue do they represent?")
            if 'sales_daily_metrics' in datasets:
                questions.append("What is the trend in daily revenue over the last 3 months and what's driving any changes?")

        elif "customer" in area.lower():
            if 'customer_profiles' in datasets:
                questions.append("Which high-value customers (top 20% by revenue) have high churn risk (>70%) and need immediate attention?")
            if 'sales_transactions' in datasets and 'customer_profiles' in datasets:
                questions.append("What is the average purchase frequency and order value by customer segment?")

        elif "product" in area.lower():
            if 'product_information' in datasets:
                questions.append("Which products have the highest profit margins and how much revenue do they generate?")
            if 'sales_transactions' in datasets and 'product_information' in datasets:
                questions.append("What products are frequently bought together and what's the cross-sell opportunity?")

        elif "operational" in area.lower():
            if 'sales_transactions' in datasets:
                questions.append("How effective is our discount strategy - what's the impact on revenue and margins?")
            if 'sales_daily_metrics' in datasets:
                questions.append("What's the ROI on marketing spend and how has it changed over time?")

        return questions if questions else ["Analyze the key metrics and trends in the available data"]
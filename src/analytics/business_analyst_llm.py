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
            genai.configure(api_key=api_key, transport='rest')  # Use REST to avoid gRPC ALTS warnings
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
        prompt = self._create_analysis_prompt(question, dataset_info, datasets)

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
                    prompt = self._create_analysis_prompt(question, dataset_info, datasets)
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
                    prompt = self._create_analysis_prompt(question, dataset_info, datasets)
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

    def _create_analysis_prompt(self, question: str, dataset_info: str, datasets: Dict[str, pd.DataFrame] = None) -> str:
        """Create a prompt for the LLM to generate analysis code."""

        # Generate dynamic example from actual datasets
        example_code = self._generate_dynamic_example(datasets) if datasets else self._generate_generic_example()

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
  # After calculating top_items
  result = {{
      'chart_type': 'bar',
      'data': top_items['value'].tolist(),
      'labels': top_items['label'].tolist(),
      'title': 'Top 10 Items'
  }}
  ```
- The result variable will be automatically saved as visualization data

WORKING EXAMPLE (using YOUR actual data):
{example_code}

Write code for: {question}

REMEMBER:
- EVERY variable must be defined before use
- Use if/elif/else to handle different data availability
- Never assume a variable exists - always define it first
"""

    def _generate_dynamic_example(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Generate example code using actual dataset names and columns."""
        if not datasets:
            return self._generate_generic_example()

        # Get first dataset
        first_dataset_name = list(datasets.keys())[0]
        first_df = datasets[first_dataset_name]

        # Find numeric columns
        numeric_cols = first_df.select_dtypes(include=['number']).columns.tolist()

        # Find categorical/text columns
        text_cols = first_df.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) > 0 and len(text_cols) > 0:
            # Example with grouping
            numeric_col = numeric_cols[0]
            group_col = text_cols[0]

            example = f"""```python
import pandas as pd
import numpy as np

# DO NOT create any test/mock data - the datasets already exist!
# Analyze the actual {first_dataset_name} dataset
if '{first_dataset_name}' in locals():
    # Calculate aggregate statistics
    grouped = {first_dataset_name}.groupby('{group_col}')['{numeric_col}'].agg(['sum', 'mean', 'count']).reset_index()
    grouped.columns = ['{group_col}', 'total_{numeric_col}', 'avg_{numeric_col}', 'count']

    # Sort by total and get top 10
    top_items = grouped.nlargest(10, 'total_{numeric_col}')

    print(f"FINDING: Top 10 {group_col} by {numeric_col}")
    print(f"FINDING: Total {numeric_col}: {{top_items['total_{numeric_col}'].sum():,.2f}}")
    print(f"FINDING: Average {numeric_col}: {{top_items['avg_{numeric_col}'].mean():,.2f}}")
else:
    print("Dataset not available")
```"""
        elif len(numeric_cols) > 0:
            # Example with numeric analysis only
            numeric_col = numeric_cols[0]

            example = f"""```python
import pandas as pd
import numpy as np

# DO NOT create any test/mock data - the datasets already exist!
# Analyze the actual {first_dataset_name} dataset
if '{first_dataset_name}' in locals():
    # Calculate statistics
    avg_value = {first_dataset_name}['{numeric_col}'].mean()
    total_value = {first_dataset_name}['{numeric_col}'].sum()
    max_value = {first_dataset_name}['{numeric_col}'].max()

    print(f"FINDING: Average {numeric_col}: {{avg_value:,.2f}}")
    print(f"FINDING: Total {numeric_col}: {{total_value:,.2f}}")
    print(f"FINDING: Maximum {numeric_col}: {{max_value:,.2f}}")
else:
    print("Dataset not available")
```"""
        else:
            # Fallback to basic analysis
            example = f"""```python
import pandas as pd
import numpy as np

# DO NOT create any test/mock data - the datasets already exist!
# Analyze the actual {first_dataset_name} dataset
if '{first_dataset_name}' in locals():
    # Basic dataset analysis
    row_count = len({first_dataset_name})
    col_count = len({first_dataset_name}.columns)

    print(f"FINDING: Dataset has {{row_count:,}} rows and {{col_count}} columns")
    print(f"FINDING: Columns: {{list({first_dataset_name}.columns)}}")
else:
    print("Dataset not available")
```"""

        return example

    def _generate_generic_example(self) -> str:
        """Generate a generic example when datasets are not available."""
        return """```python
import pandas as pd
import numpy as np

# Check if dataset exists before using it
if 'your_dataset' in locals():
    # Analyze the dataset
    result = your_dataset.describe()
    print(f"FINDING: Dataset has {len(your_dataset)} rows")
else:
    print("Dataset not available")
```"""

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
        """Generate specific questions for a focus area based on actual data structure.

        This method analyzes the actual columns and data types to generate relevant questions,
        without making ANY assumptions about dataset names or business domain.
        """

        questions = []

        # Analyze actual data structure
        data_capabilities = self._analyze_data_capabilities(datasets)

        # Generate questions based on analytical capabilities, not domain assumptions
        if "revenue" in area.lower() or "profitability" in area.lower() or "performance" in area.lower():
            # Focus on numeric aggregations and trends
            if data_capabilities['numeric_aggregatable']:
                dataset_name = data_capabilities['numeric_aggregatable'][0]['dataset']
                numeric_col = data_capabilities['numeric_aggregatable'][0]['numeric_cols'][0]
                group_col = data_capabilities['numeric_aggregatable'][0]['group_cols'][0]

                questions.append(f"What are the top performing {group_col} values in {dataset_name} based on {numeric_col}?")

                if data_capabilities['has_time_data']:
                    questions.append(f"What are the trends and patterns in {numeric_col} over time in the {dataset_name} dataset?")

            elif data_capabilities['numeric_columns']:
                # Just numeric data without good grouping columns
                dataset_name = data_capabilities['numeric_columns'][0]['dataset']
                numeric_col = data_capabilities['numeric_columns'][0]['columns'][0]
                questions.append(f"What is the distribution and statistical profile of {numeric_col} in {dataset_name}?")

        elif "segment" in area.lower() or "group" in area.lower() or "cluster" in area.lower():
            # Focus on segmentation and grouping
            if data_capabilities['numeric_aggregatable']:
                dataset_name = data_capabilities['numeric_aggregatable'][0]['dataset']
                group_col = data_capabilities['numeric_aggregatable'][0]['group_cols'][0]
                questions.append(f"What are the distinct segments in {dataset_name} when grouping by {group_col}?")

                if len(data_capabilities['numeric_aggregatable'][0]['numeric_cols']) > 0:
                    numeric_col = data_capabilities['numeric_aggregatable'][0]['numeric_cols'][0]
                    questions.append(f"How do different {group_col} segments compare in terms of {numeric_col}?")

        elif "trend" in area.lower() or "time" in area.lower() or "temporal" in area.lower():
            # Focus on time-based analysis
            if data_capabilities['has_time_data']:
                time_info = data_capabilities['time_data_info'][0]
                dataset_name = time_info['dataset']
                time_col = time_info['time_col']

                if time_info['numeric_cols']:
                    numeric_col = time_info['numeric_cols'][0]
                    questions.append(f"What are the temporal trends and patterns in {numeric_col} over {time_col} in {dataset_name}?")
                    questions.append(f"Are there any seasonal or cyclical patterns in the {dataset_name} data?")
                else:
                    questions.append(f"What are the temporal patterns and frequencies in {dataset_name} over {time_col}?")

        elif "correlation" in area.lower() or "relationship" in area.lower():
            # Focus on correlations between variables
            if data_capabilities['multi_numeric']:
                dataset_name = data_capabilities['multi_numeric'][0]['dataset']
                numeric_cols = data_capabilities['multi_numeric'][0]['columns'][:3]  # Take first 3
                questions.append(f"What are the correlations and relationships between {', '.join(numeric_cols)} in {dataset_name}?")

        elif "distribution" in area.lower() or "statistical" in area.lower():
            # Focus on statistical distributions
            if data_capabilities['numeric_columns']:
                dataset_name = data_capabilities['numeric_columns'][0]['dataset']
                numeric_cols = data_capabilities['numeric_columns'][0]['columns'][:2]
                questions.append(f"What are the statistical distributions and outliers in {', '.join(numeric_cols)} from {dataset_name}?")

        # If no questions generated, create generic analytical questions based on capabilities
        if not questions:
            if data_capabilities['numeric_aggregatable']:
                dataset_name = data_capabilities['numeric_aggregatable'][0]['dataset']
                questions.append(f"What are the key patterns and insights when analyzing {dataset_name} data?")
            elif data_capabilities['numeric_columns']:
                dataset_name = data_capabilities['numeric_columns'][0]['dataset']
                questions.append(f"What is the statistical profile and distribution of numeric values in {dataset_name}?")
            else:
                # Fallback to first dataset
                first_dataset = list(datasets.keys())[0]
                questions.append(f"What are the key patterns and characteristics in the {first_dataset} dataset?")

        return questions

    def _analyze_data_capabilities(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze what analytical capabilities exist based on actual data structure.

        Returns a dictionary describing what types of analysis are possible:
        - numeric_columns: Datasets with numeric columns
        - categorical_columns: Datasets with categorical columns
        - numeric_aggregatable: Datasets with both numeric and categorical (good for groupby)
        - has_time_data: Whether time-series analysis is possible
        - multi_numeric: Datasets with multiple numeric columns (good for correlation)
        """

        capabilities = {
            'numeric_columns': [],
            'categorical_columns': [],
            'numeric_aggregatable': [],
            'has_time_data': False,
            'time_data_info': [],
            'multi_numeric': []
        }

        for dataset_name, df in datasets.items():
            # Identify column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            # Numeric columns
            if numeric_cols:
                capabilities['numeric_columns'].append({
                    'dataset': dataset_name,
                    'columns': numeric_cols
                })

            # Categorical columns
            if categorical_cols:
                capabilities['categorical_columns'].append({
                    'dataset': dataset_name,
                    'columns': categorical_cols
                })

            # Aggregatable (both numeric and categorical)
            if numeric_cols and categorical_cols:
                capabilities['numeric_aggregatable'].append({
                    'dataset': dataset_name,
                    'numeric_cols': numeric_cols,
                    'group_cols': categorical_cols
                })

            # Time data
            if datetime_cols:
                capabilities['has_time_data'] = True
                capabilities['time_data_info'].append({
                    'dataset': dataset_name,
                    'time_col': datetime_cols[0],
                    'numeric_cols': numeric_cols
                })

            # Multiple numeric columns (for correlation analysis)
            if len(numeric_cols) >= 2:
                capabilities['multi_numeric'].append({
                    'dataset': dataset_name,
                    'columns': numeric_cols
                })

        return capabilities
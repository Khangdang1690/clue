"""LangChain tool for executing Python code in autonomous exploration."""

from typing import Optional, Type, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from src.discovery.code_executor import DataFrameExecutor, ExecutionResult
import pandas as pd


class PythonCodeInput(BaseModel):
    """Input schema for Python code execution."""
    code: str = Field(
        description="Python code to execute. The DataFrame is available as 'df'. "
                    "Store your result in a variable called 'result' if you want to return it. "
                    "Use print() to display intermediate values."
    )


class PythonExecutorTool(BaseTool):
    """
    LangChain tool that executes Python code for data exploration.

    This tool allows the LLM to write and execute Python code to analyze the dataset.
    """

    name: str = "python_executor"
    description: str = """
    Execute Python code to explore and analyze the dataset.

    Use this tool to:
    - Calculate statistics (mean, median, std, percentiles)
    - Group data by categories and compare groups
    - Find correlations and relationships
    - Detect patterns, trends, and anomalies
    - Perform statistical tests
    - Create derived metrics
    - Answer analytical questions about the data

    The DataFrame is available as 'df'. Common operations:
    - df.describe() - Statistical summary
    - df.groupby('column')['value'].mean() - Group analysis
    - df.corr() - Correlation matrix
    - df['col'].value_counts() - Frequency distribution
    - pd.crosstab(df['col1'], df['col2']) - Cross-tabulation

    Store your final result in a variable called 'result'.
    Use print() statements to show intermediate calculations.

    Example:
    ```python
    # Calculate year-over-year growth
    df['year'] = pd.to_datetime(df['date']).dt.year
    yoy_growth = df.groupby('year')['revenue'].sum().pct_change() * 100
    result = yoy_growth.to_dict()
    print(f"YoY Growth: {result}")
    ```
    """
    args_schema: Type[BaseModel] = PythonCodeInput

    # Custom attributes (not part of BaseTool)
    executor: Any = Field(default=None, exclude=True)
    execution_history: list = Field(default_factory=list, exclude=True)
    viz_output_dir: Optional[str] = Field(default=None, exclude=True)  # Declare as Field

    def __init__(self, df: pd.DataFrame, viz_output_dir: Optional[str] = None, **kwargs):
        """
        Initialize the Python executor tool.

        Args:
            df: DataFrame to analyze
            viz_output_dir: Directory to save visualizations (optional)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.executor = DataFrameExecutor(df, viz_output_dir=viz_output_dir)
        self.execution_history = []
        self.viz_output_dir = viz_output_dir

    def _run(self, code: str) -> str:
        """
        Execute Python code and return formatted result.

        Args:
            code: Python code to execute

        Returns:
            Formatted string with execution results
        """
        # Clean code: remove markdown code fences if present
        code = code.strip()
        if code.startswith('```python'):
            code = code[len('```python'):].strip()
        elif code.startswith('```'):
            code = code[len('```'):].strip()

        if code.endswith('```'):
            code = code[:-len('```')].strip()

        # Execute code
        result = self.executor.execute_analysis(code)

        # Store in history
        self.execution_history.append({
            'code': code,
            'result': result
        })

        # Format output for LLM
        output_parts = []

        if result.success:
            output_parts.append("[OK] Execution successful")

            if result.stdout:
                # Truncate stdout to prevent context overflow
                stdout_str = result.stdout
                if len(stdout_str) > 2000:
                    stdout_str = stdout_str[:2000] + "\n... (output truncated)"
                output_parts.append(f"\nOutput:\n{stdout_str}")

            if result.result is not None:
                # Truncate result to prevent context overflow
                result_str = str(result.result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... (truncated)"
                output_parts.append(f"\nResult: {result_str}")

            output_parts.append(f"\nExecution time: {result.execution_time:.2f}s")

        else:
            output_parts.append("[ERROR] Execution failed")
            output_parts.append(f"\nError: {result.error}")

            if result.stderr:
                output_parts.append(f"\nStderr:\n{result.stderr}")

        return "\n".join(output_parts)

    async def _arun(self, code: str) -> str:
        """Async version (delegates to sync)."""
        return self._run(code)

    def get_execution_summary(self) -> str:
        """
        Get summary of all code executions.

        Returns:
            Formatted summary of execution history
        """
        if not self.execution_history:
            return "No code has been executed yet."

        summary_parts = [f"Total executions: {len(self.execution_history)}\n"]

        for i, entry in enumerate(self.execution_history, 1):
            result = entry['result']
            status = "[OK]" if result.success else "[ERROR]"

            summary_parts.append(f"\n{i}. {status} Execution {i}")
            summary_parts.append(f"   Code: {entry['code'][:100]}...")

            if result.success and result.result:
                summary_parts.append(f"   Result: {str(result.result)[:200]}")

        return "\n".join(summary_parts)


class DataSummaryTool(BaseTool):
    """
    LangChain tool that provides dataset summary information.
    """

    name: str = "get_data_summary"
    description: str = """
    Get a summary of the dataset structure including:
    - Number of rows and columns
    - Column names and data types
    - Sample data (first few rows)

    Use this tool at the beginning to understand what data is available.
    """
    args_schema: Type[BaseModel] = BaseModel

    executor: Any = Field(default=None, exclude=True)

    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize the data summary tool.

        Args:
            df: DataFrame to summarize
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.executor = DataFrameExecutor(df)

    def _run(self) -> str:
        """Get dataset summary."""
        return self.executor.get_data_summary()

    async def _arun(self) -> str:
        """Async version."""
        return self._run()


class DataProfileTool(BaseTool):
    """
    LangChain tool for getting detailed data profile.
    """

    name: str = "profile_data"
    description: str = """
    Get detailed statistical profile of the dataset:
    - Numeric columns: mean, median, std, min, max, quartiles
    - Categorical columns: unique values, most common values
    - Missing value analysis
    - Data quality metrics

    Use this to understand the statistical properties of each column.
    """
    args_schema: Type[BaseModel] = BaseModel

    df: Any = Field(default=None, exclude=True)

    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize the data profile tool.

        Args:
            df: DataFrame to profile
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.df = df

    def _run(self) -> str:
        """Get detailed data profile."""
        profile_parts = []

        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            profile_parts.append("Numeric Columns:")
            profile_parts.append(self.df[numeric_cols].describe().to_string())

        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            profile_parts.append("\n\nCategorical Columns:")
            for col in cat_cols[:10]:  # Limit to 10
                unique_vals = self.df[col].nunique()
                top_vals = self.df[col].value_counts().head(3).to_dict()
                profile_parts.append(f"  {col}: {unique_vals} unique values")
                profile_parts.append(f"    Top 3: {top_vals}")

        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            profile_parts.append("\n\nMissing Values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.df)) * 100
                profile_parts.append(f"  {col}: {count} ({pct:.1f}%)")

        return "\n".join(profile_parts)

    async def _arun(self) -> str:
        """Async version."""
        return self._run()


if __name__ == "__main__":
    print("PythonTool module - ready for use")
    print("Import: from src.discovery.python_tool import PythonExecutorTool, DataSummaryTool, DataProfileTool")

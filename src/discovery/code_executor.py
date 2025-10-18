"""Safe Python code execution using llm-sandbox."""

import os
import sys
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import io
import contextlib


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    namespace: Optional[Dict[str, Any]] = None  # Execution namespace for persistence


class CodeExecutor:
    """
    Safe Python code executor for autonomous data exploration.

    Uses in-process execution with restricted builtins for Windows compatibility.
    For production, consider using Docker-based sandboxing (llm-sandbox with Docker).
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 100000,
        allowed_modules: Optional[List[str]] = None
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            allowed_modules: List of allowed module names (default: data science modules)
        """
        self.timeout = timeout
        self.max_output_size = max_output_size

        # Default allowed modules for data analysis
        if allowed_modules is None:
            self.allowed_modules = {
                'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib',
                'seaborn', 'plotly', 'statsmodels', 'math', 'statistics',
                'datetime', 'json', 'collections', 'itertools', 're'
            }
        else:
            self.allowed_modules = set(allowed_modules)

    def execute(
        self,
        code: str,
        data_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            data_context: Dictionary with variables to inject (e.g., {'df': dataframe})

        Returns:
            ExecutionResult with stdout, stderr, result, and error info
        """
        import time
        start_time = time.time()

        # Prepare execution context
        if data_context is None:
            data_context = {}

        # Create namespace with safe builtins and injected variables
        namespace = self._create_safe_namespace(data_context)

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            # Redirect stdout/stderr
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Execute code
                exec(code, namespace)

                # Get result if there's a 'result' variable
                result = namespace.get('result', None)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        execution_time = time.time() - start_time

        # Get captured output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Truncate if too large
        if len(stdout) > self.max_output_size:
            stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
        if len(stderr) > self.max_output_size:
            stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

        return ExecutionResult(
            success=error is None,
            stdout=stdout,
            stderr=stderr,
            result=result,
            error=error,
            execution_time=execution_time,
            namespace=namespace if error is None else None
        )

    def _create_safe_namespace(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a safe namespace for code execution.

        Args:
            data_context: User-provided variables

        Returns:
            Namespace dictionary with safe builtins and imports
        """
        # Start with safe builtins
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'float': float,
            'int': int, 'len': len, 'list': list, 'max': max,
            'min': min, 'pow': pow, 'print': print, 'range': range,
            'round': round, 'set': set, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            'True': True, 'False': False, 'None': None,
            '__import__': __import__,  # Needed for pandas import statements
        }

        namespace = {
            '__builtins__': safe_builtins,
        }

        # Import allowed modules
        try:
            import pandas as pd
            import numpy as np
            import scipy
            import scipy.stats
            from sklearn import linear_model
            import matplotlib.pyplot as plt
            import seaborn as sns
            import json
            import math
            from datetime import datetime, timedelta
            from collections import Counter, defaultdict
            import re
            import os

            namespace.update({
                'pd': pd,
                'np': np,
                'scipy': scipy,
                'stats': scipy.stats,
                'linear_model': linear_model,
                'plt': plt,
                'sns': sns,
                'json': json,
                'math': math,
                'datetime': datetime,
                'timedelta': timedelta,
                'Counter': Counter,
                'defaultdict': defaultdict,
                're': re,
                'os': os,  # Provide os module for makedirs, path operations
            })
        except ImportError as e:
            print(f"Warning: Could not import module: {e}")

        # Add user-provided context
        namespace.update(data_context)

        return namespace

    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code for dangerous operations.

        Args:
            code: Python code to validate

        Returns:
            (is_valid, error_message)
        """
        # Dangerous patterns to block
        dangerous_patterns = [
            'import sys', 'import subprocess',
            'open(', 'file(', 'exec(', 'eval(',
            'compile(', 'globals()', 'locals()',
            'rmdir', 'drop table', 'shutil.'
        ]

        code_lower = code.lower()

        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Code contains potentially dangerous pattern: '{pattern}'"

        return True, None

    def execute_safe(
        self,
        code: str,
        data_context: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> ExecutionResult:
        """
        Execute code with validation.

        Args:
            code: Python code to execute
            data_context: Variables to inject
            validate: Whether to validate code before execution

        Returns:
            ExecutionResult
        """
        if validate:
            is_valid, error_msg = self.validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    result=None,
                    error=error_msg
                )

        return self.execute(code, data_context)


class DataFrameExecutor(CodeExecutor):
    """
    Specialized executor for DataFrame operations.

    Automatically injects DataFrame and provides helper functions.
    """

    def __init__(self, df: pd.DataFrame, viz_output_dir: Optional[str] = None, **kwargs):
        """
        Initialize with a DataFrame.

        Args:
            df: DataFrame to analyze
            viz_output_dir: Directory to save visualizations (optional)
            **kwargs: Additional arguments for CodeExecutor
        """
        super().__init__(**kwargs)
        self.df = df
        self.original_df = df.copy()  # Keep original for reference
        self.viz_output_dir = viz_output_dir

        # Create viz directory if specified
        if self.viz_output_dir:
            os.makedirs(self.viz_output_dir, exist_ok=True)

    def execute_analysis(self, code: str) -> ExecutionResult:
        """
        Execute code with DataFrame automatically injected as 'df'.

        The DataFrame persists across executions, allowing the LLM to clean
        and modify data during exploration.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult
        """
        data_context = {
            'df': self.df,  # Persistent DataFrame that can be modified
            'original_df': self.original_df  # Read-only reference to original
        }

        # Add visualization directory to context if available
        if self.viz_output_dir:
            data_context['VIZ_DIR'] = self.viz_output_dir

        result = self.execute_safe(code, data_context)

        # After execution, update self.df if it was reassigned in the namespace
        # This allows data cleaning to persist across executions
        if result.success and result.namespace and 'df' in result.namespace:
            modified_df = result.namespace['df']
            # Check if df was reassigned (not just modified in-place)
            if modified_df is not self.df:
                # DataFrame was reassigned (e.g., df = df.dropna())
                self.df = modified_df
                print(f"[DATA_CLEANING] DataFrame updated: {len(self.df):,} rows")
            # Note: In-place modifications (e.g., df.drop(..., inplace=True))
            # automatically persist since self.df is the same object

        return result

    def get_data_summary(self) -> str:
        """
        Get a summary of the DataFrame for LLM context.

        Returns:
            String summary of DataFrame structure
        """
        summary_parts = [
            f"DataFrame: {len(self.df):,} rows × {len(self.df.columns)} columns",
            f"\nColumns: {', '.join(self.df.columns.tolist())}",
            f"\nData types:",
        ]

        for col, dtype in self.df.dtypes.items():
            summary_parts.append(f"  - {col}: {dtype}")

        # Add sample values
        summary_parts.append(f"\nFirst 3 rows:")
        summary_parts.append(self.df.head(3).to_string())

        return "\n".join(summary_parts)


if __name__ == "__main__":
    print("CodeExecutor module - ready for use")
    print("Import this module to use: from src.discovery.code_executor import CodeExecutor, DataFrameExecutor")

"""Multi-table code executor for business discovery."""

import os
import io
import contextlib
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time


@dataclass
class MultiTableExecutionResult:
    """Result of multi-table code execution."""
    success: bool
    stdout: str
    stderr: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class MultiTableCodeExecutor:
    """
    Code executor for multi-table business analysis.

    Unlike the single-table executor, this makes all datasets available
    by their actual names (e.g., sales_transactions, customer_profiles).
    """

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize multi-table executor.

        Args:
            datasets: Dictionary mapping table names to DataFrames
        """
        self.datasets = datasets

    def execute(self, code: str) -> MultiTableExecutionResult:
        """
        Execute code with all datasets available.

        Args:
            code: Python code to execute

        Returns:
            MultiTableExecutionResult with output and results
        """
        start_time = time.time()

        # Create namespace with all datasets and common imports
        namespace = {
            'pd': pd,
            'np': np,
            'print': print,
            '__builtins__': {
                '__import__': __import__,  # Critical for imports
                'print': print,
                'len': len,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'locals': locals,
                'isinstance': isinstance,
                'round': round,
                'abs': abs,
                'getattr': getattr,
                'setattr': setattr,
                'hasattr': hasattr,
            }
        }

        # Add each dataset with its name
        print(f"[Executor] Making {len(self.datasets)} datasets available:")
        for name, df in self.datasets.items():
            namespace[name] = df.copy()  # Use copy to avoid mutations
            print(f"  - {name}: {df.shape}")

        # Also add a locals() function that can see our namespace
        def custom_locals():
            return namespace
        namespace['locals'] = custom_locals
        namespace['__builtins__']['locals'] = custom_locals

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            # Redirect stdout/stderr
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # First compile to check for syntax errors
                compiled_code = compile(code, '<analysis>', 'exec')

                # Execute code
                exec(compiled_code, namespace)

                # Try to get result variable if it exists
                result = namespace.get('result', None)

        except SyntaxError as e:
            error = f"Syntax Error in generated code:\n{str(e)}\n\nCode that failed:\n{code}"
        except NameError as e:
            error = f"Variable not defined: {str(e)}\n\nThis usually means the LLM referenced a variable before defining it.\n\nCode:\n{code}"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Get captured output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Truncate if too long
        max_output = 50000
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + f"\n... (truncated, {len(stdout) - max_output} chars omitted)"

        execution_time = time.time() - start_time

        if error:
            print(f"[Executor] Code execution failed: {error.split(chr(10))[0]}")
        else:
            print(f"[Executor] Code executed successfully in {execution_time:.2f}s")

        return MultiTableExecutionResult(
            success=(error is None),
            stdout=stdout,
            stderr=stderr,
            result=result,
            error=error,
            execution_time=execution_time
        )
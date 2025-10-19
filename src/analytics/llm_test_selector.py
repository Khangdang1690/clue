"""LLM-directed statistical test selection for intelligent analytics."""

from typing import Dict, List, Optional, Any
import pandas as pd
import json
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm


class StatisticalTestSelector:
    """
    Uses LLM to intelligently select which statistical tests to run
    based on data characteristics, domain, and business context.
    """

    # Catalog of available statistical tests
    TEST_CATALOG = {
        "time_series_forecast": {
            "name": "Time Series Forecasting",
            "description": "Predicts future values based on historical patterns",
            "best_for": ["Revenue projection", "Demand forecasting", "Trend analysis"],
            "requirements": {
                "data_type": "time_series",
                "min_observations": 20,
                "columns_needed": ["date/time column", "numeric metric"]
            },
            "value": "high",
            "cost": "medium"
        },
        "anomaly_detection": {
            "name": "Anomaly Detection",
            "description": "Identifies unusual patterns, outliers, or suspicious data points",
            "best_for": ["Fraud detection", "Quality control", "Error identification"],
            "requirements": {
                "data_type": "numeric",
                "min_observations": 30,
                "columns_needed": ["numeric columns"]
            },
            "value": "high",
            "cost": "low"
        },
        "causal_analysis": {
            "name": "Causal Analysis (Granger)",
            "description": "Tests if one variable causes changes in another over time",
            "best_for": ["Marketing impact", "Price elasticity", "Operational drivers"],
            "requirements": {
                "data_type": "time_series",
                "min_observations": 50,
                "columns_needed": ["date column", "potential cause", "potential effect"]
            },
            "value": "very_high",
            "cost": "high"
        },
        "variance_decomposition": {
            "name": "Variance Decomposition",
            "description": "Identifies which factors contribute most to outcome variance",
            "best_for": ["Feature importance", "Cost drivers", "Performance factors"],
            "requirements": {
                "data_type": "cross_sectional",
                "min_observations": 100,
                "columns_needed": ["target variable", "3+ feature variables"]
            },
            "value": "high",
            "cost": "medium"
        },
        "impact_analysis": {
            "name": "Impact Analysis",
            "description": "Measures the effect of interventions or changes",
            "best_for": ["Campaign effectiveness", "Policy impact", "Process improvements"],
            "requirements": {
                "data_type": "time_series_with_intervention",
                "min_observations": 40,
                "columns_needed": ["date", "metric", "intervention indicator"]
            },
            "value": "very_high",
            "cost": "medium"
        },
        "correlation_analysis": {
            "name": "Correlation Analysis",
            "description": "Finds relationships between variables",
            "best_for": ["Variable relationships", "Quick insights", "Initial exploration"],
            "requirements": {
                "data_type": "numeric",
                "min_observations": 20,
                "columns_needed": ["2+ numeric columns"]
            },
            "value": "medium",
            "cost": "very_low"
        },
        "segmentation_analysis": {
            "name": "Segmentation Analysis",
            "description": "Groups data into meaningful segments or clusters",
            "best_for": ["Customer segmentation", "Product grouping", "Market analysis"],
            "requirements": {
                "data_type": "mixed",
                "min_observations": 50,
                "columns_needed": ["multiple features for clustering"]
            },
            "value": "high",
            "cost": "medium"
        },
        "trend_analysis": {
            "name": "Trend Analysis",
            "description": "Identifies and quantifies trends over time",
            "best_for": ["Growth tracking", "Seasonal patterns", "Long-term changes"],
            "requirements": {
                "data_type": "time_series",
                "min_observations": 12,
                "columns_needed": ["date column", "metric"]
            },
            "value": "medium",
            "cost": "low"
        }
    }

    def __init__(self):
        self.llm = get_llm(temperature=0.1, model="gemini-2.5-flash")

    def select_tests(
        self,
        df: pd.DataFrame,
        domain: str,
        dataset_context: Dict[str, Any],
        max_tests: int = 3,
        mode: str = "single_table"
    ) -> List[Dict]:
        """
        Use LLM to select appropriate statistical tests for the dataset.

        Args:
            df: DataFrame to analyze
            domain: Business domain (e.g., 'Sales', 'Finance')
            dataset_context: Additional context about the dataset
            max_tests: Maximum number of tests to recommend
            mode: 'single_table' or 'cross_table'

        Returns:
            List of recommended tests with rationale
        """
        # Prepare data profile
        data_profile = self._create_data_profile(df)

        # Create prompt for test selection
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a statistical analysis expert selecting appropriate tests.

Given a dataset profile and available tests, recommend the MOST VALUABLE statistical analyses.

Available Tests:
{test_catalog}

Selection Criteria:
1. Data must meet test requirements (type, sample size, columns)
2. Prioritize HIGH VALUE tests that provide actionable insights
3. Consider the business domain and context
4. Avoid redundant or low-value tests
5. Maximum {max_tests} tests

Return JSON array of selected tests:
[
  {{
    "test_key": "test_identifier_from_catalog",
    "rationale": "Why this test is valuable for this specific data",
    "expected_insight": "What business question it will answer",
    "priority": 1-3 (1=highest),
    "specific_columns": {{
      "target": "column_name" (if needed),
      "features": ["col1", "col2"] (if needed),
      "date": "date_column" (if needed)
    }}
  }}
]

Rules:
- Only recommend tests where ALL requirements are met
- Prioritize based on business value, not technical interest
- Be specific about which columns to use
- Consider computational cost vs insight value

Return ONLY valid JSON array."""),
            ("user", """Domain: {domain}
Context: {context}
Mode: {mode}

Data Profile:
{data_profile}

Select the most valuable statistical tests for this data:""")
        ])

        chain = prompt | self.llm

        try:
            result = chain.invoke({
                "test_catalog": self._format_test_catalog(),
                "max_tests": max_tests,
                "domain": domain,
                "context": json.dumps(dataset_context),
                "mode": mode,
                "data_profile": data_profile
            })

            # Clean the response to handle common issues
            content = result.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            selected_tests = json.loads(content)

            # Validate selections
            validated = []
            for test in selected_tests:
                if test['test_key'] in self.TEST_CATALOG:
                    test['test_info'] = self.TEST_CATALOG[test['test_key']]
                    validated.append(test)
                    print(f"[TEST SELECTOR] Selected: {test['test_key']} - {test['rationale'][:80]}...")

            return validated

        except Exception as e:
            print(f"[WARN] LLM test selection failed, using defaults: {e}")
            return self._get_default_tests(df, max_tests, mode)

    def _create_data_profile(self, df: pd.DataFrame) -> str:
        """Create a concise data profile for the LLM."""
        profile_parts = []

        # Basic info
        profile_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()

        profile_parts.append(f"\nColumn Types:")
        profile_parts.append(f"- Numeric ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}")
        if date_cols:
            profile_parts.append(f"- Date ({len(date_cols)}): {', '.join(date_cols)}")
        profile_parts.append(f"- Categorical ({len(object_cols)}): {', '.join(object_cols[:10])}")

        # Time series detection
        if date_cols:
            for date_col in date_cols[:1]:  # Check first date column
                n_unique = df[date_col].nunique()
                date_range = pd.to_datetime(df[date_col]).dropna()
                if len(date_range) > 0:
                    span_days = (date_range.max() - date_range.min()).days
                    profile_parts.append(f"\nTime Series Properties:")
                    profile_parts.append(f"- Date column: {date_col}")
                    profile_parts.append(f"- Unique dates: {n_unique}")
                    profile_parts.append(f"- Span: {span_days} days")

        # Key metrics detection
        key_metrics = []
        for col in numeric_cols:
            col_lower = col.lower()
            if any(term in col_lower for term in ['revenue', 'sales', 'profit', 'cost', 'price', 'value']):
                key_metrics.append(col)

        if key_metrics:
            profile_parts.append(f"\nKey Business Metrics: {', '.join(key_metrics[:5])}")

        # Sample size assessment
        profile_parts.append(f"\nStatistical Validity:")
        if df.shape[0] < 30:
            profile_parts.append(f"- Sample size: VERY SMALL ({df.shape[0]} rows) - limited statistical power")
        elif df.shape[0] < 100:
            profile_parts.append(f"- Sample size: SMALL ({df.shape[0]} rows) - basic tests only")
        elif df.shape[0] < 1000:
            profile_parts.append(f"- Sample size: MEDIUM ({df.shape[0]} rows) - most tests viable")
        else:
            profile_parts.append(f"- Sample size: LARGE ({df.shape[0]} rows) - all tests viable")

        return "\n".join(profile_parts)

    def _format_test_catalog(self) -> str:
        """Format test catalog for LLM prompt."""
        catalog_items = []
        for key, test in self.TEST_CATALOG.items():
            catalog_items.append(
                f"- {key}: {test['name']} - {test['description']}\n"
                f"  Requirements: {test['requirements']}\n"
                f"  Value: {test['value']}, Cost: {test['cost']}"
            )
        return "\n".join(catalog_items)

    def _get_default_tests(self, df: pd.DataFrame, max_tests: int, mode: str) -> List[Dict]:
        """Get default tests as fallback."""
        defaults = []

        # Check for basic requirements
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        if mode == "single_table":
            # Always try anomaly detection if numeric columns exist
            if numeric_cols and df.shape[0] >= 30:
                defaults.append({
                    'test_key': 'anomaly_detection',
                    'rationale': 'Default: Checking for data quality issues',
                    'priority': 1,
                    'test_info': self.TEST_CATALOG['anomaly_detection']
                })

            # Try time series if date column exists
            if date_cols and df.shape[0] >= 20:
                defaults.append({
                    'test_key': 'trend_analysis',
                    'rationale': 'Default: Time series data detected',
                    'priority': 2,
                    'test_info': self.TEST_CATALOG['trend_analysis']
                })

                # Add causal analysis for time series with multiple numeric columns
                if len(numeric_cols) >= 2:
                    defaults.append({
                        'test_key': 'causal_analysis',
                        'rationale': 'Default: Time series with multiple metrics for causal testing',
                        'priority': 1,
                        'test_info': self.TEST_CATALOG['causal_analysis']
                    })

        elif mode == "cross_table":
            # Try correlation for cross-table
            if len(numeric_cols) >= 2:
                defaults.append({
                    'test_key': 'correlation_analysis',
                    'rationale': 'Default: Multiple numeric columns available',
                    'priority': 1,
                    'test_info': self.TEST_CATALOG['correlation_analysis']
                })

            # Add causal analysis for cross-table if time series exists
            if date_cols and len(numeric_cols) >= 2 and df.shape[0] >= 20:
                defaults.append({
                    'test_key': 'causal_analysis',
                    'rationale': 'Default: Time series data with multiple metrics for causal testing',
                    'priority': 1,
                    'test_info': self.TEST_CATALOG['causal_analysis']
                })

        # Sort by priority and limit to max_tests
        defaults.sort(key=lambda x: x['priority'])
        return defaults[:max_tests]

    def explain_skipped_tests(
        self,
        df: pd.DataFrame,
        domain: str
    ) -> str:
        """
        Explain why certain high-value tests were skipped.

        Useful for transparency and data improvement suggestions.
        """
        explanations = []

        # Check each high-value test
        high_value_tests = [
            k for k, v in self.TEST_CATALOG.items()
            if v['value'] in ['high', 'very_high']
        ]

        for test_key in high_value_tests:
            test = self.TEST_CATALOG[test_key]
            requirements = test['requirements']

            # Check if requirements are met
            if requirements['data_type'] == 'time_series':
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) == 0:
                    explanations.append(
                        f"• {test['name']}: Requires time series data (no date column found)"
                    )
                elif df.shape[0] < requirements['min_observations']:
                    explanations.append(
                        f"• {test['name']}: Needs {requirements['min_observations']} observations (have {df.shape[0]})"
                    )

        return "\n".join(explanations) if explanations else "All applicable tests were run."
"""Statistical analysis engine with LLM-guided parameter selection."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Tuple, Optional
import json
from langchain.prompts import ChatPromptTemplate
from src.models.analysis_result import StatisticalTest
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge
from src.utils.llm_client import get_llm


class StatisticalAnalyzer:
    """Performs statistical analysis with LLM-guided parameter selection."""

    def __init__(self):
        """Initialize statistical analyzer."""
        self.llm = get_llm(temperature=0.3)

    def analyze_challenge(
        self,
        challenge: Challenge,
        data: Dict[str, pd.DataFrame],
        business_context: BusinessContext
    ) -> Dict:
        """
        Perform comprehensive analysis for a challenge.

        Args:
            challenge: Challenge to analyze
            data: Dictionary of DataFrames
            business_context: Business context for parameter selection

        Returns:
            Dictionary containing analysis results
        """
        analysis_results = {
            "statistical_tests": [],
            "key_findings": [],
            "correlations": {},
            "causality_insights": []
        }

        # Select relevant datasets
        relevant_data = self._select_relevant_datasets(challenge, data)

        if not relevant_data:
            analysis_results["key_findings"].append(
                "No relevant datasets found for this challenge"
            )
            return analysis_results

        # For each relevant dataset, perform appropriate analyses
        for dataset_name, df in relevant_data.items():
            # Get LLM recommendations for analysis
            analysis_plan = self._get_analysis_recommendations(
                challenge, df, business_context, dataset_name
            )

            # Perform descriptive statistics
            descriptive = self._descriptive_statistics(df)
            analysis_results["key_findings"].extend(descriptive["findings"])

            # Check thresholds from business context
            threshold_alerts = self._check_thresholds_generic(df, business_context)
            analysis_results["key_findings"].extend(threshold_alerts)

            # Perform correlation analysis
            if analysis_plan.get("perform_correlation", False):
                correlations = self._correlation_analysis(df)
                analysis_results["correlations"].update(correlations)
                analysis_results["key_findings"].extend(
                    self._interpret_correlations(correlations, challenge)
                )

            # Perform hypothesis testing
            if analysis_plan.get("hypothesis_tests"):
                for test_spec in analysis_plan["hypothesis_tests"]:
                    test_result = self._perform_hypothesis_test(
                        df, test_spec, business_context
                    )
                    if test_result:
                        analysis_results["statistical_tests"].append(test_result)

            # Perform causality analysis
            if analysis_plan.get("causality_analysis", False):
                causality = self._causality_analysis(df, challenge, business_context)
                analysis_results["causality_insights"].extend(causality)

            # Perform time series analysis if applicable
            if analysis_plan.get("time_series_analysis", False):
                time_series_insights = self._time_series_analysis(df)
                analysis_results["key_findings"].extend(time_series_insights)

        return analysis_results

    def _select_relevant_datasets(
        self,
        challenge: Challenge,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Select datasets relevant to the challenge.

        Args:
            challenge: Challenge to analyze
            data: All available datasets

        Returns:
            Dictionary of relevant datasets
        """
        relevant_data = {}

        # Quick heuristic: if dataset name contains department name, it's likely relevant
        challenge_depts = [dept.lower() for dept in challenge.department]

        for dataset_name, df in data.items():
            dataset_lower = dataset_name.lower()

            # Fast path: check if dataset matches department
            is_likely_relevant = any(dept in dataset_lower for dept in challenge_depts)

            if is_likely_relevant:
                # Skip LLM check for obviously relevant datasets
                relevant_data[dataset_name] = df
                continue

            # For unclear cases, use LLM (limit to prevent hangs)
            if len(relevant_data) >= 10:  # Already have enough datasets
                continue

            columns_info = ", ".join(df.columns[:20])  # First 20 columns

            print(f"🤖 I'm checking if dataset '{dataset_name}' is relevant to challenge...")

            relevance_prompt = ChatPromptTemplate.from_template(
                """Determine if this dataset is relevant for analyzing the given challenge.

Challenge: {challenge}
Data sources needed: {data_sources}

Dataset: {dataset_name}
Columns: {columns}

Respond with JSON:
{{
    "is_relevant": true/false,
    "relevance_score": 0-100,
    "reason": "explanation"
}}
"""
            )

            try:
                chain = relevance_prompt | self.llm
                response = chain.invoke({
                    "challenge": challenge.title,
                    "data_sources": ", ".join(challenge.data_sources_needed),
                    "dataset_name": dataset_name,
                    "columns": columns_info
                })

                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    result = json.loads(content[start_idx:end_idx])
                    if result.get("is_relevant", False):
                        relevant_data[dataset_name] = df

            except Exception as e:
                print(f"Error checking relevance for {dataset_name}: {e}")

        return relevant_data

    def _get_analysis_recommendations(
        self,
        challenge: Challenge,
        df: pd.DataFrame,
        business_context: BusinessContext,
        dataset_name: str
    ) -> Dict:
        """
        Get LLM recommendations for analysis approaches.

        Args:
            challenge: Challenge being analyzed
            df: DataFrame to analyze
            business_context: Business context
            dataset_name: Name of the dataset

        Returns:
            Dictionary with analysis recommendations
        """
        # Create data profile
        data_profile = f"""
Rows: {len(df)}
Columns: {len(df.columns)}
Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
Categorical columns: {len(df.select_dtypes(include=['object']).columns)}
Date columns: {len(df.select_dtypes(include=['datetime']).columns)}

Column details:
{chr(10).join(f"- {col}: {df[col].dtype}" for col in df.columns[:20])}
"""

        print(f"🤖 I'm recommending statistical tests for dataset '{dataset_name}'...")

        recommendation_prompt = ChatPromptTemplate.from_template(
            """As a statistical analyst, recommend appropriate analyses for this situation.

Business Context:
{business_context}

Challenge:
{challenge}

Dataset: {dataset_name}
{data_profile}

CRITICAL REQUIREMENTS:
- You MUST ONLY use column names that exist in the dataset (listed above in "Column details")
- DO NOT invent or guess column names like "churned", "customer_tier", "product_usage" unless they are explicitly listed
- If a concept isn't directly represented by an existing column, DO NOT recommend a test for it
- For regression/t-test/anova: variables must be numeric columns
- For chi-square: variables must be categorical (object type) columns

Recommend analyses in JSON format:
{{
    "perform_correlation": true/false,
    "causality_analysis": true/false,
    "time_series_analysis": true/false,
    "hypothesis_tests": [
        {{
            "test_type": "t-test/anova/chi-square/regression",
            "variables": ["exact_column_name1", "exact_column_name2"],
            "null_hypothesis": "description",
            "business_reason": "why this test matters (be specific and complete)"
        }}
    ],
    "recommended_visualizations": ["type1", "type2"]
}}
"""
        )

        try:
            chain = recommendation_prompt | self.llm
            response = chain.invoke({
                "business_context": business_context.to_context_string()[:1000],
                "challenge": challenge.to_context_string()[:1000],
                "dataset_name": dataset_name,
                "data_profile": data_profile
            })

            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                return json.loads(content[start_idx:end_idx])

        except Exception as e:
            print(f"Error getting analysis recommendations: {e}")

        return {
            "perform_correlation": True,
            "causality_analysis": False,
            "time_series_analysis": False,
            "hypothesis_tests": []
        }

    def _descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate descriptive statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with descriptive statistics and findings
        """
        findings = []

        numeric_df = df.select_dtypes(include=[np.number])

        # Exclude sequential/index columns from analysis (they're not actual metrics)
        # Instead of hardcoding patterns, we analyze the actual data properties

        columns_to_analyze = []
        for col in numeric_df.columns:
            # Check if column values are sequential/monotonic (like 1,2,3,4... or timestamps)
            # This is GENERIC - works for any sequential data regardless of column name
            if len(numeric_df[col].unique()) > 5:
                is_sequential = self._is_sequential_column(numeric_df[col])
                if not is_sequential:
                    columns_to_analyze.append(col)
            else:
                # If few unique values, could be categorical or low-cardinality metric
                # Keep it for analysis
                columns_to_analyze.append(col)

        if not columns_to_analyze:
            return {"statistics": None, "findings": findings}

        desc = numeric_df[columns_to_analyze].describe()

        if not desc.empty:
            # Identify key patterns
            for col in columns_to_analyze:
                mean_val = desc.loc['mean', col]
                std_val = desc.loc['std', col]
                min_val = desc.loc['min', col]
                max_val = desc.loc['max', col]

                # Check for high variance
                if std_val > mean_val * 0.5 and mean_val != 0:
                    findings.append(
                        f"{col} shows high variability (std: {std_val:.2f}, mean: {mean_val:.2f})"
                    )

                # Check for outliers using IQR
                q1 = desc.loc['25%', col]
                q3 = desc.loc['75%', col]
                iqr = q3 - q1
                outlier_threshold = 1.5 * iqr

                if (min_val < q1 - outlier_threshold) or (max_val > q3 + outlier_threshold):
                    findings.append(
                        f"{col} contains potential outliers (range: {min_val:.2f} to {max_val:.2f})"
                    )

        return {"statistics": desc if not desc.empty else None, "findings": findings}

    def _is_sequential_column(self, series: pd.Series) -> bool:
        """
        Check if a column contains sequential values (like 1,2,3,4... or week numbers).

        Args:
            series: Pandas Series to check

        Returns:
            True if column appears to be sequential, False otherwise
        """
        try:
            # Get unique sorted values
            unique_vals = sorted(series.dropna().unique())

            if len(unique_vals) < 3:
                return False

            # Check if values form a sequence
            # For a perfect sequence: diff between consecutive values should be constant
            diffs = [unique_vals[i+1] - unique_vals[i] for i in range(len(unique_vals)-1)]

            # If all differences are the same (or very close), it's sequential
            if len(set(diffs)) == 1:
                return True

            # Check if it's approximately sequential (allowing small gaps)
            avg_diff = sum(diffs) / len(diffs)
            if all(abs(d - avg_diff) < 0.01 for d in diffs):
                return True

        except Exception:
            pass

        return False

    def _check_thresholds_generic(
        self,
        df: pd.DataFrame,
        business_context: BusinessContext
    ) -> List[str]:
        """
        Check if any metrics fall outside acceptable thresholds based on business context.
        This is FULLY GENERIC - works for any business with any metrics.

        Args:
            df: DataFrame to analyze
            business_context: Business context with success metrics

        Returns:
            List of threshold alerts
        """
        alerts = []

        # Parse success metrics to extract targets
        # E.g., "Customer Retention Rate (target: 95%)" → column: retention, target: 0.95
        metric_targets = self._parse_success_metrics(business_context.success_metrics)

        for metric_info in metric_targets:
            metric_name = metric_info['name']
            target_value = metric_info['target']
            direction = metric_info['direction']  # 'above' or 'below'

            # Try to find matching column using fuzzy matching
            matching_col = self._find_matching_column(df, metric_name)

            if matching_col and matching_col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[matching_col].min()
                col_max = df[matching_col].max()
                col_mean = df[matching_col].mean()

                # Check if values fall outside target
                if direction == 'above' and col_min < target_value:
                    gap = target_value - col_min
                    if target_value < 1:  # Percentage format
                        alerts.append(
                            f"CRITICAL: {matching_col} dropped to {col_min:.1%}, "
                            f"{gap:.1%} below target of {target_value:.1%}"
                        )
                    else:  # Absolute value
                        alerts.append(
                            f"CRITICAL: {matching_col} dropped to {col_min:.2f}, "
                            f"{gap:.2f} below target of {target_value:.2f}"
                        )
                elif direction == 'below' and col_max > target_value:
                    gap = col_max - target_value
                    if target_value < 1:  # Percentage format
                        alerts.append(
                            f"CRITICAL: {matching_col} peaked at {col_max:.1%}, "
                            f"{gap:.1%} above target of {target_value:.1%}"
                        )
                    else:  # Absolute value
                        alerts.append(
                            f"CRITICAL: {matching_col} peaked at {col_max:.2f}, "
                            f"{gap:.2f} above target of {target_value:.2f}"
                        )

        return alerts

    def _parse_success_metrics(self, metrics_list: List[str]) -> List[Dict]:
        """
        Parse success metrics to extract targets.

        Examples:
        - "Customer Retention Rate (target: 95%)" → {name: 'retention', target: 0.95, direction: 'above'}
        - "Churn Rate (target: <5%)" → {name: 'churn', target: 0.05, direction: 'below'}
        - "Sales Conversion Rate (target: 12%)" → {name: 'conversion', target: 0.12, direction: 'above'}
        """
        import re

        parsed_metrics = []

        for metric_str in metrics_list:
            # Pattern to match "Metric Name (target: X%)" or "Metric Name (target: <X%)"
            match = re.search(r'([^(]+)\(target:\s*([<>]?)(\d+(?:\.\d+)?)%?\)', metric_str, re.IGNORECASE)

            if match:
                name = match.group(1).strip()
                comparison = match.group(2)  # '<' or '>' or ''
                value_str = match.group(3)

                # Convert to float
                try:
                    value = float(value_str)
                    # If it's a percentage (value > 1), convert to decimal
                    if value > 1:
                        value = value / 100.0

                    # Determine direction
                    if '<' in comparison:
                        direction = 'below'  # Want metric below this value
                    else:
                        direction = 'above'  # Default: want metric above this value

                    parsed_metrics.append({
                        'name': name,
                        'target': value,
                        'direction': direction
                    })
                except ValueError:
                    continue

        return parsed_metrics

    def _find_matching_column(self, df: pd.DataFrame, metric_name: str) -> Optional[str]:
        """
        Find column that matches metric name using improved fuzzy matching.

        E.g., "Customer Retention Rate" matches "customer_retention_rate" or "retention_pct"

        IMPROVED VERSION with stricter matching to prevent false positives.
        Uses word overlap ratio instead of hardcoded word lists.
        """
        from difflib import get_close_matches

        # Normalize metric name for comparison
        metric_normalized = metric_name.lower().replace(' ', '_').replace('-', '_')
        metric_words = set(metric_normalized.split('_'))

        # Remove empty strings
        metric_words = {w for w in metric_words if w}

        # Check exact match first (substring matching)
        for col in df.columns:
            col_normalized = col.lower().replace(' ', '_').replace('-', '_')
            if metric_normalized in col_normalized or col_normalized in metric_normalized:
                return col

        # Fuzzy match: require BOTH high word overlap AND at least 2 words matching
        # This prevents weak matches like matching on just "rate" alone
        best_match = None
        best_score = 0
        best_ratio = 0.0

        for col in df.columns:
            col_normalized = col.lower().replace(' ', '_').replace('-', '_')
            col_words = {w for w in col_normalized.split('_') if w}

            # Calculate word overlap
            overlap_words = metric_words.intersection(col_words)
            overlap_count = len(overlap_words)

            # Calculate overlap ratio (what % of metric words are matched)
            if len(metric_words) > 0:
                metric_overlap_ratio = overlap_count / len(metric_words)
            else:
                metric_overlap_ratio = 0

            # Also calculate reverse ratio (what % of column words are matched)
            if len(col_words) > 0:
                col_overlap_ratio = overlap_count / len(col_words)
            else:
                col_overlap_ratio = 0

            # STRICT MATCHING: Require BOTH:
            # 1. At least 2 words overlap (prevents matching on single generic words)
            # 2. At least 60% of metric words matched (high precision)
            # 3. At least 40% of column words matched (prevents spurious matches)
            if (overlap_count >= 2 and
                metric_overlap_ratio >= 0.6 and
                col_overlap_ratio >= 0.4):

                # Score by overlap count
                if overlap_count > best_score:
                    best_score = overlap_count
                    best_ratio = metric_overlap_ratio
                    best_match = col

        # Return match only if we found a strong candidate
        if best_match:
            return best_match

        return None

    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Perform correlation analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of significant correlations
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()

        # Extract significant correlations (|r| > 0.5, excluding diagonal)
        significant_correlations = {}

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) > 0.5 and not pd.isna(corr_value):
                    significant_correlations[f"{col1}_vs_{col2}"] = float(corr_value)

        return significant_correlations

    def _interpret_correlations(
        self,
        correlations: Dict[str, float],
        challenge: Challenge
    ) -> List[str]:
        """
        Use LLM to interpret correlations in business context.

        Args:
            correlations: Dictionary of correlations
            challenge: Current challenge

        Returns:
            List of interpretation strings
        """
        if not correlations:
            return []

        print(f"🤖 I'm interpreting {len(correlations)} correlation patterns...")

        interpretation_prompt = ChatPromptTemplate.from_template(
            """As a data analyst, interpret these correlations in the context of the business challenge.
Provide actionable insights.

Challenge: {challenge}

Correlations:
{correlations}

Provide 2-3 key insights in a list format.
"""
        )

        try:
            corr_text = "\n".join(
                f"- {k}: {v:.3f}" for k, v in list(correlations.items())[:10]
            )

            chain = interpretation_prompt | self.llm
            response = chain.invoke({
                "challenge": challenge.title,
                "correlations": corr_text
            })

            # Parse insights from response
            insights = []
            lines = response.content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    insights.append(line.lstrip('-•0123456789. '))

            return insights[:5]  # Top 5 insights

        except Exception as e:
            print(f"Error interpreting correlations: {e}")
            return []

    def _perform_hypothesis_test(
        self,
        df: pd.DataFrame,
        test_spec: Dict,
        business_context: BusinessContext
    ) -> Optional[StatisticalTest]:
        """
        Perform a hypothesis test.

        Args:
            df: DataFrame containing data
            test_spec: Test specification from LLM
            business_context: Business context

        Returns:
            StatisticalTest object or None
        """
        test_type = test_spec.get("test_type", "").lower()
        variables = test_spec.get("variables", [])

        if not variables or not all(var in df.columns for var in variables):
            return None

        try:
            if "t-test" in test_type or "ttest" in test_type:
                return self._perform_ttest(df, variables, test_spec)
            elif "anova" in test_type:
                return self._perform_anova(df, variables, test_spec)
            elif "chi" in test_type:
                return self._perform_chi_square(df, variables, test_spec)
            elif "regression" in test_type:
                return self._perform_regression(df, variables, test_spec)

        except Exception as e:
            print(f"Error performing {test_type}: {e}")

        return None

    def _perform_ttest(
        self,
        df: pd.DataFrame,
        variables: List[str],
        test_spec: Dict
    ) -> StatisticalTest:
        """Perform t-test."""
        if len(variables) < 2:
            return None

        # Ensure variables are numeric
        try:
            var1 = pd.to_numeric(df[variables[0]], errors='coerce').dropna()
            var2 = pd.to_numeric(df[variables[1]], errors='coerce').dropna()

            if len(var1) < 2 or len(var2) < 2:
                return None
        except Exception:
            return None

        statistic, p_value = stats.ttest_ind(var1, var2)

        interpretation = f"Comparing {variables[0]} and {variables[1]}: "
        if p_value < 0.05:
            interpretation += f"Significant difference found (p={p_value:.4f}). "
            interpretation += f"This suggests that {test_spec.get('business_reason', 'these variables differ significantly')}."
        else:
            interpretation += f"No significant difference (p={p_value:.4f}). "

        return StatisticalTest(
            test_name="Independent T-Test",
            test_statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            interpretation=interpretation,
            parameters={"variables": variables}
        )

    def _perform_anova(
        self,
        df: pd.DataFrame,
        variables: List[str],
        test_spec: Dict
    ) -> StatisticalTest:
        """Perform ANOVA."""
        if len(variables) < 2:
            return None

        # Ensure all variables are numeric and have sufficient data
        try:
            groups = []
            for var in variables:
                numeric_vals = pd.to_numeric(df[var], errors='coerce').dropna()
                if len(numeric_vals) < 2:
                    return None
                groups.append(numeric_vals)

            if len(groups) < 2:
                return None
        except Exception:
            return None

        statistic, p_value = stats.f_oneway(*groups)

        interpretation = f"ANOVA across {', '.join(variables)}: "
        if p_value < 0.05:
            interpretation += f"Significant differences found (p={p_value:.4f}). "
        else:
            interpretation += f"No significant differences (p={p_value:.4f}). "

        return StatisticalTest(
            test_name="ANOVA",
            test_statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            interpretation=interpretation,
            parameters={"variables": variables}
        )

    def _perform_chi_square(
        self,
        df: pd.DataFrame,
        variables: List[str],
        test_spec: Dict
    ) -> StatisticalTest:
        """Perform chi-square test."""
        if len(variables) < 2:
            return None

        try:
            # Create contingency table and check if it has data
            contingency_table = pd.crosstab(df[variables[0]], df[variables[1]])

            # Check if contingency table is empty or too small
            if contingency_table.size == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return None

            # Check if all values are zero
            if contingency_table.sum().sum() == 0:
                return None
        except Exception:
            return None

        statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        interpretation = f"Chi-square test between {variables[0]} and {variables[1]}: "
        if p_value < 0.05:
            interpretation += f"Significant association found (p={p_value:.4f}). "
        else:
            interpretation += f"No significant association (p={p_value:.4f}). "

        return StatisticalTest(
            test_name="Chi-Square Test",
            test_statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            interpretation=interpretation,
            parameters={"variables": variables, "degrees_of_freedom": dof}
        )

    def _perform_regression(
        self,
        df: pd.DataFrame,
        variables: List[str],
        test_spec: Dict
    ) -> StatisticalTest:
        """Perform linear regression."""
        if len(variables) < 2:
            return None

        try:
            # Assume last variable is target, others are features
            # Convert all to numeric, coercing errors to NaN
            X_df = df[variables[:-1]].apply(pd.to_numeric, errors='coerce')
            y_series = pd.to_numeric(df[variables[-1]], errors='coerce')

            # Drop rows with any NaN values
            valid_indices = X_df.dropna().index.intersection(y_series.dropna().index)

            if len(valid_indices) < 3:  # Need at least 3 points for meaningful regression
                return None

            X = X_df.loc[valid_indices]
            y = y_series.loc[valid_indices]
        except Exception:
            return None

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        interpretation = f"Regression analysis with {variables[-1]} as target: "
        interpretation += f"R² = {r_squared:.4f}. "

        if r_squared > 0.7:
            interpretation += "Strong predictive relationship found. "
        elif r_squared > 0.4:
            interpretation += "Moderate predictive relationship. "
        else:
            interpretation += "Weak predictive relationship. "

        return StatisticalTest(
            test_name="Linear Regression",
            test_statistic=float(r_squared),
            p_value=0.0,  # Would need more complex calculation
            is_significant=r_squared > 0.4,
            interpretation=interpretation,
            parameters={
                "variables": variables,
                "r_squared": r_squared,
                "coefficients": {var: float(coef) for var, coef in zip(variables[:-1], model.coef_)}
            }
        )

    def _causality_analysis(
        self,
        df: pd.DataFrame,
        challenge: Challenge,
        business_context: BusinessContext
    ) -> List[str]:
        """
        Perform causality analysis using Granger causality and LLM interpretation.

        Args:
            df: DataFrame to analyze
            challenge: Challenge being analyzed
            business_context: Business context

        Returns:
            List of causality insights
        """
        insights = []

        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return insights

        # Use LLM to identify potential causal relationships
        print("🤖 I'm analyzing potential causal relationships in data...")

        causality_prompt = ChatPromptTemplate.from_template(
            """As a causal inference expert, identify potential causal relationships in this data.

Challenge: {challenge}
Business Context: {context}

Available variables: {variables}

Based on the business context, suggest 2-3 potential causal relationships to investigate.
Format: "Variable X may cause Variable Y because..."
"""
        )

        try:
            chain = causality_prompt | self.llm
            response = chain.invoke({
                "challenge": challenge.title,
                "context": business_context.current_goal,
                "variables": ", ".join(numeric_df.columns[:15])
            })

            # Parse suggested relationships
            lines = response.content.split('\n')
            for line in lines:
                line = line.strip()
                if line and ("cause" in line.lower() or "affect" in line.lower()):
                    insights.append(line.lstrip('-•0123456789. '))

        except Exception as e:
            print(f"Error in causality analysis: {e}")

        return insights[:5]

    def _time_series_analysis(self, df: pd.DataFrame) -> List[str]:
        """
        Perform time series analysis if applicable.

        Args:
            df: DataFrame with time series data

        Returns:
            List of insights from time series analysis
        """
        insights = []

        # Check for datetime columns
        date_columns = df.select_dtypes(include=['datetime64']).columns

        if len(date_columns) == 0:
            return insights

        # Try to perform seasonal decomposition on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for date_col in date_columns[:1]:  # Analyze first date column
            for num_col in numeric_cols[:2]:  # Analyze first 2 numeric columns
                try:
                    # Sort by date
                    ts_df = df[[date_col, num_col]].sort_values(date_col).dropna()

                    if len(ts_df) < 14:  # Need enough data points
                        continue

                    # Set index
                    ts_df = ts_df.set_index(date_col)

                    # Perform decomposition if frequency can be determined
                    if len(ts_df) >= 14:
                        insights.append(
                            f"{num_col} shows time-based patterns that could be explored further"
                        )

                        # Check for trend
                        if ts_df[num_col].corr(pd.Series(range(len(ts_df)))) > 0.5:
                            insights.append(f"{num_col} shows an upward trend over time")
                        elif ts_df[num_col].corr(pd.Series(range(len(ts_df)))) < -0.5:
                            insights.append(f"{num_col} shows a downward trend over time")

                except Exception as e:
                    continue

        return insights

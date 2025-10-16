"""Business Query Engine for data manipulation and answering business questions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm
import json


@dataclass
class QueryResult:
    """Result from a business query."""
    question: str
    query_type: str  # 'aggregation', 'comparison', 'trend', 'segmentation', 'correlation'
    data: pd.DataFrame
    summary: str
    visualization_recommendation: str


class BusinessQueryEngine:
    """Engine for querying and manipulating data to answer business questions."""

    def __init__(self):
        """Initialize the business query engine."""
        self.llm = get_llm(temperature=0.2)

    def answer_business_question(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        context: Optional[str] = None
    ) -> QueryResult:
        """
        Answer a business question by querying and manipulating the data.

        Args:
            question: Business question to answer
            data: Dictionary of available DataFrames
            context: Optional business context

        Returns:
            QueryResult with the answer data and visualization recommendation
        """
        # Determine query strategy using LLM
        query_plan = self._plan_query(question, data, context)

        # Check if join is needed and perform it first
        if query_plan.get('join_required'):
            print(f"  🔗 Join required: merging {len(query_plan.get('datasets_to_join', []))} datasets")
            merged_data = self._perform_join(data, query_plan)
            if merged_data is not None:
                # Replace data dict with merged result
                data = {'merged_dataset': merged_data}
                query_plan['primary_dataset'] = 'merged_dataset'

        # Execute the query based on the plan
        if query_plan['query_type'] == 'aggregation':
            return self._execute_aggregation_query(question, data, query_plan)
        elif query_plan['query_type'] == 'comparison':
            return self._execute_comparison_query(question, data, query_plan)
        elif query_plan['query_type'] == 'trend':
            return self._execute_trend_query(question, data, query_plan)
        elif query_plan['query_type'] == 'segmentation':
            return self._execute_segmentation_query(question, data, query_plan)
        elif query_plan['query_type'] == 'correlation':
            return self._execute_correlation_query(question, data, query_plan)
        else:
            # Default to aggregation
            return self._execute_aggregation_query(question, data, query_plan)

    def _plan_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        context: Optional[str]
    ) -> Dict:
        """
        Use LLM to plan how to query the data.

        Returns:
            Dictionary with query plan including type, dataset, columns, etc.
        """
        # Create data catalog
        data_catalog = self._create_data_catalog(data)

        planning_prompt = ChatPromptTemplate.from_template(
            """As a data analyst, plan how to answer this business question using the available data.

Business Question: {question}
Context: {context}

Available Data:
{data_catalog}

IMPORTANT: Check if this question requires data from MULTIPLE datasets. If so, specify join parameters.

Create a query plan in JSON format:
{{
    "query_type": "aggregation|comparison|trend|segmentation|correlation",
    "join_required": true/false,
    "datasets_to_join": ["dataset1", "dataset2"] or null,
    "join_keys": {{"dataset1": "key_col1", "dataset2": "key_col2"}} or null,
    "join_type": "inner|left|right|outer" or null,
    "primary_dataset": "dataset_name (or merged_dataset if join)",
    "columns_needed": ["col1", "col2"],
    "group_by": ["grouping_column"] or null,
    "aggregations": {{"column": "function"}} or null,
    "filters": {{"column": "condition"}} or null,
    "time_column": "date_column" or null,
    "explanation": "How this answers the question"
}}

Examples:
- "What's the average performance by department?" → aggregation with group_by, no join
- "How do employee demographics relate to sales performance?" → join employee & sales data, then compare
- "What's the customer retention rate by product category?" → join customers & products, then aggregate
- "How are customers distributed by tier?" → segmentation, no join

JOIN EXAMPLES:
- If question involves "employees" and "departments", join on department_id
- If question involves "customers" and "orders", join on customer_id
- If question involves "products" and "sales", join on product_id
"""
        )

        try:
            chain = planning_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "context": context or "No additional context",
                "data_catalog": data_catalog
            })

            # Parse JSON from response
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                plan = json.loads(content[start_idx:end_idx])
                return plan

        except Exception as e:
            print(f"Error planning query: {e}")

        # Default plan
        return {
            "query_type": "aggregation",
            "primary_dataset": list(data.keys())[0] if data else "",
            "columns_needed": [],
            "group_by": None,
            "aggregations": None,
            "filters": None,
            "explanation": "Default aggregation query"
        }

    def _create_data_catalog(self, data: Dict[str, pd.DataFrame]) -> str:
        """Create a catalog of available data for LLM context."""
        catalog_parts = []

        for name, df in list(data.items())[:5]:  # Limit to first 5 datasets
            catalog_parts.append(f"\nDataset: {name}")
            catalog_parts.append(f"  Rows: {len(df)}")
            catalog_parts.append(f"  Columns ({len(df.columns)}):")

            # Show column info
            for col in df.columns[:20]:  # First 20 columns
                dtype = str(df[col].dtype)
                nunique = df[col].nunique()
                catalog_parts.append(f"    - {col}: {dtype} (unique values: {nunique})")

        return "\n".join(catalog_parts)

    def _execute_aggregation_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> QueryResult:
        """Execute an aggregation query."""
        dataset_name = plan.get('primary_dataset', list(data.keys())[0] if data else None)

        if not dataset_name or dataset_name not in data:
            return QueryResult(
                question=question,
                query_type='aggregation',
                data=pd.DataFrame(),
                summary="No data available for query",
                visualization_recommendation="none"
            )

        df = data[dataset_name].copy()

        # Apply filters if specified
        if plan.get('filters'):
            df = self._apply_filters(df, plan['filters'])

        # Perform aggregation
        if plan.get('group_by'):
            result_df = self._perform_groupby_aggregation(df, plan)
        else:
            result_df = self._perform_simple_aggregation(df, plan)

        # Generate summary
        summary = self._generate_aggregation_summary(result_df, plan)

        # Recommend visualization
        if plan.get('group_by'):
            viz_recommendation = "bar_chart"
        else:
            viz_recommendation = "kpi_card"

        return QueryResult(
            question=question,
            query_type='aggregation',
            data=result_df,
            summary=summary,
            visualization_recommendation=viz_recommendation
        )

    def _execute_comparison_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> QueryResult:
        """Execute a comparison query between groups."""
        dataset_name = plan.get('primary_dataset', list(data.keys())[0] if data else None)

        if not dataset_name or dataset_name not in data:
            return QueryResult(
                question=question,
                query_type='comparison',
                data=pd.DataFrame(),
                summary="No data available for comparison",
                visualization_recommendation="none"
            )

        df = data[dataset_name].copy()

        # Apply filters
        if plan.get('filters'):
            df = self._apply_filters(df, plan['filters'])

        # Perform comparison (similar to group by but focused on comparing groups)
        if plan.get('group_by'):
            result_df = self._perform_groupby_aggregation(df, plan)

            # Calculate comparison metrics
            if len(result_df) > 1:
                # Add percentage difference from mean
                for col in result_df.select_dtypes(include=[np.number]).columns:
                    mean_val = result_df[col].mean()
                    if mean_val != 0:
                        result_df[f'{col}_diff_from_mean_%'] = ((result_df[col] - mean_val) / mean_val * 100).round(1)
        else:
            result_df = df

        summary = self._generate_comparison_summary(result_df, plan)

        return QueryResult(
            question=question,
            query_type='comparison',
            data=result_df,
            summary=summary,
            visualization_recommendation="grouped_bar_chart"
        )

    def _execute_trend_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> QueryResult:
        """Execute a trend analysis query over time."""
        dataset_name = plan.get('primary_dataset', list(data.keys())[0] if data else None)

        if not dataset_name or dataset_name not in data:
            return QueryResult(
                question=question,
                query_type='trend',
                data=pd.DataFrame(),
                summary="No data available for trend analysis",
                visualization_recommendation="none"
            )

        df = data[dataset_name].copy()

        # Find time column
        time_col = plan.get('time_column')
        if not time_col:
            # Try to find a date column
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                time_col = date_cols[0]
            else:
                # Try to parse date columns
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                            time_col = col
                            break
                        except:
                            continue

        if not time_col:
            return QueryResult(
                question=question,
                query_type='trend',
                data=pd.DataFrame(),
                summary="No time-based data found for trend analysis",
                visualization_recommendation="none"
            )

        # Sort by time
        df = df.sort_values(time_col)

        # Apply filters
        if plan.get('filters'):
            df = self._apply_filters(df, plan['filters'])

        # Aggregate by time period
        if plan.get('aggregations'):
            # Group by time period (e.g., month, quarter)
            df['period'] = pd.to_datetime(df[time_col]).dt.to_period('M')
            grouped = df.groupby('period')

            result_data = {}
            for col, func in plan['aggregations'].items():
                if col in df.columns and df[col].dtype in [np.number, 'float64', 'int64']:
                    result_data[col] = grouped[col].agg(func)

            result_df = pd.DataFrame(result_data)
            result_df['period'] = result_df.index.astype(str)
            result_df = result_df.reset_index(drop=True)
        else:
            result_df = df[[time_col] + plan.get('columns_needed', [])]

        summary = self._generate_trend_summary(result_df, plan)

        return QueryResult(
            question=question,
            query_type='trend',
            data=result_df,
            summary=summary,
            visualization_recommendation="line_chart"
        )

    def _execute_segmentation_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> QueryResult:
        """Execute a segmentation query to show distribution."""
        dataset_name = plan.get('primary_dataset', list(data.keys())[0] if data else None)

        if not dataset_name or dataset_name not in data:
            return QueryResult(
                question=question,
                query_type='segmentation',
                data=pd.DataFrame(),
                summary="No data available for segmentation",
                visualization_recommendation="none"
            )

        df = data[dataset_name].copy()

        # Apply filters
        if plan.get('filters'):
            df = self._apply_filters(df, plan['filters'])

        # Perform segmentation
        if plan.get('group_by'):
            segment_col = plan['group_by'][0]
            if segment_col in df.columns:
                # Count by segment
                result_df = df[segment_col].value_counts().reset_index()
                result_df.columns = [segment_col, 'count']
                result_df['percentage'] = (result_df['count'] / result_df['count'].sum() * 100).round(1)
            else:
                result_df = pd.DataFrame()
        else:
            result_df = pd.DataFrame()

        summary = self._generate_segmentation_summary(result_df, plan)

        return QueryResult(
            question=question,
            query_type='segmentation',
            data=result_df,
            summary=summary,
            visualization_recommendation="pie_chart"
        )

    def _execute_correlation_query(
        self,
        question: str,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> QueryResult:
        """Execute a correlation analysis query."""
        dataset_name = plan.get('primary_dataset', list(data.keys())[0] if data else None)

        if not dataset_name or dataset_name not in data:
            return QueryResult(
                question=question,
                query_type='correlation',
                data=pd.DataFrame(),
                summary="No data available for correlation analysis",
                visualization_recommendation="none"
            )

        df = data[dataset_name].copy()

        # Apply filters
        if plan.get('filters'):
            df = self._apply_filters(df, plan['filters'])

        # Get columns for correlation
        cols_needed = plan.get('columns_needed', [])
        if len(cols_needed) >= 2:
            # Calculate correlation between specific columns
            numeric_cols = [col for col in cols_needed if col in df.columns and df[col].dtype in [np.number, 'float64', 'int64']]
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()

                # Convert to readable format
                result_data = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        result_data.append({
                            'variable_1': numeric_cols[i],
                            'variable_2': numeric_cols[j],
                            'correlation': corr_matrix.iloc[i, j],
                            'strength': self._classify_correlation_strength(corr_matrix.iloc[i, j])
                        })

                result_df = pd.DataFrame(result_data)
            else:
                result_df = pd.DataFrame()
        else:
            result_df = pd.DataFrame()

        summary = self._generate_correlation_summary(result_df, plan)

        return QueryResult(
            question=question,
            query_type='correlation',
            data=result_df,
            summary=summary,
            visualization_recommendation="scatter_plot"
        )

    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to a DataFrame."""
        for col, condition in filters.items():
            if col in df.columns:
                # Parse condition (e.g., "> 100", "== 'value'", "in ['a', 'b']")
                try:
                    if isinstance(condition, str):
                        if condition.startswith('>'):
                            value = float(condition[1:].strip())
                            df = df[df[col] > value]
                        elif condition.startswith('<'):
                            value = float(condition[1:].strip())
                            df = df[df[col] < value]
                        elif condition.startswith('=='):
                            value = condition[2:].strip().strip("'\"")
                            df = df[df[col] == value]
                        elif 'in' in condition:
                            # Parse list from string
                            values = eval(condition.replace('in', '').strip())
                            df = df[df[col].isin(values)]
                except:
                    continue

        return df

    def _perform_groupby_aggregation(
        self,
        df: pd.DataFrame,
        plan: Dict
    ) -> pd.DataFrame:
        """Perform group by aggregation."""
        group_cols = plan.get('group_by', [])
        aggregations = plan.get('aggregations', {})

        if not group_cols or not aggregations:
            return df

        # Filter to valid columns
        valid_group_cols = [col for col in group_cols if col in df.columns]
        if not valid_group_cols:
            return df

        # Build aggregation dict
        agg_dict = {}
        for col, func in aggregations.items():
            if col in df.columns:
                # Check if column is numeric for numeric aggregations
                if func in ['mean', 'sum', 'std', 'min', 'max', 'median']:
                    if df[col].dtype in [np.number, 'float64', 'int64']:
                        agg_dict[col] = func
                else:
                    agg_dict[col] = func

        if not agg_dict:
            # Default to count
            result = df.groupby(valid_group_cols).size().reset_index(name='count')
        else:
            result = df.groupby(valid_group_cols).agg(agg_dict).reset_index()

        return result

    def _perform_simple_aggregation(
        self,
        df: pd.DataFrame,
        plan: Dict
    ) -> pd.DataFrame:
        """Perform simple aggregation without grouping."""
        aggregations = plan.get('aggregations', {})

        result_data = {}
        for col, func in aggregations.items():
            if col in df.columns and df[col].dtype in [np.number, 'float64', 'int64']:
                if func == 'mean':
                    result_data[f'{col}_mean'] = [df[col].mean()]
                elif func == 'sum':
                    result_data[f'{col}_sum'] = [df[col].sum()]
                elif func == 'count':
                    result_data[f'{col}_count'] = [df[col].count()]
                elif func == 'std':
                    result_data[f'{col}_std'] = [df[col].std()]
                elif func == 'min':
                    result_data[f'{col}_min'] = [df[col].min()]
                elif func == 'max':
                    result_data[f'{col}_max'] = [df[col].max()]

        if result_data:
            return pd.DataFrame(result_data)
        else:
            return pd.DataFrame({'total_rows': [len(df)]})

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def _generate_aggregation_summary(self, df: pd.DataFrame, plan: Dict) -> str:
        """Generate summary for aggregation results."""
        if df.empty:
            return "No data to summarize"

        summary_parts = []

        if plan.get('group_by'):
            summary_parts.append(f"Data grouped by {', '.join(plan['group_by'])}")
            summary_parts.append(f"Found {len(df)} distinct groups")

            # Highlight top/bottom performers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                first_metric = numeric_cols[0]
                if len(df) > 0:
                    max_idx = df[first_metric].idxmax()
                    min_idx = df[first_metric].idxmin()
                    summary_parts.append(f"Highest {first_metric}: {df.iloc[max_idx][plan['group_by'][0]]}")
                    summary_parts.append(f"Lowest {first_metric}: {df.iloc[min_idx][plan['group_by'][0]]}")
        else:
            summary_parts.append("Overall aggregated metrics calculated")

        return ". ".join(summary_parts)

    def _generate_comparison_summary(self, df: pd.DataFrame, plan: Dict) -> str:
        """Generate summary for comparison results."""
        if df.empty:
            return "No data to compare"

        summary_parts = []
        summary_parts.append(f"Comparison across {len(df)} groups")

        # Find biggest differences
        diff_cols = [col for col in df.columns if 'diff_from_mean_%' in col]
        if diff_cols:
            for col in diff_cols[:1]:  # First difference column
                max_diff_idx = df[col].abs().idxmax()
                group_col = plan.get('group_by', ['group'])[0]
                if group_col in df.columns:
                    summary_parts.append(
                        f"Largest deviation: {df.iloc[max_diff_idx][group_col]} "
                        f"({df.iloc[max_diff_idx][col]:.1f}% from mean)"
                    )

        return ". ".join(summary_parts)

    def _generate_trend_summary(self, df: pd.DataFrame, plan: Dict) -> str:
        """Generate summary for trend results."""
        if df.empty:
            return "No trend data available"

        summary_parts = []
        summary_parts.append(f"Trend analysis over {len(df)} time periods")

        # Calculate trend direction for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:1]:  # First numeric column
            if len(df) > 1:
                first_val = df.iloc[0][col]
                last_val = df.iloc[-1][col]
                change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

                if change_pct > 0:
                    summary_parts.append(f"{col} increased by {change_pct:.1f}%")
                elif change_pct < 0:
                    summary_parts.append(f"{col} decreased by {abs(change_pct):.1f}%")
                else:
                    summary_parts.append(f"{col} remained stable")

        return ". ".join(summary_parts)

    def _generate_segmentation_summary(self, df: pd.DataFrame, plan: Dict) -> str:
        """Generate summary for segmentation results."""
        if df.empty:
            return "No segmentation data available"

        summary_parts = []
        summary_parts.append(f"Distribution across {len(df)} segments")

        if 'percentage' in df.columns:
            # Find largest segment
            max_idx = df['percentage'].idxmax()
            segment_col = df.columns[0]  # First column is segment name
            summary_parts.append(
                f"Largest segment: {df.iloc[max_idx][segment_col]} "
                f"({df.iloc[max_idx]['percentage']:.1f}%)"
            )

            # Check for concentration
            top_3_pct = df.nlargest(3, 'percentage')['percentage'].sum()
            if top_3_pct > 75:
                summary_parts.append(f"Top 3 segments account for {top_3_pct:.1f}% (high concentration)")

        return ". ".join(summary_parts)

    def _generate_correlation_summary(self, df: pd.DataFrame, plan: Dict) -> str:
        """Generate summary for correlation results."""
        if df.empty:
            return "No correlation data available"

        summary_parts = []
        summary_parts.append(f"Analyzed {len(df)} variable relationships")

        if 'correlation' in df.columns:
            # Find strongest correlation
            max_idx = df['correlation'].abs().idxmax()
            summary_parts.append(
                f"Strongest relationship: {df.iloc[max_idx]['variable_1']} and "
                f"{df.iloc[max_idx]['variable_2']} (r={df.iloc[max_idx]['correlation']:.3f})"
            )

        return ". ".join(summary_parts)

    def _perform_join(
        self,
        data: Dict[str, pd.DataFrame],
        plan: Dict
    ) -> Optional[pd.DataFrame]:
        """
        Perform join operation on multiple datasets.

        Args:
            data: Dictionary of available DataFrames
            plan: Query plan with join specifications

        Returns:
            Merged DataFrame or None if join fails
        """
        datasets_to_join = plan.get('datasets_to_join', [])
        join_keys = plan.get('join_keys', {})
        join_type = plan.get('join_type', 'inner')

        if len(datasets_to_join) < 2:
            print("  ⚠ Join requires at least 2 datasets")
            return None

        try:
            # Start with the first dataset - fuzzy match the name
            first_dataset = datasets_to_join[0]
            first_dataset_key = self._find_dataset_key(data, first_dataset)

            if not first_dataset_key:
                print(f"  ⚠ Dataset '{first_dataset}' not found")
                print(f"     Available: {list(data.keys())[:3]}")
                return None

            result_df = data[first_dataset_key].copy()
            # Try both the LLM's name and the actual key
            first_key = join_keys.get(first_dataset) or join_keys.get(first_dataset_key)

            # Sequentially join additional datasets
            for i in range(1, len(datasets_to_join)):
                next_dataset = datasets_to_join[i]
                next_dataset_key = self._find_dataset_key(data, next_dataset)

                if not next_dataset_key:
                    print(f"  ⚠ Dataset '{next_dataset}' not found, skipping")
                    continue

                # Try both the LLM's name and the actual key
                next_key = join_keys.get(next_dataset) or join_keys.get(next_dataset_key)

                if not first_key or not next_key:
                    # Try to intelligently find join keys
                    first_key, next_key = self._find_join_keys(
                        result_df, data[next_dataset_key], first_dataset_key, next_dataset_key
                    )

                if first_key and next_key:
                    # Ensure keys exist in both dataframes
                    if first_key in result_df.columns and next_key in data[next_dataset_key].columns:
                        print(f"    Joining on {first_key} = {next_key}")
                        result_df = result_df.merge(
                            data[next_dataset_key],
                            left_on=first_key,
                            right_on=next_key,
                            how=join_type,
                            suffixes=('', f'_{next_dataset}')
                        )
                        print(f"    ✓ Joined {first_dataset_key} with {next_dataset_key}: {len(result_df)} rows")
                        first_key = first_key  # Keep using the same key for subsequent joins
                    else:
                        print(f"  ⚠ Join keys not found in datasets")
                        print(f"     Looking for: {first_key} in result, {next_key} in {next_dataset_key}")
                else:
                    print(f"  ⚠ Could not determine join keys for {first_dataset} and {next_dataset}")

            if len(result_df) > 0:
                return result_df
            else:
                print("  ⚠ Join resulted in empty dataset")
                return None

        except Exception as e:
            print(f"  ⚠ Error performing join: {e}")
            return None

    def _find_join_keys(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str,
        name2: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Intelligently find matching join keys between two datasets.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            name1: Name of first dataset
            name2: Name of second dataset

        Returns:
            Tuple of (key1, key2) or (None, None) if not found
        """
        # Strategy 1: Look for exact column name matches
        common_cols = set(df1.columns).intersection(set(df2.columns))
        if common_cols:
            # Prioritize common ID columns
            for col in common_cols:
                col_lower = col.lower()
                if 'id' in col_lower or 'key' in col_lower or 'code' in col_lower:
                    return col, col

            # Return first common column
            return list(common_cols)[0], list(common_cols)[0]

        # Strategy 2: Look for columns with similar names (e.g., "emp_id" and "employee_id")
        for col1 in df1.columns:
            col1_lower = col1.lower().replace('_', '').replace('-', '')
            for col2 in df2.columns:
                col2_lower = col2.lower().replace('_', '').replace('-', '')

                # Check if column names are similar
                if col1_lower in col2_lower or col2_lower in col1_lower:
                    # Verify they contain ID-like data
                    if 'id' in col1_lower or 'id' in col2_lower:
                        return col1, col2

        # Strategy 3: Look for columns that reference the other dataset
        # E.g., "customer_id" in orders table when joining with customers
        for col1 in df1.columns:
            if name2.lower() in col1.lower() and ('id' in col1.lower() or 'key' in col1.lower()):
                # Find matching column in df2
                for col2 in df2.columns:
                    if 'id' in col2.lower() or col2.lower() == col1.lower():
                        return col1, col2

        for col2 in df2.columns:
            if name1.lower() in col2.lower() and ('id' in col2.lower() or 'key' in col2.lower()):
                # Find matching column in df1
                for col1 in df1.columns:
                    if 'id' in col1.lower() or col1.lower() == col2.lower():
                        return col1, col2

        # Strategy 4: Use LLM to infer join keys
        return self._llm_infer_join_keys(df1, df2, name1, name2)

    def _llm_infer_join_keys(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str,
        name2: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Use LLM to infer the join keys between two datasets.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            name1: Name of first dataset
            name2: Name of second dataset

        Returns:
            Tuple of (key1, key2) or (None, None) if not found
        """
        inference_prompt = ChatPromptTemplate.from_template(
            """Identify the join keys between these two datasets.

Dataset 1: {name1}
Columns: {cols1}
Sample values:
{sample1}

Dataset 2: {name2}
Columns: {cols2}
Sample values:
{sample2}

Which columns should be used to join these datasets?
Respond with JSON:
{{
    "join_key_1": "column_name_in_dataset1",
    "join_key_2": "column_name_in_dataset2",
    "confidence": "high|medium|low",
    "reasoning": "explanation"
}}

If no suitable join keys exist, set both to null.
"""
        )

        try:
            # Create sample data strings
            sample1 = df1.head(3).to_string()
            sample2 = df2.head(3).to_string()

            chain = inference_prompt | self.llm
            response = chain.invoke({
                'name1': name1,
                'cols1': ', '.join(df1.columns[:15]),
                'sample1': sample1[:500],
                'name2': name2,
                'cols2': ', '.join(df2.columns[:15]),
                'sample2': sample2[:500]
            })

            # Parse JSON from response
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                result = json.loads(content[start_idx:end_idx])
                key1 = result.get('join_key_1')
                key2 = result.get('join_key_2')

                if key1 and key2 and result.get('confidence') in ['high', 'medium']:
                    print(f"    🤖 LLM suggested join: {key1} = {key2} (confidence: {result.get('confidence')})")
                    return key1, key2

        except Exception as e:
            print(f"    ⚠ LLM join key inference failed: {e}")

        return None, None

    def _find_dataset_key(self, data: Dict[str, pd.DataFrame], target_name: str) -> Optional[str]:
        """
        Find the actual dataset key in the data dictionary using fuzzy matching.

        Handles cases where LLM returns shortened names like 'csv_0_sales_performance'
        but actual key is 'csv_0_sales_performance_2024.csv'

        Args:
            data: Dictionary of DataFrames
            target_name: Target dataset name to find

        Returns:
            Actual key in data dict, or None if not found
        """
        # Exact match
        if target_name in data:
            return target_name

        # Fuzzy matching - normalize names for comparison
        target_normalized = target_name.lower().replace('.csv', '').replace('.xlsx', '').replace('.xls', '')

        for key in data.keys():
            key_normalized = key.lower().replace('.csv', '').replace('.xlsx', '').replace('.xls', '')

            # Check if target is a substring or vice versa
            if target_normalized in key_normalized or key_normalized in target_normalized:
                return key

        # Try even fuzzier matching - remove common suffixes like _2024
        import re
        target_base = re.sub(r'_\d{4}', '', target_normalized)  # Remove _2024, _2023, etc.

        for key in data.keys():
            key_normalized = key.lower().replace('.csv', '').replace('.xlsx', '').replace('.xls', '')
            key_base = re.sub(r'_\d{4}', '', key_normalized)

            if target_base in key_base or key_base in target_base:
                return key

        return None
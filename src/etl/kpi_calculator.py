"""Auto-calculates KPIs based on domain and available columns."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.utils.llm_client import get_llm
from src.utils.embedding_service import get_embedding_service
from langchain_core.prompts import ChatPromptTemplate
import json


class KPICalculator:
    """Automatically identifies and calculates KPIs with embeddings."""

    # Pre-defined KPI formulas by domain
    KPI_LIBRARY = {
        'Finance': [
            {
                'name': 'Total Revenue',
                'required_columns': ['revenue'],
                'formula': 'df["revenue"].sum()',
                'aggregation': 'sum',
                'unit': '$',
                'description': 'Total revenue generated'
            },
            {
                'name': 'Profit Margin',
                'required_columns': ['revenue', 'cost'],
                'formula': '((df["revenue"] - df["cost"]) / df["revenue"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Percentage of revenue retained as profit'
            },
            {
                'name': 'ROI',
                'required_columns': ['profit', 'investment'],
                'formula': '(df["profit"] / df["investment"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Return on investment percentage'
            },
            {
                'name': 'Revenue Growth Rate',
                'required_columns': ['revenue', 'date'],
                'formula': 'calculate_growth_rate',
                'aggregation': 'custom',
                'unit': '%',
                'description': 'Period-over-period revenue growth'
            }
        ],
        'Marketing': [
            {
                'name': 'CTR',
                'required_columns': ['clicks', 'impressions'],
                'formula': '(df["clicks"] / df["impressions"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Click-through rate'
            },
            {
                'name': 'CPA',
                'required_columns': ['ad_spend', 'conversions'],
                'formula': 'df["ad_spend"] / df["conversions"]',
                'aggregation': 'mean',
                'unit': '$',
                'description': 'Cost per acquisition'
            },
            {
                'name': 'ROAS',
                'required_columns': ['revenue', 'ad_spend'],
                'formula': 'df["revenue"] / df["ad_spend"]',
                'aggregation': 'mean',
                'unit': 'x',
                'description': 'Return on ad spend'
            },
            {
                'name': 'Conversion Rate',
                'required_columns': ['conversions', 'visits'],
                'formula': '(df["conversions"] / df["visits"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Percentage of visits that convert'
            }
        ],
        'Sales': [
            {
                'name': 'Average Deal Size',
                'required_columns': ['deal_value'],
                'formula': 'df["deal_value"].mean()',
                'aggregation': 'mean',
                'unit': '$',
                'description': 'Average value of closed deals'
            },
            {
                'name': 'Win Rate',
                'required_columns': ['deals_closed', 'opportunities'],
                'formula': '(df["deals_closed"] / df["opportunities"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Percentage of opportunities converted'
            },
            {
                'name': 'Sales Velocity',
                'required_columns': ['revenue', 'days'],
                'formula': 'df["revenue"] / df["days"]',
                'aggregation': 'mean',
                'unit': '$/day',
                'description': 'Revenue generated per day'
            }
        ],
        'HR': [
            {
                'name': 'Attrition Rate',
                'required_columns': ['employees_left', 'total_employees'],
                'formula': '(df["employees_left"] / df["total_employees"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Employee turnover percentage'
            },
            {
                'name': 'Average Tenure',
                'required_columns': ['tenure_days'],
                'formula': 'df["tenure_days"].mean() / 365',
                'aggregation': 'mean',
                'unit': 'years',
                'description': 'Average employee tenure in years'
            }
        ],
        'Operations': [
            {
                'name': 'Efficiency Rate',
                'required_columns': ['actual_output', 'planned_output'],
                'formula': '(df["actual_output"] / df["planned_output"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Operational efficiency percentage'
            },
            {
                'name': 'Utilization Rate',
                'required_columns': ['hours_used', 'hours_available'],
                'formula': '(df["hours_used"] / df["hours_available"] * 100)',
                'aggregation': 'mean',
                'unit': '%',
                'description': 'Resource utilization percentage'
            }
        ]
    }

    def __init__(self, enable_custom_suggestions: bool = False, enable_embeddings: bool = False):
        """
        Initialize KPI Calculator with performance optimization flags.

        Args:
            enable_custom_suggestions: Enable LLM-based custom KPI suggestions (SLOW - adds ~2-5s per dataset)
            enable_embeddings: Enable KPI embedding creation (SLOW - adds ~1-3s per KPI)
        """
        self.enable_custom_suggestions = enable_custom_suggestions
        self.enable_embeddings = enable_embeddings

        # Lazy initialization - only create if needed
        self._llm = None
        self._embedding_service = None

    @property
    def llm(self):
        """Lazy load LLM only when needed."""
        if self._llm is None and self.enable_custom_suggestions:
            self._llm = get_llm(temperature=0.0, model="gemini-2.5-flash")
        return self._llm

    @property
    def embedding_service(self):
        """Lazy load embedding service only when needed."""
        if self._embedding_service is None and self.enable_embeddings:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def identify_kpis(
        self,
        df: pd.DataFrame,
        domain: str,
        column_semantics: Dict
    ) -> List[Dict]:
        """
        Identify which KPIs can be calculated from available columns.

        Args:
            df: DataFrame with data
            domain: Business domain
            column_semantics: Semantic metadata for columns

        Returns:
            List of applicable KPI definitions
        """
        applicable_kpis = []

        # 1. Check pre-defined KPIs (FAST - no API calls)
        domain_kpis = self.KPI_LIBRARY.get(domain, [])

        for kpi in domain_kpis:
            # Map required columns to actual columns
            column_mapping = self._find_column_mapping(
                df.columns.tolist(),
                kpi['required_columns'],
                column_semantics
            )

            if column_mapping:
                kpi_copy = kpi.copy()
                kpi_copy['column_mapping'] = column_mapping
                applicable_kpis.append(kpi_copy)
                print(f"[KPI] Found applicable: {kpi['name']}")

        # 2. OPTIONAL: Use LLM to suggest custom KPIs (disabled by default for performance)
        if self.enable_custom_suggestions:
            print(f"[KPI] Requesting custom KPI suggestions from LLM (may take 2-5 seconds)...")
            custom_kpis = self._suggest_custom_kpis(df, domain, column_semantics)
            applicable_kpis.extend(custom_kpis)
        else:
            print(f"[KPI] Custom KPI suggestions disabled (performance optimization)")

        return applicable_kpis

    def calculate_kpis(
        self,
        df: pd.DataFrame,
        kpi_definitions: List[Dict]
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Calculate KPIs and add as new columns.

        Args:
            df: DataFrame with cleaned data
            kpi_definitions: List of KPI definitions

        Returns:
            (DataFrame with KPI columns, List of calculated KPI values)
        """
        df_with_kpis = df.copy()
        calculated_kpis = []

        for kpi_def in kpi_definitions:
            try:
                kpi_name = kpi_def['name']
                kpi_col_name = f"kpi_{kpi_name.lower().replace(' ', '_')}"

                # Map columns
                col_map = kpi_def.get('column_mapping', {})

                # Create a temporary DataFrame with mapped columns
                temp_df = pd.DataFrame()
                for placeholder, actual_col in col_map.items():
                    if actual_col in df_with_kpis.columns:
                        temp_df[placeholder] = df_with_kpis[actual_col]

                # Calculate KPI
                if kpi_def['formula'] == 'calculate_growth_rate':
                    # Custom growth rate calculation
                    kpi_value = self._calculate_growth_rate(temp_df, col_map)
                else:
                    # Use eval with the temporary DataFrame
                    local_vars = {'df': temp_df, 'np': np, 'pd': pd}
                    result = eval(kpi_def['formula'], {"__builtins__": {}}, local_vars)

                    if isinstance(result, pd.Series):
                        df_with_kpis[kpi_col_name] = result
                    else:
                        # If it's a scalar, create a column with that value
                        df_with_kpis[kpi_col_name] = result

                # Calculate aggregated value for reporting
                if kpi_col_name in df_with_kpis.columns:
                    if kpi_def['aggregation'] == 'sum':
                        agg_value = df_with_kpis[kpi_col_name].sum()
                    elif kpi_def['aggregation'] == 'mean':
                        agg_value = df_with_kpis[kpi_col_name].mean()
                    else:
                        agg_value = df_with_kpis[kpi_col_name].mean()
                else:
                    agg_value = result if not isinstance(result, pd.Series) else result.mean()

                calculated_kpis.append({
                    'name': kpi_name,
                    'value': float(agg_value) if not pd.isna(agg_value) else 0,
                    'unit': kpi_def.get('unit', ''),
                    'description': kpi_def.get('description', ''),
                    'column_name': kpi_col_name
                })

                print(f"[KPI] Calculated: {kpi_name} = {agg_value:.2f} {kpi_def.get('unit', '')}")

            except Exception as e:
                print(f"[WARN] Failed to calculate KPI '{kpi_def['name']}': {e}")

        return df_with_kpis, calculated_kpis

    def _find_column_mapping(
        self,
        available_columns: List[str],
        required_columns: List[str],
        column_semantics: Dict
    ) -> Optional[Dict]:
        """
        Map required KPI columns to actual dataset columns.

        Args:
            available_columns: List of actual column names
            required_columns: List of required column names for KPI
            column_semantics: Semantic metadata for columns

        Returns:
            {placeholder: actual_column} or None if mapping fails
        """
        mapping = {}

        for req_col in required_columns:
            # Try exact match (case-insensitive)
            matched = False
            for actual_col in available_columns:
                if req_col.lower() in actual_col.lower():
                    mapping[req_col] = actual_col
                    matched = True
                    break

            # Try semantic match using business meaning
            if not matched:
                for actual_col, sem_meta in column_semantics.items():
                    if actual_col not in available_columns:
                        continue

                    business_meaning = sem_meta.get('business_meaning', '').lower()
                    if req_col.lower() in business_meaning:
                        mapping[req_col] = actual_col
                        matched = True
                        break

            if not matched:
                return None  # Required column not found

        return mapping if len(mapping) == len(required_columns) else None

    def _suggest_custom_kpis(
        self,
        df: pd.DataFrame,
        domain: str,
        column_semantics: Dict
    ) -> List[Dict]:
        """Use LLM to suggest custom KPIs based on available data."""

        # Get measure and dimension columns
        measure_cols = [
            col for col, meta in column_semantics.items()
            if meta.get('semantic_type') == 'measure' and col in df.columns
        ]

        dimension_cols = [
            col for col, meta in column_semantics.items()
            if meta.get('semantic_type') == 'dimension' and col in df.columns
        ]

        if len(measure_cols) < 2:
            return []  # Need at least 2 measures for derived KPIs

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst suggesting KPIs.

Given a domain and available columns, suggest 2-3 derived KPIs that would be valuable.

Return JSON array:
[
  {{
    "name": "KPI Name",
    "required_columns": ["col1", "col2"],
    "formula": "mathematical formula using df['column'] syntax",
    "aggregation": "sum|mean",
    "unit": "$|%|count|x|ratio",
    "description": "What this KPI measures"
  }}
]

Rules:
- Only suggest KPIs that can be calculated from available columns
- Use pandas syntax: df["column_name"]
- Formulas should be valid Python expressions
- Focus on actionable business metrics

Return ONLY valid JSON."""),
            ("user", """Domain: {domain}

Available measure columns: {measures}
Available dimension columns: {dimensions}

Suggest custom KPIs:""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "domain": domain,
            "measures": ", ".join(measure_cols[:10]),
            "dimensions": ", ".join(dimension_cols[:5])
        })

        try:
            custom_kpis = json.loads(result.content.strip())

            # Validate and map columns
            validated = []
            for kpi in custom_kpis:
                col_map = self._find_column_mapping(
                    df.columns.tolist(),
                    kpi.get('required_columns', []),
                    column_semantics
                )
                if col_map:
                    kpi['column_mapping'] = col_map
                    validated.append(kpi)
                    print(f"[KPI] Custom KPI suggested: {kpi['name']}")

            return validated

        except json.JSONDecodeError:
            return []

    def _calculate_growth_rate(self, df: pd.DataFrame, column_mapping: Dict) -> float:
        """
        Calculate growth rate for time-series data.

        Args:
            df: DataFrame with date and value columns
            column_mapping: Mapping of required columns to actual columns

        Returns:
            Growth rate percentage
        """
        try:
            if 'date' in df.columns and 'revenue' in df.columns:
                # Sort by date
                df = df.sort_values('date')

                # Get first and last values
                first_value = df['revenue'].iloc[0]
                last_value = df['revenue'].iloc[-1]

                if first_value > 0:
                    growth_rate = ((last_value - first_value) / first_value) * 100
                    return growth_rate
            return 0.0
        except:
            return 0.0

    def store_kpi_definitions(self, session, domain: str, kpi_definitions: List[Dict]):
        """
        Store KPI definitions in database for reuse.

        Args:
            session: Database session
            domain: Business domain
            kpi_definitions: List of KPI definitions to store
        """
        from src.database.models import KPIDefinition

        for kpi_def in kpi_definitions:
            try:
                # OPTIMIZATION: Skip embedding creation if disabled (saves 1-3s per KPI)
                embedding = None
                if self.enable_embeddings and self.embedding_service:
                    print(f"[KPI] Creating embedding for {kpi_def['name']}...")
                    embedding = self.embedding_service.create_kpi_embedding(
                        kpi_def['name'],
                        kpi_def.get('description', ''),
                        domain
                    )

                kpi = KPIDefinition(
                    kpi_name=kpi_def['name'],
                    domain=domain,
                    description=kpi_def.get('description', ''),
                    formula=kpi_def.get('formula', ''),
                    unit=kpi_def.get('unit', ''),
                    required_columns=kpi_def.get('required_columns', []),
                    kpi_embedding=embedding  # Will be None if embeddings disabled
                )

                session.add(kpi)

            except Exception as e:
                print(f"[WARN] Failed to store KPI definition: {e}")

        session.flush()
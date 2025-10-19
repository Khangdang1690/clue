"""LLM-powered semantic analysis for datasets using Gemini."""

from typing import Dict, List, Optional
import pandas as pd
from src.utils.llm_client import get_llm
from src.utils.embedding_service import get_embedding_service
from langchain_core.prompts import ChatPromptTemplate
import json


class SemanticAnalyzer:
    """Analyzes datasets to understand domain, entities, and business meaning."""

    def __init__(self):
        self.llm = get_llm(temperature=0.1, model="gemini-2.0-flash")
        self.embedding_service = get_embedding_service()

    def generate_table_name(self, df: pd.DataFrame, original_filename: str) -> str:
        """
        Use LLM to generate a concise, meaningful table name.

        Args:
            df: DataFrame preview
            original_filename: Original file name

        Returns:
            Generated table name (snake_case, concise)
        """
        # Get sample data
        sample = df.head(10).to_string()
        columns = list(df.columns)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst naming database tables.

Given a data sample and original filename, generate a concise, meaningful table name.

Rules:
- Use snake_case (lowercase with underscores)
- Maximum 3-4 words
- Be descriptive but concise
- Include domain hint if obvious (e.g., finance_, marketing_, hr_)
- Remove year/date from name unless essential
- Examples: "sales_performance", "customer_feedback", "marketing_campaigns"

Return ONLY the table name, nothing else."""),
            ("user", """Original filename: {filename}

Columns: {columns}

Sample data:
{sample}

Generate table name:""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "filename": original_filename,
            "columns": ", ".join(columns[:10]),  # First 10 columns
            "sample": sample[:1000]  # Limit sample size
        })

        table_name = result.content.strip().lower().replace(' ', '_').replace('-', '_')

        # Clean up - keep only alphanumeric and underscores
        table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)

        # Remove multiple underscores
        while '__' in table_name:
            table_name = table_name.replace('__', '_')

        return table_name

    def analyze_dataset(self, df: pd.DataFrame, table_name: str) -> Dict:
        """
        Perform deep semantic analysis on dataset with embeddings.

        Args:
            df: DataFrame to analyze
            table_name: Generated table name

        Returns:
            Dictionary with semantic analysis including embeddings
        """
        # Get schema info
        schema_info = self._get_schema_summary(df)
        sample_data = df.head(20).to_string()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst performing semantic analysis.

Analyze the dataset and return a JSON object with:
{{
  "domain": "Finance|Marketing|Sales|HR|Operations|Product|Support|Other",
  "department": "specific department name or null",
  "description": "1-2 sentence description of what this data represents",
  "entities": ["list", "of", "business", "entities", "found"],
  "business_terms": {{"column_name": "business meaning", ...}},
  "suggested_questions": ["What insights could be found?", ...],
  "column_semantics": {{
    "column_name": {{
      "semantic_type": "dimension|measure|key|text|date",
      "business_meaning": "what this column represents",
      "is_primary_key": true/false,
      "is_foreign_key": true/false,
      "potential_relationships": ["other_column_names_it_might_relate_to"]
    }}
  }}
}}

semantic_type definitions:
- dimension: categorical attribute (customer name, product category, region)
- measure: numeric metric to aggregate (revenue, quantity, price)
- key: identifier (customer_id, order_id, product_id)
- text: free text (description, comments, notes)
- date: temporal data (date, timestamp, year, month)

Identify ALL columns that end with _id or contain 'id' as likely keys.
Look for relationships between tables based on column names.

Return ONLY valid JSON, no explanations."""),
            ("user", """Table name: {table_name}

Schema:
{schema}

Sample data:
{sample}

Analyze this dataset:""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "table_name": table_name,
            "schema": schema_info,
            "sample": sample_data[:2000]
        })

        # Parse JSON response
        try:
            # Strip markdown code blocks if present
            content = result.content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]  # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()

            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse LLM response: {e}")
            print(f"Response: {result.content[:500]}")
            # Return minimal analysis
            analysis = {
                'domain': 'Unknown',
                'department': None,
                'description': f"Dataset from {table_name}",
                'entities': [],
                'business_terms': {},
                'suggested_questions': [],
                'column_semantics': {}
            }

        # Add embeddings for semantic search
        analysis['description_embedding'] = self.embedding_service.create_dataset_description_embedding(analysis)

        # Schema embedding for structural similarity
        analysis['schema_embedding'] = self.embedding_service.create_schema_embedding(
            list(df.columns),
            [str(df[col].dtype) for col in df.columns]
        )

        # Column embeddings for relationship detection
        for col_name in df.columns:
            if col_name in analysis['column_semantics']:
                col_meta = analysis['column_semantics'][col_name]
                col_meta['semantic_embedding'] = self.embedding_service.create_column_semantic_embedding(
                    col_name,
                    col_meta.get('business_meaning', ''),
                    col_meta.get('semantic_type', '')
                )

        # Add statistics for each column
        for col_name in df.columns:
            if col_name not in analysis['column_semantics']:
                analysis['column_semantics'][col_name] = {}

            col_stats = {
                'null_count': int(df[col_name].isnull().sum()),
                'null_percentage': float(df[col_name].isnull().mean() * 100),
                'unique_count': int(df[col_name].nunique()),
                'unique_percentage': float(df[col_name].nunique() / len(df) * 100) if len(df) > 0 else 0
            }
            analysis['column_semantics'][col_name].update(col_stats)

        return analysis

    def _get_schema_summary(self, df: pd.DataFrame) -> str:
        """Generate schema summary for LLM."""
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = df[col].isnull().mean() * 100
            unique_count = df[col].nunique()

            # Add sample values for categorical columns
            sample_values = ""
            if df[col].dtype == 'object' and unique_count < 20:
                top_values = df[col].value_counts().head(3).index.tolist()
                sample_values = f" (e.g., {', '.join(str(v) for v in top_values)})"

            lines.append(f"  {col}: {dtype} ({unique_count} unique, {null_pct:.1f}% null){sample_values}")

        return "\n".join(lines)

    def suggest_kpis(self, analysis: Dict, df: pd.DataFrame) -> List[Dict]:
        """
        Suggest KPIs based on semantic analysis.

        Args:
            analysis: Semantic analysis result
            df: DataFrame

        Returns:
            List of suggested KPIs
        """
        domain = analysis.get('domain', 'Unknown')
        column_semantics = analysis.get('column_semantics', {})

        # Get measure columns
        measure_cols = [
            col for col, meta in column_semantics.items()
            if meta.get('semantic_type') == 'measure'
        ]

        # Get date columns
        date_cols = [
            col for col, meta in column_semantics.items()
            if meta.get('semantic_type') == 'date'
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst suggesting KPIs.

Given a domain and available columns, suggest 3-5 valuable KPIs.

Return JSON array:
[
  {{
    "name": "KPI Name",
    "description": "What it measures",
    "formula": "How to calculate (using column names)",
    "unit": "$|%|count|ratio",
    "required_columns": ["col1", "col2"],
    "importance": "high|medium|low"
  }}
]

Focus on actionable KPIs relevant to the domain.
Only suggest KPIs that can be calculated from available columns.

Return ONLY valid JSON."""),
            ("user", """Domain: {domain}

Measure columns: {measures}
Date columns: {dates}
All columns: {all_columns}

Suggest KPIs:""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "domain": domain,
            "measures": ", ".join(measure_cols[:10]) if measure_cols else "None",
            "dates": ", ".join(date_cols) if date_cols else "None",
            "all_columns": ", ".join(df.columns[:20])
        })

        try:
            kpis = json.loads(result.content.strip())
            return kpis
        except json.JSONDecodeError:
            return []
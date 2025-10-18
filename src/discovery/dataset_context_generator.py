"""Dataset Context Generator - Outer agent layer to understand dataset domain."""

import pandas as pd
from typing import Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm


class DatasetContextGenerator:
    """
    Outer agent layer that scans dataset to generate supporting context.

    This helps the discovery system understand WHAT the dataset is about
    before starting analysis (e.g., stock data, customer data, financial data).
    """

    def __init__(self):
        """Initialize context generator with LLM."""
        self.llm = get_llm(temperature=0.3)  # Low temperature for factual analysis

    def generate_context(self, df: pd.DataFrame, filename: str) -> Dict:
        """
        Scan dataset and generate supporting context.

        Args:
            df: DataFrame to analyze
            filename: Original filename (may contain hints)

        Returns:
            Dictionary with:
            - domain: What domain is this data from? (finance, marketing, etc.)
            - dataset_type: What type of data? (stock prices, transactions, etc.)
            - key_entities: Main entities in data (companies, customers, products)
            - time_period: Time range if temporal data
            - suggested_questions: Domain-specific questions to investigate
            - business_context: What this data is typically used for
        """
        print("\n" + "="*70)
        print("🔍 DATASET CONTEXT GENERATION (Outer Agent Layer)")
        print("="*70)
        print(f"\nScanning dataset: {filename}")
        print(f"Size: {len(df):,} rows × {len(df.columns)} columns\n")

        # Step 1: Create data profile for LLM
        data_profile = self._create_data_profile(df, filename)

        # Step 2: Use LLM to understand domain
        context = self._analyze_domain_with_llm(data_profile, df)

        # Step 3: Print summary
        self._print_context_summary(context)

        return context

    def _create_data_profile(self, df: pd.DataFrame, filename: str) -> str:
        """Create a concise data profile for LLM analysis."""

        profile_parts = []

        # Filename hints
        profile_parts.append(f"Filename: {filename}")

        # Column names (very important signal)
        profile_parts.append(f"\nColumn names: {', '.join(df.columns.tolist())}")

        # Column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        if numeric_cols:
            profile_parts.append(f"\nNumeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        if categorical_cols:
            profile_parts.append(f"\nCategorical columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        if datetime_cols:
            profile_parts.append(f"\nDatetime columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")

        # Sample data (first 3 rows)
        profile_parts.append("\nSample data (first 3 rows):")
        profile_parts.append(df.head(3).to_string())

        # Value examples for categorical columns (important domain signals)
        if categorical_cols:
            profile_parts.append("\nUnique values in categorical columns:")
            for col in categorical_cols[:5]:  # Top 5 categorical columns
                unique_vals = df[col].dropna().unique()[:10]  # First 10 unique values
                profile_parts.append(f"  {col}: {', '.join(map(str, unique_vals))}")
                if len(df[col].unique()) > 10:
                    profile_parts.append(f"    ... and {len(df[col].unique()) - 10} more")

        # Numeric ranges (helps identify stock prices, revenues, etc.)
        if numeric_cols:
            profile_parts.append("\nNumeric column ranges:")
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                profile_parts.append(f"  {col}: {min_val:.2f} to {max_val:.2f} (mean: {mean_val:.2f})")

        # Temporal info
        if datetime_cols:
            for col in datetime_cols[:2]:
                min_date = df[col].min()
                max_date = df[col].max()
                profile_parts.append(f"\nTime range ({col}): {min_date} to {max_date}")

        return "\n".join(profile_parts)

    def _analyze_domain_with_llm(self, data_profile: str, df: pd.DataFrame) -> Dict:
        """Use LLM to identify dataset domain and context."""

        print("🤖 Analyzing dataset domain with LLM...")

        domain_prompt = ChatPromptTemplate.from_template(
            """You are a data domain expert. Analyze this dataset and identify its domain and context.

Dataset Profile:
{data_profile}

Based on the filename, column names, sample data, and value patterns, answer these questions:

1. DOMAIN: What domain/industry is this dataset from?
   Examples: Finance/Stock Market, E-commerce, Healthcare, Marketing, Operations, HR, etc.

2. DATASET_TYPE: What specific type of data is this?
   Examples: Stock prices, Financial statements, Customer transactions, Website analytics, etc.

3. KEY_ENTITIES: What are the main entities/subjects in this data?
   Examples: Companies, Customers, Products, Employees, Transactions, etc.

4. TIME_PERIOD: What time period does this data cover? (if temporal)

5. TYPICAL_USE_CASES: What is this data typically used for?
   Examples: Stock analysis, Revenue forecasting, Customer segmentation, etc.

6. SUGGESTED_QUESTIONS: Based on the domain, what are 5-7 important questions to investigate?
   Make these SPECIFIC to the domain (e.g., for stock data: "Which companies have highest revenue growth?")

7. BUSINESS_CONTEXT: What business context is important to understand this data?
   Examples: For stock data - fiscal years, reporting periods, accounting metrics, etc.

Respond in JSON format:
{{
    "domain": "domain name",
    "dataset_type": "specific type",
    "key_entities": ["entity1", "entity2"],
    "time_period": "time range or null",
    "typical_use_cases": ["use case 1", "use case 2"],
    "suggested_questions": [
        "Question 1 specific to domain",
        "Question 2 specific to domain",
        "Question 3 specific to domain",
        "Question 4 specific to domain",
        "Question 5 specific to domain"
    ],
    "business_context": {{
        "key_concept_1": "explanation",
        "key_concept_2": "explanation"
    }},
    "confidence": "high/medium/low"
}}

IMPORTANT: Make suggested questions SPECIFIC to the domain, not generic.
For example:
- Stock data: "Which companies had highest revenue in fiscal year 2024?"
- Customer data: "What customer segments have highest lifetime value?"
- Marketing data: "Which campaigns had best ROI in Q4?"
"""
        )

        try:
            chain = domain_prompt | self.llm
            response = chain.invoke({"data_profile": data_profile})

            # Parse JSON from response
            import json
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                context = json.loads(content[start_idx:end_idx])

                # Add metadata
                context['dataset_size'] = {
                    'rows': len(df),
                    'columns': len(df.columns)
                }

                return context
            else:
                print("[WARN]  Could not parse LLM response, using fallback")
                return self._create_fallback_context(df)

        except Exception as e:
            print(f"[WARN]  LLM analysis failed: {e}")
            print("   Using fallback context generation...")
            return self._create_fallback_context(df)

    def _create_fallback_context(self, df: pd.DataFrame) -> Dict:
        """Create basic context if LLM fails."""
        return {
            "domain": "Unknown",
            "dataset_type": "Generic tabular data",
            "key_entities": df.columns.tolist()[:3],
            "time_period": None,
            "typical_use_cases": ["Data exploration", "Pattern analysis"],
            "suggested_questions": [
                "What are the main patterns in the data?",
                "Are there any correlations between variables?",
                "What are the distributions of key metrics?"
            ],
            "business_context": {
                "note": "Context could not be automatically generated"
            },
            "confidence": "low",
            "dataset_size": {
                "rows": len(df),
                "columns": len(df.columns)
            }
        }

    def _print_context_summary(self, context: Dict):
        """Print a nice summary of the generated context."""

        print("\n" + "="*70)
        print("[OK] DATASET CONTEXT GENERATED")
        print("="*70)

        print(f"\n📊 Domain: {context['domain']}")
        print(f"📁 Dataset Type: {context['dataset_type']}")
        print(f"🏢 Key Entities: {', '.join(context['key_entities'])}")

        if context.get('time_period'):
            print(f"📅 Time Period: {context['time_period']}")

        print(f"\n💼 Typical Use Cases:")
        for i, use_case in enumerate(context.get('typical_use_cases', []), 1):
            print(f"   {i}. {use_case}")

        print(f"\n❓ Suggested Questions ({len(context.get('suggested_questions', []))}):")
        for i, question in enumerate(context.get('suggested_questions', []), 1):
            print(f"   {i}. {question}")

        print(f"\n📚 Business Context:")
        for key, value in context.get('business_context', {}).items():
            print(f"   • {key}: {value}")

        print(f"\n🎯 Confidence: {context.get('confidence', 'unknown').upper()}")

        print("\n" + "="*70 + "\n")

    def generate_context_summary_for_discovery(self, context: Dict) -> str:
        """
        Generate a text summary to inject into discovery workflow.

        This will help the question generator and analysis executor
        understand the domain context.
        """
        summary_parts = []

        summary_parts.append(f"DATASET DOMAIN: {context['domain']}")
        summary_parts.append(f"DATASET TYPE: {context['dataset_type']}")
        summary_parts.append(f"KEY ENTITIES: {', '.join(context['key_entities'])}")

        if context.get('time_period'):
            summary_parts.append(f"TIME PERIOD: {context['time_period']}")

        summary_parts.append("\nTYPICAL USE CASES:")
        for use_case in context.get('typical_use_cases', []):
            summary_parts.append(f"- {use_case}")

        summary_parts.append("\nBUSINESS CONTEXT:")
        for key, value in context.get('business_context', {}).items():
            summary_parts.append(f"- {key}: {value}")

        return "\n".join(summary_parts)

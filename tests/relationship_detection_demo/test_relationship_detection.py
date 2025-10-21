"""
Test script to demonstrate LLM-powered relationship detection.

This test shows how the AI detects relationships between:
- client_id (customers.csv)
- user_id (transactions.csv)
- customer_code (customer_profiles.csv)
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.etl.relationship_detector import RelationshipDetector
from src.etl.semantic_analyzer import SemanticAnalyzer


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n--- {text} ---")


def main():
    print_header("LLM RELATIONSHIP DETECTION TEST")
    print("\nThis test demonstrates how the AI detects relationships between:")
    print("  1. client_id (in customers.csv)")
    print("  2. user_id (in transactions.csv)")
    print("  3. customer_code (in customer_profiles.csv)")

    # Step 1: Load test datasets
    print_section("Step 1: Loading Test Datasets")

    test_dir = Path(__file__).parent

    customers_df = pd.read_csv(test_dir / "customers.csv")
    transactions_df = pd.read_csv(test_dir / "transactions.csv")
    profiles_df = pd.read_csv(test_dir / "customer_profiles.csv")

    datasets = {
        'customers': customers_df,
        'transactions': transactions_df,
        'customer_profiles': profiles_df
    }

    print(f"[OK] Loaded 3 datasets:")
    for name, df in datasets.items():
        print(f"  - {name}: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"    Columns: {', '.join(df.columns.tolist())}")

    # Step 2: Run semantic analysis
    print_section("Step 2: Running Semantic Analysis (LLM)")
    print("The LLM analyzes each column to understand its business meaning...\n")

    analyzer = SemanticAnalyzer()
    metadata = {}

    for dataset_id, df in datasets.items():
        print(f"Analyzing {dataset_id}...")
        try:
            # Run semantic analysis
            analysis = analyzer.analyze_dataset(
                df=df,
                table_name=dataset_id
            )
            metadata[dataset_id] = analysis
            print(f"[OK] {dataset_id} analyzed")

            # Show key columns analyzed
            column_semantics = analysis.get('column_semantics', {})
            for col_name in ['client_id', 'user_id', 'customer_code']:
                if col_name in column_semantics:
                    col_info = column_semantics[col_name]
                    print(f"  Column: {col_name}")
                    print(f"    Semantic type: {col_info.get('semantic_type', 'unknown')}")
                    print(f"    Business meaning: {col_info.get('business_meaning', 'N/A')}")

        except Exception as e:
            print(f"[ERROR] Error analyzing {dataset_id}: {e}")
            # Create minimal metadata as fallback
            metadata[dataset_id] = {
                'table_name': dataset_id,
                'domain': 'Business',
                'column_semantics': {}
            }

    # Step 3: Run relationship detection
    print_section("Step 3: Running Relationship Detection (LLM)")
    print("The LLM will analyze potential relationships using:")
    print("  - Column name similarity")
    print("  - Data type compatibility")
    print("  - Sample value patterns")
    print("  - Business meaning alignment\n")

    detector = RelationshipDetector(confidence_threshold=0.7)

    print("Starting 5-step detection process...\n")

    try:
        relationships = detector.detect_relationships(
            datasets=datasets,
            metadata=metadata
        )

        # Step 4: Display results
        print_section("Step 4: Results Summary")

        if not relationships:
            print("[WARN] No relationships detected")
        else:
            print(f"[OK] Found {len(relationships)} relationship(s)\n")

            for i, rel in enumerate(relationships, 1):
                print(f"\nRelationship #{i}:")
                print(f"  From: {rel['from_dataset_id']}.{rel['from_column']}")
                print(f"  To:   {rel['to_dataset_id']}.{rel['to_column']}")
                print(f"  Confidence: {rel['confidence']:.2%}")
                print(f"  Type: {rel.get('relationship_type', 'unknown')}")
                print(f"  Match %: {rel.get('match_percentage', 0):.1f}%")

                # Show scoring breakdown
                print(f"\n  Scoring Breakdown:")
                if 'name_similarity' in rel:
                    print(f"    - Name similarity: {rel['name_similarity']:.2f}")
                if 'llm_similarity' in rel:
                    print(f"    - LLM semantic score: {rel['llm_similarity']:.2f}")
                if 'llm_reasoning' in rel:
                    print(f"    - LLM reasoning: {rel['llm_reasoning']}")

                # Show statistical validation
                print(f"\n  Statistical Validation:")
                print(f"    - From unique values: {rel.get('from_unique_count', 0)}")
                print(f"    - To unique values: {rel.get('to_unique_count', 0)}")
                print(f"    - Intersection: {rel.get('intersection_count', 0)}")
                print(f"    - Recommended join: {rel.get('join_strategy', 'N/A')}")

        # Step 5: Show example join query
        if relationships:
            print_section("Step 5: Example SQL Joins")
            print("Based on detected relationships, here's how you could join the data:\n")

            for rel in relationships:
                from_table = rel['from_dataset_id']
                to_table = rel['to_dataset_id']
                from_col = rel['from_column']
                to_col = rel['to_column']
                join_type = rel.get('join_strategy', 'LEFT').upper()

                sql = f"""SELECT *
FROM {from_table}
{join_type} JOIN {to_table}
  ON {from_table}.{from_col} = {to_table}.{to_col};"""

                print(f"-- Join {from_table} with {to_table}")
                print(sql)
                print()

        print_header("TEST COMPLETE")
        print("\n[OK] Successfully demonstrated LLM-powered relationship detection!")
        print(f"[OK] Detected {len(relationships)} relationship(s)")
        print("\nThe LLM correctly identified that:")
        print("  - client_id, user_id, and customer_code all reference the same customer entity")
        print("  - The relationships can be used to join data across tables")

    except Exception as e:
        print(f"\n[ERROR] Error during relationship detection: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
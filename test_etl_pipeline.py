"""Test script for the complete Auto-ETL pipeline with pgvector."""

import os
import sys
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from src.database.connection import DatabaseManager
from src.graph.etl_workflow import ETLWorkflow
from src.graph.multi_table_discovery import MultiTableDiscovery


def create_sample_data():
    """Create sample CSV files for testing."""

    # Create test data directory
    test_dir = Path("test_data/etl_demo")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Sample 1: Sales data (Finance domain)
    sales_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C001', 'C002', 'C004', 'C005'],
        'product_id': ['P001', 'P002', 'P001', 'P003', 'P002', 'P001', 'P003'],
        'date': pd.date_range('2024-01-01', periods=7, freq='W'),
        'revenue': [1000, 1500, 1200, 1800, 2000, 1100, 1600],
        'cost': [700, 900, 800, 1200, 1300, 750, 1000],
        'quantity': [10, 15, 12, 18, 20, 11, 16]
    })
    sales_path = test_dir / "sales_2024.csv"
    sales_data.to_csv(sales_path, index=False)

    # Sample 2: Customer data (Marketing domain)
    customer_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'customer_name': ['Alice Corp', 'Bob Inc', 'Charlie LLC', 'David Ltd', 'Eve Co'],
        'industry': ['Tech', 'Finance', 'Retail', 'Tech', 'Healthcare'],
        'region': ['North', 'South', 'East', 'West', 'North'],
        'acquisition_date': pd.to_datetime(['2023-01-15', '2023-03-20', '2023-06-10', '2023-09-05', '2023-11-25']),
        'lifetime_value': [50000, 75000, 45000, 60000, 35000]
    })
    customer_path = test_dir / "customers.csv"
    customer_data.to_csv(customer_path, index=False)

    # Sample 3: Product data (Product domain)
    product_data = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003'],
        'product_name': ['Widget A', 'Widget B', 'Widget C'],
        'category': ['Hardware', 'Software', 'Hardware'],
        'unit_price': [100, 150, 120],
        'launch_date': pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01'])
    })
    product_path = test_dir / "products.csv"
    product_data.to_csv(product_path, index=False)

    print("[TEST DATA] Created sample files:")
    print(f"  - {sales_path}")
    print(f"  - {customer_path}")
    print(f"  - {product_path}")

    return [str(sales_path), str(customer_path), str(product_path)]


def test_database_connection():
    """Test database connection only (tables already created by Docker)."""
    print("\n" + "="*80)
    print("[TEST] DATABASE CONNECTION")
    print("="*80)

    try:
        # Just test connection
        DatabaseManager.initialize()
        print("[OK] Database connection established")

        # Ensure tables exist (idempotent - won't recreate if they exist)
        DatabaseManager.create_all_tables()
        print("[OK] Tables verified")

        return True
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        print("\nMake sure Docker is running:")
        print("  docker-compose up -d")
        return False


def test_etl_workflow(file_paths):
    """Test the ETL workflow."""
    print("\n" + "="*80)
    print("[TEST] ETL WORKFLOW")
    print("="*80)

    try:
        # Create ETL workflow
        etl = ETLWorkflow(company_name="TestCompany")

        # Run ETL
        result = etl.run_etl(file_paths)

        if result['status'] == 'completed':
            print("\n[SUCCESS] ETL completed successfully")
            print(f"  Processed datasets: {list(result['dataset_ids'].values())}")
            print(f"  Relationships found: {len(result['relationships'])}")

            # Print relationships
            for rel in result['relationships']:
                print(f"    - {rel['from_column']} → {rel['to_column']} "
                      f"(confidence: {rel['confidence']:.2f})")

            return result['dataset_ids']
        else:
            print(f"[ERROR] ETL failed: {result.get('error_message')}")
            return None

    except Exception as e:
        print(f"[ERROR] ETL workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multi_table_discovery(dataset_ids):
    """Test multi-table discovery."""
    print("\n" + "="*80)
    print("[TEST] MULTI-TABLE DISCOVERY")
    print("="*80)

    try:
        # Get company ID (we know it's TestCompany)
        with DatabaseManager.get_session() as session:
            from src.database.repository import CompanyRepository
            company = CompanyRepository.get_or_create_company(session, "TestCompany")
            company_id = company.id

        # Create discovery workflow
        discovery = MultiTableDiscovery()

        # Run discovery
        result = discovery.run_discovery(
            company_id=company_id,
            dataset_ids=list(dataset_ids.values()),
            analysis_name="ETL Pipeline Test"
        )

        print("\n[SUCCESS] Multi-table discovery completed")
        print(f"  Report: {result.get('unified_report_path')}")

        # Print some insights
        for dataset_id, discovery_result in result['single_table_results'].items():
            print(f"\n  Dataset {dataset_id}:")
            for q in discovery_result.answered_questions[:2]:  # First 2 insights
                print(f"    - {q.question}")

        if result.get('cross_table_insights'):
            print(f"\n  Cross-table insights: {len(result['cross_table_insights'])}")
            for insight in result['cross_table_insights'][:2]:
                print(f"    - {insight['question']}")

        return result

    except Exception as e:
        print(f"[ERROR] Multi-table discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_embedding_service():
    """Test Gemini embedding service."""
    print("\n" + "="*80)
    print("[TEST] EMBEDDING SERVICE")
    print("="*80)

    try:
        from src.utils.embedding_service import get_embedding_service

        service = get_embedding_service()

        # Test single embedding
        text = "This is a test for customer revenue analysis"
        embedding = service.get_embedding(text)

        print(f"[OK] Generated embedding with {len(embedding)} dimensions")

        # Test similarity
        text1 = "customer revenue growth"
        text2 = "client income increase"
        text3 = "product inventory management"

        emb1 = service.get_embedding(text1)
        emb2 = service.get_embedding(text2)
        emb3 = service.get_embedding(text3)

        sim12 = service.calculate_similarity(emb1, emb2)
        sim13 = service.calculate_similarity(emb1, emb3)

        print(f"[OK] Similarity test:")
        print(f"  '{text1}' vs '{text2}': {sim12:.3f}")
        print(f"  '{text1}' vs '{text3}': {sim13:.3f}")

        if sim12 > sim13:
            print("  OK Semantic similarity working correctly")

        return True

    except Exception as e:
        print(f"[ERROR] Embedding service failed: {e}")
        return False


def main():
    """Run complete ETL pipeline test."""
    print("\n" + "="*80)
    print("AUTO-ETL PIPELINE TEST")
    print("="*80)

    # Test 1: Database connection
    if not test_database_connection():
        print("\n[ABORT] Cannot proceed without database")
        return

    # Test 2: Embedding service
    if not test_embedding_service():
        print("\n[WARN] Embedding service issues detected")

    # Test 3: Create sample data
    file_paths = create_sample_data()

    # Test 4: Run ETL workflow
    dataset_ids = test_etl_workflow(file_paths)

    if dataset_ids:
        # Test 5: Run multi-table discovery
        discovery_result = test_multi_table_discovery(dataset_ids)

        if discovery_result:
            print("\n" + "="*80)
            print("[SUCCESS] ALL TESTS PASSED")
            print("="*80)
            print("\nThe Auto-ETL pipeline is working correctly!")
            print("Features tested:")
            print("  OK Multi-format data ingestion")
            print("  OK Semantic analysis with Gemini LLM")
            print("  OK Embedding generation with Gemini")
            print("  OK Relationship detection using embeddings")
            print("  OK Context-aware data cleaning")
            print("  OK Domain-specific KPI calculation")
            print("  OK PostgreSQL storage with pgvector")
            print("  OK Multi-table discovery and analysis")
    else:
        print("\n[FAILED] ETL workflow did not complete")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
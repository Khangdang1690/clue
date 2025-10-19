"""Test script for Dataset CRUD operations (CREATE and DELETE)."""

import pandas as pd
from pathlib import Path
from src.database.connection import DatabaseManager
from src.database.repository import DatasetRepository, RelationshipRepository
from src.database.models import Company, Dataset, TableRelationship, AnalysisSession
from src.graph.etl_workflow import ETLWorkflow


def print_header(title):
    """Print test section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def query_database_state():
    """Query and print current database state."""
    with DatabaseManager.get_session() as session:
        companies = session.query(Company).all()

        print("\n--- DATABASE STATE ---")
        for company in companies:
            print(f"\nCompany: {company.name} (ID: {company.id})")
            datasets = DatasetRepository.get_datasets_by_company(session, company.id)
            print(f"  Datasets: {len(datasets)}")
            for ds in datasets:
                print(f"    - {ds.original_filename} -> {ds.table_name} (ID: {ds.id})")

            # Count relationships
            if datasets:
                dataset_ids = [ds.id for ds in datasets]
                relationships = RelationshipRepository.get_relationships_for_datasets(
                    session, dataset_ids, min_confidence=0.0
                )
                print(f"  Relationships: {len(relationships)}")
                for rel in relationships:
                    print(f"    - {rel.from_column} -> {rel.to_column} (confidence: {rel.confidence:.2f})")


def create_test_file(filename: str, content_type: str = 'simple') -> str:
    """Create a test CSV file."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    filepath = test_dir / filename

    if content_type == 'simple':
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, 30, 40, 50]
        })
    elif content_type == 'inventory':
        # Has product_id FK to existing product_performance dataset
        df = pd.DataFrame({
            'inventory_id': [1, 2, 3, 4, 5],
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'quantity': [100, 150, 200, 75, 120],
            'warehouse': ['WH-A', 'WH-B', 'WH-A', 'WH-C', 'WH-B']
        })

    df.to_csv(filepath, index=False)
    print(f"  Created test file: {filepath}")
    return str(filepath)


def test_1_create_normal():
    """Test 1: CREATE mode - Normal case (new file)."""
    print_header("TEST 1: CREATE MODE - NORMAL CASE (NEW FILE)")

    # Create test file
    test_file = create_test_file("test_inventory.csv", content_type='inventory')

    # Run CREATE mode
    etl = ETLWorkflow(company_name="Test Company Inc")
    try:
        state = etl.run_etl(file_paths=[test_file], mode='create')

        if state['status'] == 'completed':
            print("\n✅ TEST 1 PASSED: Successfully created new dataset")
            print(f"   Dataset ID: {list(state['dataset_ids'].values())[0]}")
            print(f"   Relationships found: {len(state['relationships'])}")

            # Print relationships
            for rel in state['relationships']:
                print(f"   - Relationship: {rel['from_column']} -> {rel['to_column']} (confidence: {rel['confidence']:.2f})")
        else:
            print(f"❌ TEST 1 FAILED: ETL status = {state['status']}")

    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")

    query_database_state()


def test_2_create_duplicate():
    """Test 2: CREATE mode - Duplicate detection (should error)."""
    print_header("TEST 2: CREATE MODE - DUPLICATE DETECTION")

    # Try to create duplicate - test_inventory.csv already exists from test 1
    test_file = create_test_file("test_inventory.csv", content_type='inventory')

    # Run CREATE mode - should fail
    etl = ETLWorkflow(company_name="Test Company Inc")
    try:
        state = etl.run_etl(file_paths=[test_file], mode='create')
        print(f"❌ TEST 2 FAILED: Should have raised error but got status = {state['status']}")

    except ValueError as e:
        if "already exists" in str(e):
            print(f"✅ TEST 2 PASSED: Correctly detected duplicate")
            print(f"   Error message: {e}")
        else:
            print(f"❌ TEST 2 FAILED: Wrong error: {e}")

    except Exception as e:
        print(f"❌ TEST 2 FAILED: Unexpected error: {e}")


def test_3_delete_cascade():
    """Test 3: DELETE operation - Verify cascade deletes."""
    print_header("TEST 3: DELETE OPERATION - CASCADE DELETES")

    # Get test_inventory dataset ID
    with DatabaseManager.get_session() as session:
        company = session.query(Company).filter(Company.name == "Test Company Inc").first()
        dataset = DatasetRepository.find_dataset_by_filename(
            session, company.id, "test_inventory.csv"
        )

        if not dataset:
            print("❌ TEST 3 FAILED: test_inventory.csv not found")
            return

        dataset_id = dataset.id
        dataset_name = dataset.table_name
        print(f"  Deleting dataset: {dataset_name} (ID: {dataset_id})")

        # Count related data before delete
        columns_before = len(dataset.columns)
        rels_from = session.query(TableRelationship).filter(
            TableRelationship.from_dataset_id == dataset_id
        ).count()
        rels_to = session.query(TableRelationship).filter(
            TableRelationship.to_dataset_id == dataset_id
        ).count()

        print(f"  Before delete:")
        print(f"    Columns: {columns_before}")
        print(f"    Relationships (from): {rels_from}")
        print(f"    Relationships (to): {rels_to}")

        # Delete
        success = DatasetRepository.delete_dataset(session, dataset_id)
        session.commit()

        if success:
            print(f"\n✅ TEST 3 PASSED: Successfully deleted dataset")

            # Verify cascades
            dataset_check = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            rels_from_check = session.query(TableRelationship).filter(
                TableRelationship.from_dataset_id == dataset_id
            ).count()
            rels_to_check = session.query(TableRelationship).filter(
                TableRelationship.to_dataset_id == dataset_id
            ).count()

            print(f"  After delete:")
            print(f"    Dataset exists: {dataset_check is not None}")
            print(f"    Relationships (from): {rels_from_check}")
            print(f"    Relationships (to): {rels_to_check}")

            if dataset_check is None and rels_from_check == 0 and rels_to_check == 0:
                print("  ✅ All cascades verified")
            else:
                print("  ❌ Cascade delete incomplete!")
        else:
            print(f"❌ TEST 3 FAILED: Delete returned False")

    query_database_state()


def test_4_cross_etl_relationships():
    """Test 4: CREATE mode - Cross-ETL relationships (new + existing datasets)."""
    print_header("TEST 4: CREATE MODE - CROSS-ETL RELATIONSHIPS")

    # Create a new file with customer_id (FK to existing customer_profiles)
    test_file = create_test_file("test_orders.csv", content_type='simple')

    # Modify to have customer_id
    df = pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5],
        'customer_id': ['C001', 'C002', 'C003', 'C001', 'C002'],
        'amount': [100, 200, 150, 300, 250]
    })
    df.to_csv(test_file, index=False)

    print(f"  Created orders file with customer_id FK")

    # Run CREATE mode
    etl = ETLWorkflow(company_name="Test Company Inc")
    try:
        state = etl.run_etl(file_paths=[test_file], mode='create')

        if state['status'] == 'completed':
            print(f"\n✅ TEST 4 PASSED: Successfully created dataset")
            print(f"   Relationships found: {len(state['relationships'])}")

            # Check if relationships detected between new and existing
            cross_etl_rels = 0
            for rel in state['relationships']:
                # Check if one dataset is new (in state) and one is existing (not in state)
                from_is_new = rel['from_dataset_id'] in state['dataset_ids']
                to_is_new = rel['to_dataset_id'] in state['dataset_ids']

                if from_is_new != to_is_new:  # One is new, one is existing
                    cross_etl_rels += 1
                    print(f"   - Cross-ETL relationship: {rel['from_column']} -> {rel['to_column']} (confidence: {rel['confidence']:.2f})")

            if cross_etl_rels > 0:
                print(f"  ✅ Detected {cross_etl_rels} cross-ETL relationships!")
            else:
                print(f"  ⚠️  No cross-ETL relationships detected (may be expected)")
        else:
            print(f"❌ TEST 4 FAILED: ETL status = {state['status']}")

    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    query_database_state()


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  DATASET CRUD OPERATIONS TEST SUITE")
    print("="*80)

    # Initialize database
    DatabaseManager.initialize()

    # Show initial state
    print_header("INITIAL DATABASE STATE")
    query_database_state()

    # Run tests
    test_1_create_normal()
    test_2_create_duplicate()
    test_3_delete_cascade()
    test_4_cross_etl_relationships()

    # Final state
    print_header("FINAL DATABASE STATE")
    query_database_state()

    print("\n" + "="*80)
    print("  TEST SUITE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()

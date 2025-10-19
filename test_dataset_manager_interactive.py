"""
Interactive Dataset Manager Test

This test:
1. Runs initial ETL on simple test data (baseline)
2. Prepares test files for all 4 scenarios
3. Prompts you to test each scenario step-by-step

You can see how the DatasetManager behaves in each case.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.etl.dataset_manager import DatasetManager
from src.database.connection import DatabaseManager
from src.database.repository import CompanyRepository, DatasetRepository


def create_test_directory():
    """Create test directory."""
    test_dir = Path("data/inputs/manager_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def create_initial_sales_data(test_dir):
    """Create initial sales data for 2024 Q1-Q2."""
    print("\n[CREATING] Initial sales data (Jan-Jun 2024)...")

    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='MS')

    data = {
        'date': dates,
        'revenue': [100000, 105000, 110000, 115000, 120000, 125000],
        'costs': [60000, 63000, 66000, 69000, 72000, 75000],
        'units_sold': [1000, 1050, 1100, 1150, 1200, 1250],
        'product_category': ['Electronics', 'Electronics', 'Electronics',
                            'Furniture', 'Furniture', 'Furniture']
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "initial_sales.csv"
    df.to_csv(file_path, index=False)

    print(f"   ✓ Created: {file_path}")
    print(f"   ✓ Data: {len(df)} months (Jan-Jun 2024)")
    print(f"   ✓ Columns: {list(df.columns)}")

    return str(file_path)


def prepare_case_1_new_dataset(test_dir):
    """Prepare CASE 1: Completely new dataset (customer data)."""
    print("\n[PREPARING CASE 1] New dataset - customer_segments.csv")

    data = {
        'customer_id': range(1, 31),
        'customer_name': [f'Customer_{i}' for i in range(1, 31)],
        'segment': ['Enterprise'] * 10 + ['SMB'] * 10 + ['Startup'] * 10,
        'annual_revenue': [1000000, 950000, 900000, 850000, 800000,
                          750000, 700000, 650000, 600000, 550000,
                          100000, 95000, 90000, 85000, 80000,
                          75000, 70000, 65000, 60000, 55000,
                          10000, 9500, 9000, 8500, 8000,
                          7500, 7000, 6500, 6000, 5500],
        'signup_date': [datetime(2023, 1, 1) + timedelta(days=i*10) for i in range(30)],
        'product_category': ['Electronics'] * 15 + ['Furniture'] * 15
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "customer_segments.csv"
    df.to_csv(file_path, index=False)

    print(f"   ✓ File: {file_path}")
    print(f"   ✓ Schema: DIFFERENT from sales (customer data)")
    print(f"   ✓ Expected: Should create NEW dataset")
    print(f"   ✓ Expected: Should detect relationship (product_category column)")

    return str(file_path)


def prepare_case_2_delete(test_dir):
    """Prepare CASE 2: Instructions for deletion."""
    print("\n[PREPARING CASE 2] Dataset deletion")
    print(f"   ✓ You'll delete the customer_segments dataset after creating it")
    print(f"   ✓ Expected: Should cascade delete relationships")
    print(f"   ✓ Expected: Should remove all dependent objects")
    return None


def prepare_case_3_append(test_dir):
    """Prepare CASE 3: Append new months to sales."""
    print("\n[PREPARING CASE 3] Append data - sales_Q3.csv")

    # Q3 2024 (July-Sept) - continues from Q1-Q2
    dates = pd.date_range(start='2024-07-01', end='2024-09-30', freq='MS')

    data = {
        'date': dates,
        'revenue': [130000, 135000, 140000],
        'costs': [78000, 81000, 84000],
        'units_sold': [1300, 1350, 1400],
        'product_category': ['Electronics', 'Furniture', 'Electronics']
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "sales_Q3.csv"
    df.to_csv(file_path, index=False)

    print(f"   ✓ File: {file_path}")
    print(f"   ✓ Schema: SAME as initial_sales")
    print(f"   ✓ Time period: Jul-Sep 2024 (NEW months)")
    print(f"   ✓ Overlap: 0% (different months)")
    print(f"   ✓ Expected: Should APPEND to existing sales dataset")
    print(f"   ✓ Expected: 6 rows → 9 rows total")

    return str(file_path)


def prepare_case_4_duplicate(test_dir):
    """Prepare CASE 4: Duplicate of initial data."""
    print("\n[PREPARING CASE 4] Duplicate data - initial_sales_copy.csv")

    # Exact copy of initial data
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='MS')

    data = {
        'date': dates,
        'revenue': [100000, 105000, 110000, 115000, 120000, 125000],
        'costs': [60000, 63000, 66000, 69000, 72000, 75000],
        'units_sold': [1000, 1050, 1100, 1150, 1200, 1250],
        'product_category': ['Electronics', 'Electronics', 'Electronics',
                            'Furniture', 'Furniture', 'Furniture']
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "initial_sales_copy.csv"
    df.to_csv(file_path, index=False)

    print(f"   ✓ File: {file_path}")
    print(f"   ✓ Schema: SAME as initial_sales")
    print(f"   ✓ Time period: Jan-Jun 2024 (SAME months)")
    print(f"   ✓ Overlap: 100% (exact duplicate)")
    print(f"   ✓ Expected: Should detect DUPLICATE")
    print(f"   ✓ Expected: Should offer options (skip/replace/append)")

    return str(file_path)


def run_initial_etl():
    """Run initial ETL to establish baseline."""
    print("\n" + "="*80)
    print("STEP 1: INITIAL ETL (Baseline)")
    print("="*80)

    test_dir = create_test_directory()

    # Create initial data
    initial_file = create_initial_sales_data(test_dir)

    # Create or get company
    with DatabaseManager.get_session() as session:
        company = CompanyRepository.get_or_create_company(
            session,
            name="DatasetManager Test Co"
        )
        company_id = company.id
        company_name = company.name  # Get the name before session closes
        session.commit()

    print(f"\n[COMPANY] {company_name} (ID: {company_id})")

    # Run initial ETL
    print(f"\n[RUNNING] Initial ETL on {os.path.basename(initial_file)}...")
    manager = DatasetManager()

    result = manager.process_upload(
        company_id=company_id,
        file_path=initial_file
    )

    print(f"\n{'='*80}")
    print(f"INITIAL ETL COMPLETE")
    print(f"{'='*80}")
    print(f"Status: {result.status}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Dataset ID: {result.dataset_id}")
    print(f"Message: {result.message}")

    # Get dataset info
    with DatabaseManager.get_session() as session:
        datasets = DatasetRepository.get_datasets_by_company(session, company_id)
        print(f"\nCompany now has {len(datasets)} dataset(s):")
        for ds in datasets:
            print(f"  - {ds.table_name}: {ds.row_count} rows")

    return company_id, result.dataset_id, test_dir


def print_test_instructions(company_id, initial_dataset_id, test_dir):
    """Print instructions for manual testing."""
    print("\n" + "="*80)
    print("STEP 2: INTERACTIVE TESTING")
    print("="*80)
    print("\nThe baseline has been established. Now you can test each scenario.")
    print(f"\nCompany ID: {company_id}")
    print(f"Initial Dataset ID: {initial_dataset_id}")
    print(f"Test Directory: {test_dir}")

    # Prepare test files
    case1_file = prepare_case_1_new_dataset(test_dir)
    prepare_case_2_delete(test_dir)
    case3_file = prepare_case_3_append(test_dir)
    case4_file = prepare_case_4_duplicate(test_dir)

    print("\n" + "="*80)
    print("TESTING INSTRUCTIONS")
    print("="*80)

    print("\n" + "-"*80)
    print("TEST CASE 1: Add Completely New Dataset")
    print("-"*80)
    print(f"""
from src.etl.dataset_manager import DatasetManager

manager = DatasetManager()
result = manager.process_upload(
    company_id={company_id},
    file_path=r"{case1_file}"
)

print(f"Status: {{result.status}}")
print(f"Message: {{result.message}}")

# Expected Output:
# ✓ Status: created
# ✓ Message: Created new dataset with 30 rows
# ✓ Should detect relationship with sales (product_category)
""")

    print("\n" + "-"*80)
    print("TEST CASE 2: Delete Dataset (with cascade)")
    print("-"*80)
    print(f"""
from src.etl.dataset_manager import DatasetManager
from src.database.repository import DatasetRepository
from src.database.connection import DatabaseManager

# First, get the customer_segments dataset ID
with DatabaseManager.get_session() as session:
    datasets = DatasetRepository.get_datasets_by_company(session, {company_id})
    customer_ds = next(ds for ds in datasets if 'customer' in ds.table_name)
    customer_id = customer_ds.id

# Dry run first
manager = DatasetManager()
result = manager.delete_dataset(
    dataset_id=customer_id,
    cascade=True,
    confirm=False  # Dry run
)

print(f"Would delete: {{result['would_delete']}}")

# Confirm deletion
result = manager.delete_dataset(
    dataset_id=customer_id,
    cascade=True,
    confirm=True
)

print(f"Status: {{result['status']}}")

# Expected Output:
# ✓ Status: deleted
# ✓ Relationships deleted
# ✓ Column metadata deleted
# ✓ Table dropped
""")

    print("\n" + "-"*80)
    print("TEST CASE 3: Append to Existing Dataset")
    print("-"*80)
    print(f"""
from src.etl.dataset_manager import DatasetManager

manager = DatasetManager()
result = manager.process_upload(
    company_id={company_id},
    file_path=r"{case3_file}"
)

print(f"Status: {{result.status}}")
print(f"Message: {{result.message}}")
print(f"Old rows: {{result.metadata.get('old_row_count')}}")
print(f"New rows: {{result.metadata.get('new_row_count')}}")

# Expected Output:
# ✓ Status: appended
# ✓ Old rows: 6
# ✓ New rows: 9 (6 + 3)
# ✓ Overlap: 0%
# ✓ Message: Added 3 rows to existing dataset
# ✓ Should re-run analytics with 9 months of data
""")

    print("\n" + "-"*80)
    print("TEST CASE 4: Duplicate Detection")
    print("-"*80)
    print(f"""
from src.etl.dataset_manager import DatasetManager

manager = DatasetManager()

# First upload - should detect duplicate
result = manager.process_upload(
    company_id={company_id},
    file_path=r"{case4_file}"
)

print(f"Status: {{result.status}}")
print(f"Message: {{result.message}}")
print(f"Options: {{result.options}}")
print(f"Overlap: {{result.metadata.get('overlap_percentage')}}")

# Expected Output:
# ✓ Status: duplicate
# ✓ Overlap: ~100% (after appending Q3, it's now ~66%)
# ✓ Options: ['skip', 'replace', 'append_anyway']
# ✓ Message: This data already exists...

# Test force replace
result = manager.process_upload(
    company_id={company_id},
    file_path=r"{case4_file}",
    force_action="replace"
)

print(f"Status: {{result.status}}")

# Expected Output:
# ✓ Status: replaced
# ✓ Old data replaced with new data
""")

    print("\n" + "="*80)
    print("QUICK TEST - Run All Cases in Sequence")
    print("="*80)
    print(f"""
from src.etl.dataset_manager import DatasetManager
from src.database.repository import DatasetRepository
from src.database.connection import DatabaseManager

manager = DatasetManager()

# CASE 1: New dataset
print("\\n=== CASE 1: New Dataset ===")
r1 = manager.process_upload({company_id}, r"{case1_file}")
print(f"✓ {{r1.status}}: {{r1.message}}")

# CASE 2: Delete (get customer dataset ID first)
print("\\n=== CASE 2: Delete Dataset ===")
with DatabaseManager.get_session() as session:
    datasets = DatasetRepository.get_datasets_by_company(session, {company_id})
    customer_ds = next((ds for ds in datasets if 'customer' in ds.table_name), None)
    if customer_ds:
        r2 = manager.delete_dataset(customer_ds.id, cascade=True, confirm=True)
        print(f"✓ {{r2['status']}}: Deleted {{customer_ds.table_name}}")

# CASE 3: Append
print("\\n=== CASE 3: Append Data ===")
r3 = manager.process_upload({company_id}, r"{case3_file}")
print(f"✓ {{r3.status}}: {{r3.message}}")
print(f"  Rows: {{r3.metadata.get('old_row_count')}} → {{r3.metadata.get('new_row_count')}}")

# CASE 4: Duplicate
print("\\n=== CASE 4: Duplicate Detection ===")
r4 = manager.process_upload({company_id}, r"{case4_file}")
print(f"✓ {{r4.status}}: {{r4.message}}")
if r4.options:
    print(f"  Options: {{r4.options}}")

print("\\n✓ All tests complete!")
""")

    print("\n" + "="*80)
    print("\nCopy and paste any of the code blocks above into a Python shell to test!")
    print("="*80)


def main():
    """Main test flow."""
    print("="*80)
    print("INTERACTIVE DATASET MANAGER TEST")
    print("="*80)
    print("\nThis test will:")
    print("1. Run initial ETL on baseline data")
    print("2. Prepare test files for all 4 scenarios")
    print("3. Provide instructions for you to test manually")
    print("\nReady to start? (Press Enter to continue)")
    input()

    try:
        # Run initial ETL
        company_id, initial_dataset_id, test_dir = run_initial_etl()

        # Print instructions
        print_test_instructions(company_id, initial_dataset_id, test_dir)

        print("\n" + "="*80)
        print("✓ SETUP COMPLETE - Ready for Testing!")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

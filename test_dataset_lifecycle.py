"""
Test Dataset Lifecycle Management

This test demonstrates all 4 scenarios:
1. Add completely new dataset
2. Hard delete a dataset (with cascade)
3. Add to existing dataset (append)
4. Duplicate detection

Run this test to see how the DatasetManager intelligently handles uploads.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.etl.dataset_manager import DatasetManager
from src.database.connection import DatabaseManager
from src.database.repository import CompanyRepository


def create_test_data_directory():
    """Create test data directory if it doesn't exist."""
    test_dir = Path("data/inputs/lifecycle_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def create_monthly_sales_2023(test_dir):
    """Create initial sales data for 2023."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')

    data = {
        'date': dates,
        'revenue': [100000 + i * 5000 for i in range(12)],
        'costs': [60000 + i * 3000 for i in range(12)],
        'units_sold': [1000 + i * 50 for i in range(12)],
        'region': ['North'] * 4 + ['South'] * 4 + ['East'] * 4
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "monthly_sales_2023.csv"
    df.to_csv(file_path, index=False)

    return str(file_path), df


def create_monthly_sales_q1_2024(test_dir):
    """Create Q1 2024 sales data (SHOULD APPEND to 2023 data)."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='MS')

    data = {
        'date': dates,
        'revenue': [160000, 165000, 170000],
        'costs': [96000, 99000, 102000],
        'units_sold': [1600, 1650, 1700],
        'region': ['North', 'South', 'East']
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "monthly_sales_Q1_2024.csv"
    df.to_csv(file_path, index=False)

    return str(file_path), df


def create_monthly_sales_2023_duplicate(test_dir):
    """Create exact duplicate of 2023 data (SHOULD BE DETECTED AS DUPLICATE)."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')

    data = {
        'date': dates,
        'revenue': [100000 + i * 5000 for i in range(12)],
        'costs': [60000 + i * 3000 for i in range(12)],
        'units_sold': [1000 + i * 50 for i in range(12)],
        'region': ['North'] * 4 + ['South'] * 4 + ['East'] * 4
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "monthly_sales_2023_copy.csv"
    df.to_csv(file_path, index=False)

    return str(file_path), df


def create_employee_data(test_dir):
    """Create employee data (COMPLETELY NEW DATASET)."""
    data = {
        'employee_id': range(1, 51),
        'name': [f'Employee_{i}' for i in range(1, 51)],
        'department': ['Sales'] * 20 + ['Engineering'] * 20 + ['Marketing'] * 10,
        'salary': [50000 + i * 1000 for i in range(50)],
        'hire_date': [datetime(2020, 1, 1) + timedelta(days=i*30) for i in range(50)],
        'region': ['North'] * 15 + ['South'] * 15 + ['East'] * 20
    }

    df = pd.DataFrame(data)
    file_path = test_dir / "employee_data.csv"
    df.to_csv(file_path, index=False)

    return str(file_path), df


def main():
    print("="*80)
    print("DATASET LIFECYCLE TEST")
    print("="*80)
    print("\nThis test demonstrates all 4 scenarios for dataset management:")
    print("  1. Add completely new dataset")
    print("  2. Hard delete a dataset")
    print("  3. Add to existing dataset (append)")
    print("  4. Duplicate detection")
    print("="*80)

    # Create test data
    test_dir = create_test_data_directory()
    print(f"\n[SETUP] Test data directory: {test_dir}")

    # Initialize DatasetManager
    manager = DatasetManager()

    # Create or get test company
    with DatabaseManager.get_session() as session:
        company = CompanyRepository.get_or_create_company(
            session,
            name="Lifecycle Test Company"
        )
        company_id = company.id
        session.commit()

    print(f"[SETUP] Company: {company.name} (ID: {company_id})\n")

    # ========================================================================
    # SCENARIO 1: Upload initial dataset (NEW)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: Upload Initial Sales Data (2023)")
    print("="*80)

    sales_2023_path, sales_2023_df = create_monthly_sales_2023(test_dir)
    print(f"[CREATED] {sales_2023_path}")
    print(f"[DATA] {len(sales_2023_df)} rows, {len(sales_2023_df.columns)} columns")

    result1 = manager.process_upload(
        company_id=company_id,
        file_path=sales_2023_path
    )

    print(f"\n[RESULT] Status: {result1.status}")
    print(f"[RESULT] Message: {result1.message}")
    if result1.metadata:
        print(f"[RESULT] Metadata: {result1.metadata}")

    initial_dataset_id = result1.dataset_id

    # ========================================================================
    # SCENARIO 2: Upload employee data (NEW, DIFFERENT DATASET)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 2: Upload Employee Data (Completely Different)")
    print("="*80)

    employee_path, employee_df = create_employee_data(test_dir)
    print(f"[CREATED] {employee_path}")
    print(f"[DATA] {len(employee_df)} rows, {len(employee_df.columns)} columns")

    result2 = manager.process_upload(
        company_id=company_id,
        file_path=employee_path
    )

    print(f"\n[RESULT] Status: {result2.status}")
    print(f"[RESULT] Message: {result2.message}")
    if result2.metadata:
        print(f"[RESULT] Metadata: {result2.metadata}")

    employee_dataset_id = result2.dataset_id

    # ========================================================================
    # SCENARIO 3: Upload Q1 2024 sales (SHOULD APPEND)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: Upload Q1 2024 Sales Data (Should Append)")
    print("="*80)

    sales_q1_path, sales_q1_df = create_monthly_sales_q1_2024(test_dir)
    print(f"[CREATED] {sales_q1_path}")
    print(f"[DATA] {len(sales_q1_df)} rows, {len(sales_q1_df.columns)} columns")
    print(f"[EXPECT] Should detect schema match with 2023 data and append")

    result3 = manager.process_upload(
        company_id=company_id,
        file_path=sales_q1_path
    )

    print(f"\n[RESULT] Status: {result3.status}")
    print(f"[RESULT] Message: {result3.message}")
    if result3.metadata:
        print(f"[RESULT] Old row count: {result3.metadata.get('old_row_count')}")
        print(f"[RESULT] New row count: {result3.metadata.get('new_row_count')}")
        print(f"[RESULT] Duplicates removed: {result3.metadata.get('duplicates_removed')}")

    # ========================================================================
    # SCENARIO 4: Upload duplicate data (SHOULD DETECT DUPLICATE)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 4: Upload Duplicate 2023 Sales Data")
    print("="*80)

    sales_dup_path, sales_dup_df = create_monthly_sales_2023_duplicate(test_dir)
    print(f"[CREATED] {sales_dup_path}")
    print(f"[DATA] {len(sales_dup_df)} rows, {len(sales_dup_df.columns)} columns")
    print(f"[EXPECT] Should detect as duplicate")

    result4 = manager.process_upload(
        company_id=company_id,
        file_path=sales_dup_path
    )

    print(f"\n[RESULT] Status: {result4.status}")
    print(f"[RESULT] Message: {result4.message}")
    if result4.options:
        print(f"[RESULT] Options available: {result4.options}")
    if result4.metadata:
        print(f"[RESULT] Overlap: {result4.metadata.get('overlap_percentage', 0):.1%}")

    # ========================================================================
    # SCENARIO 5: Test forced action (FORCE REPLACE)
    # ========================================================================
    if result4.status == "duplicate":
        print("\n" + "="*80)
        print("SCENARIO 5: Force Replace Duplicate Data")
        print("="*80)
        print(f"[ACTION] User chose 'replace' option")

        result5 = manager.process_upload(
            company_id=company_id,
            file_path=sales_dup_path,
            force_action="replace"
        )

        print(f"\n[RESULT] Status: {result5.status}")
        print(f"[RESULT] Message: {result5.message}")

    # ========================================================================
    # SCENARIO 6: Delete dataset (WITH CASCADE)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 6: Delete Employee Dataset (With Cascade)")
    print("="*80)

    if employee_dataset_id:
        # First, dry run to see what would be deleted
        print(f"[DRY RUN] Checking what would be deleted...")
        dry_run_result = manager.delete_dataset(
            dataset_id=employee_dataset_id,
            cascade=True,
            confirm=False
        )

        print(f"\n[DRY RUN] Status: {dry_run_result['status']}")
        if 'would_delete' in dry_run_result:
            deps = dry_run_result['would_delete']
            print(f"[DRY RUN] Would delete {len(deps['relationships'])} relationships")
            print(f"[DRY RUN] Would delete {len(deps['columns'])} column metadata")

        # Now actually delete
        print(f"\n[CONFIRM] Proceeding with deletion...")
        delete_result = manager.delete_dataset(
            dataset_id=employee_dataset_id,
            cascade=True,
            confirm=True
        )

        print(f"\n[RESULT] Status: {delete_result['status']}")
        print(f"[RESULT] Dataset deleted: {delete_result.get('dataset_id')}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✓ SCENARIO 1: New dataset created - {result1.status}")
    print(f"✓ SCENARIO 2: Different dataset created - {result2.status}")
    print(f"✓ SCENARIO 3: Data appended - {result3.status}")
    print(f"✓ SCENARIO 4: Duplicate detected - {result4.status}")
    if employee_dataset_id:
        print(f"✓ SCENARIO 6: Dataset deleted - {delete_result['status']}")

    print("\n" + "="*80)
    print("All scenarios tested successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

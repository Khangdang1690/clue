"""Test the discovery system with synthetic data."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_test_dataset(n_rows=1000, filename="test_discovery_data.csv"):
    """
    Generate a synthetic dataset with known patterns for testing.

    Args:
        n_rows: Number of rows to generate
        filename: Output filename
    """
    np.random.seed(42)

    print(f"\nGenerating test dataset with {n_rows:,} rows...")

    # Create synthetic data with known relationships
    data = {}

    # 1. Numeric columns with relationships
    data['customer_id'] = range(1, n_rows + 1)
    data['age'] = np.random.randint(18, 80, n_rows)

    # Income correlates with age (known relationship)
    data['income'] = data['age'] * 1000 + np.random.normal(0, 10000, n_rows)

    # Purchase amount correlates with income (known relationship)
    data['purchase_amount'] = data['income'] * 0.05 + np.random.normal(0, 500, n_rows)

    # Time spent (weakly correlated with purchase)
    data['time_spent_mins'] = np.abs(data['purchase_amount'] * 0.01 + np.random.normal(10, 5, n_rows))

    # 2. Categorical columns
    categories = ['Premium', 'Standard', 'Basic']
    data['customer_tier'] = np.random.choice(categories, n_rows, p=[0.2, 0.5, 0.3])

    # Product category (independent)
    products = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    data['product_category'] = np.random.choice(products, n_rows)

    # Region (independent)
    regions = ['North', 'South', 'East', 'West']
    data['region'] = np.random.choice(regions, n_rows)

    # 3. Binary columns
    # Churned (depends on purchase amount - lower purchase = higher churn)
    churn_prob = 1 / (1 + np.exp((data['purchase_amount'] - 2000) / 500))
    data['churned'] = (np.random.random(n_rows) < churn_prob).astype(int)

    # 4. Temporal column
    data['signup_date'] = pd.date_range(start='2023-01-01', periods=n_rows, freq='h')  # lowercase 'h'

    # 5. Some columns with missing values (must use float for NaN)
    last_contact_days = np.random.randint(1, 365, n_rows).astype(float)  # Convert to float
    missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    last_contact_days[missing_idx] = np.nan
    data['last_contact_days'] = last_contact_days

    # 6. Add some outliers to purchase_amount
    outlier_idx = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    for idx in outlier_idx:
        data['purchase_amount'][idx] *= 5

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = Path("data/test") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Test dataset saved to: {output_path}")
    print(f"\nDataset characteristics:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"  Datetime columns: {len(df.select_dtypes(include=['datetime']).columns)}")

    print(f"\nKnown patterns in the data:")
    print(f"  1. Age → Income (strong positive correlation)")
    print(f"  2. Income → Purchase Amount (strong positive correlation)")
    print(f"  3. Purchase Amount → Time Spent (weak correlation)")
    print(f"  4. Purchase Amount → Churn (inverse relationship - low purchase = high churn)")
    print(f"  5. ~5% outliers in purchase_amount")
    print(f"  6. ~10% missing values in last_contact_days")

    return str(output_path)


def generate_large_dataset(n_rows=500000, filename="large_test_data.csv"):
    """
    Generate a large dataset for performance testing.

    Args:
        n_rows: Number of rows (default: 500k)
        filename: Output filename
    """
    print(f"\nGenerating LARGE test dataset with {n_rows:,} rows...")
    print("This may take a few moments...")

    return generate_test_dataset(n_rows, filename)


if __name__ == "__main__":
    print("=" * 60)
    print("DISCOVERY SYSTEM TEST DATA GENERATOR")
    print("=" * 60)

    print("\nChoose dataset size:")
    print("1. Small (1,000 rows) - Quick test")
    print("2. Medium (50,000 rows) - Moderate test")
    print("3. Large (500,000 rows) - Full test")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        path = generate_test_dataset(n_rows=1000, filename="small_test.csv")
    elif choice == "2":
        path = generate_test_dataset(n_rows=50000, filename="medium_test.csv")
    elif choice == "3":
        path = generate_large_dataset(n_rows=500000, filename="large_test.csv")
    else:
        print("Invalid choice")
        exit(1)

    print(f"\n" + "=" * 60)
    print("✅ Test data generated successfully!")
    print("=" * 60)
    print(f"\nTo run discovery on this dataset:")
    print(f"  python main.py --discovery")
    print(f"\nThen enter the path: {path}")
    print("=" * 60)

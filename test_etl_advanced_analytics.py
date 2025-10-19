"""
Enhanced ETL Test with Time Series Data for Advanced Analytics

This test creates realistic business data that showcases:
- Forecasting (monthly revenue trends)
- Anomaly Detection (unusual spikes)
- Causal Analysis (marketing -> sales)
- Variance Decomposition (what drives customer value)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.graph.etl_workflow import ETLWorkflow
from src.graph.multi_table_discovery import MultiTableDiscovery
import os


def create_enhanced_test_data():
    """Create realistic time series business data."""

    # Create output directory
    os.makedirs("data/inputs/test", exist_ok=True)

    # ================================================================
    # 1. MONTHLY SALES DATA (36 months = 3 years)
    # ================================================================
    print("Creating monthly sales data...")

    np.random.seed(42)

    # Generate 36 months of data
    dates = pd.date_range(start='2021-01-01', periods=36, freq='M')

    # Base revenue with growth trend
    base_revenue = 100000
    growth_rate = 0.02  # 2% monthly growth
    revenue = [base_revenue * (1 + growth_rate) ** i for i in range(36)]

    # Add seasonality (higher in Q4)
    seasonality = [1.0 if i % 12 < 9 else 1.3 for i in range(36)]
    revenue = [r * s for r, s in zip(revenue, seasonality)]

    # Add random noise
    revenue = [r + np.random.normal(0, 5000) for r in revenue]

    # Inject 2 anomalies (unusual spikes)
    revenue[15] = revenue[15] * 1.8  # Big spike in month 15
    revenue[28] = revenue[28] * 0.6  # Unusual drop in month 28

    # Calculate costs (70-80% of revenue)
    costs = [r * np.random.uniform(0.70, 0.80) for r in revenue]

    # Calculate marketing spend (varies)
    marketing_spend = [np.random.uniform(10000, 30000) for _ in range(36)]

    # Sales increases 2 months after marketing (causal relationship)
    for i in range(2, 36):
        revenue[i] += marketing_spend[i-2] * 0.5  # Marketing drives revenue

    sales_df = pd.DataFrame({
        'date': dates,
        'revenue': [round(r) for r in revenue],
        'costs': [round(c) for c in costs],
        'marketing_spend': [round(m) for m in marketing_spend],
        'units_sold': [int(r / 100) for r in revenue],
        'profit_margin': [(r - c) / r * 100 for r, c in zip(revenue, costs)]
    })

    # Set date as index for time series
    sales_df = sales_df.set_index('date')

    sales_df.to_csv("data/inputs/test/monthly_sales.csv")
    print(f"  Created monthly_sales.csv: {len(sales_df)} months of data")
    print(f"  Date range: {sales_df.index[0]} to {sales_df.index[-1]}")

    # ================================================================
    # 2. CUSTOMER DATA (50 customers)
    # ================================================================
    print("\nCreating customer data...")

    customer_ids = [f"CUST_{i:04d}" for i in range(1, 51)]

    # Segment customers
    segments = np.random.choice(['Enterprise', 'SMB', 'Startup'], size=50, p=[0.2, 0.5, 0.3])

    # Engagement score (1-10)
    engagement_scores = np.random.randint(1, 11, size=50)

    # Support tickets (0-20)
    support_tickets = np.random.poisson(lam=5, size=50)

    # Price tier
    price_tiers = np.random.choice(['Basic', 'Pro', 'Enterprise'], size=50, p=[0.4, 0.4, 0.2])

    # Customer lifetime value (influenced by engagement, support, tier)
    clv = []
    churn = []

    for i in range(50):
        base_clv = 10000

        # Engagement drives CLV (positive)
        base_clv *= (engagement_scores[i] / 5)

        # Support tickets reduce CLV (negative)
        base_clv *= (1 - support_tickets[i] / 50)

        # Price tier affects CLV
        tier_multiplier = {'Basic': 0.5, 'Pro': 1.0, 'Enterprise': 2.0}
        base_clv *= tier_multiplier[price_tiers[i]]

        clv.append(round(base_clv))

        # Churn is higher for low engagement + high support tickets
        churn_prob = 0.1 + (10 - engagement_scores[i]) * 0.05 + support_tickets[i] * 0.02
        churn.append(1 if np.random.random() < churn_prob else 0)

    customer_df = pd.DataFrame({
        'customer_id': customer_ids,
        'segment': segments,
        'engagement_score': engagement_scores,
        'support_tickets': support_tickets,
        'price_tier': price_tiers,
        'lifetime_value': clv,
        'churned': churn
    })

    customer_df.to_csv("data/inputs/test/customer_data.csv", index=False)
    print(f"  Created customer_data.csv: {len(customer_df)} customers")
    print(f"  Segments: {customer_df['segment'].value_counts().to_dict()}")

    # ================================================================
    # 3. PRODUCT PERFORMANCE (10 products)
    # ================================================================
    print("\nCreating product data...")

    product_ids = [f"PROD_{i:03d}" for i in range(1, 11)]
    categories = np.random.choice(['Software', 'Hardware', 'Services'], size=10)

    product_df = pd.DataFrame({
        'product_id': product_ids,
        'category': categories,
        'units_sold': np.random.randint(100, 1000, size=10),
        'revenue': np.random.randint(50000, 200000, size=10),
        'cost': np.random.randint(30000, 120000, size=10),
        'customer_rating': np.random.uniform(3.0, 5.0, size=10).round(1)
    })

    product_df['profit'] = product_df['revenue'] - product_df['cost']
    product_df['profit_margin'] = (product_df['profit'] / product_df['revenue'] * 100).round(2)

    product_df.to_csv("data/inputs/test/product_performance.csv", index=False)
    print(f"  Created product_performance.csv: {len(product_df)} products")

    print("\n" + "="*80)
    print("TEST DATA CREATED SUCCESSFULLY")
    print("="*80)
    print("\nData Features:")
    print("  - 36 months of time series data (perfect for forecasting)")
    print("  - 2 injected anomalies (to test anomaly detection)")
    print("  - Marketing spend -> Revenue causal relationship (2-month lag)")
    print("  - Customer churn driven by engagement + support tickets")
    print("  - Enough data for HIGH confidence analytics (30+ observations)")
    print("="*80)

    return [
        "data/inputs/test/monthly_sales.csv",
        "data/inputs/test/customer_data.csv",
        "data/inputs/test/product_performance.csv"
    ]


def main():
    print("\n" + "="*80)
    print("ENHANCED ETL TEST WITH ADVANCED ANALYTICS")
    print("="*80)
    print("\nThis test demonstrates:")
    print("  1. Time series forecasting (revenue prediction)")
    print("  2. Anomaly detection (unusual spikes/drops)")
    print("  3. Causal analysis (marketing -> sales)")
    print("  4. Variance decomposition (what drives churn)")
    print("\n" + "="*80)

    # Step 1: Create enhanced test data
    file_paths = create_enhanced_test_data()

    # Step 2: Run ETL
    print("\n[STEP 1] RUNNING ETL PIPELINE")
    print("="*80)

    workflow = ETLWorkflow(company_name="Advanced Analytics Demo")
    etl_result = workflow.run_etl(file_paths)

    if etl_result['status'] != 'completed':
        print("[ERROR] ETL failed")
        return

    dataset_ids = list(etl_result['dataset_ids'].values())
    company_id = "1"  # The company ID will be 1 for the first company

    print(f"\n[SUCCESS] ETL completed - {len(dataset_ids)} datasets stored")

    # Step 3: Run Discovery with Advanced Analytics
    print("\n[STEP 2] RUNNING DISCOVERY WITH ADVANCED ANALYTICS")
    print("="*80)

    discovery = MultiTableDiscovery()

    try:
        result = discovery.run_discovery(
            company_id=company_id,
            dataset_ids=dataset_ids,
            analysis_name="Advanced Analytics Demo"
        )

        print("\n" + "="*80)
        print("SUCCESS! REPORT GENERATED")
        print("="*80)
        print(f"\nReport location: {result['unified_report_path']}")
        print("\nThe report includes:")
        print("  ✓ Traditional discovery insights")
        print("  ✓ Revenue forecasts (next 6 months)")
        print("  ✓ Anomaly alerts (unusual spikes detected)")
        print("  ✓ Causal relationships (marketing -> sales)")
        print("  ✓ Churn drivers (engagement, support, pricing)")
        print("\nAll insights in plain English - NO statistics!")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Discovery failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

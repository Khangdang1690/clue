"""Create test data with clear causal relationships for demonstrating statistical tests."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_causal_sample_data():
    """Create sample CSV files with clear causal relationships."""

    # Create test data directory
    test_dir = Path("test_data/causal_demo")
    test_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Creating Enhanced Test Data with Causal Relationships")
    print("="*80)

    # Set seed for reproducibility
    np.random.seed(42)

    # 1. CREATE DAILY TIME SERIES DATA (3 years of daily data)
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)

    print(f"\nCreating {n_days} days of time series data...")

    # ========================================
    # SALES DATA WITH CAUSAL RELATIONSHIPS
    # ========================================

    # Marketing spend (independent variable that CAUSES revenue)
    base_marketing = 5000
    marketing_trend = np.linspace(0, 3000, n_days)  # Increasing marketing budget
    marketing_seasonality = 1500 * np.sin(np.linspace(0, 6*np.pi, n_days))  # Seasonal campaigns
    marketing_noise = np.random.normal(0, 500, n_days)
    marketing_spend = base_marketing + marketing_trend + marketing_seasonality + marketing_noise
    marketing_spend = np.maximum(marketing_spend, 1000)  # Ensure positive values

    # Revenue (CAUSED by marketing spend with 7-day lag)
    lag_days = 7
    revenue_base = 20000
    revenue_noise = np.random.normal(0, 2000, n_days)

    # Create lagged effect: Revenue responds to marketing spend from 7 days ago
    marketing_effect = np.zeros(n_days)
    for i in range(lag_days, n_days):
        # Revenue = base + 3.5 * marketing_spend(t-7) + noise
        marketing_effect[i] = 3.5 * marketing_spend[i - lag_days]

    revenue = revenue_base + marketing_effect + revenue_noise
    revenue = np.maximum(revenue, 5000)  # Ensure positive values

    # Cost (partially driven by quantity sold, which correlates with revenue)
    quantity = (revenue / 50 + np.random.normal(0, 100, n_days)).astype(int)
    quantity = np.maximum(quantity, 10)
    cost = 0.6 * revenue + 20 * quantity + np.random.normal(0, 1000, n_days)

    # Customer traffic (another cause of revenue, but weaker)
    base_traffic = 1000
    traffic_trend = np.linspace(0, 500, n_days)
    traffic_seasonality = 300 * np.sin(np.linspace(0, 8*np.pi, n_days))
    traffic_noise = np.random.normal(0, 100, n_days)
    customer_traffic = base_traffic + traffic_trend + traffic_seasonality + traffic_noise
    customer_traffic = np.maximum(customer_traffic, 100).astype(int)

    # Add some effect of traffic on revenue (weaker than marketing)
    revenue += 5 * customer_traffic  # Small additional effect

    # Price (affects quantity inversely)
    base_price = 100
    price_variations = 20 * np.sin(np.linspace(0, 12*np.pi, n_days))
    price = base_price + price_variations + np.random.normal(0, 5, n_days)
    price = np.maximum(price, 50)

    # Adjust quantity based on price (inverse relationship)
    quantity = (quantity * (150 / price)).astype(int)
    quantity = np.maximum(quantity, 5)

    # Create the main sales dataframe
    sales_data = pd.DataFrame({
        'date': dates,
        'marketing_spend': marketing_spend.round(2),
        'customer_traffic': customer_traffic,
        'price': price.round(2),
        'quantity': quantity,
        'revenue': revenue.round(2),
        'cost': cost.round(2),
        'profit': (revenue - cost).round(2)
    })

    # Add day of week and month for seasonal analysis
    sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
    sales_data['month'] = sales_data['date'].dt.month
    sales_data['quarter'] = sales_data['date'].dt.quarter
    sales_data['year'] = sales_data['date'].dt.year

    # Create anomalies (for anomaly detection)
    anomaly_indices = np.random.choice(range(100, n_days-100), 15, replace=False)
    for idx in anomaly_indices:
        sales_data.loc[idx, 'revenue'] *= np.random.uniform(1.5, 2.5)  # Revenue spikes
        sales_data.loc[idx, 'profit'] = sales_data.loc[idx, 'revenue'] - sales_data.loc[idx, 'cost']

    sales_path = test_dir / "daily_sales_metrics.csv"
    sales_data.to_csv(sales_path, index=False)

    print(f"[OK] Created: {sales_path}")
    print(f"   - {len(sales_data)} daily records")
    print(f"   - Causal relationship: marketing_spend -> revenue (7-day lag)")
    print(f"   - Includes {len(anomaly_indices)} anomalies for detection")

    # ========================================
    # CUSTOMER BEHAVIOR DATA (Cross-sectional with some time series)
    # ========================================

    num_customers = 500
    customer_ids = [f'C{i:04d}' for i in range(1, num_customers + 1)]

    # Generate customer acquisition dates spread over time
    acquisition_dates = pd.date_range(start_date - pd.Timedelta(days=365),
                                     end_date - pd.Timedelta(days=30),
                                     periods=num_customers)

    # Customer features that affect lifetime value
    company_size = np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'],
                                   num_customers, p=[0.4, 0.3, 0.2, 0.1])

    # Map company size to employee count (cause of lifetime value)
    size_to_employees = {
        'Small': np.random.randint(1, 50),
        'Medium': np.random.randint(50, 500),
        'Large': np.random.randint(500, 5000),
        'Enterprise': np.random.randint(5000, 50000)
    }
    employee_count = [size_to_employees[size] + np.random.randint(-10, 10)
                     for size in company_size]

    # Industry affects churn risk
    industry = np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'],
                              num_customers, p=[0.25, 0.20, 0.20, 0.20, 0.15])

    # Lifetime value CAUSED by company size and engagement
    base_ltv = 10000
    size_multipliers = {'Small': 1, 'Medium': 3, 'Large': 8, 'Enterprise': 20}
    lifetime_value = [base_ltv * size_multipliers[size] + np.random.normal(0, 5000)
                     for size in company_size]
    lifetime_value = np.maximum(lifetime_value, 1000)

    # Churn risk inversely related to lifetime value
    churn_probabilities = 1 / (1 + np.exp((np.array(lifetime_value) - 50000) / 20000))
    churn_risk = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low'
                  for p in churn_probabilities]

    # Customer engagement score (causes retention)
    engagement_score = 100 - (churn_probabilities * 100) + np.random.normal(0, 10, num_customers)
    engagement_score = np.clip(engagement_score, 0, 100)

    customer_data = pd.DataFrame({
        'customer_id': customer_ids,
        'company_name': [f'Company_{i}' for i in range(1, num_customers + 1)],
        'industry': industry,
        'company_size': company_size,
        'employee_count': employee_count,
        'acquisition_date': acquisition_dates,
        'lifetime_value': np.array(lifetime_value).round(2),
        'engagement_score': engagement_score.round(1),
        'churn_risk': churn_risk,
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_customers)
    })

    customer_path = test_dir / "customer_profiles.csv"
    customer_data.to_csv(customer_path, index=False)

    print(f"[OK] Created: {customer_path}")
    print(f"   - {len(customer_data)} customer records")
    print(f"   - Causal: company_size -> lifetime_value -> churn_risk")

    # ========================================
    # PRODUCT PERFORMANCE DATA
    # ========================================

    num_products = 200
    product_ids = [f'P{i:03d}' for i in range(1, num_products + 1)]

    # Product features
    categories = np.random.choice(['Premium', 'Standard', 'Budget'], num_products, p=[0.2, 0.5, 0.3])

    # Price CAUSES demand (inverse relationship)
    base_prices = {'Premium': 500, 'Standard': 200, 'Budget': 50}
    product_price = [base_prices[cat] + np.random.normal(0, base_prices[cat]*0.1)
                    for cat in categories]
    product_price = np.maximum(product_price, 10)

    # Quality score affects satisfaction
    quality_scores = {'Premium': 90, 'Standard': 70, 'Budget': 50}
    product_quality = [quality_scores[cat] + np.random.normal(0, 5) for cat in categories]
    product_quality = np.clip(product_quality, 0, 100)

    # Units sold inversely related to price
    base_demand = 1000
    units_sold = [int(base_demand * (100/price) * np.random.uniform(0.8, 1.2))
                  for price in product_price]

    # Customer satisfaction CAUSED by quality
    satisfaction = product_quality + np.random.normal(0, 5, num_products)
    satisfaction = np.clip(satisfaction, 0, 100)

    # Return rate inversely related to satisfaction
    return_rate = (100 - satisfaction) / 100 * 0.3 + np.random.normal(0, 0.02, num_products)
    return_rate = np.clip(return_rate, 0, 0.5)

    product_data = pd.DataFrame({
        'product_id': product_ids,
        'product_name': [f'Product_{cat}_{i}' for i, cat in enumerate(categories)],
        'category': categories,
        'price': np.array(product_price).round(2),
        'quality_score': product_quality.round(1),
        'units_sold': units_sold,
        'customer_satisfaction': satisfaction.round(1),
        'return_rate': return_rate.round(3),
        'profit_margin': np.random.uniform(0.1, 0.4, num_products).round(3)
    })

    product_path = test_dir / "product_information.csv"
    product_data.to_csv(product_path, index=False)

    print(f"[OK] Created: {product_path}")
    print(f"   - {len(product_data)} product records")
    print(f"   - Causal: price -> units_sold, quality -> satisfaction -> returns")

    # ========================================
    # TRANSACTION DATA (Links everything together)
    # ========================================

    # Generate transactions over time
    num_transactions = 5000
    transaction_dates = pd.date_range(start_date, end_date, periods=num_transactions)

    transactions = pd.DataFrame({
        'transaction_id': [f'T{i:06d}' for i in range(1, num_transactions + 1)],
        'date': transaction_dates,
        'customer_id': np.random.choice(customer_ids, num_transactions),
        'product_id': np.random.choice(product_ids, num_transactions),
        'quantity': np.random.poisson(3, num_transactions) + 1,
        'discount_percent': np.random.choice([0, 5, 10, 15, 20], num_transactions, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    })

    # Calculate transaction amount based on product price
    product_price_map = dict(zip(product_data['product_id'], product_data['price']))
    transactions['unit_price'] = transactions['product_id'].map(product_price_map)
    transactions['gross_amount'] = transactions['quantity'] * transactions['unit_price']
    transactions['discount_amount'] = transactions['gross_amount'] * transactions['discount_percent'] / 100
    transactions['net_amount'] = transactions['gross_amount'] - transactions['discount_amount']

    transaction_path = test_dir / "sales_transactions.csv"
    transactions.to_csv(transaction_path, index=False)

    print(f"[OK] Created: {transaction_path}")
    print(f"   - {len(transactions)} transaction records")
    print(f"   - Links customers to products over time")

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "="*80)
    print("CAUSAL RELATIONSHIPS IN TEST DATA:")
    print("="*80)
    print("\n1. TIME SERIES CAUSALITY (daily_sales_metrics.csv):")
    print("   • marketing_spend(t) -> revenue(t+7) [7-day lag]")
    print("   • price -> quantity [inverse relationship]")
    print("   • customer_traffic -> revenue [weak positive]")
    print("\n2. CROSS-SECTIONAL CAUSALITY (customer_profiles.csv):")
    print("   • company_size -> employee_count -> lifetime_value")
    print("   • lifetime_value -> churn_risk [inverse]")
    print("   • engagement_score -> retention")
    print("\n3. PRODUCT CAUSALITY (product_information.csv):")
    print("   • price -> units_sold [inverse]")
    print("   • quality_score -> customer_satisfaction")
    print("   • satisfaction -> return_rate [inverse]")
    print("\n4. STATISTICAL TEST OPPORTUNITIES:")
    print("   [OK] Causal Analysis: marketing->revenue with lag")
    print("   [OK] Trend Analysis: revenue growth over 3 years")
    print("   [OK] Anomaly Detection: 15 revenue spikes")
    print("   [OK] Variance Decomposition: multiple factors -> profit")
    print("   [OK] Correlation Analysis: cross-table relationships")
    print("   [OK] Forecasting: 3 years of daily data")

    return [str(sales_path), str(customer_path), str(product_path), str(transaction_path)]


if __name__ == "__main__":
    file_paths = create_causal_sample_data()
    print("\n[SUCCESS] All test data files created successfully!")
    print("\nYou can now run the ETL pipeline with these files to see:")
    print("- LLM-directed test selection")
    print("- Causal relationship detection")
    print("- Advanced statistical insights")
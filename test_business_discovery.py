"""Test the new business-focused discovery workflow."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from src.database.connection import DatabaseManager
from src.graph.etl_workflow import ETLWorkflow
from src.graph.business_discovery_workflow import BusinessDiscoveryWorkflow
from create_causal_test_data import create_causal_sample_data


def test_business_discovery():
    """Test the business discovery workflow."""

    print("\n" + "="*80)
    print("BUSINESS DISCOVERY TEST")
    print("="*80)

    # 1. Create test data
    print("\n[STEP 1] Creating test data...")
    file_paths = create_causal_sample_data()

    # 2. Initialize database
    print("\n[STEP 2] Initializing database...")
    DatabaseManager.initialize()

    # 3. Run ETL
    print("\n[STEP 3] Running ETL...")
    company_name = "Test Company Inc"
    etl = ETLWorkflow(company_name=company_name)
    etl_state = etl.run_etl(file_paths=file_paths)

    if etl_state['status'] != 'completed':
        print(f"[ERROR] ETL failed: {etl_state.get('error_message', 'Unknown error')}")
        return

    dataset_ids = list(etl_state['dataset_ids'].values())
    company_id = etl_state['company_id']
    print(f"[OK] ETL completed - {len(dataset_ids)} datasets")

    # 4. Run Business Discovery
    print("\n[STEP 4] Running Business Discovery...")
    print("-" * 40)

    discovery = BusinessDiscoveryWorkflow()
    discovery_state = discovery.run_discovery(
        company_id=company_id,
        dataset_ids=dataset_ids,
        analysis_name="Dynamic Business Analysis"
    )

    # 5. Display results
    print("\n" + "="*80)
    print("BUSINESS DISCOVERY RESULTS")
    print("="*80)

    if discovery_state['status'] == 'completed':
        print(f"\n[SUCCESS] Analysis completed successfully")
        print(f"Report: {discovery_state['report_path']}")

        # Show insights
        print(f"\nBUSINESS INSIGHTS ({len(discovery_state['insights'])}):")
        for i, insight in enumerate(discovery_state['insights'][:5], 1):
            print(f"\n{i}. {insight.get('title', 'Insight')}")
            print(f"   Finding: {insight.get('finding', 'N/A')}")

        # Show recommendations
        print(f"\nRECOMMENDATIONS ({len(discovery_state['recommendations'])}):")
        for i, rec in enumerate(discovery_state['recommendations'][:5], 1):
            print(f"\n{i}. {rec['action']}")
            print(f"   Impact: {rec.get('impact', 'Unknown')}")
            print(f"   Urgency: {rec.get('urgency', 'Unknown')}")

        # Show executive summary
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        print(discovery_state.get('executive_summary', 'No summary generated'))

    else:
        print(f"[FAILED] Analysis failed: {discovery_state.get('error', 'Unknown error')}")


def test_dynamic_exploration_only():
    """Test just the dynamic exploration component."""

    print("\n" + "="*80)
    print("DYNAMIC EXPLORATION TEST")
    print("="*80)

    from src.analytics.dynamic_explorer import DynamicDataExplorer

    # Assuming data is already in database from previous run
    DatabaseManager.initialize()

    # Get the company ID
    with DatabaseManager.get_session() as session:
        from src.database.repository import CompanyRepository

        company = CompanyRepository.get_or_create_company(session, "Test Company Inc")
        company_id = str(company.id)

    # Create explorer
    explorer = DynamicDataExplorer()

    # Load datasets
    print("\nLoading datasets...")
    datasets = explorer.load_datasets(company_id)

    for name, df in datasets.items():
        print(f"  • {name}: {df.shape}")

    # Test manual code execution
    print("\n" + "-"*40)
    print("Testing manual code execution:")
    print("-"*40)

    test_code = """
# Analyze revenue patterns
if 'sales_transactions' in locals():
    # Revenue by customer
    customer_revenue = sales_transactions.groupby('customer_id')['net_amount'].sum()
    top_10 = customer_revenue.nlargest(10)

    print(f"Top 10 Customers by Revenue:")
    print(f"Total from top 10: ${top_10.sum():,.2f}")
    print(f"Average per customer: ${top_10.mean():,.2f}")
    print(f"Revenue concentration: {(top_10.sum() / customer_revenue.sum() * 100):.1f}%")

    # Show the actual top customers
    print("\\nTop 5 customers:")
    for customer_id, revenue in top_10.head().items():
        print(f"  {customer_id}: ${revenue:,.2f}")

if 'sales_daily_metrics' in locals():
    # Marketing ROI
    print("\\nMarketing Analysis:")
    total_spend = sales_daily_metrics['marketing_spend'].sum()
    total_revenue = sales_daily_metrics['revenue'].sum()
    print(f"Total marketing spend: ${total_spend:,.2f}")
    print(f"Total revenue: ${total_revenue:,.2f}")
    print(f"Revenue per marketing dollar: ${total_revenue/total_spend:.2f}")
"""

    result = explorer.execute_code(test_code)

    if result['error']:
        print(f"ERROR: {result['error']}")
    else:
        print("OUTPUT:")
        print(result['output'])
        if result['result']:
            print("\nRESULT:")
            print(result['result'])

    # Test business question generation
    print("\n" + "-"*40)
    print("Generated Business Questions:")
    print("-"*40)

    questions = explorer.generate_business_questions()
    for i, q in enumerate(questions[:5], 1):
        print(f"{i}. {q}")

    # Test LLM exploration (if API is available)
    if os.getenv('GEMINI_API_KEY'):
        print("\n" + "-"*40)
        print("Testing LLM Exploration:")
        print("-"*40)

        results = explorer.explore_with_llm(
            objective="Find the top 3 business risks and opportunities",
            max_iterations=3
        )

        print(f"\nIterations: {results['iterations']}")
        print(f"Executions: {len(results['execution_history'])}")
        print(f"Insights found: {len(results['insights'])}")

        for insight in results['insights'][:3]:
            print(f"  • {insight}")
    else:
        print("\n[SKIP] LLM exploration - no API key configured")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--explore-only':
        test_dynamic_exploration_only()
    else:
        test_business_discovery()
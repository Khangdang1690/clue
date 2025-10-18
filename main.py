"""Main entry point for Autonomous Discovery Workflow."""

import sys
import pandas as pd
from pathlib import Path
from src.graph.discovery_workflow import DiscoveryWorkflow
# from src.discovery.discovery_reporter import DiscoveryReporter  # Disabled for testing


def run_discovery(file_path: str, dataset_name: str = None):
    """
    Run autonomous discovery on a CSV or Excel file.

    Args:
        file_path: Path to CSV or Excel file (.csv, .xlsx, .xls)
        dataset_name: Name for the dataset (optional, defaults to filename)
    """
    # Load data
    print(f"\n{'='*80}")
    print("AUTONOMOUS DATA DISCOVERY")
    print(f"{'='*80}\n")

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"Loading data from: {file_path}")

    # Determine file type and load accordingly
    file_ext = file_path.suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported formats: .csv, .xlsx, .xls")
        return

    if dataset_name is None:
        dataset_name = file_path.stem

    print(f"Dataset: {dataset_name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Create workflow
    workflow = DiscoveryWorkflow(
        max_iterations=30,       # Max exploration cycles (increased for thorough exploration)
        max_insights=3,          # Max number of insights (not a fixed requirement)
        generate_context=True    # Generate business context
    )

    # Run discovery
    print(f"\n{'='*80}")
    print("STARTING AUTONOMOUS EXPLORATION")
    print(f"{'='*80}\n")

    result = workflow.run_discovery(df, dataset_name)

    # Generate report (DISABLED FOR TESTING)
    # print(f"\n{'='*80}")
    # print("GENERATING REPORT")
    # print(f"{'='*80}\n")

    # reporter = DiscoveryReporter()
    # report_path = reporter.generate_report(
    #     result,
    #     workflow.last_dataset_context,
    #     workflow.last_exploration_result
    # )

    # Summary
    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}\n")
    print(f"✅ Insights found: {len(workflow.last_exploration_result.insights) if workflow.last_exploration_result else 0}")
    print(f"✅ Code executions: {workflow.last_exploration_result.total_executions if workflow.last_exploration_result else 0}")

    # Show viz_data path
    if workflow.last_exploration_result and workflow.last_exploration_result.viz_data_path:
        print(f"✅ Viz data JSON: {workflow.last_exploration_result.viz_data_path}")

        # Generate Plotly dashboard directly
        from src.discovery.plotly_dashboard_generator import PlotlyDashboardGenerator
        dashboard_gen = PlotlyDashboardGenerator()
        try:
            dashboard_path = dashboard_gen.generate_dashboard(workflow.last_exploration_result.viz_data_path)
            print(f"✅ Plotly Dashboard: {dashboard_path}")
        except Exception as e:
            print(f"⚠️  Dashboard generation failed: {e}")

    # Show insights summary
    if workflow.last_exploration_result and workflow.last_exploration_result.insights:
        print(f"\n📊 Insights Summary:\n")
        for idx, insight in enumerate(workflow.last_exploration_result.insights, 1):
            print(f"{idx}. {insight.question}")
            print(f"   {insight.finding[:150]}...")
            print()

    return result


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path> [dataset_name]")
        print("\nSupported formats: CSV (.csv), Excel (.xlsx, .xls)")
        print("\nExamples:")
        print("  python main.py data/test/FinDeep.xlsx")
        print("  python main.py data/sales_2024.csv sales_data")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else None

    run_discovery(file_path, dataset_name)


if __name__ == "__main__":
    main()

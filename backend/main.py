"""Main entry point for ETL to Insights AI Agent with independent phase execution."""

import sys
import pandas as pd
from pathlib import Path
from src.graph.workflow import ETLInsightsWorkflow
from src.graph.state import WorkflowState
from src.graph.discovery_workflow import DiscoveryWorkflow
from src.discovery.discovery_reporter import DiscoveryReporter
from src.models.business_context import BusinessContext, Department


def collect_business_context() -> BusinessContext:
    """Interactive collection of business context from user."""
    print("\n" + "="*60)
    print("BUSINESS CONTEXT COLLECTION")
    print("="*60)

    # Collect basic information
    print("\n--- Basic Information ---")
    company_name = input("Company Name: ").strip()
    icp = input("Ideal Customer Profile (ICP): ").strip()
    mission = input("Company Mission: ").strip()
    current_goal = input("Current Business Goal: ").strip()

    # Collect success metrics
    print("\n--- Success Metrics ---")
    print("Enter success metrics (one per line, empty line to finish):")
    success_metrics = []
    while True:
        metric = input(f"  Metric {len(success_metrics) + 1}: ").strip()
        if not metric:
            break
        success_metrics.append(metric)

    if not success_metrics:
        success_metrics = ["Revenue growth", "Customer satisfaction", "Operational efficiency"]
        print(f"Using default metrics: {', '.join(success_metrics)}")

    # Collect department information
    departments = []
    print("\n--- Department Information ---")
    print("Enter department information (leave name empty to finish):")

    while True:
        dept_name = input(f"\nDepartment {len(departments) + 1} name: ").strip()
        if not dept_name:
            break

        dept_desc = input(f"  Description (optional): ").strip()

        print(f"  Objectives for {dept_name} (empty line to finish):")
        objectives = []
        while True:
            obj = input(f"    Objective {len(objectives) + 1}: ").strip()
            if not obj:
                break
            objectives.append(obj)

        print(f"  Pain Points for {dept_name} (empty line to finish):")
        painpoints = []
        while True:
            pain = input(f"    Pain Point {len(painpoints) + 1}: ").strip()
            if not pain:
                break
            painpoints.append(pain)

        print(f"  Perspectives from {dept_name} (empty line to finish):")
        perspectives = []
        while True:
            persp = input(f"    Perspective {len(perspectives) + 1}: ").strip()
            if not persp:
                break
            perspectives.append(persp)

        department = Department(
            name=dept_name,
            description=dept_desc or None,
            objectives=objectives,
            painpoints=painpoints,
            perspectives=perspectives
        )
        departments.append(department)

    if not departments:
        print("\n⚠ No departments added. Using sample department structure.")
        departments = [
            Department(
                name="Sales",
                description="Sales and revenue generation",
                objectives=["Increase revenue", "Improve conversion rates"],
                painpoints=["Long sales cycles", "Low lead quality"],
                perspectives=["Need better lead qualification", "Sales tools are outdated"]
            ),
            Department(
                name="Marketing",
                description="Marketing and customer acquisition",
                objectives=["Generate quality leads", "Improve brand awareness"],
                painpoints=["Limited marketing budget", "Difficulty measuring ROI"],
                perspectives=["Need data-driven approach", "Want better attribution"]
            )
        ]

    # Create BusinessContext
    context = BusinessContext(
        company_name=company_name,
        icp=icp,
        mission=mission,
        current_goal=current_goal,
        success_metrics=success_metrics,
        departments=departments
    )

    return context


def get_demo_context() -> BusinessContext:
    """Get demo business context."""
    return BusinessContext(
        company_name="TechStartup Inc",
        icp="B2B SaaS companies with 50-500 employees",
        mission="Help businesses scale through data-driven decision making",
        current_goal="Increase customer retention by 20% and reduce churn",
        success_metrics=[
            "Customer retention rate",
            "Monthly recurring revenue",
            "Customer satisfaction score",
            "Time to value"
        ],
        departments=[
            Department(
                name="Sales",
                description="Sales and revenue generation",
                objectives=[
                    "Increase conversion rate by 15%",
                    "Reduce sales cycle time",
                    "Improve lead quality"
                ],
                painpoints=[
                    "Long sales cycles (avg 90 days)",
                    "Low lead-to-customer conversion rate (8%)",
                    "Difficulty identifying high-value prospects"
                ],
                perspectives=[
                    "Need better lead scoring system",
                    "Sales tools lack integration",
                    "Want more predictive insights"
                ]
            ),
            Department(
                name="Customer_Success",
                description="Customer retention and growth",
                objectives=[
                    "Reduce churn rate below 5%",
                    "Increase customer lifetime value",
                    "Improve onboarding completion rate"
                ],
                painpoints=[
                    "Churn rate at 12% (above industry average)",
                    "Difficulty predicting at-risk customers",
                    "Limited visibility into product usage"
                ],
                perspectives=[
                    "Need early warning system for churn",
                    "Want to understand usage patterns",
                    "Require automated health scoring"
                ]
            ),
            Department(
                name="Product",
                description="Product development and innovation",
                objectives=[
                    "Increase feature adoption",
                    "Reduce time to market for new features",
                    "Improve product-market fit"
                ],
                painpoints=[
                    "Low adoption of new features (30%)",
                    "Unclear which features drive retention",
                    "Limited data on user behavior"
                ],
                perspectives=[
                    "Need feature usage analytics",
                    "Want to understand user journeys",
                    "Require A/B testing capabilities"
                ]
            )
        ]
    )


def run_phase1_only():
    """Run Phase 1 only: Problem identification."""
    print("\n" + "="*60)
    print("MODE: PHASE 1 ONLY (Setup)")
    print("="*60)
    print("\nThis will:")
    print("1. Collect your business context")
    print("2. Identify all challenges")
    print("3. Prioritize them by importance")
    print("4. Store everything in ChromaDB")
    print("\nAfter this, run Phase 2 to analyze challenges one by one.")
    print("="*60)

    # Collect business context
    print("\nChoose input method:")
    print("1. Interactive (enter your data)")
    print("2. Demo (use sample data)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        print("\nUsing demo data...")
        business_context = get_demo_context()
    else:
        business_context = collect_business_context()

    print("\n✓ Business context collected")
    print(f"\nCompany: {business_context.company_name}")
    print(f"Departments: {len(business_context.departments)}")

    # Initialize workflow
    workflow = ETLInsightsWorkflow()

    # Run Phase 1
    result = workflow.run_phase1(business_context)

    # Display results
    print("\n" + "="*60)
    print("PHASE 1 RESULTS")
    print("="*60)

    if result["status"] == "completed":
        print(f"\n✓ Successfully identified {len(result['challenges'])} challenges")
        print(f"\nTop 5 priorities:")
        for i, challenge in enumerate(result["challenges"][:5], 1):
            print(f"\n{i}. {challenge.title}")
            print(f"   Priority: {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)")
            print(f"   Department: {challenge.department}")

        print(f"\n✅ Phase 1 complete! Challenges stored in ChromaDB.")
        print(f"\nNext steps:")
        print(f"  python main.py --phase2     # Analyze highest priority challenge")
        print(f"  python main.py --status     # Check challenge queue status")
    else:
        print(f"\n✗ Phase 1 failed: {result['error_message']}")

    print("="*60)


def run_phase2_once():
    """Run Phase 2 once: Analyze one challenge."""
    print("\n" + "="*60)
    print("MODE: PHASE 2 SINGLE (Analyze One Challenge)")
    print("="*60)
    print("\nThis will:")
    print("1. Pop the highest priority challenge")
    print("2. Run ETL on relevant data")
    print("3. Perform statistical analysis")
    print("4. Generate visualizations")
    print("5. Save results")
    print("="*60)

    # Initialize workflow
    workflow = ETLInsightsWorkflow()

    # Check status first
    status = workflow.get_challenge_status()
    if status["remaining"] == 0:
        print(f"\n⚠ No challenges remaining!")
        print(f"  Run 'python main.py --phase1' to set up new challenges")
        return

    print(f"\nChallenge Queue Status:")
    print(f"  Total: {status['total_challenges']}")
    print(f"  Processed: {status['processed']}")
    print(f"  Remaining: {status['remaining']}")

    if input("\nContinue with analysis? (y/n): ").strip().lower() != 'y':
        print("Cancelled.")
        return

    # Run Phase 2
    result = workflow.run_phase2_single()

    # Display results
    print("\n" + "="*60)
    print("PHASE 2 RESULTS")
    print("="*60)

    if result["status"] == "completed":
        challenge = result["challenge_processed"]
        analysis = result["analysis_result"]

        print(f"\n✓ Successfully analyzed: {challenge.title}")
        print(f"\nResults:")
        print(f"  - Key findings: {len(analysis.key_findings)}")
        print(f"  - Statistical tests: {len(analysis.statistical_tests)}")
        print(f"  - Visualizations: {len(analysis.visualizations)}")
        print(f"  - Recommendations: {len(analysis.recommendations)}")

        print(f"\n📊 Outputs saved to:")
        print(f"  - Visualizations: data/outputs/visualizations/")
        print(f"  - Processed data: data/outputs/processed/")

        print(f"\nRemaining challenges: {result['challenges_remaining']}")

        if result['challenges_remaining'] > 0:
            print(f"\n➡️  Run again to process next challenge:")
            print(f"   python main.py --phase2")
        else:
            print(f"\n✅ All challenges processed!")

        print(f"\n📄 Generate comprehensive reports:")
        print(f"   python main.py --reports")

    elif result["status"] == "no_challenges":
        print(f"\n✓ {result['error_message']}")
    else:
        print(f"\n✗ Phase 2 failed: {result['error_message']}")

    print("="*60)


def run_generate_reports():
    """Generate comprehensive reports from all analyses."""
    print("\n" + "="*60)
    print("MODE: GENERATE REPORTS")
    print("="*60)
    print("\nThis will:")
    print("1. Load all analysis results from ChromaDB")
    print("2. Generate analytical report (technical)")
    print("3. Generate business insight report (executive)")
    print("="*60)

    # Initialize workflow
    workflow = ETLInsightsWorkflow()

    # Generate reports
    result = workflow.generate_reports()

    # Display results
    print("\n" + "="*60)
    print("REPORT GENERATION RESULTS")
    print("="*60)

    if result["status"] == "completed":
        print(f"\n✓ Reports generated successfully!")
        print(f"\n📄 Reports saved to:")
        print(f"  {result['analytical_report_path']}")
        print(f"\nGenerated files:")
        print(f"  - analytical_report_[timestamp].md")
        print(f"  - business_insight_report_[timestamp].md")
    else:
        print(f"\n✗ Report generation failed: {result['error_message']}")

    print("="*60)


def show_status():
    """Show challenge queue status."""
    print("\n" + "="*60)
    print("CHALLENGE QUEUE STATUS")
    print("="*60)

    workflow = ETLInsightsWorkflow()
    status = workflow.get_challenge_status()

    if "error" in status:
        print(f"\n✗ Error: {status['error']}")
        print(f"\nRun 'python main.py --phase1' to set up challenges")
    else:
        print(f"\nTotal challenges identified: {status['total_challenges']}")
        print(f"Challenges analyzed: {status['processed']}")
        print(f"Challenges remaining: {status['remaining']}")

        if status['next_challenge']:
            next_challenge = status['next_challenge']
            print(f"\nNext challenge to analyze:")
            print(f"  {next_challenge.title}")
            print(f"  Priority: {next_challenge.priority_level.value.upper()} ({next_challenge.priority_score:.1f}/100)")
            print(f"  Department: {next_challenge.department}")

            print(f"\n➡️  Analyze next challenge:")
            print(f"   python main.py --phase2")
        elif status['total_challenges'] > 0:
            print(f"\n✅ All challenges have been analyzed!")
            print(f"\n➡️  Generate reports:")
            print(f"   python main.py --reports")
        else:
            print(f"\n⚠ No challenges found")
            print(f"\n➡️  Set up challenges:")
            print(f"   python main.py --phase1")

    print("="*60)


def run_discovery_mode():
    """Run autonomous discovery on a single dataset."""
    print("\n" + "="*60)
    print("MODE: AUTONOMOUS DISCOVERY")
    print("="*60)
    print("\nThis mode uses AI to autonomously explore your dataset.")
    print("The LLM will:")
    print("1. Understand what your data represents")
    print("2. Generate and execute Python code to explore patterns")
    print("3. Find trends, growth rates, and comparisons")
    print("4. Discover insights like Grok AI")
    print("5. Produce a business-focused report")
    print("="*60)

    # Get file path
    file_path = input("\nEnter path to CSV/Excel file: ").strip()

    if not file_path:
        print("[ERROR] No file path provided")
        return

    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    # Load data
    print(f"\nLoading data from {file_path}...")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("[ERROR] Unsupported file format. Use CSV or Excel.")
            return

        print(f"[OK] Loaded {len(df):,} rows x {len(df.columns)} columns")

    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return

    # Initialize workflow with autonomous exploration
    print("\n[AI] Initializing autonomous explorer...")
    print("The AI will autonomously decide what to explore...")
    workflow = DiscoveryWorkflow(
        max_iterations=40,  # Agent exploration steps (optimized for 10 insights)
        max_insights=10     # Target insights to find
    )

    # Run discovery
    try:
        dataset_name = file_path_obj.stem
        result = workflow.run_discovery(df, dataset_name)

        # Generate reports
        print("\n📝 Generating reports...")
        reporter = DiscoveryReporter()

        # Get dataset context from workflow state (if available)
        dataset_context = getattr(workflow, 'last_dataset_context', None)

        # Generate markdown report
        print("[REPORT] Creating markdown report...")
        report_path = reporter.generate_report(result, dataset_context)

        # Generate interactive HTML dashboard
        print("[DASHBOARD] Creating interactive HTML dashboard...")
        dashboard_path = reporter.generate_html_dashboard(result, dataset_context)

        print("\n" + "="*60)
        print("[SUCCESS] DISCOVERY COMPLETED!")
        print("="*60)
        print(f"\nInsights discovered: {len(result.answered_questions)}")
        print(f"Key findings: {len(result.key_insights)}")
        print(f"\n[OUTPUTS]")
        print(f"  Markdown Report: {report_path}")
        print(f"  HTML Dashboard:  {dashboard_path}")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Discovery failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    print("\nETL to Insights AI Agent")
    print("========================\n")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--phase1":
            run_phase1_only()
        elif command == "--phase2":
            run_phase2_once()
        elif command == "--reports":
            run_generate_reports()
        elif command == "--status":
            show_status()
        elif command == "--discovery":
            run_discovery_mode()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  --phase1     : Set up challenges (run once)")
            print("  --phase2     : Analyze next challenge (run multiple times)")
            print("  --reports    : Generate comprehensive reports")
            print("  --status     : Show challenge queue status")
            print("  --discovery  : Autonomous AI discovery (single large dataset)")
    else:
        print("Choose mode:")
        print("1. Phase 1: Setup (identify and prioritize challenges)")
        print("2. Phase 2: Analyze (process one challenge)")
        print("3. Generate Reports (create comprehensive reports)")
        print("4. Check Status (view challenge queue)")
        print("5. Discovery Mode (autonomous AI exploration of dataset)")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            run_phase1_only()
        elif choice == "2":
            run_phase2_once()
        elif choice == "3":
            run_generate_reports()
        elif choice == "4":
            show_status()
        elif choice == "5":
            run_discovery_mode()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

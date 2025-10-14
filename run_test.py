"""Test script for ETL to Insights AI Agent - Easy-Medium difficulty."""

import sys
import shutil
from pathlib import Path
from src.graph.workflow import ETLInsightsWorkflow
from src.models.business_context import BusinessContext, Department


def setup_test_environment():
    """Set up test environment by moving test data to uploads folder."""
    print("\n" + "="*60)
    print("SETTING UP TEST ENVIRONMENT")
    print("="*60)

    test_data_path = Path("test_data")
    uploads_path = Path("data/uploads")

    if not test_data_path.exists():
        print("\n✗ Test data not found!")
        print("  Run: python create_test_data.py")
        return False

    # Clear existing uploads
    if uploads_path.exists():
        print("\nClearing existing data/uploads...")
        shutil.rmtree(uploads_path)

    uploads_path.mkdir(parents=True, exist_ok=True)

    # Copy test data
    print("Copying test data to data/uploads...")
    for dept_folder in test_data_path.iterdir():
        if dept_folder.is_dir():
            dest_folder = uploads_path / dept_folder.name
            shutil.copytree(dept_folder, dest_folder)
            files = list(dest_folder.glob("*"))
            print(f"  ✓ {dept_folder.name}: {len(files)} files")

    print("\n✓ Test environment ready!")
    return True


def get_test_business_context() -> BusinessContext:
    """Create test business context for the agent."""
    return BusinessContext(
        company_name="TechVenture Solutions",
        icp="Mid-market B2B SaaS companies (50-500 employees) looking to scale operations",
        mission="Empower growing companies with data-driven insights and scalable business intelligence",
        current_goal="Increase annual recurring revenue by 40% while improving customer retention to 95%",
        success_metrics=[
            "Annual Recurring Revenue (ARR) growth",
            "Customer Retention Rate (target: 95%)",
            "Net Promoter Score (target: 8.5/10)",
            "Sales Conversion Rate (target: 12%)",
            "Average Revenue Per Customer",
            "Customer Acquisition Cost (CAC)",
            "Customer Lifetime Value (LTV)",
            "Time to Value (product onboarding)"
        ],
        departments=[
            Department(
                name="Sales",
                description="Revenue generation through new customer acquisition",
                objectives=[
                    "Increase conversion rate from 8.7% to 12%",
                    "Reduce sales cycle from 87 days to 65 days",
                    "Improve average deal size by 20%",
                    "Grow pipeline value to $12M by end of year"
                ],
                painpoints=[
                    "Sales cycles are too long (avg 87 days) affecting revenue velocity",
                    "Low conversion rate (8.7%) indicates lead quality issues",
                    "Regional performance varies significantly (West 34% better than East)",
                    "Inconsistent lead scoring leads to wasted effort on poor-fit prospects",
                    "Sales reps lack visibility into customer engagement signals"
                ],
                perspectives=[
                    "Need better lead qualification system to focus on high-potential prospects",
                    "Sales tools are disconnected - CRM doesn't integrate well with marketing automation",
                    "Want predictive analytics to identify deals likely to close",
                    "Regional training and best practice sharing could reduce performance gaps",
                    "Need real-time alerts when prospects show buying signals"
                ]
            ),
            Department(
                name="Marketing",
                description="Lead generation and brand awareness",
                objectives=[
                    "Generate 40% more marketing qualified leads (MQLs)",
                    "Improve MQL to SQL conversion rate to 65%",
                    "Reduce cost per lead by 20%",
                    "Achieve 4:1 marketing ROI across all channels"
                ],
                painpoints=[
                    "Difficult to attribute revenue to specific marketing campaigns",
                    "Some channels (email, paid ads) perform well while others lag",
                    "Lead quality varies significantly by channel",
                    "Budget allocation not optimized based on performance data",
                    "Long time lag between marketing spend and sales results"
                ],
                perspectives=[
                    "Need better attribution model to understand campaign effectiveness",
                    "Want to double down on high-performing channels (Paid, Social)",
                    "Email campaigns show consistent ROI - should increase investment",
                    "Content marketing underperforming - needs strategy refresh",
                    "Require closed-loop reporting with sales to optimize lead generation"
                ]
            ),
            Department(
                name="Customer_Success",
                description="Customer retention, satisfaction, and growth",
                objectives=[
                    "Reduce churn rate from 12% to below 5%",
                    "Increase onboarding completion rate to 85%",
                    "Improve NPS from 7.2 to 8.5+",
                    "Grow net revenue retention to 120%"
                ],
                painpoints=[
                    "Churn rate at 12% is above industry benchmark (8%)",
                    "Cannot predict which customers are at risk of churning",
                    "Only 73% of customers complete onboarding successfully",
                    "Low product usage correlates with churn but reacting too late",
                    "Support ticket escalations indicate product gaps"
                ],
                perspectives=[
                    "Need early warning system to identify at-risk customers",
                    "Product usage data should trigger proactive outreach",
                    "Onboarding process needs to be more personalized by use case",
                    "Want automated health scoring based on multiple engagement signals",
                    "Feature requests in feedback indicate opportunities for retention"
                ]
            ),
            Department(
                name="Operations",
                description="Technical operations and customer support",
                objectives=[
                    "Reduce average support response time to under 3 hours",
                    "Achieve 95% first contact resolution rate",
                    "Maintain 99.5%+ system uptime",
                    "Improve customer satisfaction score to 4.5/5"
                ],
                painpoints=[
                    "Support response time averaging 4.2 hours (target: <3 hours)",
                    "Ticket escalations (avg 12/week) indicate complex issues",
                    "System performance varies - response times spike during peak hours",
                    "Customer satisfaction at 4.1/5 needs improvement",
                    "Error rates around 0.8% causing customer frustration"
                ],
                perspectives=[
                    "Need to identify root causes of recurring support issues",
                    "Performance optimization required for peak usage periods",
                    "First contact resolution would dramatically improve satisfaction",
                    "Escalations often reveal product bugs that should be prioritized",
                    "Response time improvements would boost NPS and retention"
                ]
            )
        ]
    )


def run_complete_test():
    """Run complete end-to-end test of the agent."""
    print("\n" + "="*70)
    print("ETL TO INSIGHTS AI AGENT - COMPREHENSIVE TEST")
    print("="*70)
    print("\nTest Profile:")
    print("  Difficulty: Easy-Medium")
    print("  Data Types: CSV, Excel (multi-sheet), PDF")
    print("  Departments: 4 (Sales, Marketing, Customer Success, Operations)")
    print("  Test Mode: Full workflow with Phase 1 + Phase 2")
    print("="*70)

    # Setup environment
    if not setup_test_environment():
        return False

    # Get business context
    print("\n" + "="*70)
    print("PHASE 1: PROBLEM IDENTIFICATION")
    print("="*70)

    business_context = get_test_business_context()
    print(f"\n✓ Business Context: {business_context.company_name}")
    print(f"  Departments: {len(business_context.departments)}")
    print(f"  Success Metrics: {len(business_context.success_metrics)}")

    # Initialize workflow
    workflow = ETLInsightsWorkflow()

    # Run Phase 1
    print("\nRunning Phase 1: Identifying and prioritizing challenges...")
    phase1_result = workflow.run_phase1(business_context)

    if phase1_result["status"] != "completed":
        print(f"\n✗ Phase 1 failed: {phase1_result['error_message']}")
        return False

    challenges = phase1_result["challenges"]
    print(f"\n✓ Phase 1 Complete: {len(challenges)} challenges identified")

    print("\nTop 5 Priority Challenges:")
    for i, challenge in enumerate(challenges[:5], 1):
        print(f"\n{i}. {challenge.title}")
        print(f"   Priority: {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)")
        print(f"   Department: {challenge.department}")
        print(f"   Pain Points: {len(challenge.related_painpoints)}")

    # Run Phase 2 for top 3 challenges
    print("\n" + "="*70)
    print("PHASE 2: ANALYSIS & INSIGHTS")
    print("="*70)
    print(f"\nAnalyzing top 3 challenges (out of {len(challenges)} total)...")

    analysis_results = []
    for i in range(min(3, len(challenges))):
        print(f"\n{'─'*70}")
        print(f"ANALYZING CHALLENGE {i+1}/3")
        print(f"{'─'*70}")

        phase2_result = workflow.run_phase2_single()

        if phase2_result["status"] == "completed":
            analysis = phase2_result["analysis_result"]
            analysis_results.append(analysis)

            print(f"\n✓ Analysis Complete: {analysis.challenge_title}")
            print(f"  Key Findings: {len(analysis.key_findings)}")
            print(f"  Statistical Tests: {len(analysis.statistical_tests)}")
            print(f"  Correlations: {len(analysis.correlations)}")
            print(f"  Visualizations: {len(analysis.visualizations)}")
            print(f"  Recommendations: {len(analysis.recommendations)}")

            # Show sample findings
            if analysis.key_findings:
                print(f"\n  Sample Findings:")
                for finding in analysis.key_findings[:2]:
                    print(f"    • {finding}")

        else:
            print(f"\n⚠ Analysis failed: {phase2_result['error_message']}")

    # Generate Reports
    print("\n" + "="*70)
    print("REPORT GENERATION")
    print("="*70)

    print("\nGenerating comprehensive reports...")
    report_result = workflow.generate_reports(analysis_results)

    if report_result["status"] == "completed":
        print("\n✓ Reports Generated Successfully!")
        print(f"  Location: {report_result['analytical_report_path']}")
        print(f"\n  Files:")
        print(f"    • analytical_report_[timestamp].md")
        print(f"    • business_insight_report_[timestamp].md")
    else:
        print(f"\n⚠ Report generation failed: {report_result['error_message']}")

    # Final Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    print(f"\n✓ Test Completed Successfully!")
    print(f"\nPhase 1 Results:")
    print(f"  • Challenges Identified: {len(challenges)}")
    print(f"  • Priority Levels:")
    critical = sum(1 for c in challenges if c.priority_level.value == 'critical')
    high = sum(1 for c in challenges if c.priority_level.value == 'high')
    medium = sum(1 for c in challenges if c.priority_level.value == 'medium')
    low = sum(1 for c in challenges if c.priority_level.value == 'low')
    print(f"    - Critical: {critical}")
    print(f"    - High: {high}")
    print(f"    - Medium: {medium}")
    print(f"    - Low: {low}")

    print(f"\nPhase 2 Results:")
    print(f"  • Challenges Analyzed: {len(analysis_results)}")
    print(f"  • Total Findings: {sum(len(a.key_findings) for a in analysis_results)}")
    print(f"  • Total Statistical Tests: {sum(len(a.statistical_tests) for a in analysis_results)}")
    print(f"  • Total Visualizations: {sum(len(a.visualizations) for a in analysis_results)}")

    print(f"\n📊 Output Locations:")
    print(f"  • Reports: data/outputs/reports/")
    print(f"  • Visualizations: data/outputs/visualizations/")
    print(f"  • Processed Data: data/outputs/processed/")

    print(f"\n🎯 Test Assessment: {'PASSED' if len(analysis_results) >= 2 else 'PARTIAL'}")
    print("="*70)

    return True


def run_incremental_test():
    """Run incremental test (Phase 1, then Phase 2 separately)."""
    print("\n" + "="*70)
    print("ETL TO INSIGHTS AI AGENT - INCREMENTAL TEST")
    print("="*70)
    print("\nThis test demonstrates independent phase execution:")
    print("  1. Run Phase 1 once (setup)")
    print("  2. Run Phase 2 once (analyze top priority challenge)")
    print("="*70)

    # Setup
    if not setup_test_environment():
        return False

    business_context = get_test_business_context()
    workflow = ETLInsightsWorkflow()

    # Phase 1
    print("\n[STEP 1] Running Phase 1: Setup and Prioritization")
    phase1_result = workflow.run_phase1(business_context)

    if phase1_result["status"] != "completed":
        print(f"✗ Phase 1 failed")
        return False

    print(f"✓ Phase 1 complete: {len(phase1_result['challenges'])} challenges ready")

    # Check status
    print("\n[STEP 2] Checking Challenge Queue Status")
    status = workflow.get_challenge_status()
    print(f"  Total: {status['total_challenges']}")
    print(f"  Remaining: {status['remaining']}")

    # Run Phase 2 once
    print("\n[STEP 3] Running Phase 2: Analyzing Top Priority Challenge")
    result = workflow.run_phase2_single()
    if result["status"] == "completed":
        print(f"✓ Analyzed: {result['challenge_processed'].title}")
        print(f"  Remaining: {result['challenges_remaining']}")
    else:
        print(f"✗ Analysis failed: {result.get('error_message', 'Unknown error')}")

    # Generate reports
    print("\n[STEP 4] Generating Reports")
    workflow.generate_reports()

    print("\n✓ Incremental test complete!")
    print("\n💡 Key Insight: Phases can be run independently, allowing:")
    print("   • Setup once, analyze over time")
    print("   • Process one challenge per session")
    print("   • Generate reports after any number of analyses")

    return True


def main():
    """Main test entry point."""
    print("\nETL to Insights AI Agent - Test Suite")
    print("="*70)
    print("\nChoose test mode:")
    print("1. Complete Test (Phase 1 + analyze top 3 challenges)")
    print("2. Incremental Test (demonstrate independent phase execution)")
    print("3. Quick Test (Phase 1 only)")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-3): ").strip()

    if choice == "2":
        success = run_incremental_test()
    elif choice == "3":
        if setup_test_environment():
            workflow = ETLInsightsWorkflow()
            context = get_test_business_context()
            result = workflow.run_phase1(context)
            success = result["status"] == "completed"
    else:
        success = run_complete_test()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

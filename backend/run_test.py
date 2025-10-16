"""
Run test for ETL to Insights AI Agent - SocialPulse Media Scenario.
"""

import shutil
from pathlib import Path
from src.graph.workflow import ETLInsightsWorkflow
from src.models.business_context import BusinessContext, Department


def setup_test_environment():
    print("\n=== Setting up test environment ===")
    test_data = Path("test_data")
    uploads = Path("data/uploads")

    if not test_data.exists():
        print("✗ Test data not found! Run: python create_test_data.py")
        return False

    if uploads.exists():
        shutil.rmtree(uploads)
    uploads.mkdir(parents=True, exist_ok=True)

    for dept in test_data.iterdir():
        if dept.is_dir():
            shutil.copytree(dept, uploads / dept.name)
            print(f"  ✓ Copied {dept.name}")
    print("✓ Test environment ready.\n")
    return True


def get_test_business_context():
    return BusinessContext(
        company_name="SocialPulse Media",
        icp="Global B2C social media platform serving creators and communities",
        mission="Empower online communities through meaningful engagement and safe interactions",
        current_goal="Boost daily active users by 25% while reducing churn by 10%",
        success_metrics=[
            "DAU/MAU Ratio",
            "Retention Rate",
            "Engagement Rate per User",
            "User Satisfaction Score",
            "Ad Revenue per Active User",
            "Moderation Efficiency"
        ],
        departments=[
            Department(
                name="Engagement",
                description="Drives active usage, community activity, and content creation",
                objectives=["Increase DAU/MAU ratio", "Enhance content discoverability"],
                painpoints=["Engagement drops on weekends", "Low retention in EU region"],
                perspectives=[
                    "Need insights on peak activity hours",
                    "Identify content types driving most engagement"
                ]
            ),
            Department(
                name="Marketing",
                description="Manages campaigns to drive growth and monetization",
                objectives=["Optimize ad spend", "Improve campaign ROI by 20%"],
                painpoints=["High CAC", "Unclear attribution between campaigns and conversions"],
                perspectives=[
                    "Analyze best-performing channels by ROI",
                    "Detect seasonal marketing performance"
                ]
            ),
            Department(
                name="Product",
                description="Owns feature development and user experience",
                objectives=["Improve retention through better feature adoption"],
                painpoints=["Some features underused", "Churn risk high among low-usage users"],
                perspectives=["Correlate churn risk with feature usage patterns"]
            ),
            Department(
                name="Support",
                description="Handles moderation, customer service, and issue resolution",
                objectives=["Reduce average resolution time by 30%"],
                painpoints=["Escalations remain high", "False spam reports affecting satisfaction"],
                perspectives=["Analyze efficiency ratios", "Correlate satisfaction with resolution speed"]
            )
        ]
    )


def run_phase(phase: str):
    if not setup_test_environment():
        return
    context = get_test_business_context()
    agent = ETLInsightsWorkflow()

    if phase == "phase1":
        print("\n=== Running PHASE 1: Problem Identification ===\n")
        result = agent.run_phase1(context)
        if result["status"] == "completed":
            print(f"\n✓ Phase 1 completed successfully!")
            print(f"  {len(result['challenges'])} challenges identified and prioritized")
        else:
            print(f"\n✗ Phase 1 failed: {result['error_message']}")
    elif phase == "phase2":
        print("\n=== Running PHASE 2: Analysis & Insights (SINGLE CHALLENGE) ===\n")

        # Process ONE challenge (now includes report generation)
        result = agent.run_phase2_single()

        if result["status"] == "completed":
            print(f"\n{'='*60}")
            print(f"✓ WORKFLOW COMPLETE FOR: {result['challenge_processed'].title}")
            print(f"{'='*60}")

            if result.get("dashboard_path"):
                print(f"\n📊 Reports Generated:")
                print(f"  - Analytical Report: {result.get('analytical_report_path', 'N/A')}")
                print(f"  - Business Report: {result.get('business_report_path', 'N/A')}")
                print(f"  - Interactive Dashboard: {result.get('dashboard_path', 'N/A')}")

            print(f"\n📋 Challenges Status:")
            print(f"  - Remaining: {result['challenges_remaining']}")

            if result['challenges_remaining'] > 0:
                print(f"\n💡 To process next challenge, run: python run_test.py phase2")
        elif result["status"] == "no_challenges":
            print(f"\n✓ No more challenges to process")
        else:
            print(f"\n✗ Analysis failed: {result['error_message']}")
    else:
        print("✗ Invalid phase. Use:")
        print("  python run_test.py phase1   - Identify and prioritize challenges")
        print("  python run_test.py phase2   - Analyze challenge + generate reports/dashboard (run multiple times for each challenge)")


if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    run_phase(phase)

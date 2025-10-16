"""
Create test data for ETL to Insights AI Agent - Social Media Company Scenario.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

np.random.seed(42)

COMPANY_NAME = "SocialPulse Media"


def create_engagement_csv():
    """Create engagement performance data for Product/Engagement team."""
    print("Creating Engagement CSV data...")
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]

    df = pd.DataFrame({
        "date": dates,
        "active_users": np.random.normal(120000, 25000, 365).clip(min=30000).astype(int),
        "avg_session_duration_min": np.random.normal(18, 5, 365).clip(min=5),
        "posts_created": np.random.poisson(95000, 365),
        "comments_posted": np.random.poisson(240000, 365),
        "likes": np.random.poisson(1800000, 365),
        "shares": np.random.poisson(95000, 365),
        "region": np.random.choice(["NA", "EU", "APAC", "LATAM"], 365)
    })

    # Inject seasonal trend: higher engagement in summer and December
    df.loc[(df['date'].dt.month.isin([6, 7, 12])), 'active_users'] *= 1.15
    df.loc[df['date'].dt.dayofweek >= 5, 'active_users'] *= 0.8

    Path("test_data/Engagement").mkdir(parents=True, exist_ok=True)
    df.to_csv("test_data/Engagement/user_engagement_2024.csv", index=False)
    print(f"[OK] Engagement data: {len(df)} rows")
    return df


def create_marketing_excel():
    """Marketing campaign performance data."""
    print("Creating Marketing Excel data...")
    campaigns = [f"Campaign_{i}" for i in range(1, 13)]
    data = {
        "campaign_name": campaigns,
        "start_date": [datetime(2024, i, 1) for i in range(1, 13)],
        "budget_usd": np.random.randint(10000, 50000, 12),
        "impressions": np.random.randint(50000, 500000, 12),
        "clicks": np.random.randint(3000, 20000, 12),
        "conversions": np.random.randint(200, 1200, 12),
        "channel": np.random.choice(["Paid Ads", "Influencer", "Content", "Social"], 12),
    }

    df = pd.DataFrame(data)
    df["ctr"] = df["clicks"] / df["impressions"]
    df["cpc"] = df["budget_usd"] / df["clicks"]
    df["roi"] = ((df["conversions"] * 40) - df["budget_usd"]) / df["budget_usd"]

    Path("test_data/Marketing").mkdir(parents=True, exist_ok=True)
    df.to_excel("test_data/Marketing/marketing_performance_2024.xlsx", index=False)
    print(f"[OK] Marketing data: {len(df)} rows")
    return df


def create_product_excel():
    """Product feature usage and retention."""
    print("Creating Product Excel data...")
    users = [f"user_{i}" for i in range(1, 401)]
    features = ["Stories", "Shorts", "Reels", "Messaging", "Livestream", "Communities"]

    df_usage = pd.DataFrame({
        "user_id": np.random.choice(users, 400),
        "feature": np.random.choice(features, 400),
        "usage_hours": np.random.normal(3.5, 1.2, 400).clip(min=0),
        "sessions_per_day": np.random.poisson(4, 400),
        "days_active_per_month": np.random.randint(5, 30, 400),
        "churn_risk_score": np.random.normal(40, 25, 400).clip(min=0, max=100),
        "region": np.random.choice(["NA", "EU", "APAC", "LATAM"], 400)
    })

    df_retention = pd.DataFrame({
        "month": [f"Month_{i}" for i in range(1, 13)],
        "new_users": np.random.randint(8000, 15000, 12),
        "churned_users": np.random.randint(3000, 8000, 12),
        "avg_usage_hours": np.random.normal(3.2, 0.5, 12),
        "retention_rate": np.random.uniform(0.6, 0.85, 12)
    })

    Path("test_data/Product").mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter("test_data/Product/product_metrics.xlsx", engine='openpyxl') as writer:
        df_usage.to_excel(writer, sheet_name="Feature Usage", index=False)
        df_retention.to_excel(writer, sheet_name="Retention", index=False)

    print(f"[OK] Product data: {len(df_usage)} usage rows")
    return df_usage, df_retention


def create_support_csv():
    """Support and moderation performance."""
    print("Creating Support CSV data...")
    weeks = [f"Week_{i}" for i in range(1, 27)]
    df = pd.DataFrame({
        "week": weeks,
        "tickets_resolved": np.random.poisson(800, 26),
        "avg_resolution_time_hr": np.random.normal(18, 5, 26).clip(min=3),
        "escalations": np.random.poisson(45, 26),
        "satisfaction_score": np.random.normal(4.2, 0.5, 26).clip(min=1, max=5),
        "spam_reports": np.random.poisson(300, 26),
        "false_positives": np.random.poisson(40, 26),
    })
    df["efficiency_ratio"] = df["tickets_resolved"] / (df["escalations"] + 1)

    Path("test_data/Support").mkdir(parents=True, exist_ok=True)
    df.to_csv("test_data/Support/support_metrics_2024.csv", index=False)
    print(f"[OK] Support data: {len(df)} rows")
    return df


def create_feedback_pdf():
    """Generate PDF report summarizing user feedback."""
    print("Creating Feedback PDF...")
    output_path = Path("test_data/Engagement/user_feedback_summary.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>User Feedback Summary</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Key feedback themes include increased interest in community features, requests for better moderation tools, "
        "and performance concerns during peak hours.", styles["BodyText"]
    ))
    story.append(Spacer(1, 0.2 * inch))

    data = [
        ["Theme", "Mentions", "Sentiment"],
        ["Community & Groups", "180", "Positive"],
        ["App Performance", "120", "Negative"],
        ["Moderation Fairness", "95", "Neutral"],
        ["Feature Requests", "140", "Positive"]
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    doc.build(story)
    print(f"[OK] Created {output_path}")


def main():
    print("\n=== Creating Social Media Test Data (ETL to Insights) ===\n")
    create_engagement_csv()
    create_marketing_excel()
    create_product_excel()
    create_support_csv()
    create_feedback_pdf()
    print("\n✅ All test data generated successfully in ./test_data/")


if __name__ == "__main__":
    main()

"""Create test data for the ETL to Insights AI Agent."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# Set random seed for reproducibility
np.random.seed(42)


def create_sales_csv():
    """Create CSV file with sales data (easy complexity)."""
    print("Creating Sales CSV data...")

    # Generate 6 months of daily sales data
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(180)]

    data = {
        'date': dates,
        'revenue': np.random.normal(45000, 12000, 180).clip(min=10000),
        'num_deals': np.random.poisson(7, 180),
        'conversion_rate': np.random.normal(0.085, 0.025, 180).clip(min=0.02, max=0.20),
        'avg_deal_size': np.random.normal(6500, 1800, 180).clip(min=2000),
        'sales_cycle_days': np.random.normal(87, 25, 180).clip(min=30, max=180),
        'lead_score': np.random.normal(68, 18, 180).clip(min=0, max=100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 180),
        'sales_rep': np.random.choice(['John', 'Sarah', 'Mike', 'Lisa', 'Tom'], 180),
    }

    df = pd.DataFrame(data)

    # Add some realistic patterns
    # Weekends have lower sales
    df.loc[df['date'].dt.dayofweek >= 5, 'revenue'] *= 0.6
    # Convert num_deals to int64 first to avoid dtype warning
    df['num_deals'] = df['num_deals'].astype('int64')
    df.loc[df['date'].dt.dayofweek >= 5, 'num_deals'] = (df.loc[df['date'].dt.dayofweek >= 5, 'num_deals'] * 0.5).astype('int64')

    # Growth trend over time
    df['revenue'] = df['revenue'] * (1 + df.index / 1000)

    output_path = Path("test_data/Sales/sales_performance_2024.csv")
    df.to_csv(output_path, index=False)
    print(f"[OK] Created: {output_path} ({len(df)} rows)")
    return df


def create_customer_success_excel():
    """Create Excel file with customer health data (medium complexity)."""
    print("Creating Customer Success Excel data...")

    # Generate customer health metrics
    num_customers = 300

    customers = {
        'customer_id': [f'CUST{i:04d}' for i in range(1, num_customers + 1)],
        'signup_date': [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 400)) for _ in range(num_customers)],
        'monthly_recurring_revenue': np.random.normal(850, 350, num_customers).clip(min=200),
        'product_usage_hours': np.random.normal(42, 22, num_customers).clip(min=0),
        'feature_adoption_rate': np.random.beta(5, 2, num_customers),
        'support_tickets_month': np.random.poisson(2.5, num_customers),
        'nps_score': np.random.normal(7.2, 2.1, num_customers).clip(min=0, max=10),
        'last_login_days_ago': np.random.exponential(12, num_customers).clip(min=0, max=60),
        'onboarding_completed': np.random.choice([True, False], num_customers, p=[0.73, 0.27]),
        'customer_tier': np.random.choice(['Basic', 'Professional', 'Enterprise'], num_customers, p=[0.6, 0.3, 0.1]),
        'account_health_score': np.random.normal(72, 18, num_customers).clip(min=0, max=100),
    }

    df_customers = pd.DataFrame(customers)

    # Add churn flag based on health indicators
    churn_probability = (
        (df_customers['last_login_days_ago'] > 30) * 0.3 +
        (df_customers['nps_score'] < 5) * 0.25 +
        (df_customers['support_tickets_month'] > 5) * 0.2 +
        (~df_customers['onboarding_completed']) * 0.25
    ).clip(upper=0.8)

    df_customers['churned'] = np.random.binomial(1, churn_probability).astype(bool)

    # Create engagement metrics sheet
    engagement_data = {
        'week': [f'Week {i}' for i in range(1, 25)],
        'daily_active_users': np.random.normal(180, 35, 24).clip(min=80).astype(int),
        'weekly_active_users': np.random.normal(420, 85, 24).clip(min=200).astype(int),
        'avg_session_duration_min': np.random.normal(28, 8, 24).clip(min=5),
        'feature_usage_rate': np.random.beta(6, 3, 24),
    }

    df_engagement = pd.DataFrame(engagement_data)

    output_path = Path("test_data/Customer_Success/customer_health_metrics.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_customers.to_excel(writer, sheet_name='Customer Data', index=False)
        df_engagement.to_excel(writer, sheet_name='Engagement Metrics', index=False)

    print(f"[OK] Created: {output_path} (2 sheets: {len(df_customers)} customers, {len(df_engagement)} weeks)")
    return df_customers, df_engagement


def create_marketing_excel():
    """Create Excel file with marketing campaign data (medium complexity)."""
    print("Creating Marketing Excel data...")

    # Campaign performance data
    campaigns = ['Email_Q1', 'Social_Media_Spring', 'Content_Marketing', 'Paid_Ads_Q1',
                 'Webinar_Series', 'Email_Q2', 'Social_Media_Summer', 'Paid_Ads_Q2']

    campaign_data = {
        'campaign_name': campaigns,
        'start_date': [datetime(2024, m, 1) for m in [1, 2, 1, 1, 3, 4, 5, 4]],
        'budget': [15000, 22000, 18000, 35000, 12000, 16000, 24000, 38000],
        'impressions': [85000, 156000, 42000, 289000, 28000, 92000, 168000, 312000],
        'clicks': [3200, 6800, 1850, 12400, 1680, 3600, 7200, 13800],
        'leads_generated': [245, 380, 125, 520, 95, 268, 420, 580],
        'marketing_qualified_leads': [156, 242, 78, 312, 58, 172, 268, 348],
        'sales_qualified_leads': [89, 138, 42, 178, 32, 98, 152, 198],
        'conversions': [12, 18, 6, 24, 4, 13, 20, 26],
        'channel': ['Email', 'Social', 'Content', 'Paid', 'Event', 'Email', 'Social', 'Paid'],
    }

    df_campaigns = pd.DataFrame(campaign_data)

    # Calculate metrics
    df_campaigns['ctr'] = df_campaigns['clicks'] / df_campaigns['impressions']
    df_campaigns['cost_per_lead'] = df_campaigns['budget'] / df_campaigns['leads_generated']
    df_campaigns['conversion_rate'] = df_campaigns['conversions'] / df_campaigns['leads_generated']
    df_campaigns['roi'] = ((df_campaigns['conversions'] * 6500) - df_campaigns['budget']) / df_campaigns['budget']

    # Channel performance summary
    channel_summary = df_campaigns.groupby('channel').agg({
        'budget': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'leads_generated': 'sum',
        'conversions': 'sum'
    }).reset_index()

    channel_summary['ctr'] = channel_summary['clicks'] / channel_summary['impressions']
    channel_summary['cost_per_lead'] = channel_summary['budget'] / channel_summary['leads_generated']

    output_path = Path("test_data/Marketing/campaign_performance_2024.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_campaigns.to_excel(writer, sheet_name='Campaign Details', index=False)
        channel_summary.to_excel(writer, sheet_name='Channel Summary', index=False)

    print(f"[OK] Created: {output_path} (2 sheets: {len(df_campaigns)} campaigns)")
    return df_campaigns


def create_operations_csv():
    """Create CSV with operational metrics (easy-medium complexity)."""
    print("Creating Operations CSV data...")

    # Generate weekly operational data
    weeks = 24

    operations_data = {
        'week_number': range(1, weeks + 1),
        'week_start_date': [datetime(2024, 1, 1) + timedelta(weeks=x) for x in range(weeks)],
        'support_tickets': np.random.poisson(85, weeks),
        'avg_response_time_hours': np.random.normal(4.2, 1.8, weeks).clip(min=1, max=12),
        'avg_resolution_time_hours': np.random.normal(18, 6, weeks).clip(min=4, max=48),
        'customer_satisfaction_score': np.random.normal(4.1, 0.5, weeks).clip(min=1, max=5),
        'first_contact_resolution_rate': np.random.beta(7, 3, weeks),
        'ticket_escalations': np.random.poisson(12, weeks),
        'system_uptime_percent': np.random.normal(99.2, 0.6, weeks).clip(min=95, max=100),
        'server_response_time_ms': np.random.normal(245, 75, weeks).clip(min=100, max=800),
        'error_rate_percent': np.random.normal(0.8, 0.4, weeks).clip(min=0, max=3),
    }

    df_operations = pd.DataFrame(operations_data)

    # Add trend - improving over time
    df_operations['avg_response_time_hours'] *= (1 - df_operations.index / 100)
    df_operations['customer_satisfaction_score'] *= (1 + df_operations.index / 200)

    output_path = Path("test_data/Operations/operational_metrics_2024.csv")
    df_operations.to_csv(output_path, index=False)
    print(f"[OK] Created: {output_path} ({len(df_operations)} weeks)")
    return df_operations


def create_customer_feedback_pdf():
    """Create PDF with customer feedback summary (medium complexity)."""
    print("Creating Customer Feedback PDF...")

    output_path = Path("test_data/Customer_Success/customer_feedback_summary.pdf")
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("<b>Customer Feedback Summary Report</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3*inch))

    # Report date
    report_date = Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal'])
    story.append(report_date)
    story.append(Spacer(1, 0.2*inch))

    # Executive Summary
    summary_title = Paragraph("<b>Executive Summary</b>", styles['Heading1'])
    story.append(summary_title)

    summary_text = """This report analyzes customer feedback collected over the past quarter.
    Key findings indicate moderate satisfaction levels with opportunities for improvement in
    product features and customer support responsiveness. Overall NPS score is 7.2/10,
    indicating good customer loyalty but room for enhancement."""

    summary_para = Paragraph(summary_text, styles['BodyText'])
    story.append(summary_para)
    story.append(Spacer(1, 0.3*inch))

    # Key Metrics Table
    metrics_title = Paragraph("<b>Key Metrics</b>", styles['Heading2'])
    story.append(metrics_title)
    story.append(Spacer(1, 0.1*inch))

    metrics_data = [
        ['Metric', 'Value', 'Trend'],
        ['Net Promoter Score (NPS)', '7.2/10', '+0.3'],
        ['Customer Satisfaction (CSAT)', '8.1/10', '+0.5'],
        ['Customer Effort Score (CES)', '6.8/10', '-0.2'],
        ['Response Rate', '42%', '+5%'],
        ['Total Responses', '186', '+12'],
    ]

    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Top Pain Points
    pain_points_title = Paragraph("<b>Top Customer Pain Points</b>", styles['Heading2'])
    story.append(pain_points_title)
    story.append(Spacer(1, 0.1*inch))

    pain_points = [
        "1. <b>Feature Limitations</b> - 38% of respondents mentioned lack of advanced features",
        "2. <b>Support Response Time</b> - 32% concerned about slow support ticket resolution",
        "3. <b>User Interface</b> - 28% found the interface not intuitive enough",
        "4. <b>Integration Issues</b> - 24% experienced difficulties with third-party integrations",
        "5. <b>Performance</b> - 18% reported occasional slowness during peak hours"
    ]

    for point in pain_points:
        story.append(Paragraph(point, styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.2*inch))

    # Feature Requests
    features_title = Paragraph("<b>Most Requested Features</b>", styles['Heading2'])
    story.append(features_title)
    story.append(Spacer(1, 0.1*inch))

    features_data = [
        ['Feature Request', 'Mentions', 'Priority'],
        ['Advanced Analytics Dashboard', '72', 'High'],
        ['Mobile Application', '58', 'High'],
        ['API Rate Limit Increase', '45', 'Medium'],
        ['Real-time Collaboration', '42', 'Medium'],
        ['Dark Mode UI', '38', 'Low'],
    ]

    features_table = Table(features_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
    features_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(features_table)
    story.append(Spacer(1, 0.3*inch))

    # Recommendations
    rec_title = Paragraph("<b>Recommendations</b>", styles['Heading2'])
    story.append(rec_title)
    story.append(Spacer(1, 0.1*inch))

    recommendations = """Based on the feedback analysis, we recommend:

    1. Prioritize development of advanced analytics features to address the #1 customer request
    2. Implement SLA improvements to reduce support response times by 30%
    3. Conduct UX research to redesign key interface elements
    4. Increase engineering resources for integration stability
    5. Schedule quarterly customer feedback sessions to maintain engagement
    """

    rec_para = Paragraph(recommendations, styles['BodyText'])
    story.append(rec_para)

    # Build PDF
    doc.build(story)
    print(f"[OK] Created: {output_path}")


def create_quarterly_report_pdf():
    """Create PDF with quarterly business report (medium complexity)."""
    print("Creating Quarterly Report PDF...")

    output_path = Path("test_data/Sales/quarterly_business_report_Q1_2024.pdf")
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("<b>Q1 2024 Business Performance Report</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3*inch))

    # Executive Summary
    summary_title = Paragraph("<b>Executive Summary</b>", styles['Heading1'])
    story.append(summary_title)

    summary_text = """Q1 2024 showed strong performance with revenue growth of 18% YoY.
    Sales team closed 156 deals with an average deal size of $6,800. Conversion rates improved
    from 8% to 8.7%, though sales cycle time remains elevated at 87 days average.
    Key challenges include lead quality and sales cycle optimization."""

    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # Key Metrics
    metrics_title = Paragraph("<b>Q1 Performance Metrics</b>", styles['Heading2'])
    story.append(metrics_title)
    story.append(Spacer(1, 0.1*inch))

    metrics_data = [
        ['Metric', 'Q1 2024', 'Q4 2023', 'Change'],
        ['Total Revenue', '$3.92M', '$3.45M', '+13.6%'],
        ['New Customers', '156', '142', '+9.9%'],
        ['Avg Deal Size', '$6,800', '$6,200', '+9.7%'],
        ['Conversion Rate', '8.7%', '8.0%', '+0.7pp'],
        ['Sales Cycle (days)', '87', '82', '+5 days'],
        ['Pipeline Value', '$8.2M', '$7.4M', '+10.8%'],
    ]

    table = Table(metrics_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    # Challenges
    challenges_title = Paragraph("<b>Key Challenges</b>", styles['Heading2'])
    story.append(challenges_title)
    story.append(Spacer(1, 0.1*inch))

    challenges = """
    <b>1. Extended Sales Cycles:</b> Average cycle increased to 87 days, impacting revenue velocity.
    Requires better qualification and faster decision-making processes.

    <b>2. Lead Quality Issues:</b> Only 8.7% of leads converting. Need improved lead scoring
    and better alignment between marketing and sales on ICP.

    <b>3. Regional Performance Variance:</b> West region outperforming by 34% compared to East region.
    Requires investigation into local market dynamics and sales rep effectiveness.
    """

    story.append(Paragraph(challenges, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))

    # Build PDF
    doc.build(story)
    print(f"[OK] Created: {output_path}")


def main():
    """Create all test data files."""
    print("\n" + "="*60)
    print("CREATING TEST DATA FOR ETL TO INSIGHTS AI AGENT")
    print("="*60)
    print("\nDifficulty: Easy-Medium")
    print("Data Types: CSV, Excel, PDF")
    print("Departments: Sales, Marketing, Customer Success, Operations")
    print("="*60 + "\n")

    # Create directories
    for dept in ['Sales', 'Marketing', 'Customer_Success', 'Operations']:
        Path(f"test_data/{dept}").mkdir(parents=True, exist_ok=True)

    # Generate data
    create_sales_csv()
    create_customer_success_excel()
    create_marketing_excel()
    create_operations_csv()

    # Create PDFs
    try:
        create_customer_feedback_pdf()
        create_quarterly_report_pdf()
    except ImportError:
        print("\n[WARNING] reportlab not installed. Skipping PDF generation.")
        print("  Install with: pip install reportlab")
    except Exception as e:
        print(f"\n[WARNING] PDF generation failed: {e}")

    print("\n" + "="*60)
    print("[SUCCESS] TEST DATA CREATION COMPLETE")
    print("="*60)
    print("\nCreated files:")
    print("  test_data/Sales/")
    print("    - sales_performance_2024.csv (180 days)")
    print("    - quarterly_business_report_Q1_2024.pdf")
    print("  test_data/Marketing/")
    print("    - campaign_performance_2024.xlsx (2 sheets)")
    print("  test_data/Customer_Success/")
    print("    - customer_health_metrics.xlsx (2 sheets, 300 customers)")
    print("    - customer_feedback_summary.pdf")
    print("  test_data/Operations/")
    print("    - operational_metrics_2024.csv (24 weeks)")

    print("\n[INFO] Data Characteristics:")
    print("  - Realistic business patterns (trends, seasonality)")
    print("  - Multiple data formats (CSV, Excel multi-sheet, PDF)")
    print("  - Easy-Medium difficulty (clear patterns, some noise)")
    print("  - Cross-departmental relationships")

    print("\n[NEXT] Next step: Move test_data to data/uploads")
    print("   mv test_data/* data/uploads/")
    print("\n   Then run: python main.py --phase1")


if __name__ == "__main__":
    main()

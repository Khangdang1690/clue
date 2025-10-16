"""Interactive dashboard generator using Plotly with modern 2025 design principles."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from src.models.analysis_result import AnalysisResult, StatisticalTest
from src.models.business_context import BusinessContext


class DashboardGenerator:
    """Generates interactive HTML dashboards using Plotly with modern styling."""

    # Modern color palette (2025 design trends)
    COLORS = {
        'primary': '#6366f1',      # Vibrant indigo
        'success': '#10b981',      # Modern green
        'warning': '#f59e0b',      # Warm amber
        'danger': '#ef4444',       # Soft red
        'info': '#3b82f6',         # Bright blue
        'secondary': '#8b5cf6',    # Purple
        'bg_dark': '#0f172a',      # Slate dark
        'bg_light': '#f8fafc',     # Slate light
        'text_primary': '#1e293b', # Slate 800
        'text_secondary': '#64748b', # Slate 500
        'border': '#e2e8f0',       # Slate 200
        'gradient_start': '#6366f1',
        'gradient_end': '#8b5cf6'
    }

    def __init__(self, output_directory: str = "data/outputs/dashboards"):
        """
        Initialize dashboard generator.

        Args:
            output_directory: Directory to save dashboards
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult],
        data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> str:
        """
        Generate a modern interactive HTML dashboard.

        Args:
            business_context: Business context information
            analysis_results: List of analysis results
            data: Optional dictionary of DataFrames for detailed charts

        Returns:
            Path to generated HTML dashboard
        """
        print("\n🎨 Generating modern interactive dashboard...")

        # Create main figure with custom layout
        fig = self._create_modern_dashboard(business_context, analysis_results, data)

        # Save to HTML with custom styling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_dashboard_{timestamp}.html"
        filepath = self.output_directory / filename

        # Generate custom HTML with modern styling
        html_content = self._generate_custom_html(fig, business_context, analysis_results)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ Dashboard saved to: {filepath}")
        return str(filepath)

    def _create_modern_dashboard(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult],
        data: Optional[Dict[str, pd.DataFrame]]
    ) -> go.Figure:
        """Create modern dashboard with clean layout."""

        # Calculate KPIs
        total_challenges = len(analysis_results)
        total_tests = sum(len(r.statistical_tests) for r in analysis_results)
        significant_tests = sum(1 for r in analysis_results for t in r.statistical_tests if t.is_significant)
        total_findings = sum(len(r.key_findings) for r in analysis_results)
        significance_rate = (significant_tests / total_tests * 100) if total_tests > 0 else 0

        # Create subplot layout with better spacing
        fig = make_subplots(
            rows=4, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'bar', 'colspan': 2}, None, {'type': 'indicator'}],
                [{'type': 'scatter', 'colspan': 2}, None, {'type': 'pie'}],
                [{'type': 'bar', 'colspan': 3}, None, None]
            ],
            row_heights=[0.15, 0.25, 0.35, 0.25],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=['', '', '', 'Analysis Performance', '', 'Significance<br>Rate',
                          'Insights Distribution', '', 'Data Coverage', 'Recommendations by Challenge']
        )

        # Row 1: KPI Cards
        self._add_kpi_card(fig, "Challenges Analyzed", total_challenges, self.COLORS['primary'], row=1, col=1)
        self._add_kpi_card(fig, "Total Findings", total_findings, self.COLORS['success'], row=1, col=2)
        self._add_kpi_card(fig, "Statistical Tests", total_tests, self.COLORS['info'], row=1, col=3)

        # Row 2: Analysis Performance & Significance Gauge
        self._add_modern_bar_chart(fig, analysis_results, row=2, col=1)
        self._add_modern_gauge(fig, significance_rate, significant_tests, total_tests, row=2, col=3)

        # Row 3: Scatter Plot & Pie Chart
        self._add_insights_scatter(fig, analysis_results, row=3, col=1)
        self._add_modern_pie_chart(fig, analysis_results, row=3, col=3)

        # Row 4: Recommendations Bar
        self._add_recommendations_chart(fig, analysis_results, row=4, col=1)

        # Update overall layout with modern styling
        fig.update_layout(
            template='plotly_white',
            font=dict(
                family="'Inter', 'Segoe UI', Roboto, system-ui, -apple-system, sans-serif",
                size=13,
                color=self.COLORS['text_primary']
            ),
            paper_bgcolor=self.COLORS['bg_light'],
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=120, b=40),
            height=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=self.COLORS['border'],
                borderwidth=1
            ),
            title={
                'text': f"<b style='color:{self.COLORS['primary']};font-size:32px'>{business_context.company_name}</b><br>"
                        f"<span style='color:{self.COLORS['text_secondary']};font-size:16px;font-weight:400'>Business Intelligence Dashboard</span><br>"
                        f"<span style='color:{self.COLORS['text_secondary']};font-size:14px;font-weight:300'>{business_context.current_goal}</span>",
                'x': 0.02,
                'xanchor': 'left',
                'y': 0.98,
                'yanchor': 'top'
            },
            hovermode='closest'
        )

        # Update subplot titles styling
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color=self.COLORS['text_primary'], family="Inter")

        return fig

    def _add_kpi_card(self, fig: go.Figure, title: str, value: float, color: str, row: int, col: int):
        """Add modern KPI card."""
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={
                    'text': f"<b>{title}</b>",
                    'font': {'size': 14, 'color': self.COLORS['text_secondary']}
                },
                number={
                    'font': {'size': 42, 'color': color, 'family': 'Inter'},
                    'valueformat': '.0f'
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=row, col=col
        )

    def _add_modern_bar_chart(self, fig: go.Figure, analysis_results: List[AnalysisResult], row: int, col: int):
        """Add modern horizontal bar chart for tests vs findings."""
        challenge_titles = [r.challenge_title[:40] + '...' if len(r.challenge_title) > 40 else r.challenge_title
                           for r in analysis_results]
        test_counts = [len(r.statistical_tests) for r in analysis_results]
        findings_counts = [len(r.key_findings) for r in analysis_results]

        # Statistical Tests
        fig.add_trace(
            go.Bar(
                y=challenge_titles,
                x=test_counts,
                name='Statistical Tests',
                orientation='h',
                marker=dict(
                    color=self.COLORS['primary'],
                    line=dict(color='rgba(255,255,255,0.8)', width=2)
                ),
                text=test_counts,
                textposition='auto',
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{y}</b><br>Tests: %{x}<extra></extra>'
            ),
            row=row, col=col
        )

        # Key Findings
        fig.add_trace(
            go.Bar(
                y=challenge_titles,
                x=findings_counts,
                name='Key Findings',
                orientation='h',
                marker=dict(
                    color=self.COLORS['success'],
                    line=dict(color='rgba(255,255,255,0.8)', width=2)
                ),
                text=findings_counts,
                textposition='auto',
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{y}</b><br>Findings: %{x}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(
            title_text="Count",
            showgrid=True,
            gridcolor=self.COLORS['border'],
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=False,
            row=row, col=col
        )

    def _add_modern_gauge(self, fig: go.Figure, value: float, significant: int, total: int, row: int, col: int):
        """Add modern gauge chart."""
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={
                    'text': f"<b>{significant}/{total}</b> significant",
                    'font': {'size': 13, 'color': self.COLORS['text_secondary']}
                },
                number={
                    'suffix': "%",
                    'font': {'size': 32, 'color': self.COLORS['primary']}
                },
                gauge={
                    'axis': {
                        'range': [None, 100],
                        'tickwidth': 1,
                        'tickcolor': self.COLORS['text_secondary']
                    },
                    'bar': {'color': self.COLORS['primary'], 'thickness': 0.75},
                    'bgcolor': self.COLORS['bg_light'],
                    'borderwidth': 2,
                    'bordercolor': self.COLORS['border'],
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': self.COLORS['success'], 'width': 3},
                        'thickness': 0.8,
                        'value': 70
                    }
                }
            ),
            row=row, col=col
        )

    def _add_insights_scatter(self, fig: go.Figure, analysis_results: List[AnalysisResult], row: int, col: int):
        """Add scatter plot for correlations vs causality insights."""
        challenge_titles = [r.challenge_title[:30] + '...' if len(r.challenge_title) > 30 else r.challenge_title
                           for r in analysis_results]
        correlations_count = [len(r.correlations) for r in analysis_results]
        causality_count = [len(r.causality_insights) for r in analysis_results]
        recommendations_count = [len(r.recommendations) for r in analysis_results]

        # Size bubbles by recommendations
        sizes = [max(20, count * 15) for count in recommendations_count]

        fig.add_trace(
            go.Scatter(
                x=correlations_count,
                y=causality_count,
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=recommendations_count,
                    colorscale=[[0, self.COLORS['info']], [1, self.COLORS['secondary']]],
                    showscale=True,
                    colorbar=dict(
                        title="Recommendations",
                        x=0.65,
                        len=0.3,
                        thickness=15
                    ),
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                text=[t[:15] + '...' if len(t) > 15 else t for t in challenge_titles],
                textposition='top center',
                textfont=dict(size=10, color=self.COLORS['text_primary']),
                hovertemplate='<b>%{text}</b><br>Correlations: %{x}<br>Causality: %{y}<br>Recommendations: %{marker.color}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(
            title_text="Correlations Found",
            showgrid=True,
            gridcolor=self.COLORS['border'],
            zeroline=True,
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="Causality Insights",
            showgrid=True,
            gridcolor=self.COLORS['border'],
            zeroline=True,
            row=row, col=col
        )

    def _add_modern_pie_chart(self, fig: go.Figure, analysis_results: List[AnalysisResult], row: int, col: int):
        """Add modern donut chart for data sources."""
        data_source_counts = {}

        for result in analysis_results:
            for source in result.data_sources_used:
                # Simplify source name
                simple_name = source.replace('_data.csv', '').replace('_', ' ').title()
                data_source_counts[simple_name] = data_source_counts.get(simple_name, 0) + 1

        if not data_source_counts:
            data_source_counts = {'No Data': 1}

        # Modern color palette
        colors = [self.COLORS['primary'], self.COLORS['success'], self.COLORS['warning'],
                 self.COLORS['danger'], self.COLORS['info'], self.COLORS['secondary']]

        fig.add_trace(
            go.Pie(
                labels=list(data_source_counts.keys()),
                values=list(data_source_counts.values()),
                hole=0.5,
                marker=dict(
                    colors=colors[:len(data_source_counts)],
                    line=dict(color='white', width=3)
                ),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=12, color=self.COLORS['text_primary']),
                hovertemplate='<b>%{label}</b><br>Used: %{value} times<br>%{percent}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_recommendations_chart(self, fig: go.Figure, analysis_results: List[AnalysisResult], row: int, col: int):
        """Add waterfall-style recommendations chart."""
        challenge_titles = [r.challenge_title[:35] + '...' if len(r.challenge_title) > 35 else r.challenge_title
                           for r in analysis_results]
        recommendations_counts = [len(r.recommendations) for r in analysis_results]

        # Create gradient colors
        max_count = max(recommendations_counts) if recommendations_counts else 1
        colors = [f'rgba({99 + (139-99)*c/max_count}, {102 + (92-102)*c/max_count}, {241 + (246-241)*c/max_count}, 0.8)'
                 for c in recommendations_counts]

        fig.add_trace(
            go.Bar(
                x=challenge_titles,
                y=recommendations_counts,
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=recommendations_counts,
                textposition='outside',
                textfont=dict(size=13, color=self.COLORS['text_primary']),
                hovertemplate='<b>%{x}</b><br>Recommendations: %{y}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(
            title_text="",
            tickangle=-35,
            showgrid=False,
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="Recommendations",
            showgrid=True,
            gridcolor=self.COLORS['border'],
            row=row, col=col
        )

    def _generate_custom_html(
        self,
        fig: go.Figure,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult]
    ) -> str:
        """Generate custom HTML with modern styling."""

        print(f"\n🎨 Generating dashboard HTML...")
        print(f"  ✓ Figure contains {len(fig.data)} traces")

        # Convert figure to HTML - use full_html=False to get just the div
        # This is more reliable than regex extraction
        plotly_div = fig.to_html(
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'dashboard_{datetime.now().strftime("%Y%m%d")}',
                    'height': 1400,
                    'width': 1800,
                    'scale': 2
                }
            },
            include_plotlyjs=False,  # We'll include it via CDN in header
            full_html=False,         # Just the div, not full HTML structure
            div_id='dashboard-container'
        )

        # Debug: Check if div was generated successfully
        print(f"  ✓ Plotly div generated: {len(plotly_div)} characters")
        if len(plotly_div) < 500:
            print(f"  ⚠️  WARNING: Plotly div suspiciously short! May be empty.")
            print(f"  Preview: {plotly_div[:200]}...")
        else:
            print(f"  ✅ Plotly div looks good (first 100 chars): {plotly_div[:100]}...")

        # No need for separate script - Plotly's full_html=False includes everything in the div
        plotly_script = ""  # Empty - not needed with full_html=False

        # Create custom HTML with modern design
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{business_context.company_name} - Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        :root {{
            --color-primary: #6366f1;
            --color-secondary: #8b5cf6;
            --color-success: #10b981;
            --color-info: #3b82f6;
            --color-warning: #f59e0b;
            --color-danger: #ef4444;
            --bg-page: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
            --bg-surface: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --border: #e2e8f0;
            --shadow-lg: 0 20px 60px rgba(2, 6, 23, 0.12);
            --chip-bg: rgba(99, 102, 241, 0.1);
        }}

        [data-theme="dark"] {{
            --bg-page: linear-gradient(135deg, #0b1220 0%, #0f172a 100%);
            --bg-surface: #0b1220;
            --text-primary: #e5e7eb;
            --text-secondary: #94a3b8;
            --border: #1f2937;
            --shadow-lg: 0 20px 60px rgba(0,0,0,0.4);
            --chip-bg: rgba(139, 92, 246, 0.15);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-page);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
            transition: background 300ms ease;
        }}

        .dashboard-wrapper {{
            max-width: 1600px;
            margin: 0 auto;
            background: var(--bg-surface);
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        .dashboard-header {{
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
            padding: 28px 32px;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .header-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .brand {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .dashboard-header h1 {{
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }}

        .dashboard-header p {{
            font-size: 14px;
            opacity: 0.95;
            font-weight: 400;
        }}

        .header-actions {{
            display: flex;
            gap: 10px;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.25);
            color: white;
            background: rgba(255,255,255,0.08);
            cursor: pointer;
            backdrop-filter: blur(8px);
            transition: all 150ms ease;
        }}
        .btn:hover {{ background: rgba(255,255,255,0.16); }}

        .dashboard-meta {{
            display: flex;
            gap: 12px;
            margin-top: 14px;
            flex-wrap: wrap;
        }}

        .meta-item {{
            background: var(--chip-bg);
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.25);
        }}

        .dashboard-content {{
            padding: 24px 28px 28px 28px;
        }}

        .chart-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 12px;
            margin-bottom: 12px;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.4));
        }}

        .toolbar-actions {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .toolbar-btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-primary);
            background: #fff;
            cursor: pointer;
            transition: background 150ms ease, transform 50ms ease;
        }}
        .toolbar-btn:hover {{ background: #f8fafc; }}
        .toolbar-btn:active {{ transform: translateY(1px); }}

        #dashboard-container {{ width: 100%; }}

        .footer {{
            text-align: center;
            padding: 16px 20px;
            color: var(--text-secondary);
            font-size: 12px;
            background: linear-gradient(180deg, rgba(248,250,252,0.8), rgba(248,250,252,1));
            border-top: 1px solid var(--border);
        }}

        @media (max-width: 768px) {{
            .dashboard-header h1 {{ font-size: 22px; }}
        }}

        @media print {{
            body {{ background: #fff; }}
            .dashboard-wrapper {{ box-shadow: none; border: none; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-wrapper" id="dashboard-wrapper">
        <div class="dashboard-header">
            <div class="header-row">
                <div class="brand">
                    <h1>{business_context.company_name} — Business Intelligence Dashboard</h1>
                    <p>{business_context.current_goal}</p>
                </div>
                <div class="header-actions">
                    <button class="btn" id="theme-toggle" aria-label="Toggle theme">🌙 Theme</button>
                </div>
            </div>
            <div class="dashboard-meta">
                <div class="meta-item">📊 {len(analysis_results)} Challenges</div>
                <div class="meta-item">🔬 {sum(len(r.statistical_tests) for r in analysis_results)} Tests</div>
                <div class="meta-item">💡 {sum(len(r.key_findings) for r in analysis_results)} Findings</div>
                <div class="meta-item">📅 {datetime.now().strftime("%b %d, %Y • %I:%M %p")}</div>
            </div>
        </div>

        <div class="dashboard-content">
            <div class="chart-toolbar">
                <span style="font-size:13px;color:var(--text-secondary)">Interactive dashboard</span>
                <div class="toolbar-actions">
                    <button class="toolbar-btn" id="btn-download" aria-label="Download PNG">⬇️ Download</button>
                    <button class="toolbar-btn" id="btn-reset" aria-label="Reset view">↺ Reset</button>
                    <button class="toolbar-btn" id="btn-fullscreen" aria-label="Toggle fullscreen">⤢ Fullscreen</button>
                </div>
            </div>
            {plotly_div}
        </div>

        <div class="footer">
            <p>Generated by ETL to Insights AI Agent · {business_context.company_name}</p>
            <p style="margin-top: 4px; font-size: 11px;">Hover, zoom, and click legends to explore</p>
        </div>
    </div>

    <script>
        (function() {{
            const gd = document.getElementById('dashboard-container');
            const wrapper = document.getElementById('dashboard-wrapper');
            const btnDownload = document.getElementById('btn-download');
            const btnReset = document.getElementById('btn-reset');
            const btnFullscreen = document.getElementById('btn-fullscreen');
            const btnTheme = document.getElementById('theme-toggle');

            // Apply persisted theme
            const storedTheme = localStorage.getItem('dashboard-theme');
            if (storedTheme === 'dark') {{ document.documentElement.setAttribute('data-theme', 'dark'); }}

            btnTheme?.addEventListener('click', function() {{
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                if (isDark) {{
                    document.documentElement.removeAttribute('data-theme');
                    localStorage.setItem('dashboard-theme', 'light');
                }} else {{
                    document.documentElement.setAttribute('data-theme', 'dark');
                    localStorage.setItem('dashboard-theme', 'dark');
                }}
            }});

            btnDownload?.addEventListener('click', function() {{
                if (!window.Plotly || !gd) return;
                window.Plotly.downloadImage(gd, {{
                    format: 'png',
                    filename: '{business_context.company_name.lower().replace(" ", "_")}_dashboard_{datetime.now().strftime("%Y%m%d")}',
                    height: gd.offsetHeight,
                    width: gd.offsetWidth,
                    scale: 2
                }});
            }});

            btnReset?.addEventListener('click', function() {{
                if (!window.Plotly || !gd) return;
                const update = {{}};
                const layout = gd.layout || {{}};
                Object.keys(layout).forEach(function(key) {{
                    if (key.startsWith('xaxis') || key.startsWith('yaxis')) {{
                        update[key + '.autorange'] = true;
                    }}
                }});
                window.Plotly.relayout(gd, update);
            }});

            btnFullscreen?.addEventListener('click', function() {{
                if (!document.fullscreenElement) {{
                    wrapper.requestFullscreen?.();
                }} else {{
                    document.exitFullscreen?.();
                }}
            }});

            // Ensure responsive redraw
            window.addEventListener('resize', function() {{
                if (!window.Plotly || !gd) return;
                window.Plotly.Plots.resize(gd);
            }});
        }})();
    </script>
</body>
</html>"""

        return html

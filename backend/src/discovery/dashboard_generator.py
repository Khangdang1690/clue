"""Generate interactive HTML dashboards from discovery results."""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import json
import re
import os
from src.models.discovery_models import DiscoveryResult, AnsweredQuestion


class DashboardGenerator:
    """Generates interactive HTML dashboards with Chart.js visualizations."""

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize dashboard generator.

        Args:
            template_path: Path to HTML template file (if None, uses default in backend/template/)
        """
        if template_path is None:
            # Get the backend directory (where main.py is)
            # This file is in: backend/src/discovery/dashboard_generator.py
            current_file = Path(__file__)  # dashboard_generator.py
            backend_dir = current_file.parent.parent.parent  # Go up 3 levels to backend/
            template_path = backend_dir / "template" / "dashboard.html"

        self.template_path = Path(template_path)
        print(f"[DASHBOARD] Template path resolved to: {self.template_path.resolve()}")

    def generate_dashboard(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate interactive HTML dashboard from discovery results.

        Args:
            result: DiscoveryResult object with insights
            dataset_context: Optional context from outer agent layer
            output_path: Optional custom output path

        Returns:
            Path to generated HTML dashboard
        """
        print("\n[DASHBOARD] Generating interactive HTML dashboard...")

        # Build dashboard components
        title = f"Business Insights Report - {result.dataset_name}"
        subtitle = self._build_subtitle(dataset_context)
        generated_date = datetime.now().strftime("%B %d, %Y")
        exec_summary = self._build_exec_summary(result, dataset_context)
        stats = self._build_stats(result, dataset_context)
        insights_html, chart_scripts = self._build_insights(result)

        # Build complete HTML from scratch using template styles
        html = self._build_complete_html(
            title=title,
            subtitle=subtitle,
            generated_date=generated_date,
            exec_summary=exec_summary,
            stats=stats,
            insights_html=insights_html,
            chart_scripts=chart_scripts
        )

        # Save dashboard
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/outputs/discovery")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"discovery_dashboard_{timestamp}.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"[SUCCESS] Interactive dashboard saved to: {output_path}")
        return str(output_path)

    def _load_template(self) -> str:
        """Load HTML template file."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_subtitle(self, dataset_context: Optional[Dict]) -> str:
        """Build subtitle from dataset context."""
        if not dataset_context:
            return "AI-Powered Data Discovery"

        domain = dataset_context.get('domain', 'Unknown')
        dataset_type = dataset_context.get('dataset_type', 'Data Analysis')
        return f"{domain} | {dataset_type}"

    def _build_exec_summary(self, result: DiscoveryResult, dataset_context: Optional[Dict]) -> str:
        """Build executive summary text."""
        # Get first 3 key insights
        insights_text = ""
        if result.key_insights:
            top_insights = result.key_insights[:3]
            insights_text = " ".join([self._clean_insight_text(i) for i in top_insights])

        if not insights_text:
            insights_text = f"Analysis of {result.data_profile.num_rows:,} records revealed {len(result.answered_questions)} key insights about patterns and trends in the data."

        return insights_text

    def _clean_insight_text(self, insight: str) -> str:
        """Clean insight text for display."""
        # Remove confidence prefix like "[85%]"
        clean = re.sub(r'\[\d+%\]\s*', '', insight)
        return clean.strip()

    def _build_stats(self, result: DiscoveryResult, dataset_context: Optional[Dict]) -> Dict[str, Dict[str, str]]:
        """Build stats for stats bar."""
        # Calculate time coverage
        time_coverage = "N/A"
        if dataset_context and dataset_context.get('time_period'):
            time_coverage = dataset_context['time_period']

        return {
            'stat1': {
                'value': f"{len(result.answered_questions)}",
                'label': "Insights Found"
            },
            'stat2': {
                'value': f"{(1 - result.data_profile.overall_missing_rate):.0%}",
                'label': "Data Complete"
            },
            'stat3': {
                'value': f"{result.data_profile.num_rows:,}",
                'label': "Records"
            },
            'stat4': {
                'value': time_coverage[:10],  # Truncate if too long
                'label': "Time Period"
            }
        }

    def _build_insights(self, result: DiscoveryResult) -> tuple[str, str]:
        """
        Build insights HTML and chart scripts.

        Returns:
            (insights_html, chart_scripts)
        """
        insights_html_parts = []
        chart_scripts_parts = []

        for idx, question in enumerate(result.answered_questions, 1):
            # Build insight card
            card_html = self._build_insight_card(idx, question)
            insights_html_parts.append(card_html)

            # Build chart script if data available
            chart_script = self._build_chart_script(idx, question)
            if chart_script:
                chart_scripts_parts.append(chart_script)

        # Wrap insights in sections
        insights_html = f'''
            <div class="section-header"><div class="section-icon">📊</div><h2>Key Insights</h2></div>
            <div class="viz-grid">
                {"".join(insights_html_parts)}
            </div>
        '''

        chart_scripts = "\n".join(chart_scripts_parts)

        return insights_html, chart_scripts

    def _build_insight_card(self, idx: int, question: AnsweredQuestion) -> str:
        """Build HTML for a single insight card."""
        # Determine chart type badge
        chart_type = self._detect_chart_type(question.supporting_data)

        # Clean and format description
        description = self._clean_insight_text(question.answer)
        if len(description) > 300:
            description = description[:297] + "..."

        card_html = f'''
                <div class="insight-card">
                    <div class="insight-header">
                        <div class="insight-title-group">
                            <div class="insight-number">{idx}</div>
                            <div class="insight-title">{question.question}</div>
                        </div>
                        <div class="chart-type-badge">{chart_type}</div>
                    </div>
                    <div class="insight-body">
                        <p class="insight-description">{description}</p>
                        <div class="insight-visualization"><canvas id="chart_{idx}"></canvas></div>
                    </div>
                </div>
        '''
        return card_html

    def _detect_chart_type(self, supporting_data: Optional[Dict[str, Any]]) -> str:
        """Detect appropriate chart type from supporting data structure."""
        if not supporting_data or not isinstance(supporting_data, dict):
            return "Data"

        # Check for explicit chart_type
        if 'chart_type' in supporting_data:
            return supporting_data['chart_type'].title()

        # Detect from data structure
        if 'datasets' in supporting_data:
            num_datasets = len(supporting_data.get('datasets', []))
            if num_datasets > 1:
                return "Multi-Series"
            return "Bar Chart"

        if 'x' in supporting_data and 'y' in supporting_data:
            return "Scatter"

        if 'percentages' in supporting_data or 'shares' in supporting_data:
            return "Pie Chart"

        return "Chart"

    def _build_chart_script(self, idx: int, question: AnsweredQuestion) -> Optional[str]:
        """
        Build Chart.js initialization script for an insight.

        Returns:
            JavaScript code or None if no chart data
        """
        if not question.supporting_data or not isinstance(question.supporting_data, dict):
            return None

        data = question.supporting_data

        # Try to convert to Chart.js format
        chart_config = self._convert_to_chartjs_config(data, question.question)

        if not chart_config:
            return None

        # Generate Chart.js initialization
        script = f"new Chart(document.getElementById('chart_{idx}'), {json.dumps(chart_config)});"
        return script

    def _convert_to_chartjs_config(self, data: Dict[str, Any], title: str) -> Optional[Dict]:
        """
        Convert supporting_data to Chart.js configuration.

        Handles various data formats from agent code execution.
        """
        # Format 1: Explicit Chart.js format
        if 'chart_type' in data and 'labels' in data and 'datasets' in data:
            return {
                'type': data['chart_type'],
                'data': {
                    'labels': data['labels'],
                    'datasets': data['datasets']
                },
                'options': data.get('options', {'responsive': True, 'plugins': {'legend': {'display': True}}})
            }

        # Format 2: Simple key-value pairs (for bar chart)
        if all(isinstance(v, (int, float)) for v in data.values() if v is not None):
            labels = list(data.keys())[:10]  # Limit to 10 items
            values = [data[k] for k in labels]

            return {
                'type': 'bar',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'data': values,
                        'backgroundColor': 'rgba(102,126,234,0.8)',
                        'borderRadius': 6
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {'legend': {'display': False}}
                }
            }

        # Format 3: Time series data
        if 'dates' in data or 'time' in data:
            time_key = 'dates' if 'dates' in data else 'time'
            value_key = 'values' if 'values' in data else list(data.keys())[1] if len(data) > 1 else None

            if value_key:
                return {
                    'type': 'line',
                    'data': {
                        'labels': data[time_key][:50],  # Limit points
                        'datasets': [{
                            'label': value_key.title(),
                            'data': data[value_key][:50],
                            'borderColor': 'rgba(102,126,234,1)',
                            'borderWidth': 3,
                            'tension': 0.4
                        }]
                    },
                    'options': {'responsive': True}
                }

        # No recognizable format
        return None

    def _build_complete_html(
        self,
        title: str,
        subtitle: str,
        generated_date: str,
        exec_summary: str,
        stats: Dict,
        insights_html: str,
        chart_scripts: str
    ) -> str:
        """Build complete HTML document from template structure."""

        # Read template to copy the CSS
        template_css = self._extract_css_from_template()

        # Build stats bar HTML
        stats_html = ""
        for stat in ['stat1', 'stat2', 'stat3', 'stat4']:
            if stat in stats:
                stats_html += f'''
            <div class="stat-item"><div class="stat-value">{stats[stat]["value"]}</div><div class="stat-label">{stats[stat]["label"]}</div></div>'''

        # Build complete HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
{template_css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <div class="report-title">{title}</div>
                <div class="report-subtitle">{subtitle}</div>
                <div class="report-meta">Generated: {generated_date}</div>
            </div>
        </div>

        <div class="exec-summary">
            <h2>Executive Summary</h2>
            <p>{exec_summary}</p>
        </div>

        <div class="stats-bar">{stats_html}
        </div>

        <div class="main-content">
{insights_html}
        </div>

        <div class="footer">
            <h3>📊 AI-Powered Data Discovery</h3>
            <p>This report was autonomously generated using advanced AI and interactive visualizations.</p>
            <p style="margin-top: 10px;">All visualizations are powered by Chart.js and fully interactive.</p>
        </div>
    </div>

    <script>
{chart_scripts}
    </script>
</body>
</html>'''

        return html

    def _extract_css_from_template(self) -> str:
        """Extract CSS from template file."""
        try:
            template_html = self._load_template()
            # Extract content between <style> and </style>
            match = re.search(r'<style>(.*?)</style>', template_html, re.DOTALL)
            if match:
                css = match.group(1)
                print(f"[DASHBOARD] Using CSS from template: {self.template_path}")
                return css
            else:
                print(f"[WARN] Could not find <style> tags in template, using fallback CSS")
                return self._get_fallback_css()
        except FileNotFoundError:
            print(f"[WARN] Template not found at {self.template_path}, using fallback CSS")
            return self._get_fallback_css()
        except Exception as e:
            print(f"[ERROR] Failed to read template: {e}, using fallback CSS")
            return self._get_fallback_css()

    def _get_fallback_css(self) -> str:
        """Get fallback CSS if template not available."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }

        .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 50px 60px; position: relative; overflow: hidden; }
        .header::before { content: ''; position: absolute; top: 0; right: 0; width: 300px; height: 300px; background: rgba(255,255,255,0.05); border-radius: 50%; transform: translate(30%, -30%); }
        .header-content { position: relative; z-index: 1; }
        .report-title { font-size: 48px; font-weight: 700; margin-bottom: 10px; letter-spacing: -1px; }
        .report-subtitle { font-size: 24px; font-weight: 300; color: #a0a0ff; margin-bottom: 20px; }
        .report-meta { font-size: 14px; color: rgba(255,255,255,0.7); margin-top: 20px; }

        .exec-summary { padding: 50px 60px; background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%); border-bottom: 3px solid #e0e0e0; }
        .exec-summary h2 { font-size: 32px; color: #1a1a2e; margin-bottom: 20px; font-weight: 600; }
        .exec-summary p { font-size: 16px; line-height: 1.8; color: #555; text-align: justify; }

        .stats-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; background: #16213e; color: white; }
        .stat-item { padding: 30px; text-align: center; border-right: 1px solid rgba(255,255,255,0.1); }
        .stat-item:last-child { border-right: none; }
        .stat-value { font-size: 36px; font-weight: 700; color: #4fc3f7; margin-bottom: 8px; }
        .stat-label { font-size: 13px; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 1px; }

        .main-content { padding: 50px 60px; }
        .section-header { display: flex; align-items: center; gap: 15px; margin: 60px 0 40px 0; padding-bottom: 15px; border-bottom: 3px solid #16213e; }
        .section-header:first-child { margin-top: 0; }
        .section-icon { width: 50px; height: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; display: flex; align-items: center; justify-center; font-size: 24px; color: white; }
        .section-header h2 { font-size: 32px; color: #1a1a2e; font-weight: 600; }

        .viz-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 40px; margin-bottom: 50px; }
        .viz-grid.single-column { grid-template-columns: 1fr; }

        .insight-card { margin-bottom: 40px; border: 1px solid #e0e0e0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.3s, box-shadow 0.3s; }
        .insight-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }

        .insight-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 25px; display: flex; align-items: center; justify-content: space-between; }
        .insight-title-group { display: flex; align-items: center; gap: 15px; }
        .insight-number { width: 40px; height: 40px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; flex-shrink: 0; }
        .insight-title { font-size: 18px; font-weight: 600; line-height: 1.3; }
        .chart-type-badge { background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 15px; font-size: 11px; font-weight: 600; text-transform: uppercase; }

        .insight-body { padding: 25px; }
        .insight-description { font-size: 14px; line-height: 1.7; color: #555; margin-bottom: 20px; }
        .insight-visualization { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 15px; }
        .insight-visualization canvas { max-height: 400px; }

        .footer { background: #1a1a2e; color: white; padding: 40px 60px; text-align: center; }
        .footer h3 { font-size: 24px; margin-bottom: 15px; color: #4fc3f7; }
        .footer p { font-size: 14px; line-height: 1.8; opacity: 0.8; }

        @media (max-width: 1024px) { .viz-grid { grid-template-columns: 1fr; } }
        @media (max-width: 768px) {
            .stats-bar { grid-template-columns: repeat(2, 1fr); }
            .header, .exec-summary, .main-content, .footer { padding: 30px; }
            .report-title { font-size: 36px; }
        }
        """

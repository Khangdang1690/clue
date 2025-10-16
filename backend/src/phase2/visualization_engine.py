"""Business-oriented visualization generation engine."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.models.analysis_result import VisualizationData
from src.models.challenge import Challenge
from src.phase2.business_query_engine import QueryResult

class VisualizationEngine:
    """Generates business-oriented visualizations that answer specific questions."""

    def __init__(self, output_directory: str = "data/outputs/visualizations"):
        """
        Initialize visualization engine.

        Args:
            output_directory: Directory to save visualizations
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Set professional style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def generate_business_visualizations(
        self,
        challenge: Challenge,
        business_insights: Dict[str, Any]
    ) -> List[VisualizationData]:
        """
        Generate business-oriented visualizations based on insights.

        Args:
            challenge: Challenge being analyzed
            business_insights: Business insights containing visualizations needed

        Returns:
            List of VisualizationData objects
        """
        print(f"\n📊 Generating business visualizations for challenge: {challenge.title}")
        print(f"  Visualization requests: {len(business_insights.get('visualizations_needed', []))}")
        print(f"  Key metrics available: {len(business_insights.get('key_metrics', {}))}")

        visualizations = []

        # Process each visualization request from business analyst
        viz_requests = business_insights.get('visualizations_needed', [])
        for idx, viz_request in enumerate(viz_requests, 1):
            print(f"\n  📈 Processing visualization {idx}/{len(viz_requests)}")
            print(f"     Type: {viz_request.get('viz_type', 'unknown')}")
            print(f"     Title: {viz_request.get('title', 'N/A')[:60]}...")

            viz = self._create_business_visualization(
                viz_request,
                challenge.id
            )
            if viz:
                print(f"     ✅ Created: {viz.file_path}")
                visualizations.append(viz)
            else:
                print(f"     ⚠️  Failed to create visualization")

        # Create KPI dashboard if key metrics exist
        if business_insights.get('key_metrics'):
            print(f"\n  📊 Creating KPI Dashboard...")
            kpi_viz = self._create_kpi_dashboard(
                business_insights['key_metrics'],
                challenge
            )
            if kpi_viz:
                visualizations.append(kpi_viz)

        print(f"\n✅ Generated {len(visualizations)} visualizations total")
        return visualizations

    def _create_business_visualization(
        self,
        viz_request: Dict,
        challenge_id: str
    ) -> Optional[VisualizationData]:
        """
        Create a single business visualization based on request.

        Args:
            viz_request: Visualization request with data and type
            challenge_id: Challenge ID for file naming

        Returns:
            VisualizationData object or None
        """
        viz_type = viz_request.get('viz_type', 'bar_chart')
        data = viz_request.get('data')
        question = viz_request.get('question', '')
        title = viz_request.get('title', question)
        priority = viz_request.get('priority', 'medium')

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return None

        # Route to appropriate visualization method
        if viz_type == 'bar_chart':
            return self._create_bar_chart(data, title, question, challenge_id, priority)
        elif viz_type == 'grouped_bar_chart':
            return self._create_grouped_bar_chart(data, title, question, challenge_id, priority)
        elif viz_type == 'line_chart':
            return self._create_line_chart(data, title, question, challenge_id, priority)
        elif viz_type == 'pie_chart':
            return self._create_pie_chart(data, title, question, challenge_id, priority)
        elif viz_type == 'kpi_card':
            return self._create_kpi_card(data, title, question, challenge_id, priority)
        elif viz_type == 'scatter_plot':
            return self._create_business_scatter(data, title, question, challenge_id, priority)
        else:
            # Default to bar chart
            return self._create_bar_chart(data, title, question, challenge_id, priority)

    def _create_bar_chart(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create a business-friendly bar chart."""
        plt.figure(figsize=(12, 6))

        # Determine x and y columns
        if len(data.columns) >= 2:
            x_col = data.columns[0]
            y_col = data.columns[1]
        else:
            return None

        # Sort by value for better readability
        data_sorted = data.sort_values(y_col, ascending=False).head(15)

        # Create bar chart
        bars = plt.bar(
            range(len(data_sorted)),
            data_sorted[y_col],
            color='steelblue',
            edgecolor='darkblue',
            linewidth=1.5
        )

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, data_sorted[y_col])):
            # Format based on value magnitude
            if abs(value) >= 1000000:
                label = f'{value/1000000:.1f}M'
            elif abs(value) >= 1000:
                label = f'{value/1000:.1f}K'
            elif abs(value) < 1:
                label = f'{value:.2%}'
            else:
                label = f'{value:.0f}'

            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    label, ha='center', va='bottom', fontsize=9)

        # Customize appearance
        plt.xticks(range(len(data_sorted)), data_sorted[x_col], rotation=45, ha='right')
        plt.xlabel(self._clean_column_name(x_col))
        plt.ylabel(self._clean_column_name(y_col))
        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        # Add priority indicator
        if priority == 'high':
            plt.text(0.02, 0.98, '⚠ HIGH PRIORITY', transform=plt.gca().transAxes,
                    fontsize=10, color='red', fontweight='bold', va='top')

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_business_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="bar_chart",
            title=title,
            data={"source_data": data_sorted.to_dict()},
            file_path=str(filepath),
            description=f"Business insight: {question}"
        )

    def _create_grouped_bar_chart(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create a grouped bar chart for comparisons."""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Prepare data for grouped bars
        if len(data.columns) < 2:
            return None

        x_col = data.columns[0]
        value_cols = [col for col in data.columns[1:] if data[col].dtype in ['float64', 'int64']]

        if not value_cols:
            return None

        # Limit to reasonable number of groups
        data_plot = data.head(10)
        x_labels = data_plot[x_col].astype(str)
        x_pos = np.arange(len(x_labels))
        width = 0.8 / len(value_cols)

        # Create bars for each value column
        for i, col in enumerate(value_cols[:5]):  # Max 5 series
            offset = (i - len(value_cols)/2) * width + width/2
            bars = ax.bar(x_pos + offset, data_plot[col], width,
                          label=self._clean_column_name(col))

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    label = self._format_value(height)
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           label, ha='center', va='bottom', fontsize=8)

        # Customize appearance
        ax.set_xlabel(self._clean_column_name(x_col))
        ax.set_ylabel('Values')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend(loc='upper right')

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_comparison_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="grouped_bar_chart",
            title=title,
            data={"source_data": data_plot.to_dict()},
            file_path=str(filepath),
            description=f"Comparison analysis: {question}"
        )

    def _create_line_chart(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create a line chart for trends."""
        fig, ax = plt.subplots(figsize=(14, 7))

        if len(data.columns) < 2:
            return None

        x_col = data.columns[0]
        value_cols = [col for col in data.columns[1:] if data[col].dtype in ['float64', 'int64']]

        if not value_cols:
            return None

        # Plot lines
        for col in value_cols[:5]:  # Max 5 lines
            ax.plot(data[x_col], data[col], marker='o', linewidth=2,
                   label=self._clean_column_name(col), markersize=6)

        # Add trend indicators
        for col in value_cols[:1]:  # For the first metric
            if len(data) > 1:
                start_val = data[col].iloc[0]
                end_val = data[col].iloc[-1]
                if start_val != 0:
                    change_pct = ((end_val - start_val) / start_val) * 100
                    trend_text = f"📈 +{change_pct:.1f}%" if change_pct > 0 else f"📉 {change_pct:.1f}%"
                    ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
                           fontsize=12, fontweight='bold', va='top',
                           color='green' if change_pct > 0 else 'red')

        # Customize appearance
        ax.set_xlabel(self._clean_column_name(x_col))
        ax.set_ylabel('Values')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Rotate x labels if many points
        if len(data) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_trend_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="line_chart",
            title=title,
            data={"source_data": data.to_dict()},
            file_path=str(filepath),
            description=f"Trend analysis: {question}"
        )

    def _create_pie_chart(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create a pie chart for distribution/segmentation."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(data.columns) < 2:
            return None

        # Determine label and value columns
        label_col = data.columns[0]
        value_col = data.columns[1]

        # Limit to top segments for clarity
        data_plot = data.nlargest(8, value_col) if len(data) > 8 else data

        # Create pie chart
        colors = sns.color_palette('Set2', len(data_plot))
        wedges, texts, autotexts = ax.pie(
            data_plot[value_col],
            labels=data_plot[label_col],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(data_plot)  # Slightly separate all slices
        )

        # Enhance text
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend with values
        legend_labels = [f"{row[label_col]}: {row[value_col]:,.0f}"
                        for _, row in data_plot.iterrows()]
        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_distribution_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="pie_chart",
            title=title,
            data={"source_data": data_plot.to_dict()},
            file_path=str(filepath),
            description=f"Distribution analysis: {question}"
        )

    def _create_kpi_card(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create KPI cards for key metrics."""
        fig, axes = plt.subplots(1, min(4, len(data.columns)), figsize=(16, 4))

        if len(data.columns) == 1:
            axes = [axes]
        elif len(data.columns) == 0:
            return None

        # Create KPI cards
        for idx, col in enumerate(data.columns[:4]):
            ax = axes[idx] if len(data.columns) > 1 else axes[0]

            # Get the value
            if len(data) > 0:
                value = data[col].iloc[0]
            else:
                value = 0

            # Format value
            formatted_value = self._format_value(value)

            # Create card
            ax.text(0.5, 0.6, formatted_value, ha='center', va='center',
                   fontsize=32, fontweight='bold', color='darkblue')
            ax.text(0.5, 0.3, self._clean_column_name(col), ha='center', va='center',
                   fontsize=14, color='gray')

            # Add trend indicator if multiple rows
            if len(data) > 1:
                prev_value = data[col].iloc[1] if len(data) > 1 else value
                if prev_value != 0:
                    change = ((value - prev_value) / prev_value) * 100
                    trend_symbol = '▲' if change > 0 else '▼' if change < 0 else '─'
                    trend_color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
                    ax.text(0.5, 0.1, f"{trend_symbol} {abs(change):.1f}%",
                           ha='center', va='center', fontsize=12, color=trend_color)

            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(len(data.columns), len(axes)):
            axes[idx].axis('off')

        # Add title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_kpi_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="kpi_card",
            title=title,
            data={"source_data": data.to_dict()},
            file_path=str(filepath),
            description=f"Key metrics: {question}"
        )

    def _create_business_scatter(
        self,
        data: pd.DataFrame,
        title: str,
        question: str,
        challenge_id: str,
        priority: str
    ) -> VisualizationData:
        """Create a business-friendly scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if len(data.columns) < 2:
            return None

        # Get x and y columns (first two numeric columns)
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < 2:
            return None

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        # Create scatter plot
        scatter = ax.scatter(data[x_col], data[y_col], alpha=0.6, s=100,
                           edgecolors='darkblue', linewidth=1.5, color='lightblue')

        # Add trend line
        z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2,
               label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')

        # Calculate correlation
        correlation = data[x_col].corr(data[y_col])
        strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"

        # Add correlation text
        ax.text(0.02, 0.98, f"Correlation: {correlation:.3f} ({strength})",
               transform=ax.transAxes, fontsize=12, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Customize appearance
        ax.set_xlabel(self._clean_column_name(x_col))
        ax.set_ylabel(self._clean_column_name(y_col))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_relationship_{self._sanitize_filename(title)}.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="scatter_plot",
            title=title,
            data={"source_data": data.to_dict(), "correlation": float(correlation)},
            file_path=str(filepath),
            description=f"Relationship analysis: {question}"
        )

    def _create_kpi_dashboard(
        self,
        key_metrics: Dict,
        challenge: Challenge
    ) -> VisualizationData:
        """Create a comprehensive KPI dashboard."""
        print(f"\n🎨 Creating KPI Dashboard for challenge: {challenge.title}")
        print(f"  Input: {len(key_metrics)} metrics provided")

        # Filter and validate metrics
        clean_metrics = self._filter_valid_metrics(key_metrics)

        print(f"  Valid metrics after filtering: {len(clean_metrics)}")
        if len(clean_metrics) == 0:
            print(f"  ⚠️  No valid metrics! Dashboard will be empty or placeholder.")
        else:
            print(f"  Sample metrics:")
            for name, value in list(clean_metrics.items())[:5]:
                print(f"    • {name}: {value}")

        # If no valid metrics, create a placeholder
        if not clean_metrics:
            return self._create_placeholder_dashboard(challenge)

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"KPI Dashboard - {challenge.title}", fontsize=18, fontweight='bold')

        # Calculate grid layout
        n_metrics = len(clean_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        for idx, (metric_name, metric_data) in enumerate(list(clean_metrics.items())[:9]):
            ax = plt.subplot(n_rows, n_cols, idx + 1)

            # Extract metric value
            value = self._extract_metric_value(metric_data)

            if value is not None:
                # KPI card style
                ax.text(0.5, 0.6, self._format_value(value),
                       ha='center', va='center', fontsize=24, fontweight='bold')
                ax.text(0.5, 0.2, self._clean_metric_name(metric_name),
                       ha='center', va='center', fontsize=12, color='gray')

                # Don't show context - it's usually a question which clutters the KPI card
                # KPI cards should be clean and focused on the metric value

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        plt.tight_layout()

        # Save figure
        filename = f"{challenge.id}_kpi_dashboard.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ KPI Dashboard saved to: {filepath}")

        return VisualizationData(
            viz_type="dashboard",
            title=f"KPI Dashboard - {challenge.title}",
            data={"metrics": clean_metrics},
            file_path=str(filepath),
            description="Comprehensive KPI overview for executive decision-making"
        )

    def _filter_valid_metrics(self, key_metrics: Dict) -> Dict:
        """
        Filter out invalid metrics (questions, non-numeric data, etc.).

        Returns:
            Dict of valid business metrics only
        """
        clean_metrics = {}

        for name, data in key_metrics.items():
            # Skip if name looks like a question (contains how/what/why/?)
            if any(word in name.lower() for word in ['how', 'what', 'why', 'which', 'where', 'when', '?']):
                print(f"    ⚠️  Skipping question-like metric: '{name[:60]}...'")
                continue

            # Skip if name is excessively long (likely a sentence/description)
            if len(name) > 100:
                print(f"    ⚠️  Skipping overly long metric name: '{name[:60]}...'")
                continue

            # Try to extract numeric value
            value = self._extract_metric_value(data)
            if value is None:
                print(f"    ⚠️  Skipping non-numeric metric: '{name}' = {data}")
                continue

            clean_metrics[name] = data

        return clean_metrics

    def _extract_metric_value(self, metric_data) -> Optional[float]:
        """
        Extract numeric value from various metric data formats.

        Args:
            metric_data: Can be a number, dict, or other structure

        Returns:
            Float value or None if not extractable
        """
        # Direct numeric value
        if isinstance(metric_data, (int, float)):
            return float(metric_data)

        # Dictionary with known keys
        if isinstance(metric_data, dict):
            # Try common keys in order of preference
            for key in ['value', 'mean', 'average', 'total', 'count']:
                if key in metric_data:
                    val = metric_data[key]
                    if isinstance(val, (int, float)):
                        return float(val)

            # Try first numeric value in dict
            for val in metric_data.values():
                if isinstance(val, (int, float)):
                    return float(val)

        return None

    def _clean_metric_name(self, name: str) -> str:
        """
        Clean up metric name for better display.
        Remove file patterns, clean formatting.

        Args:
            name: Raw metric name

        Returns:
            Cleaned, human-readable metric name
        """
        import re

        # Remove common file prefixes (csv_0_, excel_1_, pdf_2_, etc.)
        name = re.sub(r'(csv|excel|pdf)_\d+_', '', name)

        # Remove file extensions
        name = re.sub(r'\.(csv|xlsx?|pdf)$', '', name, flags=re.IGNORECASE)

        # Remove year patterns like _2024
        name = re.sub(r'_\d{4}$', '', name)

        # Use existing column name cleaner
        name = self._clean_column_name(name)

        return name

    def _create_placeholder_dashboard(self, challenge: Challenge) -> VisualizationData:
        """
        Create a placeholder dashboard when no valid metrics are available.

        Args:
            challenge: Challenge object

        Returns:
            VisualizationData with placeholder message
        """
        print(f"  📝 Creating placeholder dashboard (no valid metrics available)")

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"KPI Dashboard - {challenge.title}", fontsize=18, fontweight='bold')

        ax = plt.subplot(1, 1, 1)
        ax.text(0.5, 0.5,
                "No Valid Metrics Available\n\nMetric extraction did not find\nsuitable numeric KPIs to display.",
                ha='center', va='center', fontsize=16, color='gray', style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()

        # Save figure
        filename = f"{challenge.id}_kpi_dashboard.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="dashboard",
            title=f"KPI Dashboard - {challenge.title}",
            data={"metrics": {}},
            file_path=str(filepath),
            description="Placeholder dashboard - no valid metrics found"
        )

    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name for display."""
        # Replace underscores with spaces and capitalize
        clean_name = col_name.replace('_', ' ').replace('-', ' ')
        # Capitalize first letter of each word
        clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        return clean_name

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use in filename."""
        # Remove special characters
        safe_title = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in title)
        # Replace spaces with underscores and limit length
        safe_title = safe_title.replace(' ', '_')[:50]
        return safe_title.lower()

    def _format_value(self, value: float) -> str:
        """Format numeric value for display."""
        if pd.isna(value):
            return "N/A"
        elif abs(value) >= 1000000:
            return f'{value/1000000:.1f}M'
        elif abs(value) >= 1000:
            return f'{value/1000:.1f}K'
        elif abs(value) < 1 and abs(value) > 0.01:
            return f'{value:.1%}'
        elif abs(value) < 0.01 and value != 0:
            return f'{value:.3f}'
        else:
            return f'{value:.0f}'

    def create_summary_dashboard(
        self,
        challenge: Challenge,
        visualizations: List[VisualizationData]
    ) -> str:
        """
        Create a summary dashboard HTML file.

        Args:
            challenge: Challenge being analyzed
            visualizations: List of visualizations

        Returns:
            Path to HTML file
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Business Analysis Dashboard - {challenge.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .subtitle {{
            margin-top: 10px;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .challenge-info {{
            background: #f8f9fa;
            padding: 25px;
            border-left: 5px solid #667eea;
            margin: 20px;
        }}
        .challenge-info h2 {{
            color: #333;
            margin-top: 0;
        }}
        .metrics-row {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            margin: 20px;
        }}
        .metric-card {{
            text-align: center;
            padding: 15px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            padding: 20px;
        }}
        .viz-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .viz-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }}
        .viz-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
        }}
        .viz-title {{
            font-size: 1.2em;
            font-weight: 500;
            margin: 0;
        }}
        .viz-description {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .viz-content {{
            padding: 20px;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Business Analysis Dashboard</h1>
            <div class="subtitle">Data-Driven Insights for Strategic Decision Making</div>
        </div>

        <div class="challenge-info">
            <h2>{challenge.title}</h2>
            <p><strong>Department:</strong> {', '.join(challenge.department) if isinstance(challenge.department, list) else challenge.department}</p>
            <p><strong>Priority:</strong> {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)</p>
            <p><strong>Description:</strong> {challenge.description}</p>
        </div>

        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">{len(visualizations)}</div>
                <div class="metric-label">Business Insights</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{challenge.priority_score:.0f}</div>
                <div class="metric-label">Priority Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{challenge.priority_level.value.upper()}</div>
                <div class="metric-label">Priority Level</div>
            </div>
        </div>

        <div class="viz-grid">
"""

        for viz in visualizations:
            html_content += f"""
            <div class="viz-container">
                <div class="viz-header">
                    <h3 class="viz-title">{viz.title}</h3>
                    <div class="viz-description">{viz.description}</div>
                </div>
                <div class="viz-content">
                    <img src="{Path(viz.file_path).name}" alt="{viz.title}">
                </div>
            </div>
"""

        html_content += """
        </div>

        <div class="footer">
            Generated by Business Intelligence System | Data-Driven Decision Support
        </div>
    </div>
</body>
</html>
"""

        # Save HTML file
        filename = f"{challenge.id}_business_dashboard.html"
        filepath = self.output_directory / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)
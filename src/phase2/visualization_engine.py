"""Visualization generation engine."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from src.models.analysis_result import VisualizationData
from src.models.challenge import Challenge
import json
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm


class VisualizationEngine:
    """Generates visualizations for data analysis."""

    def __init__(self, output_directory: str = "data/outputs/visualizations"):
        """
        Initialize visualization engine.

        Args:
            output_directory: Directory to save visualizations
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.llm = get_llm(temperature=0.3)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def generate_visualizations(
        self,
        challenge: Challenge,
        data: Dict[str, pd.DataFrame],
        analysis_results: Dict
    ) -> List[VisualizationData]:
        """
        Generate appropriate visualizations for the analysis.

        Args:
            challenge: Challenge being analyzed
            data: Dictionary of DataFrames
            analysis_results: Results from statistical analysis

        Returns:
            List of VisualizationData objects
        """
        visualizations = []

        # Get LLM recommendations for visualizations
        viz_plan = self._get_visualization_recommendations(
            challenge, data, analysis_results
        )

        # Generate each recommended visualization
        for dataset_name, df in data.items():
            # Distribution plots for numeric variables
            if viz_plan.get("distributions", False):
                dist_viz = self._create_distribution_plots(
                    df, dataset_name, challenge.id
                )
                visualizations.extend(dist_viz)

            # Correlation heatmap
            if viz_plan.get("correlation_heatmap", False):
                corr_viz = self._create_correlation_heatmap(
                    df, dataset_name, challenge.id
                )
                if corr_viz:
                    visualizations.append(corr_viz)

            # Time series plots
            if viz_plan.get("time_series", False):
                ts_viz = self._create_time_series_plots(
                    df, dataset_name, challenge.id
                )
                visualizations.extend(ts_viz)

            # Categorical analysis
            if viz_plan.get("categorical_analysis", False):
                cat_viz = self._create_categorical_plots(
                    df, dataset_name, challenge.id
                )
                visualizations.extend(cat_viz)

            # Scatter plots for relationships
            if viz_plan.get("scatter_plots", False):
                scatter_viz = self._create_scatter_plots(
                    df, dataset_name, challenge.id, analysis_results.get("correlations", {})
                )
                visualizations.extend(scatter_viz)

        return visualizations

    def _get_visualization_recommendations(
        self,
        challenge: Challenge,
        data: Dict[str, pd.DataFrame],
        analysis_results: Dict
    ) -> Dict:
        """
        Get LLM recommendations for visualizations.

        Args:
            challenge: Challenge being analyzed
            data: Available data
            analysis_results: Analysis results

        Returns:
            Dictionary with visualization recommendations
        """
        # Create data summary
        data_summary = []
        for name, df in list(data.items())[:3]:  # First 3 datasets
            data_summary.append(f"{name}: {len(df)} rows, columns: {', '.join(df.columns[:10])}")

        recommendation_prompt = ChatPromptTemplate.from_template(
            """As a data visualization expert, recommend appropriate visualizations for this analysis.

Challenge: {challenge}

Data available:
{data_summary}

Analysis findings:
{findings}

Recommend visualizations in JSON format:
{{
    "distributions": true/false,
    "correlation_heatmap": true/false,
    "time_series": true/false,
    "categorical_analysis": true/false,
    "scatter_plots": true/false,
    "custom_viz": ["description of any custom visualization needed"]
}}
"""
        )

        try:
            chain = recommendation_prompt | self.llm
            response = chain.invoke({
                "challenge": challenge.title,
                "data_summary": "\n".join(data_summary),
                "findings": str(analysis_results.get("key_findings", [])[:5])
            })

            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                return json.loads(content[start_idx:end_idx])

        except Exception as e:
            print(f"Error getting visualization recommendations: {e}")

        return {
            "distributions": True,
            "correlation_heatmap": True,
            "time_series": False,
            "categorical_analysis": True,
            "scatter_plots": True
        }

    def _create_distribution_plots(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        challenge_id: str
    ) -> List[VisualizationData]:
        """Create distribution plots for numeric variables."""
        visualizations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return visualizations

        # Create subplots for distributions (max 6)
        cols_to_plot = numeric_cols[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        # Ensure axes is always a flat list regardless of subplot configuration
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten().tolist()
        else:
            axes = axes.flatten().tolist()

        for idx, col in enumerate(cols_to_plot):
            if idx < len(axes):
                df[col].dropna().hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')

        # Hide extra subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_{dataset_name}_distributions.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        visualizations.append(VisualizationData(
            viz_type="histogram",
            title=f"Distribution Analysis - {dataset_name}",
            data={"columns": list(cols_to_plot)},
            file_path=str(filepath),
            description=f"Distribution plots for key numeric variables in {dataset_name}"
        ))

        return visualizations

    def _create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        challenge_id: str
    ) -> VisualizationData:
        """Create correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return None

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f'Correlation Matrix - {dataset_name}')
        plt.tight_layout()

        # Save figure
        filename = f"{challenge_id}_{dataset_name}_correlation.png"
        filepath = self.output_directory / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return VisualizationData(
            viz_type="heatmap",
            title=f"Correlation Analysis - {dataset_name}",
            data={"correlation_matrix": corr_matrix.to_dict()},
            file_path=str(filepath),
            description=f"Correlation heatmap showing relationships between variables in {dataset_name}"
        )

    def _create_time_series_plots(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        challenge_id: str
    ) -> List[VisualizationData]:
        """Create time series plots if datetime columns exist."""
        visualizations = []
        date_columns = df.select_dtypes(include=['datetime64']).columns

        if len(date_columns) == 0:
            return visualizations

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for date_col in date_columns[:1]:  # First date column
            for num_col in numeric_cols[:3]:  # First 3 numeric columns
                try:
                    # Sort and plot
                    ts_df = df[[date_col, num_col]].sort_values(date_col).dropna()

                    if len(ts_df) < 2:
                        continue

                    plt.figure(figsize=(14, 6))
                    plt.plot(ts_df[date_col], ts_df[num_col], linewidth=2)
                    plt.title(f'{num_col} over Time')
                    plt.xlabel('Date')
                    plt.ylabel(num_col)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save figure
                    filename = f"{challenge_id}_{dataset_name}_{num_col}_timeseries.png"
                    filepath = self.output_directory / filename
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()

                    visualizations.append(VisualizationData(
                        viz_type="line",
                        title=f"Time Series - {num_col}",
                        data={"date_column": date_col, "value_column": num_col},
                        file_path=str(filepath),
                        description=f"Time series plot showing trends in {num_col} over time"
                    ))

                except Exception as e:
                    print(f"Error creating time series for {num_col}: {e}")
                    continue

        return visualizations

    def _create_categorical_plots(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        challenge_id: str
    ) -> List[VisualizationData]:
        """Create plots for categorical variables."""
        visualizations = []
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) == 0:
            return visualizations

        # Create bar plots for top categories (max 4 columns)
        for col in categorical_cols[:4]:
            try:
                # Get value counts
                value_counts = df[col].value_counts().head(10)

                if len(value_counts) == 0:
                    continue

                plt.figure(figsize=(12, 6))
                value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Save figure
                filename = f"{challenge_id}_{dataset_name}_{col}_categorical.png"
                filepath = self.output_directory / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                visualizations.append(VisualizationData(
                    viz_type="bar",
                    title=f"Categorical Distribution - {col}",
                    data={"column": col, "top_values": value_counts.to_dict()},
                    file_path=str(filepath),
                    description=f"Bar chart showing distribution of categories in {col}"
                ))

            except Exception as e:
                print(f"Error creating categorical plot for {col}: {e}")
                continue

        return visualizations

    def _create_scatter_plots(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        challenge_id: str,
        correlations: Dict[str, float]
    ) -> List[VisualizationData]:
        """Create scatter plots for correlated variables."""
        visualizations = []

        if not correlations:
            return visualizations

        # Create scatter plots for top correlations
        for corr_pair, corr_value in list(correlations.items())[:3]:
            try:
                # Parse variable names
                var1, var2 = corr_pair.split('_vs_')

                if var1 not in df.columns or var2 not in df.columns:
                    continue

                # Create scatter plot
                plt.figure(figsize=(10, 8))
                plt.scatter(df[var1], df[var2], alpha=0.6, edgecolors='black')
                plt.title(f'{var1} vs {var2} (r={corr_value:.3f})')
                plt.xlabel(var1)
                plt.ylabel(var2)

                # Add trend line
                z = np.polyfit(df[var1].dropna(), df[var2].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(df[var1], p(df[var1]), "r--", alpha=0.8, linewidth=2)

                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save figure
                filename = f"{challenge_id}_{dataset_name}_{var1}_vs_{var2}_scatter.png"
                filepath = self.output_directory / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                visualizations.append(VisualizationData(
                    viz_type="scatter",
                    title=f"Relationship: {var1} vs {var2}",
                    data={"x": var1, "y": var2, "correlation": corr_value},
                    file_path=str(filepath),
                    description=f"Scatter plot showing correlation ({corr_value:.3f}) between {var1} and {var2}"
                ))

            except Exception as e:
                print(f"Error creating scatter plot for {corr_pair}: {e}")
                continue

        return visualizations

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
    <title>Analysis Dashboard - {challenge.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .viz-container {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .viz-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .viz-description {{
            color: #666;
            margin-bottom: 15px;
            font-style: italic;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .challenge-info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Analysis Dashboard</h1>

    <div class="challenge-info">
        <h2>Challenge: {challenge.title}</h2>
        <p><strong>Department:</strong> {challenge.department}</p>
        <p><strong>Priority:</strong> {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)</p>
        <p><strong>Description:</strong> {challenge.description}</p>
    </div>

    <h2>Visualizations ({len(visualizations)})</h2>
"""

        for viz in visualizations:
            html_content += f"""
    <div class="viz-container">
        <div class="viz-title">{viz.title}</div>
        <div class="viz-description">{viz.description}</div>
        <img src="{Path(viz.file_path).name}" alt="{viz.title}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Save HTML file
        filename = f"{challenge.id}_dashboard.html"
        filepath = self.output_directory / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)

"""Generate interactive Plotly dashboards from viz_data JSON."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any


class PlotlyDashboardGenerator:
    """
    Generates interactive Plotly dashboards by reading viz_data JSON.

    The viz_data JSON is written incrementally during exploration,
    so the dashboard can be generated even while exploration is running.
    """

    def __init__(self):
        """Initialize Plotly dashboard generator."""
        pass

    def generate_dashboard(
        self,
        viz_data_json_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate interactive Plotly dashboard from viz_data JSON.

        Args:
            viz_data_json_path: Path to the viz_data JSON file
            output_path: Optional custom output path for HTML

        Returns:
            Path to generated HTML dashboard
        """
        print(f"\n[PLOTLY] Generating Plotly dashboard from {viz_data_json_path}")

        # Read viz_data JSON
        with open(viz_data_json_path, 'r', encoding='utf-8') as f:
            viz_data = json.load(f)

        metadata = viz_data.get("metadata", {})
        visualizations = viz_data.get("visualizations", [])

        print(f"[PLOTLY] Found {len(visualizations)} visualizations")

        # Generate HTML
        html = self._build_html(metadata, visualizations)

        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/outputs/discovery")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"plotly_dashboard_{timestamp}.html"

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"[SUCCESS] Plotly dashboard saved to: {output_path}")
        return str(output_path)

    def _build_html(self, metadata: Dict, visualizations: List[Dict]) -> str:
        """
        Build complete HTML with embedded Plotly charts.

        Args:
            metadata: Dashboard metadata
            visualizations: List of visualization data

        Returns:
            Complete HTML string
        """
        dataset_name = metadata.get("dataset_name", "Dataset")
        domain = metadata.get("domain", "Unknown")
        dataset_type = metadata.get("dataset_type", "Data Analysis")
        generated_date = metadata.get("generated_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        total_insights = metadata.get("total_insights", len(visualizations))

        # Build chart divs and scripts
        chart_sections = []
        plotly_scripts = []

        for idx, viz in enumerate(visualizations, 1):
            viz_id = viz.get("viz_id", viz.get("insight_id", idx))  # Support both old (insight_id) and new (viz_id) format
            title = viz.get("title", f"Visualization {viz_id}")
            chart_type = viz.get("chart_type", "line")
            data = viz.get("data", {})
            description = viz.get("description", "")
            question = viz.get("question", "")  # Business question this viz answers

            # Create chart div with clean, professional structure + chart type selector + color picker
            chart_div = f'''
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title-row">
                        <h3>{title}</h3>
                        <div class="chart-controls">
                            <select class="chart-type-selector" onchange="changeChartType('{viz_id}', this.value)" data-default="{chart_type}">
                                <option value="bar" {"selected" if chart_type == "bar" else ""}>Bar Chart</option>
                                <option value="line" {"selected" if chart_type == "line" else ""}>Line Chart</option>
                                <option value="scatter" {"selected" if chart_type == "scatter" else ""}>Scatter Plot</option>
                                <option value="area" {"selected" if chart_type == "area" else ""}>Area Chart</option>
                            </select>
                            <select class="color-palette-selector" onchange="changeChartColor('{viz_id}', this.value)">
                                <option value="">Select Color</option>
                                <option value="#3b82f6" style="background-color: #3b82f6; color: white;">Blue</option>
                                <option value="#10b981" style="background-color: #10b981; color: white;">Green</option>
                                <option value="#f59e0b" style="background-color: #f59e0b; color: white;">Orange</option>
                                <option value="#ef4444" style="background-color: #ef4444; color: white;">Red</option>
                                <option value="#8b5cf6" style="background-color: #8b5cf6; color: white;">Purple</option>
                                <option value="#ec4899" style="background-color: #ec4899; color: white;">Pink</option>
                                <option value="#14b8a6" style="background-color: #14b8a6; color: white;">Teal</option>
                                <option value="#6366f1" style="background-color: #6366f1; color: white;">Indigo</option>
                                <option value="#64748b" style="background-color: #64748b; color: white;">Gray</option>
                                <option value="#000000" style="background-color: #000000; color: white;">Black</option>
                            </select>
                        </div>
                    </div>
                    <p class="chart-description">{description}</p>
                </div>
                <div id="chart_{viz_id}" class="plotly-chart"></div>
            </div>
            '''
            chart_sections.append(chart_div)

            # Create Plotly script
            plotly_script = self._create_plotly_script(viz_id, chart_type, data)
            plotly_scripts.append(plotly_script)

        # Build complete HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} - Plotly Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        /* Modern B2B Dashboard Design - Professional & Clean */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: #f8f9fa;
            color: #1a202c;
            min-height: 100vh;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 40px 24px;
        }}

        /* Header Section - Clean and Minimal */
        header {{
            background: white;
            padding: 32px 40px;
            margin-bottom: 40px;
            border-bottom: 1px solid #e2e8f0;
        }}

        h1 {{
            color: #1a202c;
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
        }}

        .subtitle {{
            color: #64748b;
            font-size: 14px;
            font-weight: 400;
            margin-bottom: 24px;
        }}

        /* Metadata Cards - Minimal and Professional */
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}

        .metadata-item {{
            background: #f8fafc;
            padding: 16px 20px;
            border-radius: 0;  /* Sharp edges */
            border-left: 3px solid #3b82f6;
        }}

        .metadata-label {{
            color: #64748b;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}

        .metadata-value {{
            color: #1a202c;
            font-size: 24px;
            font-weight: 600;
        }}

        /* Charts Section - Card Layout with Sharp Edges */
        .charts {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 32px;
            margin-top: 40px;
        }}

        /* Chart Container - White card with sharp edges */
        .chart-container {{
            background: white;
            padding: 24px;
            border: 1px solid #e2e8f0;
            border-radius: 0;  /* Sharp edges - no rounding */
        }}

        .chart-header {{
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
        }}

        .chart-title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}

        .chart-container h3 {{
            color: #1a202c;
            font-size: 18px;
            font-weight: 600;
            margin: 0;
            letter-spacing: -0.01em;
        }}

        .chart-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .chart-type-selector,
        .color-palette-selector {{
            padding: 6px 12px;
            border: 1px solid #e2e8f0;
            border-radius: 0;  /* Sharp edges */
            background: white;
            color: #1a202c;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: border-color 0.2s ease;
        }}

        .chart-type-selector:hover,
        .color-palette-selector:hover {{
            border-color: #3b82f6;
        }}

        .chart-type-selector:focus,
        .color-palette-selector:focus {{
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }}

        .color-palette-selector {{
            min-width: 120px;
        }}

        .chart-description {{
            color: #64748b;
            font-size: 13px;
            font-weight: 400;
            line-height: 1.5;
            margin-top: 6px;
        }}

        /* Plotly Chart - Sharp edges, no border inside card */
        .plotly-chart {{
            min-height: 480px;
            background: white;
            border-radius: 0;  /* Sharp edges */
            border: none;  /* No border since we're in a card */
            padding: 0;
        }}

        /* Footer - Subtle and Professional */
        footer {{
            text-align: center;
            color: #94a3b8;
            margin-top: 64px;
            padding: 24px;
            font-size: 12px;
            border-top: 1px solid #e2e8f0;
        }}

        /* Responsive Design */
        @media (min-width: 1200px) {{
            .charts {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 24px 16px;
            }}

            header {{
                padding: 24px 20px;
            }}

            h1 {{
                font-size: 24px;
            }}

            .metadata {{
                grid-template-columns: 1fr;
            }}

            .charts {{
                gap: 32px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{dataset_name}</h1>
            <div class="subtitle">{domain} | {dataset_type}</div>

            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Total Insights</div>
                    <div class="metadata-value">{total_insights}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Generated</div>
                    <div class="metadata-value">{generated_date}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Visualizations</div>
                    <div class="metadata-value">{len(visualizations)}</div>
                </div>
            </div>
        </header>

        <div class="charts">
            {"".join(chart_sections)}
        </div>

        <footer>
            <p>Generated with Autonomous AI Analyst | Plotly Dashboard</p>
        </footer>
    </div>

    <script>
        // Function to change chart type dynamically
        function changeChartType(chartId, newType) {{
            const chartDiv = 'chart_' + chartId;
            let update = {{}};

            // Configure update based on chart type
            if (newType === 'bar') {{
                update.type = 'bar';
                update.mode = null;
                update.fill = null;
            }} else if (newType === 'line') {{
                update.type = 'scatter';
                update.mode = 'lines+markers';
                update.fill = null;
                update.line = {{
                    width: 3,
                    shape: 'spline'
                }};
            }} else if (newType === 'scatter') {{
                update.type = 'scatter';
                update.mode = 'markers';
                update.fill = null;
            }} else if (newType === 'area') {{
                update.type = 'scatter';
                update.mode = 'lines';
                update.fill = 'tozeroy';
                update.line = {{
                    width: 2,
                    shape: 'spline'
                }};
            }}

            // Apply the update to all traces in the chart
            Plotly.restyle(chartDiv, update);
        }}

        // Function to change chart color dynamically
        function changeChartColor(chartId, newColor) {{
            if (!newColor) return; // Skip if "Select Color" is chosen

            const chartDiv = 'chart_' + chartId;
            const chartElement = document.getElementById(chartDiv);

            // Get number of traces in the chart
            const numTraces = chartElement.data.length;

            // Update color for all traces
            let update = {{}};

            // For markers (all chart types)
            update['marker.color'] = newColor;

            // For line charts - update line color
            update['line.color'] = newColor;

            // Apply to all traces
            const traceIndices = Array.from({{length: numTraces}}, (_, i) => i);
            Plotly.restyle(chartDiv, update, traceIndices);

            // Reset the dropdown to "Select Color"
            event.target.value = '';
        }}

        // Initialize Plotly charts
        {" ".join(plotly_scripts)}
    </script>
</body>
</html>'''

        return html

    def _create_plotly_script(self, viz_id: int, chart_type: str, data: Dict) -> str:
        """
        Create Plotly JavaScript code for a chart.

        Handles three data formats:
        1. LLM datasets format: {'labels': [...], 'datasets': [{'label': 'X', 'data': [...]}]}
        2. LLM simple format: {'chart_type': 'bar', 'data': [...], 'labels': [...]}
        3. Legacy format: {'x': [...], 'y': [...], 'labels': {'x': '...', 'y': '...'}}

        Args:
            viz_id: Unique ID for the chart
            chart_type: Type of chart (line, bar, scatter, etc.)
            data: Chart data dictionary

        Returns:
            JavaScript code string
        """
        # Detect format and normalize to x/y format
        if 'datasets' in data:
            # LLM format: {'labels': [...], 'datasets': [{'label': 'X', 'data': [...]}]}
            x_data = data.get('labels', [])
            datasets = data.get('datasets', [])

            if not datasets:
                # Empty datasets - return empty chart
                return f"console.log('No data for chart_{viz_id}');"

            # Professional varied color palette - different colors for each dataset/insight
            # Use viz_id as seed for color selection to ensure variety across charts
            colors = [
                '#667eea', '#764ba2', '#4facfe', '#43e97b', '#fa709a',
                '#fee140', '#30cfd0', '#c471ed', '#f38181', '#f093fb',
                '#66bb6a', '#ef5350', '#42a5f5', '#ab47bc', '#ffa726',
                '#26c6da', '#9ccc65', '#5c6bc0', '#ec407a', '#ffca28'
            ]

            # Build traces for each dataset with modern styling
            traces = []
            for idx, ds in enumerate(datasets):
                # Use different color for each dataset within a chart
                color = colors[(viz_id + idx) % len(colors)]
                trace = {
                    'x': x_data,
                    'y': ds.get('data', []),
                    'name': ds.get('label', 'Data'),
                    'type': self._plotly_chart_type(chart_type),
                    'mode': 'lines+markers' if chart_type == 'line' else None,
                    'line': {
                        'color': color,
                        'width': 3,
                        'shape': 'spline'  # Smooth curves
                    } if chart_type == 'line' else None,
                    'marker': {
                        'color': color,
                        'size': 8,
                        'line': {
                            'color': 'white',
                            'width': 2
                        }
                    },
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
                }
                # Remove None values
                trace = {k: v for k, v in trace.items() if v is not None}
                traces.append(trace)

            traces_json = json.dumps(traces)

            # Extract axis labels from data or use dataset labels
            x_label = data.get('x_label', '') or data.get('xlabel', '')
            y_label = data.get('y_label', '') or data.get('ylabel', '')

            # If still no labels, try to infer from dataset label
            if not x_label:
                x_label = 'Categories' if isinstance(x_data[0] if x_data else None, str) else 'X'
            if not y_label and datasets:
                # Use first dataset label as y-axis hint
                y_label = datasets[0].get('label', 'Y')

        elif 'data' in data and isinstance(data.get('labels'), list):
            # Format 3: LLM simple format {'chart_type': 'bar', 'data': [...], 'labels': [...]}
            x_data = data.get('labels', [])
            y_data = data.get('data', [])

            # Create a single trace with professional styling
            colors = [
                '#667eea', '#764ba2', '#4facfe', '#43e97b', '#fa709a',
                '#fee140', '#30cfd0', '#c471ed', '#f38181', '#f093fb',
                '#66bb6a', '#ef5350', '#42a5f5', '#ab47bc', '#ffa726',
                '#26c6da', '#9ccc65', '#5c6bc0', '#ec407a', '#ffca28'
            ]
            color = colors[viz_id % len(colors)]

            trace = {
                'x': x_data,
                'y': y_data,
                'type': self._plotly_chart_type(chart_type),
                'mode': 'lines+markers' if chart_type == 'line' else None,
                'marker': {
                    'color': color,
                    'size': 8,
                    'line': {
                        'color': 'white',
                        'width': 2
                    }
                },
                'line': {
                    'color': color,
                    'width': 3,
                    'shape': 'spline'
                } if chart_type == 'line' else None,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
            }
            # Remove None values
            trace = {k: v for k, v in trace.items() if v is not None}
            traces_json = json.dumps([trace])

            # Extract or infer axis labels
            x_label = data.get('x_label', '') or data.get('xlabel', '') or 'Categories'
            y_label = data.get('y_label', '') or data.get('ylabel', '') or 'Value'

        else:
            # Legacy format: {'x': [...], 'y': [...], 'labels': {'x': '...', 'y': '...'}}
            x_data = data.get('x', [])
            y_data = data.get('y', [])
            labels = data.get('labels', {})
            x_label = labels.get('x', 'X') if isinstance(labels, dict) else 'X'
            y_label = labels.get('y', 'Y') if isinstance(labels, dict) else 'Y'

            # Handle multi-series data (y is a list of lists)
            if y_data and isinstance(y_data[0], list):
                # Multiple series
                series_names = data.get('series_names') or [f'Series {i+1}' for i in range(len(y_data))]
                traces = []
                for idx, (y_series, name) in enumerate(zip(y_data, series_names)):
                    trace = {
                        'x': x_data,
                        'y': y_series,
                        'name': name,
                        'type': self._plotly_chart_type(chart_type),
                        'mode': 'lines+markers' if chart_type == 'line' else None
                    }
                    traces.append(trace)

                traces_json = json.dumps(traces)
            else:
                # Single series
                trace = {
                    'x': x_data,
                    'y': y_data,
                    'type': self._plotly_chart_type(chart_type),
                    'mode': 'lines+markers' if chart_type == 'line' else None,
                    'marker': {'color': '#667eea'}
                }
                traces_json = json.dumps([trace])

        # Professional B2B layout - Clean and minimal
        layout = {
            'xaxis': {
                'title': {
                    'text': x_label,
                    'font': {'size': 13, 'color': '#64748b', 'family': 'Inter, sans-serif'}
                },
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': '#f1f5f9',
                'showline': True,
                'linewidth': 1,
                'linecolor': '#e2e8f0',
                'zeroline': False,
                'tickfont': {'size': 11, 'color': '#64748b'}
            },
            'yaxis': {
                'title': {
                    'text': y_label,
                    'font': {'size': 13, 'color': '#64748b', 'family': 'Inter, sans-serif'}
                },
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': '#f1f5f9',
                'showline': False,
                'zeroline': False,
                'tickfont': {'size': 11, 'color': '#64748b'}
            },
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
                'size': 12,
                'color': '#1a202c'
            },
            'margin': {'l': 60, 'r': 20, 't': 20, 'b': 60},
            'hoverlabel': {
                'bgcolor': 'white',
                'bordercolor': '#e2e8f0',
                'font': {
                    'family': 'Inter, sans-serif',
                    'size': 12,
                    'color': '#000000'  # Black text for tooltips
                }
            }
        }
        layout_json = json.dumps(layout)

        # Clean config - only show download button (remove all other buttons)
        config = {
            'displayModeBar': True,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
                'hoverCompareCartesian', 'toggleSpikelines'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'chart_{viz_id}',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        config_json = json.dumps(config)

        script = f'''
        Plotly.newPlot('chart_{viz_id}', {traces_json}, {layout_json}, {config_json});
        '''

        return script

    def _plotly_chart_type(self, chart_type: str) -> str:
        """
        Convert chart_type to Plotly type.

        Args:
            chart_type: Chart type from viz_data

        Returns:
            Plotly chart type string
        """
        type_mapping = {
            'line': 'scatter',
            'bar': 'bar',
            'scatter': 'scatter',
            'pie': 'pie',
            'area': 'scatter',
            'histogram': 'histogram',
            'box': 'box',
            'heatmap': 'heatmap'
        }
        return type_mapping.get(chart_type.lower(), 'scatter')

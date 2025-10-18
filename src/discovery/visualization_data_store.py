"""Store visualization data incrementally for Plotly dashboards."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import Lock


class VisualizationDataStore:
    """
    Stores visualization data incrementally to a JSON file.

    This allows the Plotly dashboard to read visualizations as they're generated
    during the autonomous exploration process.
    """

    def __init__(self, dataset_name: str, dataset_context: Optional[Dict] = None):
        """
        Initialize viz data store.

        Args:
            dataset_name: Name of the dataset
            dataset_context: Optional context about the dataset
        """
        self.dataset_name = dataset_name
        self.dataset_context = dataset_context or {}

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("data/outputs/discovery/viz_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON file path
        self.json_path = self.output_dir / f"viz_data_{dataset_name}_{timestamp}.json"

        # Thread lock for safe concurrent writes
        self._lock = Lock()

        # Initialize JSON file with metadata
        self._initialize_json()

        print(f"[VIZ_DATA] Initialized: {self.json_path}")

    def _initialize_json(self):
        """Initialize JSON file with metadata structure."""
        initial_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "total_insights": 0,
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain": self.dataset_context.get("domain", "Unknown"),
                "dataset_type": self.dataset_context.get("dataset_type", "Data Analysis")
            },
            "visualizations": []
        }

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

    def add_visualization(
        self,
        execution_number: int,
        chart_data: Dict[str, Any],
        title: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Add a visualization to the JSON file incrementally.

        Args:
            execution_number: The code execution number (used as insight_id)
            chart_data: Dictionary containing chart data from code execution
            title: Optional title for the visualization
            description: Optional description
        """
        with self._lock:
            # Read current data
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract chart type
            chart_type = chart_data.get('chart_type', 'unknown')

            # Create visualization entry
            viz_entry = {
                "insight_id": execution_number,
                "title": title or f"Execution {execution_number}",
                "chart_type": chart_type,
                "data": chart_data,
                "description": description or f"Visualization from code execution {execution_number}"
            }

            # Append to visualizations array
            data["visualizations"].append(viz_entry)

            # Update metadata
            data["metadata"]["total_insights"] = len(data["visualizations"])

            # Write back to file
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[VIZ_DATA] Added visualization {execution_number}: {chart_type}")

    def get_json_path(self) -> str:
        """Get the path to the viz_data JSON file."""
        return str(self.json_path)

    def finalize(self):
        """
        Finalize the JSON file (optional cleanup/validation).

        Could add final metadata, validation, etc.
        """
        with self._lock:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update final metadata
            data["metadata"]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data["metadata"]["total_visualizations"] = len(data["visualizations"])

            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[VIZ_DATA] Finalized: {len(data['visualizations'])} visualizations")

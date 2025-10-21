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

    def __init__(self, dataset_name: str, dataset_context: Optional[Dict] = None, output_dir: Optional[str] = None):
        """
        Initialize viz data store.

        Args:
            dataset_name: Name of the dataset
            dataset_context: Optional context about the dataset
            output_dir: Optional custom output directory. If not provided, uses default.
        """
        self.dataset_name = dataset_name
        self.dataset_context = dataset_context or {}

        # Create output directory
        if output_dir:
            # Use custom output directory (e.g., analysis-specific directory)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Use simple filename in custom directory
            self.json_path = self.output_dir / "viz_data.json"
        else:
            # Use default directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("data/outputs/discovery/viz_data")
            self.output_dir.mkdir(parents=True, exist_ok=True)
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
        execution_number: int = None,  # Now optional, will auto-generate if not provided
        chart_data: Dict[str, Any] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        question: Optional[str] = None  # The business question this viz answers
    ):
        """
        Add a visualization to the JSON file incrementally.

        Args:
            execution_number: Optional execution number. If None, auto-generates unique ID
            chart_data: Dictionary containing chart data from code execution
            title: Optional title for the visualization
            description: Optional description
            question: The business question/insight this visualization supports
        """
        with self._lock:
            # Read current data
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Auto-generate unique viz_id if not provided
            if execution_number is None:
                # Use the count of existing visualizations + 1
                viz_id = len(data["visualizations"]) + 1
            else:
                viz_id = execution_number

            # Extract chart type
            chart_type = chart_data.get('chart_type', 'unknown') if chart_data else 'unknown'

            # Create visualization entry
            viz_entry = {
                "viz_id": viz_id,  # Unique ID for this visualization
                "title": title or f"Visualization {viz_id}",
                "chart_type": chart_type,
                "data": chart_data or {},
                "description": description or f"Visualization {viz_id}",
                "question": question  # Groups multiple vizs for same question
            }

            # Append to visualizations array
            data["visualizations"].append(viz_entry)

            # Update metadata
            data["metadata"]["total_insights"] = len(data["visualizations"])

            # Write back to file
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[VIZ_DATA] Added visualization {viz_id}: {chart_type} for question: {question[:50] if question else 'N/A'}")

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

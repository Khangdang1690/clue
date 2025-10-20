"""AI Agent service for running discovery and ETL workflows."""

from typing import Optional, List
import pandas as pd
from pathlib import Path

from src.graph.discovery_workflow import DiscoveryWorkflow
from src.graph.etl_workflow import ETLWorkflow
from app.schemas.agent import DiscoveryRequest, DiscoveryResponse, ETLRequest, ETLResponse


class AgentService:
    """Service for AI agent operations."""

    @staticmethod
    def run_discovery(request: DiscoveryRequest, df: pd.DataFrame, dataset_name: str) -> DiscoveryResponse:
        """
        Run AI-powered data discovery on a dataset.

        Args:
            request: Discovery request parameters
            df: Pandas DataFrame to analyze
            dataset_name: Name of the dataset

        Returns:
            DiscoveryResponse with results
        """
        # Create workflow
        workflow = DiscoveryWorkflow(
            max_iterations=request.max_iterations,
            max_insights=request.max_insights,
            generate_context=request.generate_context
        )

        # Run discovery
        result = workflow.run_discovery(df, dataset_name)

        # Extract insights
        insights = []
        if workflow.last_exploration_result and workflow.last_exploration_result.insights:
            insights = [
                {
                    "question": insight.question,
                    "finding": insight.finding,
                    "code": insight.code,
                    "viz_type": insight.viz_type,
                }
                for insight in workflow.last_exploration_result.insights
            ]

        return DiscoveryResponse(
            success=True,
            dataset_id=request.dataset_id,
            insights_count=len(insights),
            execution_count=workflow.last_exploration_result.total_executions if workflow.last_exploration_result else 0,
            viz_data_path=workflow.last_exploration_result.viz_data_path if workflow.last_exploration_result else None,
            dashboard_path=None,  # Can be generated separately
            insights=insights,
        )

    @staticmethod
    def run_etl(request: ETLRequest) -> ETLResponse:
        """
        Run ETL workflow on multiple files.

        Args:
            request: ETL request with company info and file paths

        Returns:
            ETLResponse with results
        """
        try:
            # Create ETL workflow
            workflow = ETLWorkflow(company_name=request.company_name or "Default Company")

            # Run ETL on all files
            result = workflow.run_etl(request.file_paths)

            # Extract dataset information
            datasets = []
            if hasattr(workflow, 'datasets_created'):
                datasets = [
                    {
                        "id": ds.id,
                        "name": ds.table_name,
                        "file_type": ds.file_type,
                        "row_count": ds.row_count,
                        "column_count": ds.column_count,
                    }
                    for ds in workflow.datasets_created
                ]

            return ETLResponse(
                success=True,
                company_id=request.company_id,
                datasets_created=len(datasets),
                datasets=datasets,
                errors=[],
            )

        except Exception as e:
            return ETLResponse(
                success=False,
                company_id=request.company_id,
                datasets_created=0,
                datasets=[],
                errors=[str(e)],
            )

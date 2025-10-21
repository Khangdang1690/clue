"""AI Agent routes for discovery and ETL workflows."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
from pathlib import Path

from app.api.deps import get_db
from app.services.agent_service import AgentService
from app.schemas.agent import (
    DiscoveryRequest,
    DiscoveryResponse,
    ETLRequest,
    ETLResponse,
    BusinessDiscoveryRequest,
    BusinessDiscoveryResponse
)
from src.database.models import User
from src.database.repository import DatasetRepository, AnalysisSessionRepository

router = APIRouter()


@router.post("/discovery/run", response_model=DiscoveryResponse)
async def run_discovery(
    request: DiscoveryRequest,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Run AI-powered data discovery on an uploaded file.

    Supports CSV and Excel files.
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .csv, .xlsx, .xls"
        )

    try:
        # Read file into DataFrame
        contents = await file.read()
        if file_ext == '.csv':
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))

        # Run discovery
        dataset_name = Path(file.filename).stem
        result = AgentService.run_discovery(request, df, dataset_name)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running discovery: {str(e)}"
        )


@router.post("/etl/run", response_model=ETLResponse)
async def run_etl(
    request: ETLRequest,
    db: Session = Depends(get_db)
):
    """
    Run ETL workflow on multiple files.

    Files should already be uploaded to the server or accessible via file paths.
    """
    try:
        result = AgentService.run_etl(request)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running ETL: {str(e)}"
        )


@router.get("/discovery/status/{dataset_id}")
async def get_discovery_status(dataset_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a discovery job.

    TODO: Implement job tracking and status updates.
    """
    return {
        "dataset_id": dataset_id,
        "status": "not_implemented",
        "message": "Job tracking coming soon"
    }


@router.post("/business-discovery/run", response_model=BusinessDiscoveryResponse)
async def run_business_discovery(
    request: BusinessDiscoveryRequest,
    db: Session = Depends(get_db)
):
    """
    Run business discovery workflow on user's datasets.

    Analyzes all datasets (or selected ones) to generate:
    - Business insights
    - Anomaly detection results
    - Time series forecasts
    - Causal relationships
    - Actionable recommendations
    - Interactive dashboard
    """
    try:
        # Get user and company
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not user.company_id:
            raise HTTPException(status_code=400, detail="User has no associated company")

        # Get dataset IDs to analyze
        dataset_ids = request.dataset_ids
        if not dataset_ids:
            # If no specific datasets provided, analyze all company datasets
            dataset_repo = DatasetRepository(db)
            datasets = dataset_repo.get_by_company(user.company_id)
            dataset_ids = [d.id for d in datasets]

        if not dataset_ids:
            raise HTTPException(
                status_code=400,
                detail="No datasets found for analysis. Please upload data first."
            )

        # Create AnalysisSession record before running workflow
        analysis_session = AnalysisSessionRepository.create(
            session=db,
            user_id=request.user_id,
            company_id=user.company_id,
            dataset_ids=dataset_ids,
            name=request.analysis_name or "Business Analysis",
            description=f"Analysis of {len(dataset_ids)} dataset(s)"
        )
        analysis_id = analysis_session.id

        try:
            # Run business discovery workflow with analysis_id
            from src.graph.business_discovery_workflow import BusinessDiscoveryWorkflow
            workflow = BusinessDiscoveryWorkflow()

            result = workflow.run_discovery(
                company_id=user.company_id,
                dataset_ids=dataset_ids,
                analysis_name=request.analysis_name,
                analysis_id=analysis_id  # Pass analysis_id to workflow
            )

            # Check if workflow succeeded
            if result.get('status') == 'error':
                # Mark analysis as failed
                AnalysisSessionRepository.mark_failed(
                    session=db,
                    analysis_id=analysis_id,
                    error_message=result.get('error', 'Unknown error occurred')
                )

                return BusinessDiscoveryResponse(
                    success=False,
                    analysis_id=analysis_id,
                    company_id=user.company_id,
                    dataset_count=len(dataset_ids),
                    error=result.get('error', 'Unknown error occurred')
                )

            # Mark analysis as completed with results
            insights_count = len(result.get('insights', []))
            recommendations_count = len(result.get('recommendations', []))

            analytics_results = result.get('analytics_results', {})
            analytics_summary = {
                'anomalies_count': len(analytics_results.get('anomalies', [])),
                'forecasts_count': len(analytics_results.get('forecasts', [])),
                'causal_relationships_count': len(analytics_results.get('causal', {}).get('relationships', [])),
                'variance_components_count': len(analytics_results.get('variance', {}).get('components', []))
            }

            AnalysisSessionRepository.mark_completed(
                session=db,
                analysis_id=analysis_id,
                dashboard_path=result.get('dashboard_path', ''),
                report_path=result.get('report_path', ''),
                executive_summary=result.get('executive_summary', ''),
                insights_count=insights_count,
                recommendations_count=recommendations_count,
                analytics_summary=analytics_summary
            )

            # Build response
            return BusinessDiscoveryResponse(
                success=True,
                analysis_id=analysis_id,
                company_id=user.company_id,
                dataset_count=len(dataset_ids),
                insights=result.get('insights', []),
                synthesized_insights=result.get('synthesized_insights', []),
                recommendations=result.get('recommendations', []),
                analytics_results=analytics_results,
                executive_summary=result.get('executive_summary', ''),
                dashboard_url=f"/api/analyses/{analysis_id}/dashboard",
                error=None
            )

        except Exception as workflow_error:
            # Mark analysis as failed
            AnalysisSessionRepository.mark_failed(
                session=db,
                analysis_id=analysis_id,
                error_message=str(workflow_error)
            )
            raise  # Re-raise to be caught by outer exception handler

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running business discovery: {str(e)}"
        )

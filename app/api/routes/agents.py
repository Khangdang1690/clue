"""AI Agent routes for discovery and ETL workflows."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
from pathlib import Path
import asyncio
from typing import Optional

from app.api.deps import get_db
from app.services.agent_service import AgentService
from app.services.analysis_progress_service import AnalysisProgressService
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
from src.database.connection import DatabaseManager

router = APIRouter()


async def run_business_discovery_background(
    company_id: str,
    dataset_ids: list,
    analysis_id: str,
    analysis_name: str
):
    """
    Background task to run business discovery workflow.

    This runs the full 8-step workflow and emits progress updates via AnalysisProgressService.
    """
    # Progress is already initialized in the main endpoint before this task runs
    # This prevents race condition with SSE client connection

    try:
        # Import workflow
        from src.graph.business_discovery_workflow import BusinessDiscoveryWorkflow

        # Create workflow instance with progress service
        workflow = BusinessDiscoveryWorkflow()

        # Run the workflow
        result = workflow.run_discovery(
            company_id=company_id,
            dataset_ids=dataset_ids,
            analysis_name=analysis_name,
            analysis_id=analysis_id
        )

        # Check if workflow succeeded
        print(f"[BG_TASK] Workflow returned")
        print(f"[BG_TASK] Result type: {type(result)}")
        print(f"[BG_TASK] Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        print(f"[BG_TASK] Status: {result.get('status')}")
        print(f"[BG_TASK] Dashboard path: {result.get('dashboard_path')}")
        print(f"[BG_TASK] Report path: {result.get('report_path')}")

        try:
            with DatabaseManager.get_session() as db:
                print(f"[BG_TASK] Database session acquired")

                if result.get('status') == 'error':
                    # Mark analysis as failed
                    print(f"[BG_TASK] Marking analysis as failed")
                    AnalysisSessionRepository.mark_failed(
                        session=db,
                        analysis_id=analysis_id,
                        error_message=result.get('error', 'Unknown error occurred')
                    )

                    # Emit error event to SSE clients
                    await AnalysisProgressService.mark_failed(
                        analysis_id=analysis_id,
                        error=result.get('error', 'Unknown error occurred')
                    )
                    return

                # Mark analysis as completed with results
                print(f"[BG_TASK] Status is not 'error', proceeding with completion")
                insights_count = len(result.get('insights', []))
                recommendations_count = len(result.get('recommendations', []))

                analytics_results = result.get('analytics_results', {})
                analytics_summary = {
                    'anomalies_count': len(analytics_results.get('anomalies', [])),
                    'forecasts_count': len(analytics_results.get('forecasts', [])),
                    'causal_relationships_count': len(analytics_results.get('causal', {}).get('relationships', [])),
                    'variance_components_count': len(analytics_results.get('variance', {}).get('components', []))
                }

                print(f"[BG_TASK] Calling mark_completed with:")
                print(f"  - analysis_id: {analysis_id}")
                print(f"  - dashboard_path: {result.get('dashboard_path', '')}")
                print(f"  - report_path: {result.get('report_path', '')}")
                print(f"  - insights: {insights_count}, recommendations: {recommendations_count}")

                updated_analysis = AnalysisSessionRepository.mark_completed(
                    session=db,
                    analysis_id=analysis_id,
                    dashboard_path=result.get('dashboard_path', ''),
                    report_path=result.get('report_path', ''),
                    executive_summary=result.get('executive_summary', ''),
                    insights_count=insights_count,
                    recommendations_count=recommendations_count,
                    analytics_summary=analytics_summary
                )

                print(f"[BG_TASK] mark_completed returned: {updated_analysis}")
                print(f"[BG_TASK] Exiting context manager (should auto-commit)")

            print(f"[BG_TASK] Database session closed")

            # Emit completion event to SSE clients
            print(f"[BG_TASK] Emitting completion event to SSE clients")
            await AnalysisProgressService.mark_completed(
                analysis_id=analysis_id,
                dashboard_url=f"/api/analyses/{analysis_id}/dashboard",
                report_path=result.get('report_path', ''),
                insights_count=insights_count,
                recommendations_count=recommendations_count
            )
            print(f"[BG_TASK] Completion event emitted")

        except Exception as e:
            print(f"[BG_TASK ERROR] Exception during database update: {e}")
            import traceback
            traceback.print_exc()
            raise

    except Exception as e:
        # Mark analysis as failed in database
        with DatabaseManager.get_session() as db:
            AnalysisSessionRepository.mark_failed(
                session=db,
                analysis_id=analysis_id,
                error_message=str(e)
            )

        # Emit error event to SSE clients
        await AnalysisProgressService.mark_failed(
            analysis_id=analysis_id,
            error=str(e)
        )
        print(f"[ERROR] Background workflow failed: {e}")


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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start business discovery workflow on user's datasets.

    Returns immediately with analysis_id. The workflow runs in the background
    and progress can be monitored via the /analyses/{analysis_id}/stream endpoint.

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

        # Create AnalysisSession record with "running" status
        analysis_session = AnalysisSessionRepository.create(
            session=db,
            user_id=request.user_id,
            company_id=user.company_id,
            dataset_ids=dataset_ids,
            name=request.analysis_name or "Business Analysis",
            description=f"Analysis of {len(dataset_ids)} dataset(s)"
        )
        analysis_id = analysis_session.id

        # CRITICAL: Commit the session BEFORE scheduling background task
        # The background task uses a different session and needs to see this record
        db.commit()
        print(f"[API] Analysis {analysis_id} created and committed to database")

        # Initialize progress tracking BEFORE scheduling background task
        # This prevents race condition where SSE client connects before progress is initialized
        AnalysisProgressService.initialize_analysis(analysis_id)

        # Schedule background task to run the workflow
        background_tasks.add_task(
            run_business_discovery_background,
            company_id=user.company_id,
            dataset_ids=dataset_ids,
            analysis_id=analysis_id,
            analysis_name=request.analysis_name or "Business Analysis"
        )

        # Return immediately with analysis_id and status="running"
        return BusinessDiscoveryResponse(
            success=True,
            analysis_id=analysis_id,
            company_id=user.company_id,
            dataset_count=len(dataset_ids),
            insights=[],  # Will be populated when workflow completes
            synthesized_insights=[],
            recommendations=[],
            analytics_results={},
            executive_summary="Analysis started. Monitor progress via /analyses/{analysis_id}/stream",
            dashboard_url=f"/api/analyses/{analysis_id}/dashboard",
            error=None
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting business discovery: {str(e)}"
        )

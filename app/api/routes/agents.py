"""AI Agent routes for discovery and ETL workflows."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
from pathlib import Path

from app.api.deps import get_db
from app.services.agent_service import AgentService
from app.schemas.agent import DiscoveryRequest, DiscoveryResponse, ETLRequest, ETLResponse

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

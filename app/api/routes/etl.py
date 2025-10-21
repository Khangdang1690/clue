"""ETL API routes for file upload and processing."""

import os
import json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db
from app.services.etl_service import ETLService
from app.services.company_service import CompanyService
from app.schemas.etl import DatasetResponse
from src.database.repository import DatasetRepository

router = APIRouter()


def get_user_id_from_header(authorization: Optional[str] = Header(None)) -> str:
    """Extract user ID from authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if authorization.startswith("Bearer "):
        user_id = authorization.replace("Bearer ", "")
        return user_id

    raise HTTPException(status_code=401, detail="Invalid authorization header")


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    force_actions: Optional[str] = None,  # JSON string mapping file_path to action
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Upload and process CSV/Excel files through ETL workflow.

    Streams progress updates via Server-Sent Events (SSE).

    Args:
        files: List of files to upload
        force_actions: Optional JSON string like '{"sales.csv": "replace", "inventory.csv": "skip"}'
        user_id: User ID from authorization header
        db: Database session

    The client should listen for SSE events with the following formats:
    ```
    # Normal progress
    data: {"step": "processing_file", "progress": 50, "message": "Processing file...", "status": "running"}

    # Duplicate detected (pauses upload, waits for user choice)
    data: {"step": "duplicate_detected", "status": "duplicate_detected", "options": ["skip", "replace", "append_anyway"], ...}

    # Completed
    data: {"step": "completed", "progress": 100, "status": "completed", "data": {...}}
    ```
    """
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Check if user has a company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company before uploading files"
        )

    # Validate file types
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    for file in files:
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
            )

    # Parse force_actions if provided
    force_actions_dict = {}
    if force_actions:
        try:
            force_actions_dict = json.loads(force_actions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid force_actions JSON")

    try:
        # Save uploaded files to temp directory
        file_paths = await ETLService.save_uploaded_files(files)

        # Map force actions - keep using filename as key since service expects it
        force_actions_by_filename = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            if file_name in force_actions_dict:
                force_actions_by_filename[file_name] = force_actions_dict[file_name]

        # Stream ETL progress
        async def event_stream():
            try:
                async for progress_update in ETLService.process_files_with_progress(
                    company_id=company.id,
                    file_paths=file_paths,
                    force_actions=force_actions_by_filename
                ):
                    yield progress_update
            finally:
                # Cleanup temp files after streaming completes
                ETLService.cleanup_temp_files(file_paths)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for nginx
            }
        )

    except Exception as e:
        # Cleanup on error
        if 'file_paths' in locals():
            ETLService.cleanup_temp_files(file_paths)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/datasets", response_model=List[DatasetResponse])
async def get_datasets(
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Get all datasets for the user's company.
    """
    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company first"
        )

    # Get datasets for company
    datasets = DatasetRepository.get_datasets_by_company(db, company.id)
    return datasets


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Get a specific dataset by ID.

    Validates that the dataset belongs to the user's company.
    """
    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company first"
        )

    # Get dataset
    dataset = DatasetRepository.get_dataset_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify dataset belongs to user's company
    if dataset.company_id != company.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return dataset


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    cascade: bool = True,
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Delete a dataset with optional cascade deletion.

    Args:
        dataset_id: Dataset ID to delete
        cascade: If True, cascade delete relationships and column metadata (default: True)
        user_id: User ID from authorization header
        db: Database session

    Returns:
        Deletion summary with deleted objects

    Raises:
        403: If dataset doesn't belong to user's company
        404: If dataset not found
    """
    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company first"
        )

    # Get dataset
    dataset = DatasetRepository.get_dataset_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify dataset belongs to user's company
    if dataset.company_id != company.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Save dataset name before deletion (object will be detached after delete)
    dataset_name = dataset.table_name

    # Use DatasetManager to perform cascade deletion
    from src.etl.dataset_manager import DatasetManager
    manager = DatasetManager()

    try:
        result = manager.delete_dataset(
            dataset_id=dataset_id,
            cascade=cascade,
            confirm=True  # Confirm deletion
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted": {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "cascade": cascade,
                "relationships_deleted": len(result.get("deleted_objects", {}).get("relationships", [])),
                "columns_deleted": len(result.get("deleted_objects", {}).get("columns", []))
            }
        }

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Delete dataset failed:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@router.get("/datasets/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: str,
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Get dataset data (rows) with pagination.

    Args:
        dataset_id: Dataset ID
        limit: Number of rows to return (default: 100, max: 1000)
        offset: Number of rows to skip (default: 0)
        user_id: User ID from authorization header
        db: Database session

    Returns:
        {
            "columns": ["col1", "col2", ...],
            "rows": [[val1, val2, ...], ...],
            "total_rows": 1234,
            "limit": 100,
            "offset": 0
        }

    Raises:
        403: If dataset doesn't belong to user's company
        404: If dataset not found
    """
    # Limit validation
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be at least 1")
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset cannot be negative")

    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company first"
        )

    # Get dataset
    dataset = DatasetRepository.get_dataset_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify dataset belongs to user's company
    if dataset.company_id != company.id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from src.database.connection import DatabaseManager
        import pandas as pd

        # Get database connection
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()

        # Get total row count
        total_rows = dataset.row_count or 0

        # Construct actual table name (same logic as in DatasetRepository.store_dataframe)
        # The actual PostgreSQL table name is: {company_name}_cleaned_{table_name}
        actual_table_name = f"{company.name}_cleaned_{dataset.table_name}"
        actual_table_name = actual_table_name.lower().replace(' ', '_').replace('-', '_')

        # Fetch data with pagination
        query = f'SELECT * FROM "{actual_table_name}" LIMIT {limit} OFFSET {offset}'
        df = pd.read_sql(query, engine)

        # Convert to list format
        columns = df.columns.tolist()
        rows = df.values.tolist()

        # Convert NaN/None to null for JSON serialization
        import math
        rows_cleaned = []
        for row in rows:
            cleaned_row = [None if (isinstance(v, float) and math.isnan(v)) else v for v in row]
            rows_cleaned.append(cleaned_row)

        return {
            "columns": columns,
            "rows": rows_cleaned,
            "total_rows": total_rows,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(rows_cleaned) < total_rows
        }

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Get dataset data failed:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch dataset data: {str(e)}"
        )


@router.get("/schema")
async def get_schema(
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Get complete schema view: all datasets with their columns and relationships.

    Returns:
        {
            "datasets": [
                {
                    "id": "uuid",
                    "name": "table_name",
                    "domain": "Sales",
                    "description": "...",
                    "row_count": 1234,
                    "columns": [
                        {
                            "name": "customer_id",
                            "data_type": "string",
                            "semantic_type": "key",
                            "is_primary_key": true,
                            "is_foreign_key": false,
                            "business_meaning": "..."
                        }
                    ]
                }
            ],
            "relationships": [
                {
                    "id": "uuid",
                    "from_dataset_id": "uuid1",
                    "to_dataset_id": "uuid2",
                    "from_column": "customer_id",
                    "to_column": "customer_id",
                    "relationship_type": "one_to_many",
                    "confidence": 0.95,
                    "match_percentage": 98.5
                }
            ]
        }
    """
    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="User must create a company first"
        )

    try:
        from src.database.models import ColumnMetadata, TableRelationship

        # Get all datasets for this company
        datasets = DatasetRepository.get_datasets_by_company(db, company.id)

        # Build dataset schema
        datasets_schema = []
        for dataset in datasets:
            # Get columns for this dataset
            columns = db.query(ColumnMetadata).filter(
                ColumnMetadata.dataset_id == dataset.id
            ).order_by(ColumnMetadata.position).all()

            columns_data = [
                {
                    "name": col.column_name,
                    "data_type": col.data_type,
                    "semantic_type": col.semantic_type,
                    "is_primary_key": col.is_primary_key,
                    "is_foreign_key": col.is_foreign_key,
                    "business_meaning": col.business_meaning,
                    "position": col.position
                }
                for col in columns
            ]

            datasets_schema.append({
                "id": dataset.id,
                "name": dataset.table_name,
                "original_filename": dataset.original_filename,
                "domain": dataset.domain,
                "description": dataset.description,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "columns": columns_data,
                # Business context fields
                "department": dataset.department,
                "dataset_type": dataset.dataset_type,
                "time_period": dataset.time_period,
                "entities": dataset.entities or [],
                "typical_use_cases": dataset.typical_use_cases or [],
                "business_context": dataset.business_context or {}
            })

        # Get all relationships for datasets in this company
        dataset_ids = [d.id for d in datasets]

        relationships = db.query(TableRelationship).filter(
            TableRelationship.from_dataset_id.in_(dataset_ids)
        ).all()

        relationships_data = [
            {
                "id": rel.id,
                "from_dataset_id": rel.from_dataset_id,
                "to_dataset_id": rel.to_dataset_id,
                "from_column": rel.from_column,
                "to_column": rel.to_column,
                "relationship_type": rel.relationship_type,
                "confidence": float(rel.confidence) if rel.confidence else 0.0,
                "match_percentage": float(rel.match_percentage) if rel.match_percentage else 0.0,
                "join_strategy": rel.join_strategy
            }
            for rel in relationships
        ]

        return {
            "datasets": datasets_schema,
            "relationships": relationships_data
        }

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Get schema failed:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch schema: {str(e)}"
        )

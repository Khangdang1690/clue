"""File upload and management endpoints."""

import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.models import FileUploadResponse
from api.websocket import ws_manager


router = APIRouter(prefix="/api", tags=["files"])


@router.post("/upload-files", response_model=FileUploadResponse)
async def upload_files(
    department: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload data files for a specific department.

    Args:
        department: Department name (e.g., "Marketing", "Sales", "Product", "Support")
        files: List of files to upload (CSV, Excel, PDF)

    Returns:
        FileUploadResponse with upload status and file details
    """
    try:
        # Create department directory
        uploads_dir = Path("data/uploads") / department
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Broadcast log
        await ws_manager.broadcast_log(f"Uploading files for department: {department}")

        uploaded_files = []
        total_size = 0

        for file in files:
            # Validate file type
            allowed_extensions = {'.csv', '.xlsx', '.xls', '.pdf'}
            file_ext = Path(file.filename).suffix.lower()

            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not allowed. Allowed: {allowed_extensions}"
                )

            # Save file
            file_path = uploads_dir / file.filename

            with file_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)
                total_size += len(content)

            uploaded_files.append(file.filename)
            await ws_manager.broadcast_log(f"  ✓ Uploaded: {file.filename}")

        # Format total size
        size_str = f"{total_size / (1024 * 1024):.2f} MB" if total_size > 1024 * 1024 else f"{total_size / 1024:.2f} KB"

        await ws_manager.broadcast_log(f"✓ Upload complete: {len(uploaded_files)} files ({size_str})")

        return FileUploadResponse(
            status="success",
            department=department,
            files_uploaded=uploaded_files,
            total_size=size_str
        )

    except HTTPException:
        raise
    except Exception as e:
        await ws_manager.broadcast_error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/departments/files")
async def list_department_files():
    """
    List all uploaded files grouped by department.

    Returns:
        Dictionary mapping department names to list of uploaded files
    """
    try:
        uploads_dir = Path("data/uploads")

        if not uploads_dir.exists():
            return {}

        departments = {}
        for dept_dir in uploads_dir.iterdir():
            if dept_dir.is_dir():
                files = [f.name for f in dept_dir.iterdir() if f.is_file()]
                departments[dept_dir.name] = files

        return departments

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/departments/{department}/files/{filename}")
async def delete_file(department: str, filename: str):
    """
    Delete a specific uploaded file.

    Args:
        department: Department name
        filename: Name of file to delete

    Returns:
        Success message
    """
    try:
        file_path = Path("data/uploads") / department / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_path.unlink()
        await ws_manager.broadcast_log(f"Deleted file: {department}/{filename}")

        return {"status": "success", "message": f"File {filename} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

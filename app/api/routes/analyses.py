"""Analysis routes for managing and retrieving analysis results."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import os
from pathlib import Path
import markdown
import tempfile
import asyncio
import json
import shutil

from app.api.deps import get_db
from app.services.storage_service import StorageService
from src.database.repository import AnalysisSessionRepository
from src.database.models import User

router = APIRouter()


def get_file_content(file_path: str, as_string: bool = False, encoding: str = 'utf-8'):
    """
    Get file content from storage (GCS in production, local in development).

    Args:
        file_path: Path from database
        as_string: If True, return as string; otherwise return as bytes
        encoding: Text encoding for string mode

    Returns:
        File content as string or bytes
    """
    # Use StorageService which automatically handles ENV-based storage
    storage_service = StorageService()

    # Check if path starts with "data/outputs/" (old format - backward compatibility)
    if file_path.startswith("data/outputs/"):
        # Old format: local filesystem path - read directly
        if as_string:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        else:
            with open(file_path, 'rb') as f:
                return f.read()
    else:
        # New format: storage path (GCS or local based on ENV)
        if as_string:
            return storage_service.download_as_string(file_path, encoding=encoding)
        else:
            return storage_service.download_as_bytes(file_path)


@router.get("/analyses")
async def list_analyses(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get list of analyses for a user.

    Returns most recent analyses first.
    """
    try:
        analyses = AnalysisSessionRepository.get_by_user(
            session=db,
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        # Convert to dict for JSON response
        result = []
        for analysis in analyses:
            result.append({
                "id": analysis.id,
                "name": analysis.name,
                "description": analysis.description,
                "dataset_count": len(analysis.dataset_ids) if analysis.dataset_ids else 0,
                "insights_generated": analysis.insights_generated,
                "recommendations_generated": analysis.recommendations_generated,
                "executive_summary": analysis.executive_summary,
                "analytics_summary": analysis.analytics_summary,
                "status": analysis.status,
                "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
                "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
                "error_message": analysis.error_message
            })

        return {
            "success": True,
            "analyses": result,
            "count": len(result)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching analyses: {str(e)}"
        )


@router.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Get metadata for a specific analysis.
    """
    try:
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "success": True,
            "analysis": {
                "id": analysis.id,
                "name": analysis.name,
                "description": analysis.description,
                "dataset_ids": analysis.dataset_ids,
                "dataset_count": len(analysis.dataset_ids) if analysis.dataset_ids else 0,
                "insights_generated": analysis.insights_generated,
                "recommendations_generated": analysis.recommendations_generated,
                "executive_summary": analysis.executive_summary,
                "analytics_summary": analysis.analytics_summary,
                "dashboard_path": analysis.dashboard_path,
                "report_path": analysis.report_path,
                "status": analysis.status,
                "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
                "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
                "error_message": analysis.error_message
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching analysis: {str(e)}"
        )


@router.get("/analyses/{analysis_id}/stream")
async def stream_analysis_messages(analysis_id: str):
    """
    Stream real-time narrative messages as analysis runs (Claude-style).

    Uses HTTP chunked transfer with NDJSON format.
    Each line is a complete JSON message object.

    Message types:
    - message: Narrative update (e.g., "Loading datasets from database...")
    - complete: Analysis finished
    - error: Analysis failed
    """
    from app.services.analysis_message_service import AnalysisMessageService

    async def message_generator():
        """Generate message chunks as analysis runs."""
        try:
            print(f"[STREAM] Client connecting for analysis {analysis_id}")
            queue = await AnalysisMessageService.register_client(analysis_id)
            print(f"[STREAM] Client registered")

            try:
                while True:
                    try:
                        # Wait for messages (with keepalive timeout)
                        message = await asyncio.wait_for(queue.get(), timeout=15.0)

                        # Send as NDJSON (newline-delimited JSON)
                        chunk = json.dumps(message) + "\n"
                        print(f"[STREAM] Sending: type={message.get('type')}, content={message.get('content', '')[:50]}...")
                        yield chunk

                        # Close stream on completion or error
                        if message.get('type') in ('complete', 'error'):
                            print(f"[STREAM] Stream ending: {message.get('type')}")
                            break

                    except asyncio.TimeoutError:
                        # Keepalive ping
                        yield json.dumps({"type": "keepalive"}) + "\n"

            except asyncio.CancelledError:
                print(f"[STREAM] Client disconnected")
            finally:
                await AnalysisMessageService.unregister_client(analysis_id, queue)

        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(
        message_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/analyses/{analysis_id}/dashboard", response_class=HTMLResponse)
async def get_dashboard(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Serve the interactive dashboard HTML for an analysis.
    """
    try:
        print(f"[API] Dashboard request for analysis: {analysis_id}")
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            print(f"[API] Analysis not found in database")
            raise HTTPException(status_code=404, detail="Analysis not found")

        print(f"[API] Analysis found. Status: {analysis.status}")
        print(f"[API] Dashboard path in DB: {analysis.dashboard_path}")

        if not analysis.dashboard_path:
            print(f"[API] Dashboard path is None or empty")
            raise HTTPException(status_code=404, detail="Dashboard not available for this analysis")

        # Get content from storage
        try:
            html_content = get_file_content(analysis.dashboard_path, as_string=True)
            print(f"[API] Dashboard loaded successfully from storage")
        except Exception as e:
            print(f"[API] Failed to load dashboard: {e}")
            raise HTTPException(status_code=404, detail="Dashboard file not found")

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving dashboard: {str(e)}"
        )


@router.get("/analyses/{analysis_id}/report", response_class=PlainTextResponse)
async def get_report(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Serve the markdown report for an analysis.
    """
    try:
        print(f"[API] Report request for analysis: {analysis_id}")
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            print(f"[API] Analysis not found in database")
            raise HTTPException(status_code=404, detail="Analysis not found")

        print(f"[API] Analysis found. Status: {analysis.status}")
        print(f"[API] Report path in DB: {analysis.report_path}")

        if not analysis.report_path:
            print(f"[API] Report path is None or empty")
            raise HTTPException(status_code=404, detail="Report not available for this analysis")

        # Get content from storage
        # Try UTF-8 first, fall back to cp1252 for legacy files
        try:
            markdown_content = get_file_content(analysis.report_path, as_string=True, encoding='utf-8')
            print(f"[API] Report loaded successfully from storage")
        except UnicodeDecodeError:
            try:
                markdown_content = get_file_content(analysis.report_path, as_string=True, encoding='cp1252')
            except Exception as e:
                print(f"[API] Failed to load report: {e}")
                raise HTTPException(status_code=404, detail="Report file not found")
        except Exception as e:
            print(f"[API] Failed to load report: {e}")
            raise HTTPException(status_code=404, detail="Report file not found")

        return PlainTextResponse(content=markdown_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving report: {str(e)}"
        )


@router.get("/analyses/{analysis_id}/download")
async def download_report(
    analysis_id: str,
    format: str = "html",  # html or md
    db: Session = Depends(get_db)
):
    """
    Download the report as a beautifully styled HTML file or raw markdown.
    """
    try:
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        if not analysis.report_path:
            raise HTTPException(status_code=404, detail="Report not available for this analysis")

        # Get content from GCS or local filesystem
        # Read markdown with proper encoding
        try:
            markdown_content = get_file_content(analysis.report_path, as_string=True, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                markdown_content = get_file_content(analysis.report_path, as_string=True, encoding='cp1252')
            except Exception as e:
                raise HTTPException(status_code=404, detail="Report file not found")
        except Exception as e:
            raise HTTPException(status_code=404, detail="Report file not found")

        if format == "md":
            # Return raw markdown file
            filename = f"{analysis.name.replace(' ', '_')}_{analysis_id[:8]}.md"

            # Create temp file for FileResponse
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(markdown_content)
                temp_path = temp_file.name

            return FileResponse(
                path=temp_path,
                media_type="text/markdown",
                filename=filename,
                background=lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
            )
        else:
            # Convert to beautiful HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'nl2br']
            )

            # Professional HTML template matching frontend styling
            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{analysis.name} - Business Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #334155;
            background: #f8fafc;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 60px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 1.5rem;
            margin-top: 2rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #e2e8f0;
        }}

        h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1rem;
            margin-top: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}

        h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.75rem;
            margin-top: 1.5rem;
        }}

        h4 {{
            font-size: 1.125rem;
            font-weight: 500;
            color: #475569;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }}

        p {{
            color: #475569;
            margin-bottom: 1rem;
            line-height: 1.7;
        }}

        ul, ol {{
            margin-bottom: 1rem;
            margin-left: 1.5rem;
        }}

        li {{
            color: #475569;
            margin-bottom: 0.5rem;
            line-height: 1.7;
        }}

        ul {{
            list-style-type: disc;
        }}

        ul li::marker {{
            color: #3b82f6;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }}

        thead {{
            background: #f1f5f9;
        }}

        th {{
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: #334155;
            font-size: 0.875rem;
        }}

        td {{
            padding: 12px 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 0.875rem;
            color: #475569;
        }}

        tbody tr:hover {{
            background: #f8fafc;
        }}

        strong {{
            font-weight: 600;
            color: #0f172a;
        }}

        em {{
            font-style: italic;
            color: #334155;
        }}

        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            color: #3b82f6;
        }}

        pre {{
            background: #1e293b;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}

        pre code {{
            background: none;
            padding: 0;
            color: #e2e8f0;
        }}

        blockquote {{
            border-left: 4px solid #3b82f6;
            padding-left: 16px;
            padding: 12px 16px;
            margin: 1rem 0;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 2rem 0;
        }}

        a {{
            color: #3b82f6;
            text-decoration: underline;
            text-decoration-color: #3b82f6;
        }}

        a:hover {{
            color: #2563eb;
        }}

        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 2px solid #e2e8f0;
        }}

        .header h1 {{
            border: none;
            margin: 0;
            padding: 0;
            font-size: 2.5rem;
        }}

        .header p {{
            color: #64748b;
            margin-top: 0.5rem;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{analysis.name}</h1>
            <p>Business Analysis Report</p>
        </div>
        {html_content}
    </div>
</body>
</html>"""

            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(full_html)
                temp_path = temp_file.name

            # Generate filename
            filename = f"{analysis.name.replace(' ', '_')}_{analysis_id[:8]}.html"

            # Return as file download
            return FileResponse(
                path=temp_path,
                media_type="text/html",
                filename=filename,
                background=lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading report: {str(e)}"
        )


@router.get("/analyses/{analysis_id}/viz-data")
async def get_viz_data(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Serve the visualization data JSON for an analysis.
    """
    try:
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Try to get viz_data_path from analysis state (if stored)
        # For backward compatibility, construct path if not in database
        viz_data_path = None

        # Check if analysis has viz_data_path stored (new GCS approach)
        if hasattr(analysis, 'viz_data_path') and analysis.viz_data_path:
            viz_data_path = analysis.viz_data_path
        else:
            # Backward compatibility: construct old local path
            viz_data_path = os.path.join(
                "analyses", analysis.company_id, analysis_id, "viz_data.json"
            )

        # Get content from storage
        try:
            json_bytes = get_file_content(viz_data_path, as_string=False)
            print(f"[API] Viz data loaded successfully from storage")

            # Create temp file to return as FileResponse
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as temp_file:
                temp_file.write(json_bytes)
                temp_path = temp_file.name

            return FileResponse(
                path=temp_path,
                media_type="application/json",
                filename=f"viz_data_{analysis_id[:8]}.json",
                background=lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None
            )
        except Exception as e:
            print(f"[API] Failed to load viz data: {e}")
            raise HTTPException(status_code=404, detail="Visualization data not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving visualization data: {str(e)}"
        )


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an analysis and its associated files.
    """
    try:
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Delete files from storage (GCS in production, local in development)
        storage_service = StorageService()
        storage_type = "GCS" if storage_service.use_gcs else "local storage"

        try:
            # Delete analysis files from storage
            files_deleted = []

            # Delete report
            if analysis.report_path:
                try:
                    storage_service.delete_file(analysis.report_path)
                    files_deleted.append(analysis.report_path)
                    print(f"[DELETE] Deleted report from {storage_type}: {analysis.report_path}")
                except Exception as e:
                    print(f"[DELETE WARN] Failed to delete report: {e}")

            # Delete dashboard
            if analysis.dashboard_path:
                try:
                    storage_service.delete_file(analysis.dashboard_path)
                    files_deleted.append(analysis.dashboard_path)
                    print(f"[DELETE] Deleted dashboard from {storage_type}: {analysis.dashboard_path}")
                except Exception as e:
                    print(f"[DELETE WARN] Failed to delete dashboard: {e}")

            # Delete viz_data if stored
            if hasattr(analysis, 'viz_data_path') and analysis.viz_data_path:
                try:
                    storage_service.delete_file(analysis.viz_data_path)
                    files_deleted.append(analysis.viz_data_path)
                    print(f"[DELETE] Deleted viz_data from {storage_type}: {analysis.viz_data_path}")
                except Exception as e:
                    print(f"[DELETE WARN] Failed to delete viz_data: {e}")

            print(f"[DELETE] Deleted {len(files_deleted)} files from {storage_type}")
        except Exception as e:
            print(f"[DELETE ERROR] Failed to delete analysis files from {storage_type}: {e}")

        # Also delete old local directory if it exists (backward compatibility)
        analysis_dir = os.path.join(
            "data", "outputs", "analyses",
            analysis.company_id, analysis_id
        )
        if os.path.exists(analysis_dir):
            try:
                shutil.rmtree(analysis_dir)
                print(f"[DELETE] Removed old local directory: {analysis_dir}")
            except Exception as e:
                print(f"[DELETE WARN] Failed to remove old directory {analysis_dir}: {e}")

        # Delete database record
        AnalysisSessionRepository.delete(
            session=db,
            analysis_id=analysis_id
        )

        return {
            "success": True,
            "message": "Analysis deleted successfully",
            "storage_type": storage_type,
            "files_deleted": len(files_deleted) if 'files_deleted' in locals() else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting analysis: {str(e)}"
        )

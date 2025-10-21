"""Analysis routes for managing and retrieving analysis results."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from sqlalchemy.orm import Session
from typing import List
import os
from pathlib import Path
import markdown
import tempfile

from app.api.deps import get_db
from src.database.repository import AnalysisSessionRepository
from src.database.models import User

router = APIRouter()


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


@router.get("/analyses/{analysis_id}/dashboard", response_class=HTMLResponse)
async def get_dashboard(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Serve the interactive dashboard HTML for an analysis.
    """
    try:
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        if not analysis.dashboard_path:
            raise HTTPException(status_code=404, detail="Dashboard not available for this analysis")

        # Construct full path from relative path
        full_path = os.path.join("data", "outputs", analysis.dashboard_path)

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Dashboard file not found")

        # Read and return HTML content
        with open(full_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

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
        analysis = AnalysisSessionRepository.get_by_id(
            session=db,
            analysis_id=analysis_id
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        if not analysis.report_path:
            raise HTTPException(status_code=404, detail="Report not available for this analysis")

        # Construct full path from relative path
        full_path = os.path.join("data", "outputs", analysis.report_path)

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Report file not found")

        # Read and return markdown content
        # Try UTF-8 first, fall back to Windows-1252 for legacy files
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except UnicodeDecodeError:
            # Try Windows-1252 (cp1252) which is the Windows default encoding
            # This handles legacy reports with bullet points and other special chars
            try:
                with open(full_path, 'r', encoding='cp1252') as f:
                    markdown_content = f.read()
            except UnicodeDecodeError:
                # Last resort: UTF-8 with error replacement
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    markdown_content = f.read()

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

        # Construct full path from relative path
        full_path = os.path.join("data", "outputs", analysis.report_path)

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Report file not found")

        # Read markdown with proper encoding
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='cp1252') as f:
                    markdown_content = f.read()
            except UnicodeDecodeError:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    markdown_content = f.read()

        if format == "md":
            # Return raw markdown file
            filename = f"{analysis.name.replace(' ', '_')}_{analysis_id[:8]}.md"
            return FileResponse(
                path=full_path,
                media_type="text/markdown",
                filename=filename
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

        # Construct path to viz_data.json
        # Format: data/outputs/analyses/{company_id}/{analysis_id}/viz_data.json
        viz_data_path = os.path.join(
            "data", "outputs", "analyses",
            analysis.company_id, analysis_id, "viz_data.json"
        )

        if not os.path.exists(viz_data_path):
            raise HTTPException(status_code=404, detail="Visualization data not found")

        return FileResponse(
            path=viz_data_path,
            media_type="application/json",
            filename=f"viz_data_{analysis_id[:8]}.json"
        )

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

        # Delete files if they exist
        files_deleted = []
        if analysis.dashboard_path:
            dashboard_full_path = os.path.join("data", "outputs", analysis.dashboard_path)
            if os.path.exists(dashboard_full_path):
                os.remove(dashboard_full_path)
                files_deleted.append("dashboard")

        if analysis.report_path:
            report_full_path = os.path.join("data", "outputs", analysis.report_path)
            if os.path.exists(report_full_path):
                os.remove(report_full_path)
                files_deleted.append("report")

        # Delete viz_data.json
        viz_data_path = os.path.join(
            "data", "outputs", "analyses",
            analysis.company_id, analysis_id, "viz_data.json"
        )
        if os.path.exists(viz_data_path):
            os.remove(viz_data_path)
            files_deleted.append("viz_data")

        # Try to delete the analysis directory if empty
        if analysis.dashboard_path:
            analysis_dir = os.path.dirname(os.path.join("data", "outputs", analysis.dashboard_path))
            try:
                if os.path.exists(analysis_dir) and not os.listdir(analysis_dir):
                    os.rmdir(analysis_dir)
            except:
                pass  # Directory not empty or other error, ignore

        # Delete database record
        AnalysisSessionRepository.delete(
            session=db,
            analysis_id=analysis_id
        )

        return {
            "success": True,
            "message": "Analysis deleted successfully",
            "files_deleted": files_deleted
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting analysis: {str(e)}"
        )

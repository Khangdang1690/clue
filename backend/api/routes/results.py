"""Results and outputs endpoints - challenges, visualizations, reports, dashboard."""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from api.models import ChallengeStatusResponse, ChallengeInfo
from src.graph.workflow import ETLInsightsWorkflow
from src.models.challenge import Challenge


router = APIRouter(prefix="/api", tags=["results"])
workflow = ETLInsightsWorkflow()


@router.get("/challenges", response_model=ChallengeStatusResponse)
async def get_challenges_status():
    """
    Get challenge queue status.

    Returns:
        Total challenges, processed count, remaining count, and next challenge
    """
    try:
        status = workflow.get_challenge_status()

        next_challenge = None
        if status.get("next_challenge"):
            challenge = status["next_challenge"]
            next_challenge = ChallengeInfo(
                id=challenge.id,
                title=challenge.title,
                priority_score=challenge.priority_score,
                priority_level=challenge.priority_level.value,
                department=challenge.department,
                description=challenge.description
            )

        return ChallengeStatusResponse(
            total_challenges=status.get("total_challenges", 0),
            processed=status.get("processed", 0),
            remaining=status.get("remaining", 0),
            next_challenge=next_challenge,
            error=status.get("error")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get challenge status: {str(e)}")


@router.get("/challenges/all")
async def get_all_challenges():
    """
    Get all challenges (both processed and remaining).

    Returns:
        List of all challenges with their details
    """
    try:
        all_challenges_dict = workflow.chroma_manager.get_all_challenges()

        if not all_challenges_dict:
            return []

        challenges = []
        for c_dict in all_challenges_dict:
            challenge = Challenge(**c_dict)
            challenges.append({
                "id": challenge.id,
                "title": challenge.title,
                "description": challenge.description,
                "priority_score": challenge.priority_score,
                "priority_level": challenge.priority_level.value,
                "department": challenge.department,
                "success_metrics": challenge.success_metrics,
                "data_sources": challenge.data_sources_needed
            })

        # Sort by priority score (descending)
        challenges.sort(key=lambda x: x["priority_score"], reverse=True)

        return challenges

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get challenges: {str(e)}")


@router.get("/visualizations")
async def list_visualizations():
    """
    List all generated visualizations.

    Returns:
        List of visualization files with metadata
    """
    try:
        viz_dir = Path("data/outputs/visualizations")

        if not viz_dir.exists():
            return []

        visualizations = []
        for viz_file in viz_dir.glob("*.png"):
            visualizations.append({
                "id": viz_file.stem,
                "filename": viz_file.name,
                "path": f"/api/visualizations/{viz_file.name}",
                "size": viz_file.stat().st_size,
                "created": viz_file.stat().st_ctime
            })

        # Sort by creation time (most recent first)
        visualizations.sort(key=lambda x: x["created"], reverse=True)

        return visualizations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list visualizations: {str(e)}")


@router.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """
    Download a specific visualization image.

    Args:
        filename: Name of the visualization file

    Returns:
        Image file
    """
    try:
        viz_path = Path("data/outputs/visualizations") / filename

        if not viz_path.exists() or not viz_path.is_file():
            raise HTTPException(status_code=404, detail="Visualization not found")

        return FileResponse(
            viz_path,
            media_type="image/png",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualization: {str(e)}")


@router.get("/reports")
async def list_reports():
    """
    List all generated reports.

    Returns:
        List of report files with metadata
    """
    try:
        reports_dir = Path("data/outputs/reports")

        if not reports_dir.exists():
            return {"analytical_reports": [], "business_reports": []}

        analytical_reports = []
        business_reports = []

        # Look for both .md and .pdf files
        for report_file in reports_dir.glob("*"):
            if report_file.suffix not in ['.md', '.pdf']:
                continue

            file_size_kb = report_file.stat().st_size / 1024
            size_str = f"{file_size_kb:.2f} KB" if file_size_kb < 1024 else f"{file_size_kb / 1024:.2f} MB"

            report_info = {
                "filename": report_file.name,
                "path": f"/api/reports/{report_file.name}",
                "size": size_str,
                "type": report_file.suffix.lstrip('.'),
                "created": report_file.stat().st_ctime
            }

            if "analytical" in report_file.name.lower():
                analytical_reports.append(report_info)
            elif "business" in report_file.name.lower():
                business_reports.append(report_info)

        return {
            "analytical_reports": sorted(analytical_reports, key=lambda x: x["created"], reverse=True),
            "business_reports": sorted(business_reports, key=lambda x: x["created"], reverse=True)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@router.get("/reports/{filename}")
async def get_report(filename: str):
    """
    Download a specific report file (MD or PDF).

    Args:
        filename: Name of the report file

    Returns:
        Report file
    """
    try:
        report_path = Path("data/outputs/reports") / filename

        if not report_path.exists() or not report_path.is_file():
            raise HTTPException(status_code=404, detail="Report not found")

        # Determine media type based on file extension
        media_type = "application/pdf" if filename.endswith('.pdf') else "text/markdown"

        return FileResponse(
            report_path,
            media_type=media_type,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """
    Get the interactive dashboard HTML.

    Returns:
        HTML content of the dashboard
    """
    try:
        # Find the most recent dashboard file
        dashboard_dir = Path("data/outputs/dashboards")

        if not dashboard_dir.exists():
            raise HTTPException(status_code=404, detail="No dashboard generated yet")

        # Look for both naming patterns
        dashboard_files = list(dashboard_dir.glob("*.html"))

        if not dashboard_files:
            raise HTTPException(status_code=404, detail="No dashboard generated yet")

        # Get most recent dashboard
        latest_dashboard = max(dashboard_files, key=lambda f: f.stat().st_ctime)

        with latest_dashboard.open("r", encoding="utf-8") as f:
            html_content = f.read()

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard: {str(e)}")


@router.get("/analysis-results")
async def get_analysis_results():
    """
    Get all analysis results from completed challenges.

    Returns:
        List of analysis results with findings, recommendations, and visualizations
    """
    try:
        analyses = workflow._load_all_analyses_from_chromadb()

        if not analyses:
            return []

        results = []
        for analysis in analyses:
            results.append({
                "challenge_id": analysis.challenge_id,
                "challenge_title": analysis.challenge_title,
                "data_sources_used": analysis.data_sources_used,
                "key_findings": analysis.key_findings,
                "recommendations": analysis.recommendations,
                "visualizations": analysis.visualizations,
                "statistical_tests": [
                    {
                        "test_name": test.test_name,
                        "description": test.description,
                        "p_value": test.p_value,
                        "is_significant": test.is_significant
                    }
                    for test in analysis.statistical_tests
                ],
                "timestamp": str(analysis.timestamp)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")

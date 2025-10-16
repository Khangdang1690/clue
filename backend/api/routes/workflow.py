"""Workflow execution endpoints for Phase 1 and Phase 2."""

import sys
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models import (
    Phase1Response,
    Phase2Response,
    ReportsResponse,
    ChallengeInfo,
    AnalysisResultResponse,
    VisualizationInfo,
    ReportInfo
)
from src.graph.workflow import ETLInsightsWorkflow
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge
from api.websocket import ws_manager, PrintCapture


router = APIRouter(prefix="/api", tags=["workflow"])

# Global workflow instance and state
workflow = ETLInsightsWorkflow()
workflow_state = {
    "phase": "idle",  # idle, phase1, phase2, generating_reports, completed
    "is_running": False,
    "current_operation": None,
    "progress_percent": 0.0
}


class AsyncPrintCapture:
    """Async-compatible print capture for background tasks."""

    def __init__(self, ws_manager):
        self.ws_manager = ws_manager
        self.original_stdout = sys.stdout
        self.buffer = []

    def write(self, message: str):
        """Capture print output."""
        self.original_stdout.write(message)
        self.original_stdout.flush()

        if message.strip():
            # Schedule broadcast
            self.buffer.append(message.strip())

    def flush(self):
        """Flush output."""
        self.original_stdout.flush()

    async def flush_to_websocket(self):
        """Flush buffered messages to WebSocket."""
        for message in self.buffer:
            await self.ws_manager.broadcast_log(message)
        self.buffer.clear()


def _challenge_to_info(challenge: Challenge) -> ChallengeInfo:
    """Convert Challenge model to ChallengeInfo response."""
    return ChallengeInfo(
        id=challenge.id,
        title=challenge.title,
        priority_score=challenge.priority_score,
        priority_level=challenge.priority_level.value,
        department=challenge.department,
        description=challenge.description
    )


async def _run_phase1_task():
    """Background task for Phase 1 execution."""
    global workflow_state

    try:
        workflow_state["phase"] = "phase1"
        workflow_state["is_running"] = True
        workflow_state["current_operation"] = "Identifying and prioritizing challenges"

        await ws_manager.broadcast_log("\n=== STARTING PHASE 1: PROBLEM IDENTIFICATION ===\n")
        await ws_manager.broadcast_progress("phase1", 0)

        # Get business context
        business_context_dict = workflow.chroma_manager.get_business_context()
        if not business_context_dict:
            raise ValueError("No business context found. Please submit business context first.")

        business_context = BusinessContext(**business_context_dict)

        # Capture print output
        capture = AsyncPrintCapture(ws_manager)
        original_stdout = sys.stdout
        sys.stdout = capture

        try:
            # Run Phase 1
            await ws_manager.broadcast_progress("phase1", 10)
            result = workflow.run_phase1(business_context)

            # Flush captured output
            sys.stdout = original_stdout
            await capture.flush_to_websocket()

            if result["status"] == "completed":
                challenges = result["challenges"]
                await ws_manager.broadcast_progress("phase1", 100)
                await ws_manager.broadcast_phase_complete("phase1", {
                    "challenges_count": len(challenges)
                })

                workflow_state["phase"] = "idle"
                workflow_state["current_operation"] = None
                workflow_state["progress_percent"] = 0.0

                await ws_manager.broadcast_log(f"\n✓ Phase 1 completed: {len(challenges)} challenges identified\n")
            else:
                raise ValueError(result.get("error_message", "Unknown error"))

        finally:
            sys.stdout = original_stdout

    except Exception as e:
        await ws_manager.broadcast_error(f"Phase 1 failed: {str(e)}", str(e))
        workflow_state["phase"] = "idle"
        workflow_state["current_operation"] = None
    finally:
        workflow_state["is_running"] = False


async def _run_phase2_task():
    """Background task for Phase 2 execution (single challenge)."""
    global workflow_state

    try:
        workflow_state["phase"] = "phase2"
        workflow_state["is_running"] = True
        workflow_state["current_operation"] = "Analyzing next challenge"

        await ws_manager.broadcast_log("\n=== STARTING PHASE 2: ANALYSIS (SINGLE CHALLENGE) ===\n")
        await ws_manager.broadcast_progress("phase2", 0)

        # Capture print output
        capture = AsyncPrintCapture(ws_manager)
        original_stdout = sys.stdout
        sys.stdout = capture

        try:
            # Run Phase 2
            result = workflow.run_phase2_single()

            # Flush captured output
            sys.stdout = original_stdout
            await capture.flush_to_websocket()

            if result["status"] == "completed":
                challenge = result["challenge_processed"]
                await ws_manager.broadcast_progress("phase2", 100)
                await ws_manager.broadcast_challenge_complete(
                    challenge.id,
                    challenge.title,
                    result["challenges_remaining"]
                )

                # Broadcast visualizations
                if result["analysis_result"] and result["analysis_result"].visualizations:
                    for viz in result["analysis_result"].visualizations:
                        await ws_manager.broadcast_visualization_created(
                            viz.get("id", ""),
                            viz.get("path", ""),
                            viz.get("title", "")
                        )

                # Broadcast reports and dashboard if generated
                if result.get("dashboard_path"):
                    await ws_manager.broadcast_report_generated("analytical", result.get("analytical_report_path", ""))
                    await ws_manager.broadcast_report_generated("business", result.get("business_report_path", ""))
                    await ws_manager.broadcast_report_generated("dashboard", result.get("dashboard_path", ""))

                    await ws_manager.broadcast_log(
                        f"\n✓ Reports and Dashboard Generated:\n"
                        f"  - Analytical Report: {result.get('analytical_report_path', 'N/A')}\n"
                        f"  - Business Report: {result.get('business_report_path', 'N/A')}\n"
                        f"  - Dashboard: {result.get('dashboard_path', 'N/A')}\n"
                    )

                workflow_state["phase"] = "idle"
                workflow_state["current_operation"] = None
                workflow_state["progress_percent"] = 0.0

                await ws_manager.broadcast_log(
                    f"\n✓ Phase 2 completed for: {challenge.title}\n"
                    f"  Remaining challenges: {result['challenges_remaining']}\n"
                )

            elif result["status"] == "no_challenges":
                await ws_manager.broadcast_log("\n✓ No more challenges to process\n")
                workflow_state["phase"] = "completed"
                workflow_state["current_operation"] = None
            else:
                raise ValueError(result.get("error_message", "Unknown error"))

        finally:
            sys.stdout = original_stdout

    except Exception as e:
        await ws_manager.broadcast_error(f"Phase 2 failed: {str(e)}", str(e))
        workflow_state["phase"] = "idle"
        workflow_state["current_operation"] = None
    finally:
        workflow_state["is_running"] = False


async def _generate_reports_task():
    """Background task for report generation."""
    global workflow_state

    try:
        workflow_state["phase"] = "generating_reports"
        workflow_state["is_running"] = True
        workflow_state["current_operation"] = "Generating comprehensive reports"

        await ws_manager.broadcast_log("\n=== GENERATING COMPREHENSIVE REPORTS ===\n")
        await ws_manager.broadcast_progress("reports", 0)

        # Capture print output
        capture = AsyncPrintCapture(ws_manager)
        original_stdout = sys.stdout
        sys.stdout = capture

        try:
            # Generate reports
            result = workflow.generate_reports()

            # Flush captured output
            sys.stdout = original_stdout
            await capture.flush_to_websocket()

            if result["status"] == "completed":
                await ws_manager.broadcast_progress("reports", 100)

                # Broadcast each report
                await ws_manager.broadcast_report_generated("analytical", result["analytical_report_path"])
                await ws_manager.broadcast_report_generated("business", result["business_report_path"])
                await ws_manager.broadcast_report_generated("dashboard", result["dashboard_path"])

                await ws_manager.broadcast_phase_complete("reports", {
                    "analytical_report": result["analytical_report_path"],
                    "business_report": result["business_report_path"],
                    "dashboard": result["dashboard_path"]
                })

                workflow_state["phase"] = "completed"
                workflow_state["current_operation"] = None
                workflow_state["progress_percent"] = 0.0

                await ws_manager.broadcast_log("\n✓ All reports generated successfully\n")
            else:
                raise ValueError(result.get("error_message", "Unknown error"))

        finally:
            sys.stdout = original_stdout

    except Exception as e:
        await ws_manager.broadcast_error(f"Report generation failed: {str(e)}", str(e))
        workflow_state["phase"] = "idle"
        workflow_state["current_operation"] = None
    finally:
        workflow_state["is_running"] = False


@router.post("/phase1/start")
async def start_phase1(background_tasks: BackgroundTasks):
    """
    Start Phase 1: Problem Identification.

    This analyzes the business context and uploaded data to identify
    and prioritize challenges across departments.

    Requires:
    - Business context to be submitted
    - Department files to be uploaded

    Returns:
        Immediate response confirming task started
    """
    if workflow_state["is_running"]:
        raise HTTPException(status_code=409, detail="Another workflow operation is already running")

    # Check business context exists
    context = workflow.chroma_manager.get_business_context()
    if not context:
        raise HTTPException(
            status_code=400,
            detail="Business context not found. Please submit business context first."
        )

    # Start background task
    background_tasks.add_task(_run_phase1_task)

    return {
        "status": "started",
        "message": "Phase 1 execution started. Connect to WebSocket for real-time updates."
    }


@router.post("/phase2/start")
async def start_phase2(background_tasks: BackgroundTasks):
    """
    Start Phase 2: Analysis (processes one challenge).

    This analyzes the next highest-priority challenge:
    - Runs ETL pipeline on relevant data
    - Performs statistical analysis
    - Generates business insights
    - Creates visualizations

    Can be called multiple times to process all challenges.

    Returns:
        Immediate response confirming task started
    """
    if workflow_state["is_running"]:
        raise HTTPException(status_code=409, detail="Another workflow operation is already running")

    # Check if challenges exist
    status = workflow.get_challenge_status()
    if status.get("remaining", 0) == 0:
        raise HTTPException(
            status_code=400,
            detail="No challenges to process. Run Phase 1 first or all challenges have been processed."
        )

    # Start background task
    background_tasks.add_task(_run_phase2_task)

    return {
        "status": "started",
        "message": "Phase 2 execution started. Connect to WebSocket for real-time updates."
    }


@router.post("/phase2/generate-reports")
async def generate_reports(background_tasks: BackgroundTasks):
    """
    Generate comprehensive reports from all analyses.

    Generates:
    - Analytical Report (PDF): Technical details and statistics
    - Business Insight Report (PDF): Executive summary and recommendations
    - Interactive Dashboard (HTML): Visualizations and metrics

    Returns:
        Immediate response confirming task started
    """
    if workflow_state["is_running"]:
        raise HTTPException(status_code=409, detail="Another workflow operation is already running")

    # Start background task
    background_tasks.add_task(_generate_reports_task)

    return {
        "status": "started",
        "message": "Report generation started. Connect to WebSocket for real-time updates."
    }


@router.get("/status")
async def get_workflow_status():
    """
    Get current workflow status.

    Returns:
        Current phase, running status, and progress information
    """
    challenge_status = workflow.get_challenge_status()

    return {
        "phase": workflow_state["phase"],
        "is_running": workflow_state["is_running"],
        "current_operation": workflow_state["current_operation"],
        "progress_percent": workflow_state["progress_percent"],
        "challenges_status": challenge_status
    }

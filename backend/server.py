"""FastAPI server for ETL to Insights AI Agent."""

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import files, context, workflow, results
from api.websocket import ws_manager


# Create FastAPI app
app = FastAPI(
    title="ETL to Insights AI Agent API",
    description="Backend server for business intelligence and data analysis agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # Alternative port if needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router)
app.include_router(context.router)
app.include_router(workflow.router)
app.include_router(results.router)

# Ensure output directories exist
def ensure_directories():
    """Create necessary directories for data storage."""
    directories = [
        "data/uploads",
        "data/outputs/visualizations",
        "data/outputs/reports",
        "data/outputs/dashboards",
        "data/chroma_db"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

ensure_directories()


# WebSocket endpoint for real-time updates
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time progress updates and logs.

    Clients connect here to receive:
    - Real-time log messages from agent execution
    - Progress updates (percentage complete)
    - Phase completion notifications
    - Challenge completion notifications
    - Visualization creation notifications
    - Report generation notifications
    - Error messages

    Message format:
    {
        "type": "log" | "progress" | "phase_complete" | "challenge_complete" | "visualization_created" | "report_generated" | "error",
        "message": "Log message text",
        "data": { ... additional data ... },
        "timestamp": "ISO 8601 timestamp"
    }
    """
    await ws_manager.connect(websocket)
    try:
        # Keep connection alive and handle incoming messages (if needed)
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            # For now, this is primarily a one-way broadcast from server to client

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        print("WebSocket client disconnected")


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "ETL to Insights AI Agent API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/api/docs",
        "websocket": "/ws/progress"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "etl-insights-agent"
    }


# API info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints."""
    return {
        "endpoints": {
            "websocket": {
                "progress": "ws://localhost:8000/ws/progress"
            },
            "business_context": {
                "submit": "POST /api/business-context",
                "get": "GET /api/business-context",
                "clear": "DELETE /api/business-context"
            },
            "files": {
                "upload": "POST /api/upload-files",
                "list": "GET /api/departments/files",
                "delete": "DELETE /api/departments/{department}/files/{filename}"
            },
            "workflow": {
                "phase1_start": "POST /api/phase1/start",
                "phase2_start": "POST /api/phase2/start",
                "generate_reports": "POST /api/phase2/generate-reports",
                "status": "GET /api/status"
            },
            "results": {
                "challenges": "GET /api/challenges",
                "challenges_all": "GET /api/challenges/all",
                "visualizations_list": "GET /api/visualizations",
                "visualization_get": "GET /api/visualizations/{filename}",
                "reports_list": "GET /api/reports",
                "report_get": "GET /api/reports/{filename}",
                "dashboard": "GET /api/dashboard",
                "analysis_results": "GET /api/analysis-results"
            }
        },
        "usage_flow": [
            "1. Submit business context: POST /api/business-context",
            "2. Upload department files: POST /api/upload-files (for each department)",
            "3. Connect to WebSocket: ws://localhost:8000/ws/progress",
            "4. Start Phase 1: POST /api/phase1/start",
            "5. Monitor via WebSocket (receive real-time logs)",
            "6. Check challenges: GET /api/challenges",
            "7. Start Phase 2: POST /api/phase2/start (repeat for each challenge)",
            "8. Generate reports: POST /api/phase2/generate-reports",
            "9. Download reports: GET /api/reports/{filename}",
            "10. View dashboard: GET /api/dashboard"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("ETL to Insights AI Agent - Backend Server")
    print("="*60)
    print("\nStarting server...")
    print("  - API Documentation: http://localhost:8000/api/docs")
    print("  - WebSocket Endpoint: ws://localhost:8000/ws/progress")
    print("  - API Info: http://localhost:8000/api/info")
    print("\n" + "="*60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

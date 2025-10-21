"""Main FastAPI application."""

import time
import psutil
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import health, auth, agents, company, etl, analyses
from src.database.connection import DatabaseManager

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Performance logging middleware
@app.middleware("http")
async def log_performance(request: Request, call_next):
    """Log request execution time and memory usage."""
    # Get process
    process = psutil.Process(os.getpid())

    # Record start metrics
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Process request
    response = await call_next(request)

    # Record end metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Calculate metrics
    duration = end_time - start_time
    duration_ms = duration * 1000
    memory_used = end_memory - start_memory

    # Color code based on duration
    if duration_ms < 100:
        time_color = "\033[92m"  # Green
    elif duration_ms < 500:
        time_color = "\033[93m"  # Yellow
    else:
        time_color = "\033[91m"  # Red

    # Color code based on memory usage
    if abs(memory_used) < 10:
        mem_color = "\033[92m"  # Green
    elif abs(memory_used) < 50:
        mem_color = "\033[93m"  # Yellow
    else:
        mem_color = "\033[91m"  # Red

    reset = "\033[0m"

    # Format log message
    method = request.method
    path = request.url.path
    status = response.status_code

    # Format memory change with sign
    mem_sign = "+" if memory_used >= 0 else ""
    mem_str = f"{mem_sign}{memory_used:.2f}MB"

    # Log with color and formatting
    print(
        f"{time_color}[PERF]{reset} "
        f"{method:6s} {path:50s} | "
        f"{status} | "
        f"{time_color}{duration_ms:7.2f}ms{reset} | "
        f"{mem_color}{mem_str:>10s}{reset} | "
        f"RSS: {end_memory:.1f}MB"
    )

    return response


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    DatabaseManager.initialize()
    print(f"✅ {settings.app_name} v{settings.app_version} started")
    print(f"✅ Database connection initialized")
    print(f"📚 API docs available at /docs")
    print(f"⚡ Performance logging enabled:")
    print(f"   Time:   Green: <100ms | Yellow: 100-500ms | Red: >500ms")
    print(f"   Memory: Green: <10MB  | Yellow: 10-50MB   | Red: >50MB")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"👋 {settings.app_name} shutting down")


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(company.router, prefix="/api/company", tags=["Company"])
app.include_router(etl.router, prefix="/api/etl", tags=["ETL"])
app.include_router(agents.router, prefix="/api/agents", tags=["AI Agents"])
app.include_router(analyses.router, prefix="/api", tags=["Analyses"])

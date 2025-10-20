"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import health, auth, agents, company, etl
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


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    DatabaseManager.initialize()
    print(f"✅ {settings.app_name} v{settings.app_version} started")
    print(f"✅ Database connection initialized")
    print(f"📚 API docs available at /docs")


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

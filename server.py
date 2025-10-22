"""
iClue API Server - Main entry point for the FastAPI application.

This script initializes the database and starts the FastAPI server.
"""

import sys
from src.database.connection import DatabaseManager
from app.core.config import get_settings

def main():
    """Initialize database and run server."""
    settings = get_settings()

    print(f"🚀 Starting {settings.app_name} v{settings.app_version}...")
    print()

    # Initialize database
    print("📊 Initializing database...")
    try:
        DatabaseManager.initialize()
        DatabaseManager.create_all_tables()
        print("✅ Database initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    # Start server
    print(f"🌐 Starting FastAPI server on http://{settings.host}:{settings.port}")
    print(f"📚 API documentation available at http://{settings.host}:{settings.port}/docs")
    print(f"📖 ReDoc documentation at http://{settings.host}:{settings.port}/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",  # Updated to use new app structure
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()

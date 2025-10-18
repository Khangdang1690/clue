"""PostgreSQL connection manager."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    """Manages PostgreSQL connections and sessions."""

    _engine = None
    _session_factory = None

    @classmethod
    def initialize(cls):
        """Initialize database connection pool."""
        if cls._engine is not None:
            return

        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not set in environment")

        cls._engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=int(os.getenv('DB_POOL_SIZE', 10)),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 20)),
            echo=os.getenv('DB_ECHO', 'False').lower() == 'true',
            pool_pre_ping=True,  # Verify connections before using
        )

        cls._session_factory = sessionmaker(bind=cls._engine)

        print("[DB] Database connection initialized")

    @classmethod
    def get_engine(cls):
        """Get SQLAlchemy engine."""
        if cls._engine is None:
            cls.initialize()
        return cls._engine

    @classmethod
    @contextmanager
    def get_session(cls) -> Session:
        """Get database session (context manager)."""
        if cls._session_factory is None:
            cls.initialize()

        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def create_all_tables(cls):
        """Create all tables (for initial setup)."""
        from src.database.models import Base

        engine = cls.get_engine()

        # Enable pgvector extension
        with engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

        # Create tables
        Base.metadata.create_all(engine)
        print("[DB] All tables created")

    @classmethod
    def drop_all_tables(cls):
        """Drop all tables (for testing/reset)."""
        from src.database.models import Base
        Base.metadata.drop_all(cls.get_engine())
        print("[DB] All tables dropped")
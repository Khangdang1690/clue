"""SQLAlchemy models for AI Analyst platform with pgvector support."""

from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text, ForeignKey, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

# Gemini embedding dimension (new free SDK)
EMBEDDING_DIM = 3072


class Company(Base):
    """Company/Organization entity"""
    __tablename__ = 'companies'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    datasets = relationship("Dataset", back_populates="company", cascade="all, delete-orphan")
    analysis_sessions = relationship("AnalysisSession", back_populates="company", cascade="all, delete-orphan")


class Dataset(Base):
    """Represents a single data source (file)"""
    __tablename__ = 'datasets'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey('companies.id'), nullable=False)

    # File info
    original_filename = Column(String, nullable=False)
    table_name = Column(String, nullable=False)  # LLM-generated name
    file_type = Column(String, nullable=False)  # csv, excel, json, parquet
    file_path = Column(String, nullable=False)

    # Semantic info (from LLM)
    domain = Column(String)  # Finance, Marketing, Sales, HR, Operations
    department = Column(String)
    description = Column(Text)
    entities = Column(JSON)  # ["Customer", "Product", "Transaction"]

    # Embeddings (Gemini 3072 dimensions - new free SDK)
    description_embedding = Column(Vector(EMBEDDING_DIM))  # Semantic embedding of description
    schema_embedding = Column(Vector(EMBEDDING_DIM))  # Embedding of column names/types

    # Processing status
    status = Column(String, default='uploaded')  # uploaded, profiled, cleaned, ready
    row_count = Column(Integer)
    column_count = Column(Integer)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    # Relationships
    company = relationship("Company", back_populates="datasets")
    columns = relationship("ColumnMetadata", back_populates="dataset", cascade="all, delete-orphan")
    relationships_from = relationship("TableRelationship", foreign_keys="TableRelationship.from_dataset_id")
    relationships_to = relationship("TableRelationship", foreign_keys="TableRelationship.to_dataset_id")

    __table_args__ = (
        Index('idx_company_table', 'company_id', 'table_name'),
    )


class ColumnMetadata(Base):
    """Semantic metadata for each column"""
    __tablename__ = 'column_metadata'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey('datasets.id'), nullable=False)

    # Column info
    column_name = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    position = Column(Integer)

    # Data type info
    data_type = Column(String)  # int64, float64, object, datetime64
    semantic_type = Column(String)  # dimension, measure, key, text, date

    # Semantic understanding (from LLM)
    business_meaning = Column(Text)
    is_primary_key = Column(Boolean, default=False)
    is_foreign_key = Column(Boolean, default=False)

    # Embedding for semantic similarity
    semantic_embedding = Column(Vector(EMBEDDING_DIM))

    # Statistics
    null_count = Column(Integer)
    null_percentage = Column(Float)
    unique_count = Column(Integer)
    unique_percentage = Column(Float)

    # Relationships
    dataset = relationship("Dataset", back_populates="columns")


class TableRelationship(Base):
    """Detected relationships between tables"""
    __tablename__ = 'table_relationships'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Relationship
    from_dataset_id = Column(String, ForeignKey('datasets.id'), nullable=False)
    to_dataset_id = Column(String, ForeignKey('datasets.id'), nullable=False)
    from_column = Column(String, nullable=False)
    to_column = Column(String, nullable=False)

    # Relationship metadata
    relationship_type = Column(String)  # one-to-one, one-to-many, many-to-many
    confidence = Column(Float)  # 0.0 to 1.0
    join_strategy = Column(String, default='inner')  # inner, left, right, outer

    # Validation stats
    match_percentage = Column(Float)  # % of FK values that exist in PK

    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_relationship', 'from_dataset_id', 'to_dataset_id'),
    )


class KPIDefinition(Base):
    """Pre-calculated KPI definitions per domain"""
    __tablename__ = 'kpi_definitions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # KPI info
    kpi_name = Column(String, nullable=False)
    domain = Column(String, nullable=False)  # Finance, Marketing, etc.
    description = Column(Text)
    formula = Column(Text)  # SQL or Python formula
    unit = Column(String)  # %, $, count, etc.

    # Column mappings (for auto-calculation)
    required_columns = Column(JSON)  # {"revenue": "total_revenue", "cost": "total_cost"}

    # Embedding for semantic search
    kpi_embedding = Column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        Index('idx_domain_kpi', 'domain', 'kpi_name'),
    )


class InsightPattern(Base):
    """Store successful insight patterns for reuse"""
    __tablename__ = 'insight_patterns'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Pattern info
    pattern_type = Column(String)  # 'single_table', 'cross_table', 'time_series'
    domain = Column(String)

    # The insight
    insight_text = Column(Text)
    insight_embedding = Column(Vector(EMBEDDING_DIM))

    # Code that generated it
    code_template = Column(Text)  # Python code with placeholders
    required_columns = Column(JSON)  # Column types needed

    # Performance
    confidence = Column(Float)
    times_used = Column(Integer, default=0)
    avg_user_rating = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisSession(Base):
    """Tracks analysis sessions (multi-table discoveries)"""
    __tablename__ = 'analysis_sessions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey('companies.id'), nullable=False)

    # Session info
    name = Column(String)
    description = Column(Text)
    dataset_ids = Column(JSON)  # List of dataset IDs included

    # Results
    insights_generated = Column(Integer, default=0)
    report_path = Column(String)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    status = Column(String, default='running')  # running, completed, failed

    # Relationships
    company = relationship("Company", back_populates="analysis_sessions")
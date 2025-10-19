"""Data access layer for database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
from src.database.models import (
    Company, Dataset, ColumnMetadata,
    TableRelationship, KPIDefinition, AnalysisSession, InsightPattern
)
import pandas as pd
from datetime import datetime


class CompanyRepository:
    """Repository for company operations."""

    @staticmethod
    def create_company(session: Session, name: str) -> Company:
        """Create a new company."""
        company = Company(name=name)
        session.add(company)
        session.flush()
        return company

    @staticmethod
    def get_or_create_company(session: Session, name: str) -> Company:
        """Get existing company or create new one."""
        company = session.query(Company).filter(Company.name == name).first()
        if not company:
            company = CompanyRepository.create_company(session, name)
        return company

    @staticmethod
    def get_company_by_id(session: Session, company_id: str) -> Optional[Company]:
        """Get company by ID."""
        return session.query(Company).filter(Company.id == company_id).first()


class DatasetRepository:
    """Repository for dataset operations."""

    @staticmethod
    def create_dataset(
        session: Session,
        company_id: str,
        original_filename: str,
        table_name: str,
        file_type: str,
        file_path: str,
        domain: Optional[str] = None,
        department: Optional[str] = None,
        description: Optional[str] = None,
        entities: Optional[List[str]] = None,
        description_embedding: Optional[List[float]] = None,
        schema_embedding: Optional[List[float]] = None,
        # New unified context fields
        dataset_type: Optional[str] = None,
        time_period: Optional[str] = None,
        typical_use_cases: Optional[List[str]] = None,
        business_context: Optional[Dict] = None
    ) -> Dataset:
        """Create a new dataset record with unified context."""
        dataset = Dataset(
            company_id=company_id,
            original_filename=original_filename,
            table_name=table_name,
            file_type=file_type,
            file_path=file_path,
            domain=domain,
            department=department,
            description=description,
            entities=entities or [],
            description_embedding=description_embedding,
            schema_embedding=schema_embedding,
            # New unified context fields
            dataset_type=dataset_type,
            time_period=time_period,
            typical_use_cases=typical_use_cases or [],
            business_context=business_context or {}
        )
        session.add(dataset)
        session.flush()
        return dataset

    @staticmethod
    def get_datasets_by_company(session: Session, company_id: str) -> List[Dataset]:
        """Get all datasets for a company."""
        return session.query(Dataset).filter(
            Dataset.company_id == company_id
        ).all()

    @staticmethod
    def get_dataset_by_id(session: Session, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID."""
        return session.query(Dataset).filter(Dataset.id == dataset_id).first()

    @staticmethod
    def update_dataset_status(
        session: Session,
        dataset_id: str,
        status: str,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None
    ):
        """Update dataset processing status."""
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            dataset.status = status
            if row_count is not None:
                dataset.row_count = row_count
            if column_count is not None:
                dataset.column_count = column_count
            if status == 'ready':
                dataset.processed_at = datetime.utcnow()
            session.flush()

    @staticmethod
    def store_dataframe(
        session: Session,
        dataset_id: str,
        df: pd.DataFrame,
        table_prefix: str = "cleaned"
    ) -> str:
        """
        Store DataFrame to PostgreSQL.

        Returns: table name
        """
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Create safe table name
        company = session.query(Company).filter(Company.id == dataset.company_id).first()
        table_name = f"{company.name}_{table_prefix}_{dataset.table_name}"
        table_name = table_name.lower().replace(' ', '_').replace('-', '_')

        # Store to PostgreSQL
        from src.database.connection import DatabaseManager
        engine = DatabaseManager.get_engine()
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        return table_name

    @staticmethod
    def load_dataframe(session: Session, dataset_id: str, table_prefix: str = "cleaned") -> pd.DataFrame:
        """Load DataFrame from PostgreSQL."""
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        company = session.query(Company).filter(Company.id == dataset.company_id).first()
        table_name = f"{company.name}_{table_prefix}_{dataset.table_name}"
        table_name = table_name.lower().replace(' ', '_').replace('-', '_')

        from src.database.connection import DatabaseManager
        engine = DatabaseManager.get_engine()
        return pd.read_sql_table(table_name, engine)


class ColumnMetadataRepository:
    """Repository for column metadata operations."""

    @staticmethod
    def create_column_metadata(
        session: Session,
        dataset_id: str,
        column_name: str,
        original_name: str,
        position: int,
        data_type: str,
        semantic_type: Optional[str] = None,
        business_meaning: Optional[str] = None,
        is_primary_key: bool = False,
        is_foreign_key: bool = False,
        semantic_embedding: Optional[List[float]] = None,
        null_count: Optional[int] = None,
        null_percentage: Optional[float] = None,
        unique_count: Optional[int] = None,
        unique_percentage: Optional[float] = None
    ) -> ColumnMetadata:
        """Create column metadata."""
        column_meta = ColumnMetadata(
            dataset_id=dataset_id,
            column_name=column_name,
            original_name=original_name,
            position=position,
            data_type=data_type,
            semantic_type=semantic_type,
            business_meaning=business_meaning,
            is_primary_key=is_primary_key,
            is_foreign_key=is_foreign_key,
            semantic_embedding=semantic_embedding,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage
        )
        session.add(column_meta)
        session.flush()
        return column_meta

    @staticmethod
    def get_columns_by_dataset(session: Session, dataset_id: str) -> List[ColumnMetadata]:
        """Get all columns for a dataset."""
        return session.query(ColumnMetadata).filter(
            ColumnMetadata.dataset_id == dataset_id
        ).order_by(ColumnMetadata.position).all()


class RelationshipRepository:
    """Repository for table relationships."""

    @staticmethod
    def create_relationship(
        session: Session,
        from_dataset_id: str,
        to_dataset_id: str,
        from_column: str,
        to_column: str,
        relationship_type: str,
        confidence: float,
        match_percentage: float,
        join_strategy: str = 'inner'
    ) -> TableRelationship:
        """Create a new relationship."""
        rel = TableRelationship(
            from_dataset_id=from_dataset_id,
            to_dataset_id=to_dataset_id,
            from_column=from_column,
            to_column=to_column,
            relationship_type=relationship_type,
            confidence=confidence,
            match_percentage=match_percentage,
            join_strategy=join_strategy
        )
        session.add(rel)
        session.flush()
        return rel

    @staticmethod
    def get_relationships_for_datasets(
        session: Session,
        dataset_ids: List[str],
        min_confidence: float = 0.8
    ) -> List[TableRelationship]:
        """Get all relationships between given datasets."""
        return session.query(TableRelationship).filter(
            TableRelationship.from_dataset_id.in_(dataset_ids),
            TableRelationship.to_dataset_id.in_(dataset_ids),
            TableRelationship.confidence >= min_confidence
        ).all()


class SimilarityRepository:
    """Repository for similarity searches using pgvector."""

    @staticmethod
    def find_similar_datasets(
        session: Session,
        embedding: List[float],
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find datasets with similar descriptions.

        Useful for:
        - Finding related datasets for cross-analysis
        - Suggesting datasets to include in analysis
        """
        # Convert numpy array to list if necessary
        import numpy as np
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        query = text("""
            SELECT
                id,
                table_name,
                domain,
                description,
                1 - (description_embedding <=> CAST(:embedding AS vector)) as similarity
            FROM datasets
            WHERE 1 - (description_embedding <=> CAST(:embedding AS vector)) > :threshold
            ORDER BY description_embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)

        result = session.execute(
            query,
            {
                'embedding': embedding,
                'threshold': threshold,
                'limit': limit
            }
        )

        return [
            {
                'id': row.id,
                'table_name': row.table_name,
                'domain': row.domain,
                'description': row.description,
                'similarity': row.similarity
            }
            for row in result
        ]

    @staticmethod
    def find_similar_columns(
        session: Session,
        embedding: List[float],
        exclude_dataset_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.8
    ) -> List[Dict]:
        """
        Find semantically similar columns across datasets.

        Useful for:
        - Relationship detection
        - Finding equivalent metrics across departments
        """
        # Convert numpy array to list if necessary
        import numpy as np
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        query_parts = [
            """
            SELECT
                cm.id,
                cm.dataset_id,
                cm.column_name,
                cm.business_meaning,
                d.table_name,
                1 - (cm.semantic_embedding <=> CAST(:embedding AS vector)) as similarity
            FROM column_metadata cm
            JOIN datasets d ON cm.dataset_id = d.id
            WHERE 1 - (cm.semantic_embedding <=> CAST(:embedding AS vector)) > :threshold
            """
        ]

        params = {
            'embedding': embedding,
            'threshold': threshold,
            'limit': limit
        }

        if exclude_dataset_id:
            query_parts.append("AND cm.dataset_id != :exclude_id")
            params['exclude_id'] = exclude_dataset_id

        query_parts.append("""
            ORDER BY cm.semantic_embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)

        query = text(" ".join(query_parts))
        result = session.execute(query, params)

        return [
            {
                'id': row.id,
                'dataset_id': row.dataset_id,
                'table_name': row.table_name,
                'column_name': row.column_name,
                'business_meaning': row.business_meaning,
                'similarity': row.similarity
            }
            for row in result
        ]

    @staticmethod
    def find_similar_insights(
        session: Session,
        embedding: List[float],
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find similar historical insights.

        Useful for:
        - Reusing successful analysis patterns
        - Suggesting relevant analyses
        """
        query_parts = [
            """
            SELECT
                id,
                pattern_type,
                domain,
                insight_text,
                code_template,
                confidence,
                1 - (insight_embedding <=> CAST(:embedding AS vector)) as similarity
            FROM insight_patterns
            WHERE 1 - (insight_embedding <=> CAST(:embedding AS vector)) > :threshold
            """
        ]

        params = {
            'embedding': embedding,
            'threshold': threshold,
            'limit': limit
        }

        if domain:
            query_parts.append("AND domain = :domain")
            params['domain'] = domain

        query_parts.append("""
            ORDER BY insight_embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)

        query = text(" ".join(query_parts))
        result = session.execute(query, params)

        return [
            {
                'id': row.id,
                'pattern_type': row.pattern_type,
                'domain': row.domain,
                'insight_text': row.insight_text,
                'code_template': row.code_template,
                'confidence': row.confidence,
                'similarity': row.similarity
            }
            for row in result
        ]


class AnalysisSessionRepository:
    """Repository for analysis sessions."""

    @staticmethod
    def create_session(
        session: Session,
        company_id: str,
        name: str,
        dataset_ids: List[str],
        description: Optional[str] = None
    ) -> AnalysisSession:
        """Create a new analysis session."""
        analysis_session = AnalysisSession(
            company_id=company_id,
            name=name,
            dataset_ids=dataset_ids,
            description=description
        )
        session.add(analysis_session)
        session.flush()
        return analysis_session

    @staticmethod
    def update_session_status(
        session: Session,
        session_id: str,
        status: str,
        insights_generated: Optional[int] = None,
        report_path: Optional[str] = None
    ):
        """Update analysis session status."""
        analysis_session = session.query(AnalysisSession).filter(
            AnalysisSession.id == session_id
        ).first()

        if analysis_session:
            analysis_session.status = status
            if insights_generated is not None:
                analysis_session.insights_generated = insights_generated
            if report_path is not None:
                analysis_session.report_path = report_path
            if status == 'completed':
                analysis_session.completed_at = datetime.utcnow()
            session.flush()
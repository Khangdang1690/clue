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
from app.services.storage_service import StorageService


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

    @staticmethod
    def find_dataset_by_filename(
        session: Session,
        company_id: str,
        filename: str
    ) -> Optional[Dataset]:
        """
        Check if dataset with this filename already exists for company.
        Used for duplicate detection in CREATE mode.
        """
        return session.query(Dataset).filter(
            Dataset.company_id == company_id,
            Dataset.original_filename == filename
        ).first()

    @staticmethod
    def load_existing_datasets_for_company(
        session: Session,
        company_id: str,
        exclude_dataset_ids: List[str] = None
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Load all existing datasets for a company as DataFrames and metadata.
        Used for cross-ETL relationship detection.

        Args:
            session: Database session
            company_id: Company ID
            exclude_dataset_ids: Optional list of dataset IDs to exclude

        Returns:
            Tuple of (datasets_dict, metadata_dict)
            - datasets_dict: {dataset_id: DataFrame}
            - metadata_dict: {dataset_id: semantic_analysis_dict}
        """
        exclude_dataset_ids = exclude_dataset_ids or []

        # Query existing datasets
        query = session.query(Dataset).filter(Dataset.company_id == company_id)
        if exclude_dataset_ids:
            query = query.filter(~Dataset.id.in_(exclude_dataset_ids))

        datasets = query.all()

        datasets_dict = {}
        metadata_dict = {}

        for dataset in datasets:
            try:
                # Load DataFrame
                df = DatasetRepository.load_dataframe(session, dataset.id)
                datasets_dict[dataset.id] = df

                # Build metadata dict (similar format to semantic_analyzer output)
                metadata_dict[dataset.id] = {
                    'table_name': dataset.table_name,
                    'domain': dataset.domain,
                    'description': dataset.description,
                    'entities': dataset.entities or [],
                    'dataset_type': dataset.dataset_type,
                    'time_period': dataset.time_period,
                    'typical_use_cases': dataset.typical_use_cases or [],
                    'business_context': dataset.business_context or {}
                }
            except Exception as e:
                print(f"  Warning: Could not load dataset {dataset.table_name}: {e}")
                continue

        return datasets_dict, metadata_dict

    @staticmethod
    def delete_dataset(session: Session, dataset_id: str) -> bool:
        """
        Delete dataset with full cascade:
        - File from storage (GCS or local filesystem)
        - Table relationships (from/to)
        - Analysis sessions containing this dataset
        - Actual data table (cleaned_xxx)
        - Dataset record (auto-cascades column metadata)

        Args:
            session: Database session
            dataset_id: Dataset ID to delete

        Returns:
            True if deleted, False if not found
        """
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return False

        print(f"[DELETE] Deleting dataset: {dataset.table_name} (ID: {dataset_id})")

        # 1. Delete file from storage
        try:
            storage_service = StorageService()
            print(f"  [DEBUG] Storage type: {'GCS' if storage_service.use_gcs else 'local'}")
            print(f"  [DEBUG] Local storage root: {storage_service.local_storage_root}")
            print(f"  [DEBUG] Dataset file_path: {dataset.file_path}")

            if dataset.file_path:
                file_exists = storage_service.file_exists(dataset.file_path)
                print(f"  [DEBUG] File exists check: {file_exists}")

                if file_exists:
                    # Get full local path for debugging
                    if not storage_service.use_gcs:
                        import os
                        full_path = os.path.join(storage_service.local_storage_root, dataset.file_path)
                        print(f"  [DEBUG] Full local path: {full_path}")

                    storage_service.delete_file(dataset.file_path)
                    storage_type = "GCS" if storage_service.use_gcs else "local storage"
                    print(f"  ✓ Deleted file from {storage_type}: {dataset.file_path}")

                    # Verify deletion
                    still_exists = storage_service.file_exists(dataset.file_path)
                    if still_exists:
                        print(f"  ⚠ WARNING: File still exists after deletion!")
                    else:
                        print(f"  ✓ Verified: File successfully removed from {storage_type}")
                else:
                    print(f"  File not found in storage: {dataset.file_path}")
            else:
                print(f"  No file_path set for this dataset")
        except Exception as e:
            print(f"  ⚠ ERROR: Could not delete file from storage: {e}")
            import traceback
            traceback.print_exc()

        # 2. Delete relationships (both from and to)
        from_rels = session.query(TableRelationship).filter(
            TableRelationship.from_dataset_id == dataset_id
        ).all()
        to_rels = session.query(TableRelationship).filter(
            TableRelationship.to_dataset_id == dataset_id
        ).all()

        for rel in from_rels + to_rels:
            session.delete(rel)
        print(f"  Deleted {len(from_rels) + len(to_rels)} relationships")

        # 3. Delete or update analysis sessions containing this dataset
        sessions_with_dataset = session.query(AnalysisSession).all()
        sessions_deleted = 0
        sessions_updated = 0

        for analysis_session in sessions_with_dataset:
            if analysis_session.dataset_ids and dataset_id in analysis_session.dataset_ids:
                # Remove dataset_id from the list
                analysis_session.dataset_ids.remove(dataset_id)

                # If no datasets left, delete the session
                if not analysis_session.dataset_ids:
                    session.delete(analysis_session)
                    sessions_deleted += 1
                else:
                    sessions_updated += 1

        if sessions_deleted > 0:
            print(f"  Deleted {sessions_deleted} empty analysis sessions")
        if sessions_updated > 0:
            print(f"  Updated {sessions_updated} analysis sessions (removed dataset reference)")

        # 4. Drop actual data table
        try:
            company = session.query(Company).filter(Company.id == dataset.company_id).first()
            table_name = f"{company.name}_cleaned_{dataset.table_name}"
            table_name = table_name.lower().replace(' ', '_').replace('-', '_')

            session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            print(f"  Dropped data table: {table_name}")
        except Exception as e:
            print(f"  Warning: Could not drop data table: {e}")

        # 5. Delete dataset record (cascades to column metadata automatically)
        session.delete(dataset)
        print(f"  Deleted dataset record and column metadata")

        session.flush()
        return True


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
    def bulk_create_column_metadata(
        session: Session,
        columns_data: List[Dict[str, Any]]
    ) -> None:
        """
        Bulk insert column metadata for better performance.

        Args:
            session: Database session
            columns_data: List of dictionaries containing column metadata
                Each dict should have keys: dataset_id, column_name, original_name,
                position, data_type, and optional semantic fields

        Performance: Reduces N individual INSERTs to 1 bulk INSERT.
        Example: 50 columns = 50 roundtrips → 1 roundtrip (50x faster)
        """
        if not columns_data:
            return

        # Use bulk_insert_mappings for optimal performance
        session.bulk_insert_mappings(ColumnMetadata, columns_data)
        session.flush()

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
    def bulk_create_relationships(
        session: Session,
        relationships_data: List[Dict[str, Any]]
    ) -> None:
        """
        Bulk insert relationships for better performance.

        Args:
            session: Database session
            relationships_data: List of dictionaries containing relationship data
                Each dict should have keys: from_dataset_id, to_dataset_id,
                from_column, to_column, relationship_type, confidence, etc.

        Performance: Reduces N individual INSERTs to 1 bulk INSERT.
        Example: 10 relationships = 10 roundtrips → 1 roundtrip (10x faster)
        """
        if not relationships_data:
            return

        # Use bulk_insert_mappings for optimal performance
        session.bulk_insert_mappings(TableRelationship, relationships_data)
        session.flush()

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

        Note: Returns empty list if embedding is None (embeddings disabled).
        """
        # Return empty if embeddings are disabled
        if embedding is None:
            return []

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
            WHERE description_embedding IS NOT NULL
              AND 1 - (description_embedding <=> CAST(:embedding AS vector)) > :threshold
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

        Note: Returns empty list if embedding is None (embeddings disabled).
        """
        # Return empty if embeddings are disabled
        if embedding is None:
            return []

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
            WHERE cm.semantic_embedding IS NOT NULL
              AND 1 - (cm.semantic_embedding <=> CAST(:embedding AS vector)) > :threshold
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
    def create(
        session: Session,
        user_id: str,
        company_id: str,
        dataset_ids: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> AnalysisSession:
        """Create a new analysis session."""
        analysis_session = AnalysisSession(
            user_id=user_id,
            company_id=company_id,
            name=name or "Business Analysis",
            dataset_ids=dataset_ids,
            description=description,
            status='running'
        )
        session.add(analysis_session)
        session.flush()
        return analysis_session

    @staticmethod
    def get_by_id(session: Session, analysis_id: str) -> Optional[AnalysisSession]:
        """Get analysis session by ID."""
        return session.query(AnalysisSession).filter(
            AnalysisSession.id == analysis_id
        ).first()

    @staticmethod
    def get_by_user(
        session: Session,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[AnalysisSession]:
        """Get all analyses for a user, ordered by most recent first."""
        return session.query(AnalysisSession).filter(
            AnalysisSession.user_id == user_id
        ).order_by(
            AnalysisSession.started_at.desc()
        ).limit(limit).offset(offset).all()

    @staticmethod
    def get_by_company(
        session: Session,
        company_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AnalysisSession]:
        """Get all analyses for a company."""
        return session.query(AnalysisSession).filter(
            AnalysisSession.company_id == company_id
        ).order_by(
            AnalysisSession.started_at.desc()
        ).limit(limit).offset(offset).all()

    @staticmethod
    def mark_completed(
        session: Session,
        analysis_id: str,
        dashboard_path: str,
        report_path: str,
        executive_summary: str,
        insights_count: int,
        recommendations_count: int,
        analytics_summary: dict
    ) -> AnalysisSession:
        """Mark analysis as completed with results."""
        print(f"[REPO] mark_completed called for analysis: {analysis_id}")
        print(f"[REPO]   dashboard_path: {dashboard_path}")
        print(f"[REPO]   report_path: {report_path}")

        analysis = session.query(AnalysisSession).filter(
            AnalysisSession.id == analysis_id
        ).first()

        if analysis:
            print(f"[REPO] Analysis found, updating fields")
            analysis.status = 'completed'
            analysis.completed_at = datetime.utcnow()
            analysis.dashboard_path = dashboard_path
            analysis.report_path = report_path
            analysis.executive_summary = executive_summary
            analysis.insights_generated = insights_count
            analysis.recommendations_generated = recommendations_count
            analysis.analytics_summary = analytics_summary

            print(f"[REPO] Calling session.flush()")
            session.flush()
            print(f"[REPO] Flush completed. Analysis.dashboard_path = {analysis.dashboard_path}")
        else:
            print(f"[REPO WARNING] Analysis {analysis_id} not found in database!")

        return analysis

    @staticmethod
    def mark_failed(
        session: Session,
        analysis_id: str,
        error_message: str
    ) -> AnalysisSession:
        """Mark analysis as failed with error message."""
        analysis = session.query(AnalysisSession).filter(
            AnalysisSession.id == analysis_id
        ).first()

        if analysis:
            analysis.status = 'failed'
            analysis.completed_at = datetime.utcnow()
            analysis.error_message = error_message
            session.flush()

        return analysis

    @staticmethod
    def update(
        session: Session,
        analysis_id: str,
        **kwargs
    ) -> AnalysisSession:
        """Update analysis session with arbitrary fields."""
        analysis = session.query(AnalysisSession).filter(
            AnalysisSession.id == analysis_id
        ).first()

        if analysis:
            for key, value in kwargs.items():
                if hasattr(analysis, key):
                    setattr(analysis, key, value)
            session.flush()

        return analysis

    @staticmethod
    def delete(session: Session, analysis_id: str) -> bool:
        """Delete analysis session."""
        analysis = session.query(AnalysisSession).filter(
            AnalysisSession.id == analysis_id
        ).first()

        if analysis:
            session.delete(analysis)
            session.flush()
            return True

        return False

    # Legacy method for backward compatibility
    @staticmethod
    def create_session(
        session: Session,
        company_id: str,
        name: str,
        dataset_ids: List[str],
        description: Optional[str] = None
    ) -> AnalysisSession:
        """Create a new analysis session (legacy method)."""
        # Assume system user if not provided
        analysis_session = AnalysisSession(
            user_id='system',  # Default for legacy calls
            company_id=company_id,
            name=name,
            dataset_ids=dataset_ids,
            description=description
        )
        session.add(analysis_session)
        session.flush()
        return analysis_session

    # Legacy method for backward compatibility
    @staticmethod
    def update_session_status(
        session: Session,
        session_id: str,
        status: str,
        insights_generated: Optional[int] = None,
        report_path: Optional[str] = None
    ):
        """Update analysis session status (legacy method)."""
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
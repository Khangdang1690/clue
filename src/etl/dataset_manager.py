"""
Dataset Lifecycle Manager

Handles the 4 core scenarios for dataset management:
1. Add completely new dataset
2. Hard delete a dataset (with cascade)
3. Add to existing dataset (append)
4. Duplicate detection

This is the smart layer that sits on top of the ETL pipeline.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.connection import DatabaseManager
from src.database.repository import (
    DatasetRepository,
    CompanyRepository,
    RelationshipRepository
)
from src.database.models import Dataset, Company
from src.graph.etl_workflow import ETLWorkflow


class UploadResult:
    """Result of processing an upload."""

    def __init__(
        self,
        status: str,  # "created", "appended", "duplicate", "error"
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        message: str = "",
        options: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        self.status = status
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.message = message
        self.options = options or []
        self.metadata = metadata or {}


class DatasetManager:
    """
    Intelligent dataset lifecycle management.

    Automatically detects whether uploaded data is:
    - Completely new dataset
    - Extension of existing dataset (append)
    - Duplicate of existing data

    And handles deletions with proper cascade.
    """

    def __init__(self):
        pass


    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def quick_duplicate_check(
        self,
        company_id: int,
        file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Lightweight duplicate detection - NO full ETL, just schema comparison.

        This is much faster than process_upload() because it only:
        - Reads file headers + first 100 rows (sample)
        - Compares schemas with existing datasets
        - Estimates overlap if schema matches

        Args:
            company_id: Company ID
            file_path: Path to file to check

        Returns:
            Dict with duplicate info if found, None otherwise
            {
                "dataset_id": str,
                "dataset_name": str,
                "overlap_percentage": float,
                "new_rows": int
            }
        """
        try:
            # Read just a sample (fast)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                new_df = pd.read_csv(file_path, nrows=100)
            else:  # Excel
                new_df = pd.read_excel(file_path, nrows=100)

            new_filename = os.path.basename(file_path)

        except Exception as e:
            print(f"[WARN] Could not quick-check {file_path}: {e}")
            return None

        # Get existing datasets for this company
        with DatabaseManager.get_session() as session:
            existing_datasets = DatasetRepository.get_datasets_by_company(
                session,
                company_id
            )

            if not existing_datasets:
                return None

            # Check each existing dataset for schema match
            for existing_ds in existing_datasets:
                try:
                    # Load sample from existing dataset
                    existing_df = DatasetRepository.load_dataframe(session, existing_ds.id)
                    if existing_df is None or len(existing_df) == 0:
                        continue

                    # Check if schemas match
                    if not self._schemas_match(new_df, existing_df):
                        continue

                    # Schema matches! Calculate overlap
                    overlap_pct = self._calculate_overlap(new_df, existing_df)

                    # Only report if significant overlap (>90% = duplicate)
                    if overlap_pct > 0.9:
                        # Get full row count from new file for metadata
                        if file_ext == '.csv':
                            full_df = pd.read_csv(file_path)
                        else:
                            full_df = pd.read_excel(file_path)

                        return {
                            "dataset_id": existing_ds.id,
                            "dataset_name": existing_ds.table_name,
                            "overlap_percentage": overlap_pct,
                            "new_rows": len(full_df)
                        }

                except Exception as e:
                    print(f"[WARN] Error checking {existing_ds.table_name}: {e}")
                    continue

        return None  # No duplicate found

    def process_upload(
        self,
        company_id: int,
        file_path: str,
        force_action: Optional[str] = None  # "new", "append", "replace", "skip"
    ) -> UploadResult:
        """
        Main entry point for handling file uploads.

        Args:
            company_id: ID of the company uploading data
            file_path: Path to the uploaded file
            force_action: Override auto-detection with explicit action

        Returns:
            UploadResult with status and metadata
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING UPLOAD: {os.path.basename(file_path)}")
        print(f"{'='*80}\n")

        # Load the new data
        try:
            new_df = pd.read_csv(file_path)
            new_filename = os.path.basename(file_path)
        except Exception as e:
            return UploadResult(
                status="error",
                message=f"Failed to read file: {str(e)}"
            )

        print(f"[FILE] Loaded {len(new_df)} rows × {len(new_df.columns)} columns")

        # Check if user forced an action
        if force_action == "skip":
            return UploadResult(status="skipped", message="Upload skipped by user")

        # Get company
        with DatabaseManager.get_session() as session:
            company = session.query(Company).filter_by(id=company_id).first()
            if not company:
                return UploadResult(
                    status="error",
                    message=f"Company ID {company_id} not found"
                )

        # Detect what type of upload this is
        if force_action:
            print(f"[FORCED ACTION] User requested: {force_action}")
            match_result = {"type": force_action.upper()}
        else:
            print(f"[AUTO-DETECT] Analyzing dataset...")
            match_result = self._find_matching_dataset(company_id, new_df, new_filename)

        # Handle each case
        if match_result["type"] == "DUPLICATE":
            return self._handle_duplicate(match_result, new_df, file_path)

        elif match_result["type"] == "APPEND":
            return self._handle_append(match_result, new_df, company_id, file_path)

        elif match_result["type"] == "REPLACE":
            return self._handle_replace(match_result, new_df, company_id, file_path)

        else:  # NEW
            return self._handle_new_dataset(company_id, file_path, new_df)


    def delete_dataset(
        self,
        dataset_id: str,
        cascade: bool = True,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a dataset and optionally cascade to dependent objects.

        Args:
            dataset_id: ID of dataset to delete
            cascade: If True, delete all dependent objects
            confirm: If False, return what would be deleted without deleting

        Returns:
            Dict with deletion summary
        """
        print(f"\n{'='*80}")
        print(f"DELETE DATASET: {dataset_id}")
        print(f"{'='*80}\n")

        with DatabaseManager.get_session() as session:
            dataset = DatasetRepository.get_dataset_by_id(session, dataset_id)
            if not dataset:
                return {
                    "status": "error",
                    "message": f"Dataset {dataset_id} not found"
                }

            print(f"[DATASET] {dataset.table_name} ({dataset.row_count} rows)")

            # Find all dependencies
            dependencies = self._find_dependencies(session, dataset_id)

            print(f"\n[DEPENDENCIES]")
            print(f"  Relationships: {len(dependencies['relationships'])}")
            print(f"  Column metadata: {len(dependencies['columns'])}")

            if not cascade:
                print(f"\n[CASCADE=FALSE] Only dataset will be deleted")

            if not confirm:
                print(f"\n[DRY RUN] No changes made. Set confirm=True to proceed.")
                return {
                    "status": "dry_run",
                    "would_delete": dependencies,
                    "dataset": {
                        "id": dataset.id,
                        "name": dataset.table_name
                    }
                }

            # Perform deletion
            if cascade:
                self._cascade_delete_dependencies(session, dependencies)

            # Save info before deleting (object will be detached after delete)
            table_name = dataset.table_name
            file_path = dataset.file_path

            # Delete file from storage
            if file_path:
                try:
                    from app.services.storage_service import StorageService
                    storage_service = StorageService()
                    storage_type = "GCS" if storage_service.use_gcs else "local storage"

                    if storage_service.file_exists(file_path):
                        storage_service.delete_file(file_path)
                        print(f"[STORAGE] Deleted file from {storage_type}: {file_path}")
                    else:
                        print(f"[STORAGE] File not found: {file_path}")
                except Exception as e:
                    print(f"[STORAGE] Warning: Could not delete file: {e}")

            # Delete the dataset record
            session.delete(dataset)
            session.commit()

            # Drop the actual database table (after committing session)
            self._drop_table(table_name)

            print(f"\n[SUCCESS] Dataset deleted")

            return {
                "status": "deleted",
                "dataset_id": dataset_id,
                "cascade": cascade,
                "deleted_objects": dependencies
            }


    # ========================================================================
    # CASE HANDLERS
    # ========================================================================

    def _handle_new_dataset(
        self,
        company_id: int,
        file_path: str,
        df: pd.DataFrame
    ) -> UploadResult:
        """CASE 1: Create completely new dataset."""
        print(f"\n[CASE 1] COMPLETELY NEW DATASET")
        print(f"{'='*80}")

        # Run full ETL pipeline
        print(f"\n[ETL] Running full pipeline...")
        etl_workflow = ETLWorkflow(company_id=str(company_id))
        etl_result = etl_workflow.run_etl(file_paths=[file_path])

        if etl_result.get('status') != 'completed':
            return UploadResult(
                status="error",
                message=f"ETL failed: {etl_result.get('error_message', 'Unknown error')}"
            )

        # Get the newly created dataset
        with DatabaseManager.get_session() as session:
            datasets = DatasetRepository.get_datasets_by_company(session, company_id)
            # Assume the newest dataset is the one we just created
            new_dataset = max(datasets, key=lambda d: d.uploaded_at)
            new_dataset_id = new_dataset.id
            new_dataset_name = new_dataset.table_name

        # Get all existing datasets (excluding this new one)
        with DatabaseManager.get_session() as session:
            all_datasets = DatasetRepository.get_datasets_by_company(session, company_id)
            existing_dataset_ids = [
                ds.id for ds in all_datasets if ds.id != new_dataset_id
            ]

        # If there are existing datasets, check for cross-dataset relationships
        if existing_dataset_ids:
            print(f"\n[RELATIONSHIPS] Checking relationships with {len(existing_dataset_ids)} existing datasets...")
            # This is already done in ETL pipeline's relationship detection
            # But we could add incremental relationship detection here if needed

        print(f"\n[SUCCESS] New dataset created: {new_dataset_name}")

        return UploadResult(
            status="created",
            dataset_id=new_dataset_id,
            dataset_name=new_dataset_name,
            message=f"Created new dataset with {len(df)} rows",
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )


    def _handle_append(
        self,
        match_result: Dict,
        new_df: pd.DataFrame,
        company_id: int,
        file_path: str
    ) -> UploadResult:
        """CASE 3: Append to existing dataset."""
        print(f"\n[CASE 3] APPEND TO EXISTING DATASET")
        print(f"{'='*80}")

        dataset_id = match_result["dataset_id"]
        dataset_name = match_result["dataset_name"]

        print(f"[TARGET] {dataset_name}")
        print(f"[OVERLAP] {match_result['overlap']:.1%} of new data already exists")

        with DatabaseManager.get_session() as session:
            # Load existing dataset
            existing_df = DatasetRepository.load_dataframe(session, dataset_id)
            print(f"[EXISTING] {len(existing_df)} rows")
            print(f"[NEW] {len(new_df)} rows")

            # Combine data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Deduplicate
            original_size = len(combined_df)
            combined_df = self._deduplicate(combined_df)
            duplicates_removed = original_size - len(combined_df)

            if duplicates_removed > 0:
                print(f"[DEDUP] Removed {duplicates_removed} duplicate rows")

            print(f"[COMBINED] {len(combined_df)} total rows")

            # Get dataset info
            dataset = DatasetRepository.get_dataset_by_id(session, dataset_id)
            table_name = dataset.table_name

            # Replace table in database
            print(f"\n[DATABASE] Updating table {table_name}...")
            combined_df.to_sql(
                table_name,
                DatabaseManager.engine,
                if_exists='replace',
                index=False
            )

            # Update dataset metadata
            dataset.row_count = len(combined_df)
            dataset.column_count = len(combined_df.columns)
            dataset.updated_at = datetime.now()
            session.commit()

            print(f"[SUCCESS] Dataset updated")

        return UploadResult(
            status="appended",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            message=f"Added {len(new_df)} rows to existing dataset",
            metadata={
                "old_row_count": len(existing_df),
                "new_row_count": len(combined_df),
                "duplicates_removed": duplicates_removed
            }
        )


    def _handle_duplicate(
        self,
        match_result: Dict,
        new_df: pd.DataFrame,
        file_path: str
    ) -> UploadResult:
        """CASE 4: Duplicate detection."""
        print(f"\n[CASE 4] DUPLICATE DETECTED")
        print(f"{'='*80}")

        dataset_name = match_result["dataset_name"]
        overlap = match_result["overlap"]

        print(f"[DUPLICATE] {overlap:.1%} of data already exists in '{dataset_name}'")

        return UploadResult(
            status="duplicate",
            dataset_id=match_result["dataset_id"],
            dataset_name=dataset_name,
            message=f"This data already exists in '{dataset_name}' ({overlap:.1%} overlap)",
            options=["skip", "replace", "append_anyway"],
            metadata={
                "overlap_percentage": overlap,
                "new_rows": len(new_df)
            }
        )


    def _handle_replace(
        self,
        match_result: Dict,
        new_df: pd.DataFrame,
        company_id: int,
        file_path: str
    ) -> UploadResult:
        """CASE 4b: Replace existing data (when user chooses 'replace' on duplicate)."""
        print(f"\n[CASE 4b] REPLACE EXISTING DATA")
        print(f"{'='*80}")

        dataset_id = match_result.get("dataset_id")

        with DatabaseManager.get_session() as session:
            dataset = DatasetRepository.get_dataset_by_id(session, dataset_id)
            table_name = dataset.table_name

            print(f"[REPLACING] {table_name}")
            print(f"[OLD] {dataset.row_count} rows")
            print(f"[NEW] {len(new_df)} rows")

            # Replace table
            new_df.to_sql(
                table_name,
                DatabaseManager.engine,
                if_exists='replace',
                index=False
            )

            # Update metadata
            dataset.row_count = len(new_df)
            dataset.column_count = len(new_df.columns)
            dataset.updated_at = datetime.now()
            session.commit()

            print(f"[SUCCESS] Data replaced")

        return UploadResult(
            status="replaced",
            dataset_id=dataset_id,
            dataset_name=table_name,
            message=f"Replaced existing data with {len(new_df)} rows",
            metadata={
                "row_count": len(new_df)
            }
        )


    # ========================================================================
    # DETECTION LOGIC
    # ========================================================================

    def _find_matching_dataset(
        self,
        company_id: int,
        new_df: pd.DataFrame,
        new_filename: str
    ) -> Dict[str, Any]:
        """
        Detect if uploaded data matches existing dataset.

        Returns:
            {
                "type": "NEW" | "APPEND" | "DUPLICATE",
                "dataset_id": str (if match found),
                "dataset_name": str,
                "overlap": float (0-1),
                "confidence": float
            }
        """
        with DatabaseManager.get_session() as session:
            existing_datasets = DatasetRepository.get_datasets_by_company(
                session,
                company_id
            )

            if not existing_datasets:
                return {"type": "NEW"}

            # Check each existing dataset
            for existing_ds in existing_datasets:
                # Load existing data
                try:
                    existing_df = DatasetRepository.load_dataframe(session, existing_ds.id)
                except Exception as e:
                    print(f"[WARN] Could not load {existing_ds.table_name}: {e}")
                    continue

                # Check if schemas match
                if not self._schemas_match(new_df, existing_df):
                    continue  # Different schema = different dataset

                print(f"[MATCH] Schema matches existing dataset: {existing_ds.table_name}")

                # Schemas match! Calculate data overlap
                overlap_pct = self._calculate_overlap(new_df, existing_df)

                print(f"[OVERLAP] {overlap_pct:.1%}")

                if overlap_pct > 0.9:  # >90% duplicate
                    return {
                        "type": "DUPLICATE",
                        "dataset_id": existing_ds.id,
                        "dataset_name": existing_ds.table_name,
                        "overlap": overlap_pct,
                        "confidence": 0.95
                    }

                elif overlap_pct > 0:  # Partial overlap
                    return {
                        "type": "APPEND",
                        "dataset_id": existing_ds.id,
                        "dataset_name": existing_ds.table_name,
                        "overlap": overlap_pct,
                        "confidence": 0.85,
                        "will_deduplicate": True
                    }

                else:  # No overlap, clean append
                    return {
                        "type": "APPEND",
                        "dataset_id": existing_ds.id,
                        "dataset_name": existing_ds.table_name,
                        "overlap": 0.0,
                        "confidence": 0.90,
                        "will_deduplicate": False
                    }

            # No matching dataset found
            return {"type": "NEW"}


    def _schemas_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Check if two dataframes have matching schemas."""
        # Ignore internal columns
        ignore_cols = {'_data_quality_flag', 'index'}

        cols1 = set(df1.columns) - ignore_cols
        cols2 = set(df2.columns) - ignore_cols

        return cols1 == cols2


    def _calculate_overlap(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> float:
        """
        Calculate what percentage of new data already exists.

        Returns:
            Float between 0 and 1 (0% to 100% overlap)
        """
        # Strategy 1: Check datetime overlap (for time series)
        datetime_cols = new_df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            # Try to detect date columns by name
            date_cols = [col for col in new_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                try:
                    new_df[date_cols[0]] = pd.to_datetime(new_df[date_cols[0]])
                    existing_df[date_cols[0]] = pd.to_datetime(existing_df[date_cols[0]])
                    datetime_cols = date_cols
                except:
                    pass

        if len(datetime_cols) > 0:
            date_col = datetime_cols[0]
            new_dates = set(pd.to_datetime(new_df[date_col]).dt.date)
            existing_dates = set(pd.to_datetime(existing_df[date_col]).dt.date)

            if len(new_dates) > 0:
                overlap = len(new_dates & existing_dates) / len(new_dates)
                return overlap

        # Strategy 2: Check ID column overlap
        id_cols = [col for col in new_df.columns if 'id' in col.lower()]
        for col in id_cols:
            if col in existing_df.columns:
                new_ids = set(new_df[col].astype(str))
                existing_ids = set(existing_df[col].astype(str))

                if len(new_ids) > 0:
                    overlap = len(new_ids & existing_ids) / len(new_ids)
                    return overlap

        # Strategy 3: Row hash overlap (fallback)
        try:
            new_hashes = set(
                new_df.apply(lambda x: hash(tuple(x.astype(str))), axis=1)
            )
            existing_hashes = set(
                existing_df.apply(lambda x: hash(tuple(x.astype(str))), axis=1)
            )

            if len(new_hashes) > 0:
                overlap = len(new_hashes & existing_hashes) / len(new_hashes)
                return overlap
        except:
            pass

        # No reliable overlap detection
        return 0.0


    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from dataframe."""
        # Try to deduplicate intelligently

        # If there's a datetime column, use it as primary key with other columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return df.drop_duplicates(subset=list(datetime_cols), keep='last')

        # If there's an ID column, use it
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        if id_cols:
            return df.drop_duplicates(subset=id_cols, keep='last')

        # Fallback: drop exact row duplicates
        return df.drop_duplicates(keep='last')


    # ========================================================================
    # DELETION HELPERS
    # ========================================================================

    def _find_dependencies(self, session: Session, dataset_id: str) -> Dict[str, List]:
        """Find all objects that depend on this dataset."""
        from src.database.models import TableRelationship, ColumnMetadata

        dependencies = {
            "relationships": [],
            "columns": []
        }

        # Find relationships that reference this dataset
        relationships = session.query(TableRelationship).filter(
            (TableRelationship.from_dataset_id == dataset_id) |
            (TableRelationship.to_dataset_id == dataset_id)
        ).all()
        dependencies["relationships"] = [r.id for r in relationships]

        # Find column metadata
        columns = session.query(ColumnMetadata).filter_by(
            dataset_id=dataset_id
        ).all()
        dependencies["columns"] = [c.id for c in columns]

        return dependencies


    def _cascade_delete_dependencies(
        self,
        session: Session,
        dependencies: Dict[str, List]
    ):
        """Delete all dependent objects."""
        from src.database.models import TableRelationship, ColumnMetadata

        # Delete relationships
        for rel_id in dependencies["relationships"]:
            rel = session.query(TableRelationship).filter_by(id=rel_id).first()
            if rel:
                session.delete(rel)

        # Delete column metadata
        for col_id in dependencies["columns"]:
            col = session.query(ColumnMetadata).filter_by(id=col_id).first()
            if col:
                session.delete(col)

        print(f"[CASCADE] Deleted {len(dependencies['relationships'])} relationships")
        print(f"[CASCADE] Deleted {len(dependencies['columns'])} column metadata")


    def _drop_table(self, table_name: str):
        """Drop the actual database table."""
        from sqlalchemy import text

        try:
            engine = DatabaseManager.get_engine()
            with engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
            print(f"[DATABASE] Dropped table {table_name}")
        except Exception as e:
            print(f"[ERROR] Could not drop table {table_name}: {e}")
            # Don't raise - dataset record is already deleted, table cleanup is best effort

"""ETL service for file upload and processing."""

import os
import tempfile
import uuid
from typing import AsyncGenerator, List, Optional, Dict, Any
from pathlib import Path
import json
import asyncio

from src.etl.dataset_manager import DatasetManager, UploadResult
from src.graph.etl_workflow import ETLWorkflow
from src.database.connection import DatabaseManager
from app.services.storage_service import StorageService
from app.services.etl_message_service import ETLMessageService, ETLJobTracker


class ETLService:
    """Service for ETL operations with progress streaming."""

    @staticmethod
    async def process_files_with_progress(
        company_id: int,
        file_paths: List[str],
        force_actions: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process uploaded files through DatasetManager with SSE progress updates.

        Uses two-phase approach:
        1. Quick scan all files for duplicates (lightweight, fast)
        2. If duplicates found, return ALL at once for user decision
        3. Batch process all approved files

        Args:
            company_id: Company ID
            file_paths: List of temporary file paths
            force_actions: Optional dict mapping file_path to force_action ("skip", "replace", "append")

        Yields:
            SSE formatted progress updates
        """
        force_actions = force_actions or {}
        manager = DatasetManager()

        total_files = len(file_paths)
        processed_results = []

        try:
            # Initial progress
            yield ETLService._format_sse({
                "step": "initialization",
                "progress": 0,
                "message": f"Starting upload of {total_files} file(s)...",
                "current_step": "Initialization",
                "status": "running"
            })

            # ================================================================
            # PHASE 1: Quick duplicate scan of ALL files
            # ================================================================
            duplicates_found = {}

            yield ETLService._format_sse({
                "step": "scanning",
                "progress": 5,
                "message": f"Scanning {total_files} file(s) for duplicates...",
                "current_step": "Duplicate Check",
                "status": "running"
            })

            await asyncio.sleep(0.1)

            for file_path in file_paths:
                file_name = os.path.basename(file_path)

                # Skip if user already made a decision for this file
                if file_name in force_actions:
                    continue

                # Quick duplicate check (lightweight)
                dup_info = manager.quick_duplicate_check(company_id, file_path)

                if dup_info:
                    duplicates_found[file_name] = {
                        "file_path": file_path,
                        "dataset_id": dup_info["dataset_id"],
                        "dataset_name": dup_info["dataset_name"],
                        "overlap_percentage": dup_info["overlap_percentage"],
                        "new_rows": dup_info["new_rows"]
                    }

            # If duplicates found, report ALL at once and wait for user
            if duplicates_found:
                yield ETLService._format_sse({
                    "step": "duplicates_detected",
                    "progress": 10,
                    "message": f"Found {len(duplicates_found)} duplicate file(s)",
                    "current_step": "Duplicates Detected",
                    "status": "duplicates_detected",
                    "duplicates": duplicates_found,
                    "options": ["skip", "replace", "append_anyway"]
                })
                return  # Stop and wait for user to resolve ALL duplicates

            # ================================================================
            # PHASE 2: Batch process all approved files
            # ================================================================
            # Handle force_actions by deleting old datasets or merging data
            # Then process ALL files through single batch ETL

            clean_files = []
            datasets_to_delete = []

            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                action = force_actions.get(file_name)

                if action == "skip":
                    continue  # Skip entirely

                elif action == "replace":
                    # For replace: delete old dataset, then process new file via ETL
                    dup_info = manager.quick_duplicate_check(company_id, file_path)
                    if dup_info:
                        datasets_to_delete.append({
                            "dataset_id": dup_info["dataset_id"],
                            "dataset_name": dup_info["dataset_name"],
                            "reason": "replace"
                        })
                    clean_files.append(file_path)

                elif action == "append_anyway":
                    # For append: load old data, merge with new, save to temp file, delete old dataset
                    dup_info = manager.quick_duplicate_check(company_id, file_path)
                    if dup_info:
                        # Load old data from database
                        with DatabaseManager.get_session() as session:
                            from src.database.repository import DatasetRepository
                            old_df = DatasetRepository.load_dataframe(session, dup_info["dataset_id"])

                        # Load new data
                        import pandas as pd
                        new_df = pd.read_csv(file_path)

                        # Merge old + new
                        combined_df = pd.concat([old_df, new_df], ignore_index=True)

                        # Deduplicate
                        combined_df = combined_df.drop_duplicates()

                        # Save to temp file (replace original)
                        combined_df.to_csv(file_path, index=False)

                        datasets_to_delete.append({
                            "dataset_id": dup_info["dataset_id"],
                            "dataset_name": dup_info["dataset_name"],
                            "reason": "append"
                        })

                    clean_files.append(file_path)

                else:
                    clean_files.append(file_path)  # New files, no duplicates

            # Delete old datasets before running ETL
            if datasets_to_delete:
                yield ETLService._format_sse({
                    "step": "deleting_old_datasets",
                    "progress": 10,
                    "message": f"Removing {len(datasets_to_delete)} old dataset(s)...",
                    "current_step": "Cleanup",
                    "status": "running"
                })

                await asyncio.sleep(0.1)

                for ds_info in datasets_to_delete:
                    try:
                        manager.delete_dataset(
                            dataset_id=ds_info["dataset_id"],
                            cascade=True,
                            confirm=True
                        )
                        print(f"[DELETED] {ds_info['dataset_name']} (reason: {ds_info['reason']})")
                    except Exception as e:
                        yield ETLService._format_sse({
                            "step": "error",
                            "progress": 10,
                            "message": f"Failed to delete {ds_info['dataset_name']}: {str(e)}",
                            "current_step": "Deletion Error",
                            "status": "error",
                            "error": str(e)
                        })
                        return

            total_to_process = len(clean_files)

            # Process clean files in BATCH with ETLWorkflow (much faster!)
            if clean_files:
                yield ETLService._format_sse({
                    "step": "batch_processing",
                    "progress": 15,
                    "message": f"Batch processing {len(clean_files)} new file(s)...",
                    "current_step": "Batch ETL",
                    "status": "running"
                })

                # Small delay to ensure SSE is flushed to client
                await asyncio.sleep(0.5)

                try:
                    # For now, use synchronous ETL with manual progress updates
                    # Custom streaming only emits after each node completes (not during execution)
                    workflow = ETLWorkflow(company_id=str(company_id))

                    # Send progress for each major ETL step
                    steps_progress = [
                        (20, "Loading files..."),
                        (35, "Analyzing semantics..."),
                        (50, "Detecting relationships..."),
                        (65, "Cleaning data..."),
                        (80, "Calculating KPIs..."),
                        (90, "Storing to database...")
                    ]

                    # Start ETL in background and manually update progress
                    etl_task = asyncio.create_task(
                        asyncio.to_thread(workflow.run_etl, file_paths=clean_files, mode='create')
                    )

                    # Simulate progress while ETL runs
                    # Don't check if done too frequently - let each progress update show
                    for i, (progress, message) in enumerate(steps_progress):
                        # Yield progress update
                        yield ETLService._format_sse({
                            "step": "processing",
                            "progress": progress,
                            "message": message,
                            "current_step": message,
                            "status": "running"
                        })

                        # Ensure the update is flushed
                        await asyncio.sleep(0.1)

                        # Check if ETL finished early
                        if etl_task.done():
                            # Check for exceptions
                            try:
                                await etl_task  # This will raise if there was an error
                            except Exception as e:
                                # ETL failed - will be handled below
                                etl_result = {"status": "error", "error_message": str(e)}
                                break
                            # ETL completed successfully early
                            break

                        # Wait before next update (only if not last step and ETL not done)
                        # For ~5 min ETL, wait ~45 seconds between updates
                        if i < len(steps_progress) - 1:
                            wait_time = 45
                            elapsed = 0
                            while elapsed < wait_time and not etl_task.done():
                                await asyncio.sleep(5)
                                elapsed += 5

                    # Wait for ETL to complete (if not already done)
                    if not etl_task.done():
                        etl_result = await etl_task
                    elif 'etl_result' not in locals():
                        # Task is done but we didn't set result yet (successful early completion)
                        etl_result = await etl_task

                    if etl_result.get('status') != 'completed':
                        yield ETLService._format_sse({
                            "step": "error",
                            "progress": 15,
                            "message": f"Batch ETL failed: {etl_result.get('error_message', 'Unknown error')}",
                            "current_step": "Batch ETL Error",
                            "status": "error",
                            "error": etl_result.get('error_message', 'Unknown error')
                        })
                        return

                    # Success - add to results
                    for file_id, dataset_id in etl_result.get('dataset_ids', {}).items():
                        meta = etl_result.get('semantic_metadata', {}).get(file_id, {})
                        processed_results.append(UploadResult(
                            status="created",
                            dataset_id=dataset_id,
                            dataset_name=meta.get('table_name', 'unknown'),
                            message=f"Created {meta.get('table_name', 'unknown')}"
                        ))

                except Exception as e:
                    yield ETLService._format_sse({
                        "step": "error",
                        "progress": 15,
                        "message": f"Batch processing error: {str(e)}",
                        "current_step": "Error",
                        "status": "error",
                        "error": str(e)
                    })
                    return

            # All files processed successfully
            yield ETLService._format_sse({
                "step": "completed",
                "progress": 100,
                "message": f"Successfully processed {len(processed_results)} file(s)",
                "current_step": "Complete",
                "status": "completed",
                "data": {
                    "company_id": company_id,
                    "total_files": total_files,
                    "processed_files": len(processed_results),
                    "results": [
                        {
                            "dataset_id": r.dataset_id,
                            "dataset_name": r.dataset_name,
                            "status": r.status,
                            "metadata": r.metadata
                        }
                        for r in processed_results
                    ]
                }
            })

        except Exception as e:
            yield ETLService._format_sse({
                "step": "error",
                "progress": 0,
                "message": f"Upload failed: {str(e)}",
                "current_step": "Error",
                "status": "error",
                "error": str(e)
            })

    @staticmethod
    def _format_sse(data: dict) -> str:
        """Format data as Server-Sent Event."""
        return f"data: {json.dumps(data)}\n\n"

    # ========================================================================
    # NEW: Fire-and-Forget Pattern with NDJSON Streaming
    # ========================================================================

    @staticmethod
    async def start_etl_job(
        company_id: int,
        file_paths: List[str],
        force_actions: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start ETL job and return job_id immediately (fire-and-forget pattern).

        Args:
            company_id: Company ID
            file_paths: List of file paths to process
            force_actions: Optional dict mapping filename to action

        Returns:
            job_id: Unique identifier for tracking this ETL job
        """
        job_id = str(uuid.uuid4())
        file_names = [os.path.basename(f) for f in file_paths]

        # Track job
        ETLJobTracker.create_job(job_id, company_id, len(file_paths), file_names)

        # Run ETL in background (don't await)
        asyncio.create_task(
            ETLService._run_etl_background(job_id, company_id, file_paths, force_actions or {})
        )

        print(f"[ETL-SERVICE] Started job {job_id} for {len(file_paths)} file(s)")
        return job_id

    @staticmethod
    async def _run_etl_background(
        job_id: str,
        company_id: int,
        file_paths: List[str],
        force_actions: Dict[str, str]
    ) -> None:
        """
        Background ETL execution with message broadcasting.

        This method runs the complete ETL workflow and emits progress updates
        via ETLMessageService for clients to stream.

        Args:
            job_id: Job identifier
            company_id: Company ID
            file_paths: Files to process
            force_actions: User actions for duplicate files
        """
        manager = DatasetManager()
        processed_results = []

        try:
            # CRITICAL: Wait 1 second for frontend to connect to stream
            # This prevents race condition where messages are sent before client connects
            print(f"[ETL-BACKGROUND] Waiting 1s for client to connect to stream...")
            await asyncio.sleep(1.0)

            # Initial message
            await ETLMessageService.emit_message(job_id, "Starting upload...", "info")
            ETLJobTracker.update_job(job_id, message="Starting upload...")

            # Phase 1: Duplicate scan
            await ETLMessageService.emit_message(
                job_id, f"Scanning {len(file_paths)} file(s) for duplicates...", "info"
            )
            ETLJobTracker.update_job(job_id, message="Scanning for duplicates...")

            await asyncio.sleep(0.1)

            duplicates_found = {}
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                if file_name in force_actions:
                    continue

                dup_info = manager.quick_duplicate_check(company_id, file_path)
                if dup_info:
                    duplicates_found[file_name] = {
                        "file_path": file_path,
                        "dataset_id": dup_info["dataset_id"],
                        "dataset_name": dup_info["dataset_name"],
                        "overlap_percentage": dup_info["overlap_percentage"],
                        "new_rows": dup_info["new_rows"]
                    }

            # If duplicates found, emit error and pause
            if duplicates_found:
                await ETLMessageService.emit_error(
                    job_id,
                    f"Found {len(duplicates_found)} duplicate file(s). Please resolve duplicates and try again."
                )
                ETLJobTracker.update_job(
                    job_id,
                    status='duplicates_detected',
                    message=f"Found {len(duplicates_found)} duplicate file(s)"
                )
                return  # Stop and wait for user resolution

            # Phase 2: Process files
            clean_files = []
            datasets_to_delete = []

            # Handle force actions (same logic as original)
            import pandas as pd
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                action = force_actions.get(file_name)

                if action == "skip":
                    continue
                elif action == "replace":
                    dup_info = manager.quick_duplicate_check(company_id, file_path)
                    if dup_info:
                        datasets_to_delete.append({
                            "dataset_id": dup_info["dataset_id"],
                            "dataset_name": dup_info["dataset_name"],
                            "reason": "replace"
                        })
                    clean_files.append(file_path)
                elif action == "append_anyway":
                    dup_info = manager.quick_duplicate_check(company_id, file_path)
                    if dup_info:
                        with DatabaseManager.get_session() as session:
                            from src.database.repository import DatasetRepository
                            old_df = DatasetRepository.load_dataframe(session, dup_info["dataset_id"])

                        new_df = pd.read_csv(file_path)
                        combined_df = pd.concat([old_df, new_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates()
                        combined_df.to_csv(file_path, index=False)

                        datasets_to_delete.append({
                            "dataset_id": dup_info["dataset_id"],
                            "dataset_name": dup_info["dataset_name"],
                            "reason": "append"
                        })
                    clean_files.append(file_path)
                else:
                    clean_files.append(file_path)

            # Delete old datasets if needed
            if datasets_to_delete:
                await ETLMessageService.emit_message(
                    job_id, f"Removing {len(datasets_to_delete)} old dataset(s)...", "info"
                )
                ETLJobTracker.update_job(job_id, message="Removing old datasets...")

                for ds_info in datasets_to_delete:
                    manager.delete_dataset(
                        dataset_id=ds_info["dataset_id"],
                        cascade=True,
                        confirm=True
                    )

            # Process clean files with ETL workflow
            if clean_files:
                await ETLMessageService.emit_message(
                    job_id, f"Processing {len(clean_files)} file(s)...", "info"
                )
                ETLJobTracker.update_job(job_id, message=f"Processing {len(clean_files)} file(s)...")

                workflow = ETLWorkflow(company_id=str(company_id))

                # Define progress messages (no percentages)
                progress_messages = [
                    "Loading files...",
                    "Analyzing data semantics...",
                    "Detecting relationships between datasets...",
                    "Cleaning and validating data...",
                    "Calculating key performance indicators...",
                    "Storing to database..."
                ]

                # Start ETL in background
                etl_task = asyncio.create_task(
                    asyncio.to_thread(workflow.run_etl, file_paths=clean_files, mode='create')
                )

                # Emit messages as ETL progresses
                for i, message in enumerate(progress_messages):
                    await ETLMessageService.emit_message(job_id, message, "info")
                    ETLJobTracker.update_job(job_id, message=message)
                    await asyncio.sleep(0.1)

                    if etl_task.done():
                        try:
                            await etl_task
                        except Exception as e:
                            etl_result = {"status": "error", "error_message": str(e)}
                            break
                        break

                    # Wait between updates
                    if i < len(progress_messages) - 1:
                        wait_time = 45
                        elapsed = 0
                        while elapsed < wait_time and not etl_task.done():
                            await asyncio.sleep(5)
                            elapsed += 5

                # Wait for ETL completion
                if not etl_task.done():
                    etl_result = await etl_task
                elif 'etl_result' not in locals():
                    etl_result = await etl_task

                if etl_result.get('status') != 'completed':
                    raise Exception(etl_result.get('error_message', 'Unknown error'))

                # Build result data
                for file_id, dataset_id in etl_result.get('dataset_ids', {}).items():
                    meta = etl_result.get('semantic_metadata', {}).get(file_id, {})
                    processed_results.append(UploadResult(
                        status="created",
                        dataset_id=dataset_id,
                        dataset_name=meta.get('table_name', 'unknown'),
                        message=f"Created {meta.get('table_name', 'unknown')}"
                    ))

            # Success - emit completion
            result_data = {
                "company_id": company_id,
                "total_files": len(file_paths),
                "processed_files": len(processed_results),
                "results": [
                    {
                        "dataset_id": r.dataset_id,
                        "dataset_name": r.dataset_name,
                        "status": r.status,
                        "metadata": r.metadata
                    }
                    for r in processed_results
                ]
            }

            await ETLMessageService.emit_complete(job_id, result_data)
            ETLJobTracker.update_job(
                job_id,
                status='completed',
                message=f"Successfully processed {len(processed_results)} file(s)",
                data=result_data
            )

            # Cleanup temp files
            ETLService.cleanup_temp_files(file_paths)

        except Exception as e:
            error_msg = str(e)
            await ETLMessageService.emit_error(job_id, error_msg)
            ETLJobTracker.update_job(
                job_id,
                status='error',
                error=error_msg,
                message=f"ETL failed: {error_msg}"
            )

            # Cleanup on error
            ETLService.cleanup_temp_files(file_paths)

    @staticmethod
    async def stream_etl_messages(job_id: str):
        """
        Stream NDJSON messages for an ETL job.

        Yields:
            NDJSON formatted messages (one JSON object per line)

        Message types:
            - message: {"type": "message", "message_type": "info|success|error", "content": "..."}
            - complete: {"type": "complete", "data": {...}}
            - error: {"type": "error", "error": "..."}
            - keepalive: {"type": "keepalive"}
        """
        queue = await ETLMessageService.register_client(job_id)

        try:
            while True:
                try:
                    # Wait for message with 15-second timeout
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield json.dumps(message) + '\n'

                    # Stop streaming on terminal events
                    if message.get('type') in ['complete', 'error']:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive ping to prevent connection timeout
                    yield json.dumps({'type': 'keepalive'}) + '\n'

        finally:
            await ETLMessageService.unregister_client(job_id, queue)

    @staticmethod
    async def save_uploaded_files(files, company_id: int) -> List[str]:
        """
        Save uploaded files to storage and temporary directory.

        Strategy:
        - Upload to storage (GCS in production, local in development)
        - Also save to temp directory for ETL processing
        - ETL workflow processes from temp files
        - Temp files cleaned up after processing

        Args:
            files: List of UploadFile objects from FastAPI
            company_id: Company ID for organizing storage

        Returns:
            List of temporary file paths (for ETL processing)
        """
        storage_service = StorageService()
        temp_dir = tempfile.mkdtemp(prefix="iclue_etl_")
        file_paths = []

        for file in files:
            # Read file content once
            content = await file.read()

            # Upload to storage for permanent storage
            storage_path = StorageService.get_gcs_path(company_id, file.filename)
            storage_service.upload_file(content, storage_path)
            storage_type = "GCS" if storage_service.use_gcs else "local storage"
            print(f"[STORAGE] Uploaded {file.filename} to {storage_type}: {storage_path}")

            # Also save to temp directory for ETL processing
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, 'wb') as f:
                f.write(content)

            file_paths.append(temp_path)

        return file_paths

    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """Clean up temporary files after processing."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp file {file_path}: {e}")

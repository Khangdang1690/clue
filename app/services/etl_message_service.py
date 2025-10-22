"""Service for streaming real-time ETL messages (Claude-style)."""

import asyncio
from typing import Dict, List
from threading import Lock
from collections import defaultdict


class ETLMessageService:
    """
    Simple service for streaming narrative messages as ETL runs.

    Matches the AnalysisMessageService pattern - just stream text messages,
    no complex progress tracking with percentages and steps.
    """

    # Message queues: {job_id: [queue1, queue2, ...]}
    _client_queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)
    _queues_lock = Lock()

    @classmethod
    async def emit_message(cls, job_id: str, content: str, message_type: str = 'info') -> None:
        """
        Emit a narrative message for streaming to clients.

        Args:
            job_id: Unique ETL job identifier
            content: The message content to stream
            message_type: Type of message ('info', 'success', 'error', 'thinking')
        """
        print(f"[ETL-STREAM {job_id}] {content}")

        message = {
            'type': 'message',
            'message_type': message_type,
            'content': content
        }

        # Broadcast to all connected clients
        with cls._queues_lock:
            queues = cls._client_queues.get(job_id, [])
            print(f"[ETL-STREAM] Broadcasting to {len(queues)} client(s)")

            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[ETL-STREAM ERROR] Failed to queue message: {e}")

    @classmethod
    async def emit_complete(cls, job_id: str, data: dict = None) -> None:
        """
        Emit completion message.

        Args:
            job_id: Unique ETL job identifier
            data: Optional result data (datasets_processed, relationships_found, etc.)
        """
        print(f"[ETL-STREAM {job_id}] Emitting completion")

        message = {
            'type': 'complete',
            'data': data or {}
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(job_id, [])
            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[ETL-STREAM ERROR] Failed to queue completion: {e}")

    @classmethod
    async def emit_error(cls, job_id: str, error: str) -> None:
        """
        Emit error message.

        Args:
            job_id: Unique ETL job identifier
            error: Error message
        """
        print(f"[ETL-STREAM {job_id}] Emitting error: {error}")

        message = {
            'type': 'error',
            'error': error
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(job_id, [])
            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[ETL-STREAM ERROR] Failed to queue error: {e}")

    @classmethod
    async def register_client(cls, job_id: str) -> asyncio.Queue:
        """
        Register a new client and return its message queue.

        Args:
            job_id: ETL job ID to listen to

        Returns:
            asyncio.Queue for receiving messages
        """
        print(f"[ETL-STREAM] Registering client for job {job_id}")

        queue = asyncio.Queue()
        with cls._queues_lock:
            cls._client_queues[job_id].append(queue)
            print(f"[ETL-STREAM] Total clients for {job_id}: {len(cls._client_queues[job_id])}")

        return queue

    @classmethod
    async def unregister_client(cls, job_id: str, queue: asyncio.Queue) -> None:
        """
        Unregister a client.

        Args:
            job_id: ETL job ID
            queue: Client's queue to remove
        """
        with cls._queues_lock:
            if job_id in cls._client_queues:
                if queue in cls._client_queues[job_id]:
                    cls._client_queues[job_id].remove(queue)
                    print(f"[ETL-STREAM] Client unregistered. Remaining: {len(cls._client_queues[job_id])}")

    @classmethod
    def cleanup_job(cls, job_id: str) -> None:
        """
        Clean up message queues for a job.

        Args:
            job_id: ETL job ID
        """
        with cls._queues_lock:
            if job_id in cls._client_queues:
                del cls._client_queues[job_id]
                print(f"[ETL-STREAM] Cleaned up queues for {job_id}")


class ETLJobTracker:
    """
    Track ETL job status in memory.

    Provides a simple in-memory store for job status that can be polled
    as a fallback if streaming fails.
    """

    # Job status: {job_id: {status, message, ...}}
    _jobs: Dict[str, Dict[str, any]] = {}
    _jobs_lock = Lock()

    @classmethod
    def create_job(cls, job_id: str, company_id: int, file_count: int, file_names: List[str]) -> None:
        """Create a new job tracking entry."""
        from datetime import datetime

        with cls._jobs_lock:
            cls._jobs[job_id] = {
                'job_id': job_id,
                'company_id': company_id,
                'file_count': file_count,
                'file_names': file_names,
                'status': 'running',
                'message': 'Starting ETL...',
                'created_at': datetime.utcnow().isoformat(),
                'completed_at': None,
                'error': None,
                'data': None
            }
        print(f"[JOB-TRACKER] Created job {job_id}: {file_count} file(s)")

    @classmethod
    def update_job(cls, job_id: str, **kwargs) -> None:
        """Update job status fields."""
        from datetime import datetime

        with cls._jobs_lock:
            if job_id in cls._jobs:
                cls._jobs[job_id].update(kwargs)
                if kwargs.get('status') in ['completed', 'error']:
                    cls._jobs[job_id]['completed_at'] = datetime.utcnow().isoformat()

    @classmethod
    def get_job(cls, job_id: str) -> dict:
        """Get job status."""
        with cls._jobs_lock:
            return cls._jobs.get(job_id)

    @classmethod
    def delete_job(cls, job_id: str) -> None:
        """Delete a job from tracking."""
        with cls._jobs_lock:
            if job_id in cls._jobs:
                del cls._jobs[job_id]
                print(f"[JOB-TRACKER] Deleted job {job_id}")

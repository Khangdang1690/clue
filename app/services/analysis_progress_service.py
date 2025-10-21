"""Service for tracking and streaming analysis progress in real-time."""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import Lock
from collections import defaultdict


class AnalysisProgressService:
    """
    Service for tracking progress of business discovery workflows.

    Stores progress updates in memory and provides methods for:
    - Emitting progress updates from workflow nodes
    - Streaming progress to SSE clients
    - Tracking step completion and status
    """

    # In-memory storage for progress data
    # Format: {analysis_id: {step_name: step_data, ...}}
    _progress_store: Dict[str, Dict[str, Any]] = {}
    _store_lock = Lock()

    # SSE client queues: {analysis_id: [queue1, queue2, ...]}
    _client_queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)
    _queues_lock = Lock()

    # Step metadata
    WORKFLOW_STEPS = [
        {
            'name': 'load_data',
            'display_name': 'Load Data',
            'order': 1,
        },
        {
            'name': 'understand_business',
            'display_name': 'Understand Business Context',
            'order': 2,
        },
        {
            'name': 'explore_dynamically',
            'display_name': 'Dynamic Exploration',
            'order': 3,
        },
        {
            'name': 'run_analytics',
            'display_name': 'Run Advanced Analytics',
            'order': 4,
        },
        {
            'name': 'generate_insights',
            'display_name': 'Generate Insights',
            'order': 5,
        },
        {
            'name': 'synthesize_insights',
            'display_name': 'Synthesize Insights',
            'order': 6,
        },
        {
            'name': 'create_recommendations',
            'display_name': 'Create Recommendations',
            'order': 7,
        },
        {
            'name': 'generate_report',
            'display_name': 'Generate Report & Dashboard',
            'order': 8,
        },
    ]

    @classmethod
    def initialize_analysis(cls, analysis_id: str) -> None:
        """Initialize progress tracking for a new analysis."""
        with cls._store_lock:
            cls._progress_store[analysis_id] = {
                'analysis_id': analysis_id,
                'status': 'running',
                'current_step': 0,
                'total_steps': len(cls.WORKFLOW_STEPS),
                'started_at': datetime.utcnow().isoformat(),
                'steps': {
                    step['name']: {
                        'name': step['name'],
                        'display_name': step['display_name'],
                        'order': step['order'],
                        'status': 'pending',
                        'started_at': None,
                        'completed_at': None,
                        'details': {}
                    }
                    for step in cls.WORKFLOW_STEPS
                }
            }

    @classmethod
    async def emit_progress(
        cls,
        analysis_id: str,
        step_name: str,
        status: str,  # 'running', 'completed', 'failed'
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit a progress update for a workflow step.

        Args:
            analysis_id: Unique analysis identifier
            step_name: Name of the workflow step
            status: Current status of the step
            details: Additional details about the progress
        """
        print(f"[PROGRESS] emit_progress called: {analysis_id} - {step_name} - {status}")
        timestamp = datetime.utcnow().isoformat()

        with cls._store_lock:
            if analysis_id not in cls._progress_store:
                print(f"[PROGRESS] Analysis {analysis_id} not in store, initializing")
                cls.initialize_analysis(analysis_id)

            progress_data = cls._progress_store[analysis_id]

            # Update step data
            if step_name in progress_data['steps']:
                step_data = progress_data['steps'][step_name]
                step_data['status'] = status
                step_data['details'] = details or {}

                if status == 'running' and not step_data['started_at']:
                    step_data['started_at'] = timestamp
                    progress_data['current_step'] = step_data['order']
                elif status == 'completed':
                    step_data['completed_at'] = timestamp
                elif status == 'failed':
                    step_data['completed_at'] = timestamp
                    progress_data['status'] = 'failed'
                    progress_data['error'] = details.get('error', 'Unknown error')

                print(f"[PROGRESS] Updated step {step_name} to {status}")
            else:
                print(f"[PROGRESS WARNING] Step {step_name} not found in workflow steps")

        # Send update to all connected SSE clients
        print(f"[PROGRESS] Broadcasting update to clients")
        await cls._broadcast_update(analysis_id)

    @classmethod
    async def mark_completed(
        cls,
        analysis_id: str,
        dashboard_url: str,
        report_path: str,
        insights_count: int = 0,
        recommendations_count: int = 0
    ) -> None:
        """Mark an analysis as completed."""
        with cls._store_lock:
            if analysis_id in cls._progress_store:
                progress_data = cls._progress_store[analysis_id]
                progress_data['status'] = 'completed'
                progress_data['completed_at'] = datetime.utcnow().isoformat()
                progress_data['dashboard_url'] = dashboard_url
                progress_data['report_path'] = report_path
                progress_data['insights_count'] = insights_count
                progress_data['recommendations_count'] = recommendations_count

        # Send completion event
        await cls._broadcast_completion(analysis_id)

    @classmethod
    async def mark_failed(
        cls,
        analysis_id: str,
        error: str,
        failed_step: Optional[str] = None
    ) -> None:
        """Mark an analysis as failed."""
        with cls._store_lock:
            if analysis_id in cls._progress_store:
                progress_data = cls._progress_store[analysis_id]
                progress_data['status'] = 'failed'
                progress_data['error'] = error
                progress_data['failed_step'] = failed_step
                progress_data['completed_at'] = datetime.utcnow().isoformat()

        # Send error event
        await cls._broadcast_error(analysis_id)

    @classmethod
    def get_progress(cls, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for an analysis."""
        with cls._store_lock:
            return cls._progress_store.get(analysis_id)

    @classmethod
    async def register_client(cls, analysis_id: str) -> asyncio.Queue:
        """Register a new SSE client and return its queue."""
        print(f"[PROGRESS] Registering new client for analysis {analysis_id}")
        queue = asyncio.Queue()
        with cls._queues_lock:
            cls._client_queues[analysis_id].append(queue)
            print(f"[PROGRESS] Total clients for {analysis_id}: {len(cls._client_queues[analysis_id])}")

        # Send current progress immediately
        progress = cls.get_progress(analysis_id)
        if progress:
            print(f"[PROGRESS] Sending initial progress to new client: current_step={progress.get('current_step')}, status={progress.get('status')}")
            await queue.put({
                'event': 'progress',
                'data': progress
            })
        else:
            print(f"[PROGRESS WARNING] No progress data found for {analysis_id} when registering client")

        return queue

    @classmethod
    async def unregister_client(cls, analysis_id: str, queue: asyncio.Queue) -> None:
        """Unregister an SSE client."""
        with cls._queues_lock:
            if analysis_id in cls._client_queues:
                if queue in cls._client_queues[analysis_id]:
                    cls._client_queues[analysis_id].remove(queue)

    @classmethod
    async def _broadcast_update(cls, analysis_id: str) -> None:
        """Broadcast progress update to all connected clients."""
        progress = cls.get_progress(analysis_id)
        if not progress:
            print(f"[PROGRESS] No progress data found for {analysis_id}")
            return

        event = {
            'event': 'progress',
            'data': progress
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            print(f"[PROGRESS] Broadcasting to {len(queues)} connected clients")
            for queue in queues:
                try:
                    await queue.put(event)
                    print(f"[PROGRESS] Event added to queue successfully")
                except Exception as e:
                    print(f"[PROGRESS ERROR] Error broadcasting to client: {e}")

    @classmethod
    async def _broadcast_completion(cls, analysis_id: str) -> None:
        """Broadcast completion event to all connected clients."""
        progress = cls.get_progress(analysis_id)
        if not progress:
            return

        event = {
            'event': 'complete',
            'data': {
                'analysis_id': analysis_id,
                'status': 'completed',
                'dashboard_url': progress.get('dashboard_url'),
                'report_path': progress.get('report_path'),
                'insights_count': progress.get('insights_count', 0),
                'recommendations_count': progress.get('recommendations_count', 0),
            }
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            for queue in queues:
                try:
                    await queue.put(event)
                except Exception as e:
                    print(f"Error broadcasting completion: {e}")

    @classmethod
    async def _broadcast_error(cls, analysis_id: str) -> None:
        """Broadcast error event to all connected clients."""
        progress = cls.get_progress(analysis_id)
        if not progress:
            return

        event = {
            'event': 'error',
            'data': {
                'analysis_id': analysis_id,
                'status': 'failed',
                'error': progress.get('error', 'Unknown error'),
                'failed_step': progress.get('failed_step'),
            }
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            for queue in queues:
                try:
                    await queue.put(event)
                except Exception as e:
                    print(f"Error broadcasting error: {e}")

    @classmethod
    def cleanup_analysis(cls, analysis_id: str) -> None:
        """Clean up progress data for a completed/failed analysis."""
        with cls._store_lock:
            if analysis_id in cls._progress_store:
                del cls._progress_store[analysis_id]

        with cls._queues_lock:
            if analysis_id in cls._client_queues:
                del cls._client_queues[analysis_id]

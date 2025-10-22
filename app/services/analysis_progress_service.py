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
    - Streaming progress to HTTP clients (NDJSON format)
    - Tracking step completion and status
    """

    # In-memory storage for progress data
    # Format: {analysis_id: {step_name: step_data, ...}}
    _progress_store: Dict[str, Dict[str, Any]] = {}
    _store_lock = Lock()

    # Streaming client queues: {analysis_id: [queue1, queue2, ...]}
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
        print(f"[PROGRESS] initialize_analysis called for {analysis_id}")

        # CRITICAL: Clear any existing client queues for this analysis
        # This prevents old streaming connections from receiving stale data
        with cls._queues_lock:
            if analysis_id in cls._client_queues:
                print(f"[PROGRESS] Clearing {len(cls._client_queues[analysis_id])} existing client queue(s)")
                cls._client_queues[analysis_id] = []

        # Initialize fresh progress state
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
            print(f"[PROGRESS] Initialized progress store: status='running', current_step=0, total_steps={len(cls.WORKFLOW_STEPS)}")

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

        # Send update to all connected streaming clients
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
        """Register a new streaming client and return its queue."""
        print(f"[PROGRESS] Registering new client for analysis {analysis_id}")
        queue = asyncio.Queue()
        with cls._queues_lock:
            cls._client_queues[analysis_id].append(queue)
            print(f"[PROGRESS] Total clients for {analysis_id}: {len(cls._client_queues[analysis_id])}")

        # Send ONLY current progress state (not full history)
        # This prevents overwhelming the client with all past events at once
        progress = cls.get_progress(analysis_id)
        if progress:
            print(f"[PROGRESS] Sending current progress to new client:")
            print(f"  - current_step: {progress.get('current_step')}")
            print(f"  - total_steps: {progress.get('total_steps')}")
            print(f"  - status: {progress.get('status')}")
            print(f"  - analysis_id: {progress.get('analysis_id')}")

            # Log step statuses to debug
            steps_summary = {}
            if 'steps' in progress:
                for step_name, step_data in progress['steps'].items():
                    steps_summary[step_name] = step_data.get('status', 'unknown')
            print(f"  - steps: {steps_summary}")

            # If the analysis is already completed/failed, immediately emit terminal event
            status = progress.get('status')
            if status in ('completed', 'failed'):
                if status == 'completed':
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
                else:
                    event = {
                        'event': 'error',
                        'data': {
                            'analysis_id': analysis_id,
                            'status': 'failed',
                            'error': progress.get('error', 'Unknown error'),
                            'failed_step': progress.get('failed_step'),
                        }
                    }
                await queue.put(event)
            else:
                # CRITICAL: Send a deep copy so mutations don't affect queued events
                import copy
                progress_snapshot = copy.deepcopy(progress)
                await queue.put({
                    'event': 'progress',
                    'data': progress_snapshot
                })
        else:
            print(f"[PROGRESS WARNING] No progress data found for {analysis_id} when registering client")
            print(f"[PROGRESS] Initializing progress for {analysis_id}")
            cls.initialize_analysis(analysis_id)
            progress = cls.get_progress(analysis_id)
            if progress:
                # CRITICAL: Send a deep copy
                import copy
                progress_snapshot = copy.deepcopy(progress)
                await queue.put({
                    'event': 'progress',
                    'data': progress_snapshot
                })

        return queue

    @classmethod
    async def unregister_client(cls, analysis_id: str, queue: asyncio.Queue) -> None:
        """Unregister a streaming client."""
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

        # CRITICAL: Create a deep copy of the progress data
        # Without this, all queued events point to the same mutable dict
        # and get the LATEST state instead of the state at broadcast time
        import copy
        progress_snapshot = copy.deepcopy(progress)

        event = {
            'event': 'progress',
            'data': progress_snapshot
        }

        # Broadcast to connected clients
        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            current_step = progress_snapshot.get('current_step', 0)
            status = progress_snapshot.get('status', 'unknown')
            print(f"[PROGRESS BROADCAST] Step {current_step}/8, Status: {status}, Clients: {len(queues)}")

            if len(queues) == 0:
                print(f"[PROGRESS WARNING] No clients connected to receive broadcast!")

            for i, queue in enumerate(queues):
                try:
                    await queue.put(event)
                    print(f"[PROGRESS BROADCAST] Event queued for client #{i+1} ✓")
                except Exception as e:
                    print(f"[PROGRESS BROADCAST ERROR] Failed to queue for client #{i+1}: {e}")

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

        # Broadcast to connected clients
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

        # Broadcast to connected clients
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

"""Service for streaming real-time analysis messages (Claude-style)."""

import asyncio
from typing import Dict, List
from threading import Lock
from collections import defaultdict


class AnalysisMessageService:
    """
    Simple service for streaming narrative messages as analysis runs.

    Unlike progress tracking, this emits human-readable narrative messages
    as the workflow executes - similar to how Claude streams responses.
    """

    # Message queues: {analysis_id: [queue1, queue2, ...]}
    _client_queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)
    _queues_lock = Lock()

    @classmethod
    async def emit_message(cls, analysis_id: str, content: str, message_type: str = 'info') -> None:
        """
        Emit a narrative message for streaming to clients.

        Args:
            analysis_id: Unique analysis identifier
            content: The message content to stream
            message_type: Type of message ('info', 'success', 'error', 'thinking', 'insight')
        """
        print(f"[MESSAGE-STREAM] Emitting: {content[:80]}...")

        message = {
            'type': 'message',
            'message_type': message_type,
            'content': content
        }

        # Broadcast to all connected clients
        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            print(f"[MESSAGE-STREAM] Broadcasting to {len(queues)} client(s)")

            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[MESSAGE-STREAM ERROR] Failed to queue message: {e}")

    @classmethod
    async def emit_complete(
        cls,
        analysis_id: str,
        insights_count: int = 0,
        recommendations_count: int = 0
    ) -> None:
        """Emit completion message."""
        print(f"[MESSAGE-STREAM] Emitting completion")

        message = {
            'type': 'complete',
            'insights_count': insights_count,
            'recommendations_count': recommendations_count
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[MESSAGE-STREAM ERROR] Failed to queue completion: {e}")

    @classmethod
    async def emit_error(cls, analysis_id: str, error: str) -> None:
        """Emit error message."""
        print(f"[MESSAGE-STREAM] Emitting error: {error}")

        message = {
            'type': 'error',
            'error': error
        }

        with cls._queues_lock:
            queues = cls._client_queues.get(analysis_id, [])
            for queue in queues:
                try:
                    await queue.put(message)
                except Exception as e:
                    print(f"[MESSAGE-STREAM ERROR] Failed to queue error: {e}")

    @classmethod
    async def register_client(cls, analysis_id: str) -> asyncio.Queue:
        """Register a new client and return its message queue."""
        print(f"[MESSAGE-STREAM] Registering client for {analysis_id}")

        queue = asyncio.Queue()
        with cls._queues_lock:
            cls._client_queues[analysis_id].append(queue)
            print(f"[MESSAGE-STREAM] Total clients for {analysis_id}: {len(cls._client_queues[analysis_id])}")

        return queue

    @classmethod
    async def unregister_client(cls, analysis_id: str, queue: asyncio.Queue) -> None:
        """Unregister a client."""
        with cls._queues_lock:
            if analysis_id in cls._client_queues:
                if queue in cls._client_queues[analysis_id]:
                    cls._client_queues[analysis_id].remove(queue)
                    print(f"[MESSAGE-STREAM] Client unregistered. Remaining: {len(cls._client_queues[analysis_id])}")

    @classmethod
    def cleanup_analysis(cls, analysis_id: str) -> None:
        """Clean up message queues for an analysis."""
        with cls._queues_lock:
            if analysis_id in cls._client_queues:
                del cls._client_queues[analysis_id]
                print(f"[MESSAGE-STREAM] Cleaned up queues for {analysis_id}")

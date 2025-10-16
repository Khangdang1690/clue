"""WebSocket manager for real-time progress updates and log streaming."""

import sys
import asyncio
from typing import List, Optional
from fastapi import WebSocket
from datetime import datetime
import json


class WebSocketManager:
    """Manages WebSocket connections and broadcasts messages to connected clients."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._original_stdout = None
        self._original_stderr = None

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.send_message(websocket, {
            "type": "connected",
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message to websocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to websocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_log(self, message: str, level: str = "info"):
        """Broadcast a log message to all connected clients."""
        await self.broadcast({
            "type": "log",
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_progress(self, phase: str, percent: float, step: Optional[str] = None):
        """Broadcast progress update to all connected clients."""
        data = {
            "phase": phase,
            "percent": percent
        }
        if step:
            data["step"] = step

        await self.broadcast({
            "type": "progress",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_phase_complete(self, phase: str, data: Optional[dict] = None):
        """Broadcast phase completion to all connected clients."""
        await self.broadcast({
            "type": "phase_complete",
            "phase": phase,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_challenge_complete(self, challenge_id: str, challenge_title: str, remaining: int):
        """Broadcast challenge completion to all connected clients."""
        await self.broadcast({
            "type": "challenge_complete",
            "data": {
                "challenge_id": challenge_id,
                "challenge_title": challenge_title,
                "challenges_remaining": remaining
            },
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_visualization_created(self, viz_id: str, path: str, title: str):
        """Broadcast visualization creation to all connected clients."""
        await self.broadcast({
            "type": "visualization_created",
            "data": {
                "id": viz_id,
                "path": path,
                "title": title
            },
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_report_generated(self, report_type: str, path: str):
        """Broadcast report generation to all connected clients."""
        await self.broadcast({
            "type": "report_generated",
            "data": {
                "type": report_type,
                "path": path
            },
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_error(self, error: str, detail: Optional[str] = None):
        """Broadcast error message to all connected clients."""
        await self.broadcast({
            "type": "error",
            "message": error,
            "detail": detail,
            "timestamp": datetime.now().isoformat()
        })


class PrintCapture:
    """Capture print statements and broadcast them via WebSocket."""

    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, message: str):
        """Intercept write calls and broadcast to WebSocket."""
        # Write to original stdout
        self.original_stdout.write(message)
        self.original_stdout.flush()

        # Broadcast to WebSocket if it's a meaningful message
        if message.strip():
            # Create async task to broadcast
            asyncio.create_task(self.ws_manager.broadcast_log(message.strip()))

    def flush(self):
        """Flush the output."""
        self.original_stdout.flush()


# Global WebSocket manager instance
ws_manager = WebSocketManager()

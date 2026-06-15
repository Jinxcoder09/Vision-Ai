"""
Eyeva AI – WebSocket Connection Manager

Manages active WebSocket connections and provides typed message routing.
Each connection carries a session state (voice, speed, lang config).
"""
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket
from loguru import logger


@dataclass
class SessionState:
    """Per-connection session configuration and state."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en"
    mode: str = "voice"
    vad_threshold: float = 0.5
    last_frame_b64: Optional[str] = None  # most recent camera frame
    status: str = "idle"
    last_analyzed_frame_b64: Optional[str] = None
    scene_memory: dict = field(default_factory=lambda: {
        "timestamp": 0.0,
        "last_update": 0.0,
        "scene_summary": "",
        "summary": "",
        "objects": [],
        "people_count": 0,
        "detected_text": "",
        "text": "",
        "confidence": 0
    })


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        # Maps websocket → session state
        self._connections: dict[WebSocket, SessionState] = {}

    async def connect(self, websocket: WebSocket) -> SessionState:
        await websocket.accept()
        session = SessionState()
        self._connections[websocket] = session
        logger.info("WS connected — session={} total={}", session.session_id, len(self._connections))
        return session

    def disconnect(self, websocket: WebSocket) -> None:
        session = self._connections.pop(websocket, None)
        if session:
            logger.info("WS disconnected — session={} total={}", session.session_id, len(self._connections))

    def get_session(self, websocket: WebSocket) -> Optional[SessionState]:
        return self._connections.get(websocket)

    async def send_json(self, websocket: WebSocket, payload: dict) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning("Failed to send JSON to WS: {}", e)

    async def send_bytes(self, websocket: WebSocket, data: bytes) -> None:
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            logger.warning("Failed to send bytes to WS: {}", e)

    async def send_status(self, websocket: WebSocket, state: str) -> None:
        await self.send_json(websocket, {"type": "status", "state": state})

    async def send_error(self, websocket: WebSocket, message: str) -> None:
        await self.send_json(websocket, {"type": "error", "message": message})

    async def send_transcript(self, websocket: WebSocket, text: str) -> None:
        await self.send_json(websocket, {"type": "transcript", "data": text})

    async def send_response(self, websocket: WebSocket, text: str) -> None:
        await self.send_json(websocket, {"type": "response", "data": text})

    async def send_ocr_result(self, websocket: WebSocket, text: str) -> None:
        await self.send_json(websocket, {"type": "ocr_result", "data": text})

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# Module-level singleton shared across all route handlers
manager = ConnectionManager()

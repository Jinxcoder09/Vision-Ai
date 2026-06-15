"""
Eyeva AI – Vision Routes

POST /api/vision/analyze  – single-shot scene description
POST /api/vision/question – single-shot Q&A about image
WS   /ws/vision           – streaming vision with frame throttling
"""
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from models.schemas import VisionRequest, VisionResponse, WSMessage
from services import vision_service
from websocket.manager import manager

router = APIRouter(tags=["vision"])


@router.post("/api/vision/analyze", response_model=VisionResponse, summary="Analyze camera frame")
async def analyze_frame(req: VisionRequest) -> VisionResponse:
    """
    Analyze a base64-encoded image and return a concise scene description.
    Responses are cached for 30 seconds (configurable) to avoid redundant API calls.
    """
    try:
        description, cached, latency = await vision_service.analyze_frame(
            req.image_b64, req.prompt
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return VisionResponse(description=description, cached=cached, latency_ms=latency)


@router.post("/api/vision/question", response_model=VisionResponse, summary="Ask a question about the image")
async def ask_question(req: VisionRequest) -> VisionResponse:
    """
    Answer a specific question about the provided image frame.
    Not cached (questions can vary on the same frame).
    """
    try:
        answer, latency = await vision_service.answer_question(req.image_b64, req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return VisionResponse(description=answer, cached=False, latency_ms=latency)


@router.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket) -> None:
    """
    Streaming vision WebSocket.

    Client sends JSON: {"type": "frame", "data": "<base64_jpeg>", "action": "describe|question", "lang": "en"}
    Server sends JSON: {"type": "response", "data": "<description>"}
                  or: {"type": "error", "message": "..."}
                  or: {"type": "status", "state": "processing|idle"}
    """
    session = await manager.connect(websocket)
    await manager.send_status(websocket, "idle")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WSMessage(**json.loads(raw))
            except Exception:
                await manager.send_error(websocket, "Invalid message format.")
                continue

            # ── Config update ─────────────────────────────────────────────────
            if msg.type == "config":
                if msg.voice:
                    session.voice = msg.voice
                if msg.speed:
                    session.speed = msg.speed
                if msg.lang:
                    session.lang = msg.lang
                continue

            # ── Frame analysis ────────────────────────────────────────────────
            if msg.type == "frame" and msg.data:
                session.last_frame_b64 = msg.data
                await manager.send_status(websocket, "processing")

                action = msg.action or "describe"
                prompt = (
                    msg.data  # reuse data field for question text if action=="question"
                    if False  # placeholder — question text comes from msg.action details
                    else "Describe the scene concisely for a visually impaired person."
                )

                if action == "question" and msg.lang:
                    prompt = msg.lang  # repurposed: "lang" carries question text in question mode

                try:
                    description, _, latency = await vision_service.analyze_frame(msg.data, prompt)
                    await manager.send_response(websocket, description)
                except RuntimeError as e:
                    await manager.send_error(websocket, str(e))
                finally:
                    await manager.send_status(websocket, "idle")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("Vision WS unexpected error: {}", e)
        await manager.send_error(websocket, "An unexpected error occurred.")
        manager.disconnect(websocket)

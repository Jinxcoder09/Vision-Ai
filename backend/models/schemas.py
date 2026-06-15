"""
Eyeva AI – Pydantic request/response schemas.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Vision ────────────────────────────────────────────────────────────────────

class VisionRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    prompt: str = Field(
        default="Describe the scene concisely for a visually impaired person.",
        max_length=500,
    )
    lang: str = Field(default="en", description="Response language code")


class VisionResponse(BaseModel):
    description: str
    cached: bool = False
    latency_ms: Optional[float] = None


# ── OCR ───────────────────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    lang: Literal["en", "hi"] = Field(default="en", description="OCR language")


class OCRResponse(BaseModel):
    text: str
    word_count: int
    lang: str
    latency_ms: Optional[float] = None


# ── Speech-to-Text ────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None


# ── Text-to-Speech ────────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    voice: str = Field(default="af_heart")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


# ── WebSocket Message Protocol ────────────────────────────────────────────────

class WSMessage(BaseModel):
    """Typed WebSocket JSON message envelope."""
    type: Literal[
        "frame",
        "command",
        "config",
        "transcript",
        "response",
        "ocr_result",
        "error",
        "status",
        "audio",
    ]
    data: Optional[str] = None       # base64 frame or text payload
    action: Optional[str] = None     # for command type
    voice: Optional[str] = None      # for config type
    speed: Optional[float] = None    # for config type
    lang: Optional[str] = None       # language override
    mode: Optional[str] = None       # mode config (voice, navigation, money, text)
    vad_threshold: Optional[float] = None # VAD sensitivity threshold
    message: Optional[str] = None    # for error type
    state: Optional[str] = None      # for status type


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    services: dict[str, bool] = {}

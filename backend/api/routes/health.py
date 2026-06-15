"""
Eyeva AI – Health Check Route
GET /api/health — returns overall status and per-service availability.
"""
from fastapi import APIRouter
from models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Returns overall API health and the availability of each backend service.
    Used by Docker HEALTHCHECK and frontend status indicators.
    """
    services: dict[str, bool] = {}

    # ── Check Whisper (Groq) ──────────────────────────────────────────────────
    try:
        import openai  # noqa
        from core.config import get_settings
        cfg = get_settings()
        services["stt"] = bool(cfg.groq_api_key)
    except Exception:
        services["stt"] = False

    # ── Check Orpheus TTS (Groq) ──────────────────────────────────────────────
    try:
        import openai  # noqa
        from core.config import get_settings
        cfg = get_settings()
        services["tts"] = bool(cfg.groq_api_key)
    except Exception:
        services["tts"] = False

    # ── Check OCR (NVIDIA VLM) ────────────────────────────────────────────────
    try:
        from core.config import get_settings
        cfg = get_settings()
        services["ocr"] = bool(cfg.effective_api_key)
    except Exception:
        services["ocr"] = False

    # ── Check Groq API (Vision VLM) ───────────────────────────────────────────
    try:
        import openai  # noqa
        from core.config import get_settings
        cfg = get_settings()
        services["vision"] = bool(cfg.groq_api_key)
    except Exception:
        services["vision"] = False

    overall = "ok" if all(services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)

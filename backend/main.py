"""
Eyeva AI – FastAPI Application Entry Point

Wires up:
  - CORS middleware
  - All API routers
  - Lifespan events (model warm-up, cleanup)
  - Global exception handlers
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

from api.routes import health, ocr, speech, vision
from core.config import get_settings
from core.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    setup_logging(settings.log_level)
    os.makedirs("logs", exist_ok=True)

    logger.info("====================================")
    logger.info("  Eyeva AI V1 -- Starting up")
    logger.info("====================================")
    logger.info("  VLM model   : {}", settings.vlm_model)
    logger.info("  Whisper     : {}", settings.whisper_model)
    logger.info("  TTS voice   : {}", settings.kokoro_voice)
    logger.info("  OCR lang    : {}", settings.paddle_lang)
    logger.info("  API key set : {}", bool(settings.effective_api_key))
    logger.info("====================================")

    # Eagerly initialize models on startup
    logger.info("Initializing models on startup...")
    try:
        from services import stt_service, tts_service, ocr_service, vad_service

        # Load Whisper (STT), Kokoro (TTS), and Silero (VAD) concurrently
        logger.info("Loading STT, TTS, and VAD models...")
        await asyncio.gather(
            stt_service.init_model(),
            tts_service.init_model(),
            asyncio.to_thread(vad_service.init_model)
        )

        # Load PaddleOCR (OCR) in the thread pool to avoid blocking the main thread
        logger.info("Loading OCR models...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ocr_service.init_model)

        logger.info("All models successfully warmed up and ready!")
        logger.info("====================================")
    except Exception as e:
        logger.critical("Model initialization failed on startup: {}", e)

    yield  # app is running

    logger.info("Eyeva AI V1 - Shutting down")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Eyeva AI V1",
    description="AI-powered visual assistant API for visually impaired users.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(vision.router)
app.include_router(ocr.router)
app.include_router(speech.router)

# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on {}: {}", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Root ──────────────────────────────────────────────────────────────────────

# ── Static Frontend Files (Catch-all) ─────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    # If static directory doesn't exist, fallback to base API metadata or JSON 404
    if not os.path.isdir(STATIC_DIR):
        if not path:
            return {"name": "Eyeva AI V1", "docs": "/docs", "health": "/api/health"}
        return JSONResponse(status_code=404, content={"detail": f"Path '{path}' not found."})

    # 1. Empty path -> index.html
    if not path:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # 2. Check exact file
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    # 3. Clean URLs (e.g., /camera -> static/camera.html)
    html_file = file_path + ".html"
    if os.path.isfile(html_file):
        return FileResponse(html_file)

    # 4. Fallback to index.html for client-side routing
    fallback = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(fallback):
        return FileResponse(fallback)

    return JSONResponse(status_code=404, content={"detail": "Not Found"})

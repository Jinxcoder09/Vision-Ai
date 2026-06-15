"""
Eyeva AI – OCR Routes

POST /api/ocr/read – extract text from a base64-encoded image
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from models.schemas import OCRRequest, OCRResponse
from services import ocr_service

router = APIRouter(prefix="/api/ocr", tags=["ocr"])


@router.post("/read", response_model=OCRResponse, summary="Extract text from image")
async def read_text(req: OCRRequest) -> OCRResponse:
    """
    Run PaddleOCR on the provided image and return cleaned text.

    Supports English ('en') and Hindi ('hi').
    Low-confidence detections are automatically filtered out.
    """
    try:
        text, word_count, latency = await ocr_service.extract_text(req.image_b64, req.lang)
    except RuntimeError as e:
        logger.error("OCR route error: {}", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text:
        return OCRResponse(text="No text found in the image.", word_count=0, lang=req.lang, latency_ms=latency)

    return OCRResponse(text=text, word_count=word_count, lang=req.lang, latency_ms=latency)

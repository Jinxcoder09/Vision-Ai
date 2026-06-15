"""
Eyeva AI – OCR Service
Wraps NVIDIA LLama 3.1 Nemotron Nano VL for precise text extraction.
"""
import time
from typing import Optional
from loguru import logger

from core.config import get_settings

settings = get_settings()
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.effective_api_key or "no-key",
        )
    return _client


def init_model():
    """No-op for VLM-based OCR model."""
    pass


async def extract_text(
    image_b64: str,
    lang: str = "en",
) -> tuple[str, int, float]:
    """
    Extract all text from a base64-encoded image using the NVIDIA LLama 3.1 VLM model.
    """
    start = time.monotonic()
    
    clean_b64 = image_b64
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",", 1)[1]
    clean_b64 = clean_b64.strip()
    
    client = _get_client()
    
    prompt = (
        "Transcribe all visible text in the image. "
        "Do not summarize, do not describe the scene, do not explain. "
        "Return ONLY the transcribed text. If no text is found, return nothing."
    )
    
    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{clean_b64}",
                "detail": "auto",
            },
        },
        {"type": "text", "text": prompt},
    ]
    
    import asyncio
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                messages=[
                    {"role": "system", "content": "You are a precise OCR tool. Transcribe text exactly as it appears. Output only the extracted text, or nothing if no text exists."},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=settings.vlm_max_tokens,
                temperature=0.1,
                stream=False,
            ),
            timeout=12.0,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("LLama VLM OCR failed: {}", e)
        text = ""
        
    latency_ms = (time.monotonic() - start) * 1000
    word_count = len(text.split()) if text else 0
    
    logger.info(
        "LLama OCR complete — words={} latency={:.0f}ms",
        word_count, latency_ms,
    )
    return text, word_count, latency_ms

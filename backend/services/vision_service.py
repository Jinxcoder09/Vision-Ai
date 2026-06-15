"""
Eyeva AI – Vision Service
Wraps NVIDIA NIM Qwen2.5-VL via the OpenAI-compatible SDK.

Responsibilities:
  - Frame throttling (max N calls/sec)
  - Response caching (LRU + TTL)
  - Concise prompt engineering for visually impaired users
"""
import asyncio
import base64
import time
from typing import Optional, AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from core.cache import response_cache, make_cache_key
from core.config import get_settings

settings = get_settings()

# ── OpenAI-compatible client pointed at NVIDIA NIM ───────────────────────────
_client: Optional[AsyncOpenAI] = None
_last_call_time: float = 0.0
_throttle_lock = asyncio.Lock()
def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = settings.groq_api_key
        if not api_key:
            logger.warning("No API key configured for Groq vision service — calls will fail")
        _client = AsyncOpenAI(
            base_url=settings.groq_base_url,
            api_key=api_key or "no-key",
        )
        logger.info(
            "Vision service initialized — model={} base_url={}",
            settings.vlm_model,
            settings.groq_base_url,
        )
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are Eyeva, an AI assistant for visually impaired users. "
    "Describe what you see in the image clearly and concisely — 1-2 short sentences maximum. "
    "Focus on: people, objects, text, hazards, spatial relationships. "
    "Use directions like left, right, ahead, behind. "
    "Do not use markdown. Do not say 'I see' or 'The image shows'. Just describe directly."
)


async def _enforce_throttle() -> None:
    """Enforce max frame rate by sleeping if necessary."""
    global _last_call_time
    async with _throttle_lock:
        min_interval = 1.0 / settings.max_frame_rate
        elapsed = time.monotonic() - _last_call_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        _last_call_time = time.monotonic()


async def analyze_frame(
    image_b64: str,
    prompt: str = "Describe the scene for a visually impaired person.",
    use_cache: bool = True,
    system_prompt: str = _SYSTEM_PROMPT,
) -> tuple[str, bool, float]:
    """
    Analyze an image frame and return a concise description.

    Returns:
        (description, was_cached, latency_ms)

    Raises:
        RuntimeError on API failure after logging.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    if use_cache:
        cache_key = make_cache_key(image_b64[:64], prompt)
        cached = await response_cache.get(cache_key)
        if cached is not None:
            return cached, True, 0.0

    # ── Throttle ──────────────────────────────────────────────────────────────
    await _enforce_throttle()

    start = time.monotonic()
    client = _get_client()

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            },
        },
        {"type": "text", "text": prompt},
    ]

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=settings.vlm_max_tokens,
                temperature=0.3,
                stream=False,
            ),
            timeout=10.0,
        )
        description = (response.choices[0].message.content or "").strip()
    except (APITimeoutError, asyncio.TimeoutError):
        logger.error("Vision service timeout")
        raise RuntimeError("Vision API timed out. Please try again.")
    except RateLimitError:
        logger.error("Vision service rate limited")
        raise RuntimeError("Too many requests. Please wait a moment.")
    except APIError as e:
        logger.error("Vision API error: {}", e)
        raise RuntimeError(f"Vision API error: {e.message}")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info("Vision response in {:.0f}ms — {}", latency_ms, description[:60])

    # ── Store in cache ────────────────────────────────────────────────────────
    if use_cache:
        await response_cache.set(cache_key, description)

    return description, False, latency_ms


async def answer_question(
    image_b64: str,
    question: str,
) -> tuple[str, float]:
    """
    Answer a specific question about the image content.
    Not cached (questions can vary on same frame).
    """
    system_prompt = (
        "You are Eyeva, a helpful AI assistant for visually impaired users. "
        "Answer the user's question about the image directly and conversationally. "
        "Be extremely concise — limit your response to 1 short sentence."
    )
    description, _, latency = await analyze_frame(
        image_b64, question, use_cache=False, system_prompt=system_prompt
    )
    return description, latency


async def answer_text_query(
    question: str,
) -> tuple[str, float]:
    """
    Answer a general text query without an image (conversational fallback).
    """
    start = time.monotonic()
    client = _get_client()
    system_prompt = (
        "You are Eyeva, a helpful AI assistant for visually impaired users. "
        "Answer the user's question directly and conversationally. "
        "Be extremely concise — limit your response to 1 short sentence."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=settings.vlm_max_tokens,
                temperature=0.3,
                stream=False,
            ),
            timeout=10.0,
        )
        description = (response.choices[0].message.content or "").strip()
    except (APITimeoutError, asyncio.TimeoutError):
        logger.error("Vision service timeout during text query")
        raise RuntimeError("AI service timed out. Please try again.")
    except RateLimitError:
        logger.error("Vision service rate limited during text query")
        raise RuntimeError("Too many requests. Please wait a moment.")
    except APIError as e:
        logger.error("Vision API error during text query: {}", e)
        raise RuntimeError(f"AI service error: {e.message}")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info("Text-only query response in {:.0f}ms — {}", latency_ms, description[:60])
    return description, latency_ms


async def stream_answer_question(
    image_b64: str,
    question: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the answer to a specific question about the image content.
    """
    # Throttle
    await _enforce_throttle()

    client = _get_client()
    system_prompt = (
        "You are Eyeva, a helpful AI assistant for visually impaired users. "
        "Answer the user's question about the image directly and conversationally. "
        "Be extremely concise — limit your response to 1 short sentence."
    )

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}",
            },
        },
        {"type": "text", "text": question},
    ]

    try:
        response_stream = await client.chat.completions.create(
            model=settings.vlm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=settings.vlm_max_tokens,
            temperature=0.3,
            stream=True,
        )
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("Vision streaming API error: {}", e)
        raise RuntimeError(f"Vision streaming API error: {e}")


async def stream_answer_text_query(
    question: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the answer to a general text query.
    """
    client = _get_client()
    system_prompt = (
        "You are Eyeva, a helpful AI assistant for visually impaired users. "
        "Answer the user's question directly and conversationally. "
        "Be extremely concise — limit your response to 1 short sentence."
    )
    try:
        response_stream = await client.chat.completions.create(
            model=settings.vlm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=settings.vlm_max_tokens,
            temperature=0.3,
            stream=True,
        )
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("Vision text streaming API error: {}", e)
        raise RuntimeError(f"Vision text streaming API error: {e}")

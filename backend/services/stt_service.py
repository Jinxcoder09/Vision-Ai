"""
Eyeva AI – Speech-to-Text Service
Converts speech to text using Groq's Hosted Whisper API (via OpenAI SDK).

Features:
  - Lazy client initialization
  - Accepts raw WAV bytes or numpy float32 arrays
  - Returns transcript + normalized language code
  - Runs requests in an executor to prevent blocking
"""
import asyncio
import io
import time
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from openai import OpenAI

from core.config import get_settings

settings = get_settings()

# ── Lazy-loaded Groq client ───────────────────────────────────────────────────
_groq_client = None
_client_lock = asyncio.Lock()


async def _get_client():
    """Lazily load the Groq OpenAI SDK client."""
    global _groq_client
    async with _client_lock:
        if _groq_client is None:
            logger.info(
                "Initializing Groq client for STT — model={} base_url={}",
                settings.whisper_model,
                settings.groq_base_url,
            )
            _groq_client = OpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
    return _groq_client


async def init_model():
    """Eagerly initialize the client."""
    await _get_client()


def normalize_language_code(lang: Optional[str]) -> str:
    """Normalize language name or code returned by Groq to a 2-character ISO code."""
    if not lang:
        return "en"
    lang_lower = lang.lower()
    mapping = {
        "english": "en",
        "hindi": "hi",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "japanese": "ja",
        "chinese": "zh",
        "russian": "ru",
        "portuguese": "pt",
    }
    return mapping.get(lang_lower, lang_lower[:2])


def _transcribe_bytes_groq(audio_bytes: bytes, client: OpenAI, language: Optional[str] = None) -> tuple[str, str, Optional[float]]:
    """
    Synchronous transcription using Groq API — called in executor.
    Returns (transcript, language_code, confidence).
    """
    # Detect audio format from magic bytes for proper file naming
    suffix = "speech.wav"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        suffix = "speech.webm"
    elif audio_bytes.startswith(b"RIFF"):
        suffix = "speech.wav"
    elif audio_bytes.startswith(b"ID3") or (len(audio_bytes) > 1 and audio_bytes[0] == 0xff and (audio_bytes[1] & 0xe0) == 0xe0):
        suffix = "speech.mp3"
    elif audio_bytes.startswith(b"OggS"):
        suffix = "speech.ogg"
    elif audio_bytes.startswith(b"fLaC"):
        suffix = "speech.flac"

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = suffix

    try:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=settings.whisper_model,
            language=language,
            response_format="verbose_json",
            prompt="Eva, Eyeva, Ava, Iva, Ifa, Ayeva, Even, Ever."
        )
        
        transcript = response.text.strip()
        lang_str = getattr(response, "language", "en")
        lang_code = normalize_language_code(lang_str)
        confidence = 1.0  # Default confidence representation
        
        return transcript, lang_code, confidence
    except Exception as e:
        logger.error("Groq Whisper API call failed: {}", e)
        raise RuntimeError(f"Groq transcription request failed: {e}")


async def transcribe_audio(
    audio_bytes: bytes,
    language: Optional[str] = None,
) -> tuple[str, str, Optional[float], float]:
    """
    Transcribe audio bytes to text.

    Args:
        audio_bytes: Raw audio bytes.
        language: Optional language override.

    Returns:
        (transcript, language, confidence, latency_ms)
    """
    if not audio_bytes:
        raise ValueError("Empty audio data")

    start = time.monotonic()
    client = await _get_client()
    loop = asyncio.get_event_loop()

    try:
        transcript, lang, confidence = await asyncio.wait_for(
            loop.run_in_executor(None, _transcribe_bytes_groq, audio_bytes, client, language),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("Groq Whisper transcription timed out")
        raise RuntimeError("Transcription timed out. Please try again.")
    except Exception as e:
        logger.error("Groq Whisper error: {}", e)
        raise RuntimeError(f"Transcription failed: {e}")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Transcribed via Groq in {:.0f}ms — lang={} text='{}'",
        latency_ms, lang, transcript[:60],
    )
    return transcript, lang, confidence, latency_ms


def _transcribe_numpy_groq(audio_numpy: np.ndarray, client: OpenAI, language: Optional[str] = None) -> tuple[str, str, Optional[float]]:
    """
    Synchronous transcription from a NumPy Float32 mono 16kHz array — called in executor.
    Converts numpy array to WAV bytes first.
    """
    try:
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_numpy, 16000, format="WAV", subtype="PCM_16")
        wav_bytes = wav_buf.getvalue()
        
        return _transcribe_bytes_groq(wav_bytes, client, language)
    except Exception as e:
        logger.error("Error transcribing numpy array with Groq: {}", e)
        raise


async def transcribe_audio_numpy(
    audio_numpy: np.ndarray,
    language: Optional[str] = None,
) -> tuple[str, str, Optional[float], float]:
    """
    Transcribe raw NumPy Float32 mono 16kHz audio array to text.

    Returns:
        (transcript, language, confidence, latency_ms)
    """
    if audio_numpy is None or len(audio_numpy) == 0:
        raise ValueError("Empty audio numpy array")

    start = time.monotonic()
    client = await _get_client()
    loop = asyncio.get_event_loop()

    try:
        transcript, lang, confidence = await asyncio.wait_for(
            loop.run_in_executor(None, _transcribe_numpy_groq, audio_numpy, client, language),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("Groq Whisper transcription timed out")
        raise RuntimeError("Transcription timed out. Please try again.")
    except Exception as e:
        logger.error("Groq Whisper numpy error: {}", e)
        raise RuntimeError(f"Transcription failed: {e}")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Transcribed numpy via Groq in {:.0f}ms — lang={} text='{}'",
        latency_ms, lang, transcript[:60],
    )
    return transcript, lang, confidence, latency_ms

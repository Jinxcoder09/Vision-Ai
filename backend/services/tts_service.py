"""
Eyeva AI – Text-to-Speech Service using Groq Orpheus API with local Kokoro fallback.

Features:
  - Lazy client initialization
  - Fully asynchronous cloud-based speech synthesis
  - Automated local Kokoro TTS fallback when terms are not accepted on Groq Console or offline
  - Returns WAV bytes compatible with frontend AudioContext players
"""
import asyncio
import io
import time
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from core.config import get_settings

settings = get_settings()

# ── Lazy-loaded Groq client ───────────────────────────────────────────────────
_groq_client = None
_client_lock = asyncio.Lock()

# ── Lazy-loaded Local Kokoro ──────────────────────────────────────────────────
_pipelines: dict[str, object] = {}
_pipeline_lock = asyncio.Lock()


async def _get_client() -> AsyncOpenAI:
    global _groq_client
    async with _client_lock:
        if _groq_client is None:
            logger.info("Initializing Groq client for TTS with base_url={}", settings.groq_base_url)
            _groq_client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url
            )
    return _groq_client


async def _get_pipeline(lang_code: str = "a"):
    """Get or create a Kokoro pipeline for the given lang_code (lazy-loaded fallback)."""
    global _pipelines
    async with _pipeline_lock:
        if lang_code not in _pipelines:
            try:
                from kokoro import KPipeline  # type: ignore
            except ImportError:
                logger.error("Kokoro package not installed. Local fallback is unavailable.")
                raise RuntimeError("Kokoro package not installed. Local fallback is unavailable.")
            
            logger.info("Loading local Kokoro TTS pipeline fallback — lang_code={}", lang_code)
            loop = asyncio.get_event_loop()
            pipeline = await loop.run_in_executor(
                None,
                lambda: KPipeline(lang_code=lang_code),
            )
            _pipelines[lang_code] = pipeline
            logger.info("Local Kokoro TTS pipeline loaded — lang_code={}", lang_code)
    return _pipelines[lang_code]


def _synthesize_sync_local(text: str, pipeline, voice: str, speed: float) -> bytes:
    """Synchronous local TTS — called in executor."""
    import soundfile as sf  # type: ignore
    import numpy as np

    audio_chunks = []
    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"(?<=[.!?])\s+")
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)
    except Exception as e:
        raise RuntimeError(f"Local TTS synthesis failed: {e}")

    if not audio_chunks:
        raise RuntimeError("Local TTS produced no audio output")

    combined = np.concatenate(audio_chunks)
    
    # Normalize volume to prevent low volume issues
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = (combined / max_val) * 0.95

    buf = io.BytesIO()
    # Kokoro generates audio at 24000 Hz sample rate
    sf.write(buf, combined, 24000, format="WAV")
    buf.seek(0)
    return buf.read()


def normalize_wav_bytes(wav_bytes: bytes) -> bytes:
    """Read WAV bytes, normalize amplitude peak to 0.95, and return updated WAV bytes."""
    import soundfile as sf
    import numpy as np

    try:
        data, samplerate = sf.read(io.BytesIO(wav_bytes))
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 0.95
        
        buf = io.BytesIO()
        sf.write(buf, data, samplerate, format="WAV")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning("Failed to normalize WAV bytes: {}", e)
        return wav_bytes


async def init_model():
    """Verify client and eagerly pre-warm local Kokoro pipeline fallback in background."""
    await _get_client()
    import os
    if os.environ.get("RENDER"):
        logger.info("Running on Render - skipping eager local Kokoro pre-warming to conserve memory.")
        return

    try:
        import kokoro  # noqa
    except ImportError:
        logger.info("Kokoro package is not installed - skipping eager local Kokoro pre-warming.")
        return

    try:
        # Load local Kokoro pipeline asynchronously on startup to avoid loading latency on first fallback
        asyncio.create_task(_get_pipeline(settings.kokoro_lang_code))
    except Exception as e:
        logger.warning("Failed to eagerly pre-warm local Kokoro pipeline: {}", e)


def _map_voice(voice: Optional[str]) -> str:
    """Map voice configuration to valid Groq Orpheus voice persona."""
    if not voice:
        return "troy"
    voice_lower = voice.lower()
    
    # Map Kokoro voices to Orpheus voices
    if "bell" in voice_lower or "sky" in voice_lower or "heart" in voice_lower or "sarah" in voice_lower or "nicole" in voice_lower:
        return "hannah"
    elif "emma" in voice_lower or "isabella" in voice_lower:
        return "diana"
    elif "adam" in voice_lower or "michael" in voice_lower or "lewis" in voice_lower:
        return "troy"
    elif "george" in voice_lower:
        return "daniel"
    elif voice_lower in {"abdullah", "aisha", "fahad", "sultan", "lulwa", "noura", "autumn", "diana", "hannah", "austin", "daniel", "troy"}:
        return voice_lower
    return "troy"


async def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    lang_code: Optional[str] = None,
) -> tuple[bytes, float]:
    """
    Convert text to speech and return WAV bytes.
    First attempts high-speed Groq cloud synthesis, falls back to local Kokoro if terms not accepted or API offline.

    Returns:
        (wav_bytes, latency_ms)
    """
    if not text.strip():
        raise ValueError("Empty text for synthesis")

    start = time.monotonic()
    groq_voice = _map_voice(voice or settings.kokoro_voice)
    effective_lang = lang_code or settings.kokoro_lang_code
    
    # Determine the model ID based on the language
    if "ar" in effective_lang.lower():
        model = "canopylabs/orpheus-arabic-saudi"
        if groq_voice not in {"abdullah", "aisha", "fahad", "sultan", "lulwa", "noura"}:
            groq_voice = "abdullah"
    else:
        model = "canopylabs/orpheus-v1-english"

    # 1. Attempt Groq Cloud TTS
    try:
        client = await _get_client()
        response = await client.audio.speech.create(
            model=model,
            voice=groq_voice,
            input=text,
            response_format="wav"
        )
        wav_bytes = response.content
        # Normalize the WAV bytes to prevent low/uneven volume issues
        wav_bytes = normalize_wav_bytes(wav_bytes)
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Synthesized TTS via Groq in {:.0f}ms — model={} voice='{}' text='{}'",
            latency_ms, model, groq_voice, text[:40]
        )
        return wav_bytes, latency_ms
    except Exception as e:
        # Standard error messaging for terms acceptance:
        if "terms acceptance" in str(e) or "model_terms_required" in str(e):
            logger.warning(
                "Groq Orpheus model requires terms acceptance. "
                "Accept terms at https://console.groq.com/playground?model={} "
                "to enable ultra-low latency cloud TTS. Falling back to local Kokoro...",
                model
            )
        else:
            logger.warning("Groq Orpheus TTS failed ({}). Falling back to local Kokoro...", e)

        # 2. Local Fallback
        try:
            pipeline = await _get_pipeline(effective_lang)
            
            # Map Orpheus voice back to Kokoro voice
            kokoro_voice = "af_heart"
            voice_lower = groq_voice.lower()
            if voice_lower in {"troy", "daniel", "austin"}:
                kokoro_voice = "am_michael"
            elif voice_lower in {"hannah", "diana", "autumn"}:
                kokoro_voice = "af_heart"
            elif voice_lower in {"abdullah", "fahad", "sultan"}:
                kokoro_voice = "am_adam"
            elif voice_lower in {"aisha", "lulwa", "noura"}:
                kokoro_voice = "af_sky"
            else:
                kokoro_voice = voice or settings.kokoro_voice

            loop = asyncio.get_event_loop()
            wav_bytes = await loop.run_in_executor(
                None,
                _synthesize_sync_local,
                text,
                pipeline,
                kokoro_voice,
                speed or 1.0
            )
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("Local fallback TTS synthesized in {:.0f}ms", latency_ms)
            return wav_bytes, latency_ms
        except Exception as local_err:
            logger.warning("Local Kokoro fallback failed or unavailable: {}. Trying gTTS...", local_err)
            
            # 3. Cloud Free Fallback: gTTS (Google Text-to-Speech)
            try:
                from gtts import gTTS  # type: ignore
                logger.info("Falling back to gTTS (Google Text-to-Speech)...")
                # Normalize lang code to 2 letters (e.g., 'a' -> 'en', 'b' -> 'en')
                gtts_lang = effective_lang.lower()
                if gtts_lang in {"a", "b"}:
                    gtts_lang = "en"
                elif len(gtts_lang) > 2:
                    gtts_lang = gtts_lang[:2]

                # Map voice accents if possible
                tld = "com"
                if "b" in effective_lang.lower() or "uk" in groq_voice.lower():
                    tld = "co.uk"

                tts = gTTS(text=text, lang=gtts_lang, tld=tld)
                buf = io.BytesIO()
                # Run the blocking gTTS API call in the thread executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: tts.write_to_fp(buf))
                buf.seek(0)
                audio_bytes = buf.read()
                latency_ms = (time.monotonic() - start) * 1000
                logger.info("gTTS fallback synthesized in {:.0f}ms", latency_ms)
                return audio_bytes, latency_ms
            except Exception as gtts_err:
                logger.critical("All TTS options failed. Groq: {}, Kokoro: {}, gTTS: {}", e, local_err, gtts_err)
                raise RuntimeError(f"Speech synthesis failed: {e}")


async def stream_synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    lang_code: Optional[str] = None,
):
    """
    Convert text to speech and yield WAV bytes.
    Yields the entire WAV file as a single complete chunk to ensure the browser 
    can decode it successfully via AudioContext.decodeAudioData().
    """
    if not text.strip():
        raise ValueError("Empty text for synthesis")

    try:
        wav_bytes, _ = await synthesize_speech(text, voice, speed, lang_code)
        yield wav_bytes
    except Exception as e:
        logger.error("TTS stream synthesis failed: {}", e)
        raise RuntimeError(f"TTS stream synthesis failed: {e}")

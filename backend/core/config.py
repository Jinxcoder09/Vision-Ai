"""
Eyeva AI – Application Configuration
Reads all settings from environment variables / .env file.
Never hardcode credentials here.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from project root (parent of /backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── NVIDIA NIM / Vision ────────────────────────────────────────────────
    nvidia_api_key: str = ""
    qwen_api_key: str = ""          # alias – falls back to nvidia_api_key if empty
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    vlm_model: str = "qwen/qwen2.5-vl-7b-instruct"
    vlm_max_tokens: int = 200       # keep responses concise

    # ── Groq Whisper (STT) & Orpheus (TTS) ──────────────────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    tts_model: str = "canopylabs/orpheus-v1-english"

    # ── Faster Whisper (Legacy/Local) ──────────────────────────────────────
    whisper_model: str = "whisper-large-v3"  # tiny.en | base.en | small.en | medium.en | whisper-large-v3
    whisper_device: Literal["cpu", "cuda", "auto"] = "cpu"
    whisper_compute_type: str = "int8"

    # ── Kokoro TTS ─────────────────────────────────────────────────────────
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    kokoro_lang_code: str = "a"     # 'a'=American English, 'b'=British English

    # ── PaddleOCR ─────────────────────────────────────────────────────────
    paddle_lang: str = "en"         # en | hi
    paddle_use_gpu: bool = False

    # ── Performance ────────────────────────────────────────────────────────
    max_frame_rate: float = 2.0     # max VLM calls per second
    response_cache_ttl: int = 30    # seconds
    response_cache_size: int = 128  # LRU entries

    # ── App ────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://frontend:3000"]
    log_level: str = "INFO"

    @property
    def effective_api_key(self) -> str:
        """Return QWEN_API_KEY if set, otherwise fall back to NVIDIA_API_KEY."""
        return self.qwen_api_key or self.nvidia_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

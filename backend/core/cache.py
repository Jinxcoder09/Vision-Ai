"""
Eyeva AI – In-memory LRU response cache with TTL support.

Used to avoid sending identical frames / prompts to the VLM repeatedly.
Thread-safe via asyncio lock.
"""
import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional

from loguru import logger


class TTLCache:
    """
    Async-safe LRU cache with per-entry TTL.
    Entries expire after `ttl` seconds and are evicted LRU when `maxsize` is reached.
    """

    def __init__(self, maxsize: int = 128, ttl: int = 30) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            value, expiry = self._cache[key]
            if time.monotonic() > expiry:
                del self._cache[key]
                logger.debug("Cache miss (expired): {}", key[:16])
                return None
            # Move to end (most-recently used)
            self._cache.move_to_end(key)
            logger.debug("Cache hit: {}", key[:16])
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic() + self._ttl)
            if len(self._cache) > self._maxsize:
                evicted = self._cache.popitem(last=False)
                logger.debug("Cache evicted LRU entry: {}", str(evicted[0])[:16])

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def make_cache_key(*parts: str) -> str:
    """Create a deterministic cache key from one or more string parts."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


# Module-level singleton — shared across all requests
response_cache: TTLCache = TTLCache()

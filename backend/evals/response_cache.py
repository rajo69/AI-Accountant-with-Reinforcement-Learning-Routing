"""
Disk-based response cache for LLM calls during evals.

Caches API responses to JSON files keyed by SHA-256(model + prompt).
Re-running evals against unchanged fixtures costs $0.

Usage:
    cache = ResponseCache()

    cached = cache.get(model, prompt)
    if cached:
        return cached  # free — no API call

    response = client.messages.create(...)
    cache.set(model, prompt, response_dict)
    return response_dict

Control via environment variables:
    EVAL_CACHE=1   (default) — use cache
    EVAL_CACHE=0   — bypass cache (force fresh API calls)
    EVAL_CACHE_DIR — path to cache directory (default: evals/.cache/)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

_DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


class ResponseCache:
    def __init__(self, cache_dir: Optional[Path] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.getenv("EVAL_CACHE", "1") != "0"
        self.enabled = enabled

        if cache_dir is None:
            env_dir = os.getenv("EVAL_CACHE_DIR")
            cache_dir = Path(env_dir) if env_dir else _DEFAULT_CACHE_DIR

        self.cache_dir = cache_dir
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0

    def _key(self, model: str, prompt: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, model: str, prompt: str) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self._path(self._key(model, prompt))
        if path.exists():
            self._hits += 1
            return json.loads(path.read_text())
        self._misses += 1
        return None

    def set(self, model: str, prompt: str, response: dict) -> None:
        if not self.enabled:
            return
        path = self._path(self._key(model, prompt))
        path.write_text(json.dumps(response, indent=2))

    def clear(self) -> int:
        """Delete all cached responses. Returns count of deleted files."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count

    def stats(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "Cache: no lookups yet"
        hit_rate = self._hits / total * 100
        size = sum(1 for _ in self.cache_dir.glob("*.json")) if self.cache_dir.exists() else 0
        return (
            f"Cache: {self._hits}/{total} hits ({hit_rate:.0f}%), "
            f"{size} entries on disk, dir={self.cache_dir}"
        )

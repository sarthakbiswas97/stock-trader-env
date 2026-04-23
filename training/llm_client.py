"""Provider-agnostic async LLM client for judge scoring.

Wraps OpenAI-compatible chat endpoints. Supports GPT-4o-mini (default),
DeepSeek, Gemini, Groq, Cerebras — any API that accepts the OpenAI
chat completions format.

Usage:
    from training.llm_client import LLMJudge, openai_4o_mini

    judge = LLMJudge(openai_4o_mini())
    score = await judge.score(messages)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")
CACHE_DB = CACHE_DIR / "llm_cache.db"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM endpoint."""

    base_url: str
    api_key: str
    model: str
    max_rpm: int = 500
    timeout: float = 30.0
    temperature: float = 0.1


def openai_4o_mini() -> LLMConfig:
    """GPT-4o-mini — best quality/cost for judge scoring."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY env var required")
    return LLMConfig(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-4o-mini",
        max_rpm=500,
    )


def deepseek_v3() -> LLMConfig:
    """DeepSeek V3 — cheaper alternative."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY env var required")
    return LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        model="deepseek-chat",
        max_rpm=300,
    )


def _cache_key(messages: list[dict], model: str) -> str:
    """Deterministic hash for caching."""
    content = json.dumps(messages, sort_keys=True) + model
    return hashlib.sha256(content.encode()).hexdigest()


class ScoreCache:
    """SQLite-backed cache for LLM judge scores."""

    def __init__(self, db_path: Path = CACHE_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS scores "
            "(key TEXT PRIMARY KEY, response TEXT)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM scores WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores (key, response) VALUES (?, ?)",
            (key, response),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class LLMJudge:
    """Async LLM judge that scores trading decisions."""

    def __init__(
        self,
        config: LLMConfig,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._config = config
        self._cache = ScoreCache() if use_cache else None
        self._max_retries = max_retries
        self._semaphore: asyncio.Semaphore | None = None

    async def score(self, messages: list[dict]) -> str:
        """Send messages to LLM, return raw response text.

        Uses cache if available. Retries with exponential backoff.
        """
        key = _cache_key(messages, self._config.model)

        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(10)

        async with self._semaphore:
            response_text = await self._call_api(messages)

        if self._cache and response_text:
            self._cache.put(key, response_text)

        return response_text

    async def _call_api(self, messages: list[dict]) -> str:
        """Make API call with retry."""
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                    resp = await client.post(
                        f"{self._config.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self._config.model,
                            "messages": messages,
                            "temperature": self._config.temperature,
                            "max_tokens": 100,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as e:
                wait = 2 ** attempt
                logger.warning(
                    "LLM API attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, self._max_retries, str(e)[:100], wait,
                )
                await asyncio.sleep(wait)

        logger.error("LLM API failed after %d retries", self._max_retries)
        return ""

    async def score_batch(
        self,
        message_lists: list[list[dict]],
        concurrency: int = 10,
    ) -> list[str]:
        """Score a batch of message lists concurrently."""
        self._semaphore = asyncio.Semaphore(concurrency)
        tasks = [self.score(msgs) for msgs in message_lists]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        if self._cache:
            self._cache.close()

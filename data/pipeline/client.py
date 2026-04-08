#!/usr/bin/env python3
"""
Generic OpenAI-compatible client pool for the Memento data pipeline.

Supports any provider with an OpenAI-compatible API:
  - OpenAI (default)
  - Azure OpenAI
  - Together AI, Fireworks, Groq, DeepInfra, Mistral, OpenRouter
  - Local servers: vLLM, Ollama, llama.cpp, SGLang, TGI

Configuration via environment variables:
  OPENAI_API_KEY   — API key (required for cloud providers)
  OPENAI_BASE_URL  — Base URL (optional; defaults to OpenAI)

Or via CLI arguments passed to the pipeline scripts:
  --model, --api-key, --base-url
"""

import os
import threading
from typing import List, Optional, Tuple

from openai import OpenAI


# Default model if none specified
DEFAULT_MODEL = "gpt-4o"


class LLMClient:
    """Wrapper that pairs an OpenAI client with its model name."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def create_chat_completion(self, **kwargs):
        """Convenience: call chat.completions.create with the stored model."""
        kwargs.setdefault("model", self.model)
        return self.client.chat.completions.create(**kwargs)


class ClientPool:
    """Thread-safe round-robin client pool for load balancing across endpoints."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoints: Optional[List[dict]] = None,
    ):
        """
        Initialize client pool.

        Args:
            model: Model name to use (e.g., "gpt-4o", "Qwen/Qwen3-32B").
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            base_url: Base URL. Falls back to OPENAI_BASE_URL env var.
            endpoints: Optional list of {"base_url": ..., "api_key": ..., "model": ...}
                       for multi-endpoint round-robin. Overrides model/api_key/base_url.
        """
        self.clients: List[OpenAI] = []
        self.models: List[str] = []

        if endpoints:
            for ep in endpoints:
                ep_key = ep.get("api_key", api_key or os.environ.get("OPENAI_API_KEY"))
                if not ep_key:
                    raise ValueError(
                        "No API key provided. Set OPENAI_API_KEY or pass --api-key. "
                        "For local servers (vLLM, Ollama), pass --api-key no-key."
                    )
                client = OpenAI(
                    api_key=ep_key,
                    base_url=ep.get("base_url", base_url),
                )
                self.clients.append(client)
                self.models.append(ep.get("model", model))
        else:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "No API key provided. Set OPENAI_API_KEY or pass --api-key. "
                    "For local servers (vLLM, Ollama), pass --api-key no-key."
                )
            resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")
            kwargs = {"api_key": resolved_key}
            if resolved_url:
                kwargs["base_url"] = resolved_url
            client = OpenAI(**kwargs)
            self.clients.append(client)
            self.models.append(model)

        self.pool_size = len(self.clients)
        self.current_index = 0
        self.lock = threading.Lock()

        print(f"Initialized client pool with {self.pool_size} endpoint(s):")
        for i in range(self.pool_size):
            url = self.clients[i].base_url or "https://api.openai.com/v1"
            print(f"  - {url} + {self.models[i]}")

    def get_client(self) -> Tuple[OpenAI, str]:
        """Get next client and model using round-robin.

        Returns:
            Tuple of (OpenAI client, model name).
        """
        with self.lock:
            client = self.clients[self.current_index]
            model = self.models[self.current_index]
            self.current_index = (self.current_index + 1) % self.pool_size
            return client, model

    def get_llm_client(self) -> LLMClient:
        """Get next client wrapped with its model name."""
        client, model = self.get_client()
        return LLMClient(client, model)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_pool: Optional[ClientPool] = None
_pool_lock = threading.Lock()


def init_client_pool(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoints: Optional[List[dict]] = None,
) -> ClientPool:
    """Create (or replace) the global client pool singleton."""
    global _pool
    with _pool_lock:
        _pool = ClientPool(
            model=model,
            api_key=api_key,
            base_url=base_url,
            endpoints=endpoints,
        )
    return _pool


def get_client_pool() -> ClientPool:
    """Get or lazily create the global client pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ClientPool()
    return _pool


def get_next_client() -> Tuple[OpenAI, str]:
    """Get the next available (client, model) from the pool."""
    pool = get_client_pool()
    return pool.get_client()


def get_llm_client() -> LLMClient:
    """Get the next client wrapped with its model name."""
    pool = get_client_pool()
    return pool.get_llm_client()

import httpx
import json
from typing import AsyncGenerator, List, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


class NVIDIAClient:
    """Async client for NVIDIA NIM API — LLM + Embeddings."""

    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": settings.NVIDIA_EMBEDDING_MODEL,
                "input": texts,
                "input_type": "passage",
                "encoding_format": "float",
                "truncate": "END",
            }
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": settings.NVIDIA_EMBEDDING_MODEL,
                "input": [query],
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "END",
            }
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    # ── Chat Completion ───────────────────────────────────────────────────────

    async def chat_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        max_tokens: int = None,
        temperature: float = None,
    ) -> dict:
        """Non-streaming chat completion."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": settings.NVIDIA_LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens or settings.MAX_TOKENS,
                "temperature": temperature or settings.TEMPERATURE,
                "stream": False,
            }
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def chat_completion_stream(
        self,
        messages: List[dict],
        max_tokens: int = None,
        temperature: float = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion — yields token strings."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": settings.NVIDIA_LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens or settings.MAX_TOKENS,
                "temperature": temperature or settings.TEMPERATURE,
                "stream": True,
            }
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def check_connectivity(self) -> bool:
        """Ping the NVIDIA API to verify the key is valid."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"NVIDIA API connectivity check failed: {e}")
            return False


# Singleton
nvidia_client = NVIDIAClient()

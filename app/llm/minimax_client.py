from __future__ import annotations

import asyncio
import re
import time

import httpx

from app.config import settings


class MinimaxClient:
    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        self.last_usage: dict | None = None
        self.last_latency_ms: int | None = None

    def enabled(self) -> bool:
        return bool(self.api_key)

    async def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1200, json_mode: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError(f"{self.provider} API key is not configured")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if self.provider.lower() == "minimax":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = await self._post_chat_with_retry(self.base_url, payload)
        if should_retry_minimax_chat_endpoint(self.base_url, response):
            response = await self._post_chat_with_retry("https://api.minimax.chat/v1", payload)
            self.base_url = "https://api.minimax.chat/v1"
        self.last_latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(safe_error_message(response)) from exc
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.provider} returned no choices: {data}")
        usage = data.get("usage") or {}
        self.last_usage = {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": self.last_latency_ms,
        }
        return strip_thinking(choices[0].get("message", {}).get("content", "")).strip()

    async def _post_chat(self, base_url: str, payload: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

    async def _post_chat_with_retry(self, base_url: str, payload: dict) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._post_chat(base_url, payload)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.6)
                    continue
                raise
            if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                if attempt == 0:
                    await asyncio.sleep(0.6)
                    continue
            return response
        if last_exc:
            raise last_exc
        raise RuntimeError(f"{self.provider} retry exhausted")


def should_retry_minimax_chat_endpoint(base_url: str, response: httpx.Response) -> bool:
    if "api.minimax.io" not in base_url or response.status_code != 401:
        return False
    try:
        data = response.json()
    except ValueError:
        return False
    message = ((data.get("error") or {}).get("message") or "").lower()
    return "invalid api key" in message


def safe_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = response.text[:300]
    if isinstance(data, dict):
        error = data.get("error") or data
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or str(error)
            return f"LLM HTTP {response.status_code}: {message}"
    return f"LLM HTTP {response.status_code}: {data}"


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


async def chat_grounded_json(client, messages: list[dict]) -> str:
    try:
        return await client.chat(messages, max_tokens=settings.llm_grounded_max_tokens, json_mode=True)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword" not in message and "positional" not in message:
            raise
        return await client.chat(messages)

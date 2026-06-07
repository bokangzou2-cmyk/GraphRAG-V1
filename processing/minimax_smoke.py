from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm.minimax_client import safe_error_message  # noqa: E402


async def smoke(message: str) -> dict:
    if not settings.llm_api_key:
        return {"ok": False, "status_code": None, "error": "LLM_API_KEY is not configured"}
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    if settings.llm_provider.lower() == "minimax":
        payload["max_completion_tokens"] = 128
    else:
        payload["max_tokens"] = 128
    tried = []
    bases = [settings.llm_base_url]
    if settings.llm_provider.lower() == "minimax" and settings.llm_base_url != "https://api.minimax.chat/v1":
        bases.append("https://api.minimax.chat/v1")
    response = None
    for base_url in bases:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        tried.append({"base_url": base_url, "status_code": response.status_code})
        if not response.is_error:
            break
    assert response is not None
    if response.is_error:
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": safe_error_message(response),
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "tried": tried,
        }
    data = response.json()
    return {
        "ok": True,
        "status_code": response.status_code,
        "base_url": tried[-1]["base_url"],
        "tried": tried,
        "model": data.get("model"),
        "content_preview": ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe LLM API smoke test without printing the API key.")
    parser.add_argument("message", nargs="?", default="Hello")
    args = parser.parse_args()
    result = asyncio.run(smoke(args.message))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

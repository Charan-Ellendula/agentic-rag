from __future__ import annotations

from typing import List, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    pass


def _url(path: str) -> str:
    if not settings.gemini_api_key:
        raise GeminiError("GEMINI_API_KEY is missing")
    return f"{GEMINI_BASE_URL}/{path}?key={settings.gemini_api_key}"


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, GeminiError)),
)
async def embed_texts(texts: List[str]) -> List[List[float]]:
    payload: Dict[str, Any] = {
        "requests": [
            {
                "model": f"models/{settings.gemini_embed_model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            _url(f"models/{settings.gemini_embed_model}:batchEmbedContents"),
            json=payload,
        )
        if resp.status_code >= 400:
            raise GeminiError(f"Embeddings error {resp.status_code}: {resp.text}")
        data = resp.json()

    return [e["values"] for e in data["embeddings"]]


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=1, max=12),
    retry=retry_if_exception_type((httpx.HTTPError, GeminiError)),
)
async def chat(messages: List[Dict[str, str]]) -> str:
    system_text = None
    contents: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        text = m.get("content", "")
        if role == "system":
            system_text = text
        elif role in ("user", "assistant"):
            gem_role = "user" if role == "user" else "model"
            contents.append({"role": gem_role, "parts": [{"text": text}]})

    payload: Dict[str, Any] = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            _url(f"models/{settings.gemini_chat_model}:generateContent"),
            json=payload,
        )
        if resp.status_code >= 400:
            raise GeminiError(f"Chat error {resp.status_code}: {resp.text}")
        data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise GeminiError(f"Unexpected Gemini response format: {data}")


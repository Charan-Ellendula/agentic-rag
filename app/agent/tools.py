from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import ValidationError

from app.llm.client import chat
from app.rag.retrieve import retrieve
from app.rag.vectorstore import FaissStore
from app.schemas.models import ActionItemsResponse


async def tool_search_docs(store: FaissStore, query: str, top_k: int = 5) -> Dict[str, Any]:
    results = await retrieve(store, query, top_k=top_k)
    # Return lightweight evidence
    evidence = [
        {
            "score": score,
            "source": ch.source,
            "chunk_id": ch.chunk_id,
            "text": ch.text[:800],  # cap to reduce token usage
        }
        for score, ch in results
    ]
    return {"query": query, "top_k": top_k, "evidence": evidence}


async def tool_summarize(text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "Summarize clearly and concisely. Do not add facts."},
        {"role": "user", "content": text},
    ]
    summary = await chat(messages)
    return {"summary": summary}


async def tool_extract_action_items(text: str) -> Dict[str, Any]:
    """
    Extract action items as strict JSON: {"items":[{"task":"...", "owner":"...", "due_date":"..."}]}
    Validate with Pydantic.
    """
    messages = [
        {"role": "system", "content": "Return ONLY valid JSON. No markdown. No extra text."},
        {
            "role": "user",
            "content": (
                "Extract action items from the text.\n"
                "Return JSON exactly in this schema:\n"
                '{"items":[{"task":"...", "owner":"...", "due_date":"..."}]}\n\n'
                f"TEXT:\n{text}"
            ),
        },
    ]
    raw = await chat(messages)

    # Defensive: sometimes models wrap JSON in text—try to isolate JSON
    raw_str = raw.strip()
    start = raw_str.find("{")
    end = raw_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_str = raw_str[start : end + 1]

    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode failed: {e}", "raw": raw}

    try:
        validated = ActionItemsResponse(**data)
    except ValidationError as e:
        return {"error": f"Schema validation failed: {e}", "raw": data}

    return validated.model_dump()


from __future__ import annotations
from typing import List, Tuple, Dict, Any
from app.llm.client import chat
from app.rag.vectorstore import StoredChunk

def _build_context(results: List[Tuple[float, StoredChunk]]) -> str:
    parts = []
    for score, ch in results:
        parts.append(f"[source={ch.source} chunk={ch.chunk_id} score={score:.3f}]\n{ch.text}")
    return "\n\n---\n\n".join(parts)

async def answer_with_rag(question: str, results: List[Tuple[float, StoredChunk]]) -> Dict[str, Any]:
    context = _build_context(results)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If missing, say you don't know."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nReturn a short answer plus bullet citations like (source, chunk_id)."},
    ]
    text = await chat(messages)

    citations = [{"source": ch.source, "chunk_id": ch.chunk_id, "score": score} for score, ch in results]
    return {"answer": text, "citations": citations}


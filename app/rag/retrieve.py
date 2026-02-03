from __future__ import annotations
from typing import List, Tuple
import numpy as np

from app.llm.client import embed_texts
from app.rag.vectorstore import FaissStore, StoredChunk

async def retrieve(store: FaissStore, query: str, top_k: int = 5) -> List[Tuple[float, StoredChunk]]:
    emb = await embed_texts([query])
    q = np.array(emb, dtype=np.float32)
    return store.search(q, top_k=top_k)


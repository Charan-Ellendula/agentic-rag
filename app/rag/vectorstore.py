from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import faiss

@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str

class FaissStore:
    """
    Uses IndexFlatIP (inner product). If we L2-normalize embeddings,
    inner product ~= cosine similarity.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: Dict[int, StoredChunk] = {}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, vectors: np.ndarray, chunks: List[StoredChunk]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n,{self.dim}), got {vectors.shape}")
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks length must match vectors rows")

        vectors = vectors.astype(np.float32)
        vectors = self._normalize(vectors)

        start_id = len(self.meta)
        self.index.add(vectors)

        for i, ch in enumerate(chunks):
            self.meta[start_id + i] = ch

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, StoredChunk]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = query_vec.astype(np.float32)
        query_vec = self._normalize(query_vec)

        scores, ids = self.index.search(query_vec, top_k)

        results: List[Tuple[float, StoredChunk]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[int(idx)]))
        return results

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): self.meta[k].__dict__ for k in self.meta}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str) -> "FaissStore":
        index_path = os.path.join(folder, "index.faiss")
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Missing index.faiss or meta.json in data/index")

        index = faiss.read_index(index_path)
        dim = index.d
        store = cls(dim=dim)
        store.index = index

        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        store.meta = {int(k): StoredChunk(**v) for k, v in raw.items()}
        return store


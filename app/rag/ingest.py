from __future__ import annotations

import os
from typing import List

import numpy as np
from pypdf import PdfReader

from app.utils.text import clean_text
from app.rag.chunk import chunk_text
from app.rag.vectorstore import FaissStore, StoredChunk
from app.llm.client import embed_texts


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    """
    Robust PDF text extraction.
    - strict=False avoids many malformed-PDF crashes
    - page-by-page extraction prevents one page from killing the whole file
    """
    reader = PdfReader(path, strict=False)
    parts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
        except Exception:
            # Skip problematic pages
            continue
    return "\n".join(parts)


async def ingest_folder(raw_folder: str = "data/raw", index_folder: str = "data/index") -> FaissStore:
    files = [
        os.path.join(raw_folder, f)
        for f in os.listdir(raw_folder)
        if f.lower().endswith((".txt", ".md", ".pdf"))
    ]

    if not files:
        raise RuntimeError("No .txt/.md/.pdf files found in data/raw")

    all_chunks: List[StoredChunk] = []
    all_vectors: List[List[float]] = []

    for file_path in files:
        source = os.path.basename(file_path)
        doc_id = os.path.splitext(source)[0]

        try:
            if file_path.lower().endswith(".pdf"):
                text = _read_pdf_file(file_path)
            else:
                text = _read_text_file(file_path)
        except Exception as e:
            # IMPORTANT: don't crash ingestion because one PDF is bad
            print(f"[ingest] Skipping {source} due to read error: {e}")
            continue

        text = clean_text(text)

        if not text.strip():
            print(f"[ingest] Skipping {source}: extracted empty text")
            continue

        chunks = chunk_text(text, doc_id=doc_id)

        for ch in chunks:
            all_chunks.append(
                StoredChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=doc_id,
                    text=ch.text,
                    source=source,
                )
            )

    if not all_chunks:
        raise RuntimeError("No chunks were created. PDFs may be scanned/locked or extraction failed.")

    # Embed in batches
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embs = await embed_texts([c.text for c in batch])
        all_vectors.extend(embs)

    vectors = np.array(all_vectors, dtype=np.float32)
    dim = vectors.shape[1]

    store = FaissStore(dim=dim)
    store.add(vectors=vectors, chunks=all_chunks)
    store.save(index_folder)

    return store


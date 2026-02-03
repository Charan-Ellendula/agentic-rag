from dataclasses import dataclass
from typing import List
import tiktoken

@dataclass
class Chunk:
    chunk_id: str
    text: str

def chunk_text(text: str, doc_id: str, max_tokens: int = 350, overlap_tokens: int = 60, model: str = "gpt-4o-mini") -> List[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: List[Chunk] = []
    start = 0
    i = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)

        chunks.append(Chunk(chunk_id=f"{doc_id}_c{i}", text=chunk_str))
        i += 1

        if end == len(tokens):
            break

        start = max(0, end - overlap_tokens)

    return chunks


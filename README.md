# Agentic RAG System

A Retrieval-Augmented Generation (RAG) system that lets you query your own
PDFs and text files using Gemini embeddings, FAISS vector search, and FastAPI.

This system grounds LLM responses in real documents and returns answers
with citations.

---

## Architecture

User Question
      ↓
Embed Query (Gemini)
      ↓
FAISS Vector Search (Top-K Chunks)
      ↓
Context Injection
      ↓
Gemini LLM
      ↓
Grounded Answer + Citations

---

## Features
- PDF + text ingestion
- Text cleaning & chunking
- Gemini embeddings
- FAISS vector store
- Grounded answers with citations
- FastAPI backend
- Multi-document knowledge base

---

## Project Phases
1. Environment & API setup
2. Document ingestion (PDF/text)
3. Cleaning & token safety
4. Chunking strategy
5. Embedding generation (batched)
6. FAISS vector indexing
7. Retrieval + generation (RAG)
8. Multi-document scaling
9. Web UI (basic)
10. GitHub + secret protection

---

## Run Locally

```bash
git clone https://github.com/Charan-Ellendula/agentic-rag.git
cd agentic-rag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your GEMINI_API_KEY in .env

python -m uvicorn app.main:app --reload

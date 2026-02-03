

# Agentic RAG System (PDF Question Answering)

A production-style **Retrieval-Augmented Generation (RAG)** system that allows users to upload and query multiple PDF/text documents using:

* **Gemini Embeddings**
* **FAISS vector database**
* **FastAPI backend**
* **Simple Web UI**

This project demonstrates the full lifecycle of an AI knowledge system: ingestion → chunking → embedding → retrieval → generation.

---

## Tech Stack

| Layer      | Technology                  |
| ---------- | --------------------------- |
| Backend    | FastAPI                     |
| LLM        | Gemini                      |
| Embeddings | Gemini `text-embedding-004` |
| Vector DB  | FAISS                       |
| Parsing    | PyPDF                       |
| Language   | Python 3.9+                 |
| UI         | HTML + JS                   |
| Deployment | Uvicorn                     |

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

## API Endpoints

### Health Check

```
GET /health
```

### Ingest Documents

Reads all files from `data/raw/`, chunks them, embeds them, and builds the FAISS index.

```
POST /ingest
```

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### Ask a Question

Retrieves the most relevant chunks and generates an answer.

```
POST /query
```

**Request:**

```json
{
  "question": "What is AI?",
  "top_k": 8
}
```

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this book about?", "top_k": 8}'
```

---

## Run Locally

```bash
git clone https://github.com/Charan-Ellendula/agentic-rag.git
cd agentic-rag

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your GEMINI_API_KEY inside .env

python -m uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

## Data Ingestion

Place any PDF or text files inside:

```
data/raw/
```

Then run:

```bash
curl -X POST "http://127.0.0.1:8000/ingest"
```

---

## Web UI

A minimal UI is available at:

```
http://127.0.0.1:8000
```

You can:

* Ask questions
* Re-ingest documents
* View answers with citations

---

## Security

* `.env` files are ignored
* Secrets are never committed
* GitHub push protection enabled

---

## Example Use Cases

* AI textbook QA
* Legal document search
* Company policy assistant
* Research paper search
* Knowledge base chatbot

---

## Author

**Sai Charan Ellendula**
Senior AI/ML Engineer (GenAI, RAG, LLM Systems)


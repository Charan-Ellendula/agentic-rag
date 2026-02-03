from app.agent.runner import run_agent
from fastapi import APIRouter, HTTPException
from app.schemas.models import IngestResponse, QueryRequest, QueryResponse
from app.rag.vectorstore import FaissStore
from app.rag.ingest import ingest_folder
from app.rag.retrieve import retrieve
from app.rag.answer import answer_with_rag

router = APIRouter()

from typing import Optional

STORE: Optional[FaissStore] = None


@router.get("/health")
def health():
    return {"ok": True}

@router.post("/ingest", response_model=IngestResponse)
async def ingest():
    global STORE
    try:
        STORE = await ingest_folder()
        return IngestResponse(status="ok", message="Ingested data/raw and built index in data/index")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    global STORE
    try:
        if STORE is None:
            # Try load existing index
            STORE = FaissStore.load("data/index")
        results = await retrieve(STORE, req.question, top_k=req.top_k)
        out = await answer_with_rag(req.question, results)
        return QueryResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/agent")
async def agent(req: QueryRequest):
    global STORE
    try:
        if STORE is None:
            STORE = FaissStore.load("data/index")
        out = await run_agent(STORE, req.question, max_steps=3)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


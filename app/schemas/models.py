from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class IngestResponse(BaseModel):
    status: str
    message: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = 5

class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]




class ActionItem(BaseModel):
    task: str
    owner: str = "unassigned"
    due_date: str = "unspecified"

class ActionItemsResponse(BaseModel):
    items: List[ActionItem]


#!/bin/bash
cd /Users/saicharanellendula/Projects/agentic-rag
source .venv/bin/activate
python -m uvicorn app.main:app --reload


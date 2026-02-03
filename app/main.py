from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from app.api.routes import router
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="Agentic RAG (Starter)")
app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def home():
    return FileResponse("app/static/index.html")



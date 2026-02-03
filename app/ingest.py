from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path

RAW_DIR = Path("data/raw")
docs = []

for file in RAW_DIR.iterdir():
    if file.suffix.lower() in [".md", ".txt"]:
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())
    elif file.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())


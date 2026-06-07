from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from app.api.routes import router
from app.rag.pipeline import RAGPipeline
from app.core.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ClassMind...")
    if not settings.faiss_index_path.exists() and settings.metadata_path.exists():
        logger.info("Rebuilding FAISS index from metadata...")
        import json, numpy as np
        from app.rag.embedder import Embedder
        from app.rag.vector_store import VectorStore
        embedder = Embedder()
        with open(settings.metadata_path, encoding="utf-8") as f:
            chunks = json.load(f)
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed(texts)
        store = VectorStore()
        store.add(vectors, chunks)
        store.save()
        logger.info("FAISS index rebuilt.")
    app.state.pipeline = RAGPipeline()
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="ClassMind",
    description="AI Teacher Assistant for Indian Schools — NCERT RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "service": "ClassMind"}
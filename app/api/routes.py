from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import shutil
from pathlib import Path
from app.ingestion.chunker import extract_text_from_pdf, chunk_pages
from app.rag.embedder import Embedder
from app.core.logger import get_logger

router = APIRouter(prefix="/api")
logger = get_logger(__name__)

class QueryRequest(BaseModel):
    query: str

@router.post("/ask")
async def ask(request: Request, body: QueryRequest):
    pipeline = request.app.state.pipeline
    result = pipeline.run(body.query)
    return result

@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        logger.info(f"PDF uploaded: {file.filename} → {tmp_path}")

        # Extract and chunk
        pages = extract_text_from_pdf(tmp_path)
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. Make sure it's not a scanned image-only PDF.")

        metadata = {
            "filename": file.filename,
            "class": "Custom",
            "subject": "Upload",
            "chapter": "—",
            "title": file.filename.replace(".pdf", ""),
        }
        chunks = chunk_pages(pages, metadata)

        # Embed new chunks
        embedder = Embedder()
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed(texts)

        # Add to existing live index — no rebuild needed
        pipeline = request.app.state.pipeline
        pipeline.retriever.store.add(vectors, chunks)
        pipeline.retriever.store.save()

        logger.info(f"Added {len(chunks)} chunks from '{file.filename}' to live index.")

        return JSONResponse({
            "message": f"Successfully indexed '{file.filename}'",
            "chunks_added": len(chunks),
            "pages_processed": len(pages),
            "total_vectors": pipeline.retriever.store.index.ntotal
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)
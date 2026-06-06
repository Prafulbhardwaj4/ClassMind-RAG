from app.ingestion.fetcher import NCERT_BOOKS, download_pdfs
from app.ingestion.chunker import extract_text_from_pdf, chunk_pages
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def run_ingestion(force_download: bool = False):
    """
    Full ingestion pipeline:
    1. Download NCERT PDFs
    2. Extract text page by page
    3. Sliding window chunk
    4. Embed all chunks
    5. Save FAISS index + metadata to disk
    """

    # Step 1 — Download
    logger.info("=== PHASE 1: Downloading PDFs ===")
    download_pdfs(force=force_download)

    # Step 2+3 — Extract and chunk all books
    logger.info("=== PHASE 2: Extracting and chunking ===")
    all_chunks = []

    for book in NCERT_BOOKS:
        pdf_path = settings.raw_data_dir / book["filename"]

        if not pdf_path.exists():
            logger.warning(f"PDF not found, skipping: {book['filename']}")
            continue

        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages, metadata=book)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks across all books: {len(all_chunks)}")

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check PDF downloads in data/raw/")

    # Step 4 — Embed
    logger.info("=== PHASE 3: Embedding chunks ===")
    embedder = Embedder()

    texts = [c["text"] for c in all_chunks]
    
    # Embed in batches to avoid memory pressure
    import numpy as np
    batch_size = 256
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        vectors = embedder.embed(batch)
        all_vectors.append(vectors)
        logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    all_vectors = np.vstack(all_vectors)

    # Step 5 — Store and save
    logger.info("=== PHASE 4: Building FAISS index ===")
    try:
        store = VectorStore()
        logger.info("VectorStore created")
        store.add(all_vectors, all_chunks)
        logger.info("Vectors added to store")
        store.save()
        logger.info("=== Ingestion complete ===")
        logger.info(f"Total vectors in index: {store.index.ntotal}")
    except Exception as e:
        logger.error(f"FAILED at Phase 4: {e}", exc_info=True)
        raise
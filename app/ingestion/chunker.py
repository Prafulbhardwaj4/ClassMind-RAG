import fitz  # PyMuPDF
import re
from pathlib import Path
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text page by page from a PDF.
    Returns list of {page_num, text} dicts.
    PyMuPDF preserves reading order better than pdfplumber for NCERT layouts.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = _clean_text(text)
        if len(text.strip()) > 50:  # Skip near-empty pages (images, blank)
            pages.append({"page_num": page_num, "text": text})
    doc.close()
    logger.info(f"Extracted {len(pages)} non-empty pages from {pdf_path.name}")
    return pages


def _clean_text(text: str) -> str:
    """Remove artifacts common in NCERT PDFs."""
    text = re.sub(r'\n{3,}', '\n\n', text)       # Collapse excessive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)        # Collapse horizontal whitespace
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Rejoin hyphenated line breaks
    return text.strip()


def chunk_pages(
    pages: list[dict],
    metadata: dict,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[dict]:
    """
    Sliding window chunking over the full book text.

    Why word-level sliding window over page boundaries?
    Pages are arbitrary visual units — a concept like 'Newton's Laws'
    often spans 3-4 pages. Chunking at page boundaries would split
    concepts mid-explanation, destroying retrieval coherence.
    Word-level windows with overlap ensure every concept appears
    fully in at least one chunk.

    Each chunk carries source metadata so retrieved results are
    traceable back to exact book + page range — essential for
    teacher trust and citation.
    """
    # Flatten all pages into a single word list, tracking page boundaries
    word_page_map = []  # (word, page_num) for every word in book
    for page in pages:
        words = page["text"].split()
        for word in words:
            word_page_map.append((word, page["page_num"]))

    chunks = []
    start = 0
    total_words = len(word_page_map)

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = word_page_map[start:end]

        text = " ".join(w for w, _ in chunk_words)
        start_page = chunk_words[0][1]
        end_page = chunk_words[-1][1]

        chunks.append({
            "text": text,
            "source": metadata["filename"],
            "class": metadata["class"],
            "subject": metadata["subject"],
            "chapter": metadata.get("chapter"),
            "title": metadata.get("title"),
            "start_page": start_page,
            "end_page": end_page,
            "chunk_id": len(chunks),
        })

        # Slide forward by (chunk_size - overlap)
        start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} chunks from {metadata['filename']}")
    return chunks
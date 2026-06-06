import httpx
from pathlib import Path
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# NCERT PDF URLs — Classes 6-10, Science and Mathematics
# Source: ncert.nic.in (freely available, government published)
NCERT_BOOKS = [
    {"filename": "iesc101.pdf", "class": 9, "subject": "Science", "chapter": 1, "title": "Exploration: Entering the World of Secondary Science"},
    {"filename": "iesc102.pdf", "class": 9, "subject": "Science", "chapter": 2, "title": "Origin of Life"},
    {"filename": "iesc103.pdf", "class": 9, "subject": "Science", "chapter": 3, "title": "Tissues in Action"},
    {"filename": "iesc104.pdf", "class": 9, "subject": "Science", "chapter": 4, "title": "Describing Motion Around Us"},
    {"filename": "iesc105.pdf", "class": 9, "subject": "Science", "chapter": 5, "title": "Exploring Mixtures and their Separation"},
    {"filename": "iesc106.pdf", "class": 9, "subject": "Science", "chapter": 6, "title": "How Forces Affect Motion"},
    {"filename": "iesc107.pdf", "class": 9, "subject": "Science", "chapter": 7, "title": "Work, Energy and Simple Machines"},
    {"filename": "iesc108.pdf", "class": 9, "subject": "Science", "chapter": 8, "title": "Journey Inside the Atom"},
    {"filename": "iesc109.pdf", "class": 9, "subject": "Science", "chapter": 9, "title": "Atomic Foundations of Matter"},
    {"filename": "iesc110.pdf", "class": 9, "subject": "Science", "chapter": 10, "title": "Sound Waves: Characteristics and Applications"},
    {"filename": "iesc111.pdf", "class": 9, "subject": "Science", "chapter": 11, "title": "Reproduction: How Life Continues"},
    {"filename": "iesc112.pdf", "class": 9, "subject": "Science", "chapter": 12, "title": "Adaptation and Survival"},
    {"filename": "iesc113.pdf", "class": 9, "subject": "Science", "chapter": 13, "title": "Earth as a System: Energy, Matter and Life"},
]

def download_pdfs(force: bool = False):
    """PDFs are already in data/raw/ — no download needed."""
    logger.info("PDFs already present in data/raw/, skipping download.")
    
def _extract_pdf_from_zip(zip_bytes: bytes, dest: Path):
    """Extract the first PDF found inside a ZIP archive."""
    import zipfile
    import io

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        pdf_files = [f for f in zf.namelist() if f.endswith(".pdf")]
        if not pdf_files:
            raise ValueError("No PDF found inside ZIP archive.")
        # Take the largest PDF (usually the full book, not a chapter)
        pdf_name = max(pdf_files, key=lambda f: zf.getinfo(f).file_size)
        dest.write_bytes(zf.read(pdf_name))
        logger.info(f"Extracted '{pdf_name}' from ZIP.")
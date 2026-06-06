import json
import faiss
import numpy as np
from pathlib import Path
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(settings.embedding_dim)
        self.metadata: list[dict] = []  # Parallel list to FAISS vectors

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        """Add vectors and their metadata to the store."""
        assert vectors.shape[0] == len(metadata), "Vector count must match metadata count."
        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(metadata)} chunks. Total: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = settings.top_k) -> list[dict]:
        """
        Return top-k chunks with scores and metadata.
        query_vector shape: (1, 384)
        """
        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty. Run build_index.py first.")

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save(self):
        """Persist FAISS index and metadata to disk."""
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(settings.faiss_index_path))
        with open(settings.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Index saved. Vectors: {self.index.ntotal}, Metadata: {len(self.metadata)}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        if not settings.faiss_index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {settings.faiss_index_path}. Run build_index.py first.")
        self.index = faiss.read_index(str(settings.faiss_index_path))
        with open(settings.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Index loaded. Vectors: {self.index.ntotal}, Metadata: {len(self.metadata)}")
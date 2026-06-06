import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dim = settings.embedding_dim
        logger.info(f"Embedding model loaded. Output dim: {self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns float32 array of shape (len(texts), 384).
        Normalized for cosine similarity via dot product on IndexFlatIP.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalize — cosine sim becomes dot product
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding — returns shape (1, 384) for FAISS search."""
        return self.embed([query])
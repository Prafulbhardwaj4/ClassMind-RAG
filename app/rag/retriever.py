from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.store.load()

    def retrieve(self, query: str, k: int = settings.top_k) -> list[dict]:
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, k=k)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
        return results
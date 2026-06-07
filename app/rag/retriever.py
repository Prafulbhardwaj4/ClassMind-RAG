import re
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
        # Check if user mentioned a specific chapter number
        chapter_match = re.search(r'chapter\s*(\d+)', query.lower())
        
        query_vector = self.embedder.embed_query(query)
        
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            # Get more results then filter by chapter
            results = self.store.search(query_vector, k=30)
            chapter_results = [r for r in results if r.get('chapter') == chapter_num]
            
            if len(chapter_results) >= 3:
                logger.info(f"Chapter filter: found {len(chapter_results)} chunks for chapter {chapter_num}")
                return chapter_results[:k]
        
        # Default semantic search
        results = self.store.search(query_vector, k=k)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
        return results
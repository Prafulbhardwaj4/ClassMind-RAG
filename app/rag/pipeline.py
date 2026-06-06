from app.rag.retriever import Retriever
from app.rag.generator import generate
from app.core.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG pipeline...")
        self.retriever = Retriever()
        logger.info("RAG pipeline ready.")

    def run(self, query: str) -> dict:
        chunks = self.retriever.retrieve(query)
        answer = generate(query, chunks)
        sources = [
            {
                "file": c["source"],
                "class": c.get("class"),
                "subject": c.get("subject"),
                "chapter": c.get("chapter"),
                "title": c.get("title"),
                "pages": f"{c.get('start_page')}-{c.get('end_page')}",
                "score": round(c["score"], 4)
            }
            for c in chunks
        ]
        return {"answer": answer, "sources": sources}
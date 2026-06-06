from groq import Groq
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are ClassMind, an AI assistant for teachers.
You answer strictly based on the provided document excerpts given to you.
Always structure your response clearly for classroom use.
If the context doesn't contain enough information, say: "The uploaded document does not cover this in the available excerpts."
Never mention NCERT unless the document is actually an NCERT textbook."""

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for c in chunks:
        header = f"[Source: Class {c.get('class')} {c.get('subject')} | Chapter {c.get('chapter')}: {c.get('title')} | Pages {c.get('start_page')}-{c.get('end_page')} | Relevance: {c.get('score', 0):.2f}]"
        context_parts.append(f"{header}\n{c['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    return f"""Here are the relevant excerpts from NCERT Class 9 Science textbook:

{context}

---

Teacher's Request: {query}

Important: Base your entire response on the NCERT excerpts above. If the excerpts don't contain enough information for this specific topic, say so clearly and mention which chapter would be more relevant.

Provide a detailed, classroom-ready response:"""

def generate(query: str, chunks: list[dict]) -> str:
    prompt = build_prompt(query, chunks)
    logger.info(f"Sending request to Groq — model: {settings.groq_model}")
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=settings.groq_temperature,
    )
    return response.choices[0].message.content
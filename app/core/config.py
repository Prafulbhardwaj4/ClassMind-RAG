from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Project root is two levels up from this file (app/core/config.py -> ClassMind/)
ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.3

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k: int = 5

    # Chunking
    chunk_size: int = 400      # words
    chunk_overlap: int = 80    # words

    # Paths
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    index_dir: Path = ROOT_DIR / "data" / "index"
    faiss_index_path: Path = ROOT_DIR / "data" / "index" / "ncert.faiss"
    metadata_path: Path = ROOT_DIR / "data" / "index" / "metadata.json"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Single instance imported everywhere — no repeated .env parsing
settings = Settings()
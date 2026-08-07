"""Application settings loaded from environment variables / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = folder that contains app/, streamlit_app.py, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central config — change values in .env, not in code."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    redis_url: str = "redis://localhost:6379/0"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Relative paths are resolved against PROJECT_ROOT (fixes Windows Celery CWD issues)
    upload_dir: str = "uploads"

    # true = run jobs in a local thread (easiest on Windows, no Celery needed)
    # false = use Celery + Redis
    sync_jobs: bool = True

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieve more candidates, then keep top_k after re-ranking
    retrieve_k: int = 10
    top_k: int = 6
    # ms-marco scores: keep mild filter but always retain at least min_keep
    min_rerank_score: float = 0.0
    min_keep_hits: int = 3
    max_correction_loops: int = 3

    # Embedding model used for Qdrant vectors
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Cross-encoder for re-ranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def upload_path(self) -> Path:
        """Absolute upload directory — always the same no matter which process starts."""
        path = Path(self.upload_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


settings = Settings()


def has_openai_key() -> bool:
    """True only when a real-looking OpenAI key is configured."""
    key = (settings.openai_api_key or "").strip()
    if not key:
        return False
    # Ignore .env.example placeholders
    if "your-key" in key.lower() or key.startswith("sk-your"):
        return False
    return True

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    chat_api_key: str | None = os.getenv("CHAT_API_KEY")
    chat_base_url: str | None = os.getenv("CHAT_BASE_URL", "https://api.deepseek.com")
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-chat")

    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    source_dir: Path = Path(os.getenv("DOCAGENT_SOURCE_DIR", "data"))
    persist_dir: Path = Path(os.getenv("DOCAGENT_PERSIST_DIR", ".chroma"))
    collection_name: str = os.getenv("DOCAGENT_COLLECTION", "docagent")

    top_k: int = _int_env("DOCAGENT_TOP_K", 4)
    min_relevant_docs: int = _int_env("DOCAGENT_MIN_RELEVANT_DOCS", 1)
    max_retries: int = _int_env("DOCAGENT_MAX_RETRIES", 2)
    chunk_size: int = _int_env("DOCAGENT_CHUNK_SIZE", 800)
    chunk_overlap: int = _int_env("DOCAGENT_CHUNK_OVERLAP", 120)


settings = Settings()

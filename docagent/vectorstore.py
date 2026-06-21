from __future__ import annotations

from langchain_chroma import Chroma

from docagent.config import Settings, settings
from docagent.models import build_embeddings


def get_vectorstore(config: Settings = settings) -> Chroma:
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=build_embeddings(config),
        persist_directory=str(config.persist_dir),
    )

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from docagent.config import Settings, settings


def build_chat_model(config: Settings = settings) -> ChatOpenAI:
    if not config.chat_api_key:
        raise RuntimeError("CHAT_API_KEY is required. Copy .env.example to .env and fill it in.")

    return ChatOpenAI(
        api_key=config.chat_api_key,
        base_url=config.chat_base_url,
        model=config.chat_model,
        temperature=0,
    )


def build_embeddings(config: Settings = settings) -> OpenAIEmbeddings:
    if not config.embedding_api_key:
        raise RuntimeError("EMBEDDING_API_KEY is required. Copy .env.example to .env and fill it in.")

    kwargs = {
        "api_key": config.embedding_api_key,
        "model": config.embedding_model,
    }
    if config.embedding_base_url:
        kwargs["base_url"] = config.embedding_base_url

    return OpenAIEmbeddings(**kwargs)

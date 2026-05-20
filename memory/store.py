from dataclasses import dataclass

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver


@dataclass(slots=True)
class EmbeddedChunk:
    document: Document
    embedding: list[float]


INGESTED_CORPUS: list[EmbeddedChunk] = []


def store_documents(documents: list[EmbeddedChunk]) -> None:
    INGESTED_CORPUS[:] = documents


def get_documents() -> list[Document]:
    return [item.document for item in INGESTED_CORPUS]


def get_embeddings() -> list[list[float]]:
    return [item.embedding for item in INGESTED_CORPUS]


def clear_documents() -> None:
    INGESTED_CORPUS.clear()


def build_checkpointer():
    return InMemorySaver()

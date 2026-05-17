from __future__ import annotations

import logging
from collections.abc import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from supportmind.models import ChunkRecord

logger = logging.getLogger(__name__)

COLLECTION_NAME = "supportmind_kb"


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()


class ChromaIndex:
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            logger.debug("Collection did not exist during reset", exc_info=True)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: list[list[float]], batch_size: int = 256) -> None:
        chunk_list = list(chunks)
        for start in range(0, len(chunk_list), batch_size):
            batch = chunk_list[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=batch_embeddings,
            )

    def query(self, query_embedding: list[float], top_k: int) -> dict:
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

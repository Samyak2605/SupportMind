from __future__ import annotations

from supportmind.ingestion.vectorstore import ChromaIndex, EmbeddingService
from supportmind.models import RetrievedChunk


class SemanticRetriever:
    def __init__(self, embedding_service: EmbeddingService, vector_index: ChromaIndex):
        self.embedding_service = embedding_service
        self.vector_index = vector_index

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_query(query)
        result = self.vector_index.query(query_embedding=query_embedding, top_k=top_k)
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        ids = result["ids"][0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                score=1.0 - float(distance),
                source="semantic",
                metadata=metadata,
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False)
        ]

from __future__ import annotations

from sentence_transformers import CrossEncoder

from supportmind.models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(chunks, scores, strict=False), key=lambda item: float(item[1]), reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                source="rerank",
                metadata=chunk.metadata,
            )
            for chunk, score in reranked
        ]

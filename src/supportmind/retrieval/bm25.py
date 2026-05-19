from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from supportmind.models import ChunkRecord, RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[ChunkRecord]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self.index.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    source="bm25",
                    metadata=chunk.metadata,
                )
            )
        return results

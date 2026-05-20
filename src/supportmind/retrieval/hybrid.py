from __future__ import annotations

from collections import defaultdict

from supportmind.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]],
    rrf_k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    lookup: dict[str, RetrievedChunk] = {}

    for results in result_sets:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            lookup[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused_results = []
    for chunk_id, score in ranked:
        original = lookup[chunk_id]
        fused_results.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                text=original.text,
                score=float(score),
                source="hybrid",
                metadata=original.metadata,
            )
        )
    return fused_results

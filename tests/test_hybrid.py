from supportmind.models import RetrievedChunk
from supportmind.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_returns_ranked_chunks():
    semantic = [
        RetrievedChunk(chunk_id="a", text="one", score=0.9, source="semantic", metadata={}),
        RetrievedChunk(chunk_id="b", text="two", score=0.8, source="semantic", metadata={}),
    ]
    bm25 = [
        RetrievedChunk(chunk_id="b", text="two", score=12.0, source="bm25", metadata={}),
        RetrievedChunk(chunk_id="a", text="one", score=11.0, source="bm25", metadata={}),
    ]

    result = reciprocal_rank_fusion([semantic, bm25], rrf_k=60, top_k=2)

    assert [item.chunk_id for item in result] == ["a", "b"]

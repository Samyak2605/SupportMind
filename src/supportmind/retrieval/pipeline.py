from __future__ import annotations

import logging
from pathlib import Path
from collections import Counter
import re

from supportmind.config import SupportMindConfig
from supportmind.ingestion.vectorstore import ChromaIndex, EmbeddingService
from supportmind.models import ChunkRecord, RetrievedChunk
from supportmind.retrieval.bm25 import BM25Retriever
from supportmind.retrieval.hybrid import reciprocal_rank_fusion
from supportmind.retrieval.reranker import CrossEncoderReranker
from supportmind.retrieval.semantic import SemanticRetriever
from supportmind.utils.io import read_jsonl

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, config: SupportMindConfig):
        manifest_path = Path(config.paths.processed_chunks)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Chunk manifest not found at {manifest_path}. Run scripts/build_index.py before serving queries."
            )

        chunk_records = [ChunkRecord.model_validate(item) for item in read_jsonl(manifest_path)]
        logger.info("Loaded %s chunk records", len(chunk_records))

        embedding_service = EmbeddingService(config.models.embedding_model)
        vector_index = ChromaIndex(str(config.paths.vectorstore_dir))

        self.config = config
        self.chunk_records = chunk_records
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunk_records}
        self.semantic = SemanticRetriever(embedding_service, vector_index)
        self.bm25 = BM25Retriever(chunk_records)
        self.reranker = CrossEncoderReranker(config.models.reranker_model)

    def _full_context_text(self, chunk: ChunkRecord) -> str:
        return (
            f"Customer issue:\n{chunk.source_instruction}\n\n"
            f"Approved support response:\n{chunk.source_response}"
        )

    def _response_signature(self, text: str) -> str:
        if "Approved support response:" in text:
            text = text.split("Approved support response:", maxsplit=1)[1]
        normalized = re.sub(r"\{\{[^}]+\}\}", " ", text.lower())
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:220]

    def _hydrate_row_results(self, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        hydrated: list[RetrievedChunk] = []
        seen_rows: set[int] = set()
        seen_signatures: set[str] = set()

        for result in results:
            source_chunk = self.chunk_by_id[result.chunk_id]
            row_id = source_chunk.row_id
            full_text = self._full_context_text(source_chunk)
            signature = self._response_signature(full_text)

            if row_id in seen_rows:
                continue
            if result.source == "rerank" and result.score <= 0:
                continue
            if signature in seen_signatures:
                continue

            seen_rows.add(row_id)
            seen_signatures.add(signature)
            hydrated.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=full_text,
                    score=result.score,
                    source=result.source,
                    metadata={
                        **result.metadata,
                        "row_id": source_chunk.row_id,
                        "category": source_chunk.category,
                        "intent": source_chunk.intent,
                        "instruction": source_chunk.source_instruction,
                        "source_response": source_chunk.source_response,
                    },
                )
            )
            if len(hydrated) >= top_k:
                break

        if hydrated:
            return hydrated

        return results[:top_k]

    def _filter_by_dominant_intent(self, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not results:
            return results

        intent_counts = Counter(str(item.metadata.get("intent", "")) for item in results if item.metadata.get("intent"))
        if not intent_counts:
            return results[:top_k]

        dominant_intent, dominant_count = intent_counts.most_common(1)[0]
        if dominant_count < 2:
            return results[:top_k]

        filtered = [item for item in results if item.metadata.get("intent") == dominant_intent]
        return filtered[:top_k] if filtered else results[:top_k]

    def get_base_results(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, list[RetrievedChunk]]]:
        requested_k = top_k or self.config.retrieval.final_top_k
        semantic_results = self.semantic.search(query=query, top_k=self.config.retrieval.semantic_top_k)

        if mode == "semantic":
            final_results = semantic_results[:requested_k]
            bm25_results: list[RetrievedChunk] = []
        else:
            bm25_results = self.bm25.search(query=query, top_k=self.config.retrieval.bm25_top_k)
            final_results = reciprocal_rank_fusion(
                [semantic_results, bm25_results],
                rrf_k=self.config.retrieval.rrf_k,
                top_k=requested_k,
            )

        traces = {
            "semantic": semantic_results,
            "bm25": bm25_results,
            "final": final_results,
        }
        return final_results, traces

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, list[RetrievedChunk]], float]:
        requested_k = top_k or self.config.retrieval.final_top_k
        base_results, traces = self.get_base_results(
            query=query,
            mode=mode,
            top_k=max(requested_k, self.config.retrieval.rerank_top_k),
        )
        rerank_ms = 0.0

        if use_reranker:
            import time

            start = time.perf_counter()
            reranked = self.reranker.rerank(
                query=query,
                chunks=base_results[: self.config.retrieval.rerank_top_k],
                top_k=self.config.retrieval.rerank_top_k,
            )
            rerank_ms = (time.perf_counter() - start) * 1000
            hydrated = self._hydrate_row_results(reranked, requested_k * 2)
            final_results = self._filter_by_dominant_intent(hydrated, requested_k)
        else:
            hydrated = self._hydrate_row_results(base_results, requested_k * 2)
            final_results = self._filter_by_dominant_intent(hydrated, requested_k)

        traces["final"] = final_results
        return final_results, traces, rerank_ms

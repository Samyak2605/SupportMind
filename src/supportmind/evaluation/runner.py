from __future__ import annotations

import asyncio
import logging
import sys
import types
from statistics import mean

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq

# ragas==0.2.14's ragas/llms/base.py unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai, a submodule removed in current
# langchain-community (superseded by the standalone langchain-google-vertexai
# package). We never use Vertex AI (Groq only), so a stub satisfies the
# import without pulling in an unrelated Google Cloud dependency.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from supportmind.config import EnvironmentSettings, SupportMindConfig, load_config, load_settings
from supportmind.evaluation.dataset import create_evaluation_dataset
from supportmind.generation import SupportMindService
from supportmind.ingestion.vectorstore import EmbeddingService
from supportmind.models import QueryRequest

logger = logging.getLogger(__name__)


class _SentenceTransformerRagasEmbeddings(BaseRagasEmbeddings):
    """Wraps the project's own EmbeddingService (bge-small) for RAGAS metrics
    that need embeddings (answer_relevancy), so the eval stays on the same
    embedding model already used for retrieval instead of pulling in OpenAI
    (ragas' default) or a second embeddings dependency."""

    def __init__(self, embedding_service: EmbeddingService):
        super().__init__()
        self._service = embedding_service

    def embed_query(self, text: str) -> list[float]:
        return self._service.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._service.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._service.embed_documents, texts)


RAGAS_JUDGE_MODEL = "llama-3.1-8b-instant"


def _build_judge(config: SupportMindConfig, settings: EnvironmentSettings):
    """RAGAS metrics need their own judge LLM + embeddings.

    This eval runs on a dedicated GROQ_EVAL_API_KEY (see
    EnvironmentSettings.for_eval), so judge + generation calls sharing a
    model's quota no longer risks starving the live demo. What *does* matter
    now is Groq's free-tier daily token budget per model: llama-3.3-70b-
    versatile is capped at 100k TPD, and a single RAGAS variant (30 rows x 4
    metrics, each involving statement generation + verdicts) burns almost
    exactly that in one run — enough for one variant, not three. llama-3.1-
    8b-instant has a 500k TPD budget, comfortably covering all three
    variants, and is more than capable of the structured judgments RAGAS
    metrics need. We also reuse the project's own embedding model rather
    than pulling in OpenAI (ragas' default), so the eval has no extra API
    dependency."""
    llm = LangchainLLMWrapper(ChatGroq(model_name=RAGAS_JUDGE_MODEL, groq_api_key=settings.groq_api_key, temperature=0.0))
    embeddings = _SentenceTransformerRagasEmbeddings(EmbeddingService(config.models.embedding_model))
    return llm, embeddings


def _run_variant(service: SupportMindService, dataset_rows: list[dict], mode: str, use_reranker: bool, llm, embeddings) -> dict:
    outputs = []
    latencies = []
    skipped = 0
    for row in dataset_rows:
        try:
            response = service.answer(QueryRequest(query=row["question"], mode=mode, use_reranker=use_reranker))
        except Exception as exc:
            # A single rate-limited/timed-out row (Groq free-tier daily quotas
            # are easy to exhaust mid-batch) shouldn't sink the whole variant —
            # skip it and keep going; the row count is reported honestly below.
            logger.warning("Skipping row during %s/reranker=%s generation: %s", mode, use_reranker, exc)
            skipped += 1
            continue
        outputs.append(
            {
                "question": row["question"],
                "answer": response.answer,
                "contexts": [chunk.text for chunk in response.retrieved_chunks[:2]] or [""],
                "ground_truth": row["ground_truth"],
            }
        )
        latencies.append(response.latency.total_ms)

    if not outputs:
        raise RuntimeError(f"Every row failed to generate for mode={mode} reranker={use_reranker}; nothing to evaluate")

    ragas_dataset = Dataset.from_pandas(pd.DataFrame(outputs))
    metrics = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=45, max_retries=3, max_wait=20),
    )
    result = metrics.to_pandas().mean(numeric_only=True).to_dict()
    result["average_latency_ms"] = mean(latencies)
    result["mode"] = mode
    result["reranker"] = use_reranker
    result["rows_evaluated"] = len(outputs)
    result["rows_skipped"] = skipped
    return result


def run_evaluation() -> pd.DataFrame:
    config = load_config()
    settings = load_settings().for_eval()
    service = SupportMindService(config, settings)
    judge_llm, judge_embeddings = _build_judge(config, settings)
    eval_rows = [row.model_dump() for row in create_evaluation_dataset(config)]

    variants = [
        ("semantic", False),
        ("hybrid", False),
        ("hybrid", True),
    ]

    results = []
    for mode, use_reranker in variants:
        try:
            results.append(
                _run_variant(service, eval_rows, mode=mode, use_reranker=use_reranker, llm=judge_llm, embeddings=judge_embeddings)
            )
        except Exception:
            logger.exception("Variant mode=%s reranker=%s failed entirely; skipping it in the published table", mode, use_reranker)

    if not results:
        raise RuntimeError("Every retrieval variant failed to evaluate; nothing to publish")

    frame = pd.DataFrame(results)
    logger.info("Evaluation completed (%s/%s variants)", len(results), len(variants))
    return frame

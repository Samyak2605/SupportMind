from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    row_id: int
    source_instruction: str
    source_response: str
    category: str
    intent: str
    text: str
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: Literal["semantic", "bm25", "hybrid", "rerank"]
    metadata: dict[str, Any]


class LatencyBreakdown(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    mode: Literal["semantic", "hybrid"] = "hybrid"
    use_reranker: bool = True
    top_k: int | None = None


class Citation(BaseModel):
    chunk_id: str
    category: str
    intent: str
    row_id: int
    excerpt: str


class QueryResponse(BaseModel):
    query: str
    mode: Literal["semantic", "hybrid"]
    use_reranker: bool
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    latency: LatencyBreakdown


class EvaluationRow(BaseModel):
    question: str
    ground_truth: str
    category: str
    intent: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class PendingApproval(BaseModel):
    approval_id: str
    order_id: int
    amount: float
    reason: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    escalated: bool = False
    pending_approval: PendingApproval | None = None


class ApprovalSummary(BaseModel):
    id: str
    thread_id: str
    order_id: int
    amount: float
    status: str
    created_at: str
    expires_at: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool


class ApprovalDecisionResponse(BaseModel):
    approval_id: str
    status: str
    reply: str

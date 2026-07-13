from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command

from supportmind.agent.runtime import AgentRuntime
from supportmind.config import load_config, load_settings
from supportmind.generation import SupportMindService
from supportmind.logging_config import configure_logging
from supportmind.models import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalSummary,
    ChatRequest,
    ChatResponse,
    PendingApproval,
    QueryRequest,
    QueryResponse,
)
from supportmind.retrieval.pipeline import RetrievalService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "webui" / "static"

# Built at import time so both /query|/chat share one retrieval stack instead
# of constructing it twice in the same process.
_config = load_config()
_settings = load_settings()
configure_logging(_config.app.log_level)
_retrieval_service = RetrievalService(_config)
service = SupportMindService(_config, _settings, retrieval_service=_retrieval_service)
agent_runtime = AgentRuntime(_config, _settings, retrieval_service=_retrieval_service)
logger.info("SupportMind API initialized")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    agent_runtime.close()


app = FastAPI(title="SupportMind API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        return service.answer(payload)
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent_runtime.graph.invoke(
            {"thread_id": thread_id, "messages": [{"role": "user", "content": payload.message}]},
            config=graph_config,
        )
    except Exception as exc:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        reply = (
            f"Your refund request for order #{interrupt_payload['order_id']} "
            f"(${interrupt_payload['amount']:.2f}) needs manager approval before I can process it. "
            "I'll follow up once it has been reviewed."
        )
        return ChatResponse(
            thread_id=thread_id,
            reply=reply,
            pending_approval=PendingApproval(
                approval_id=interrupt_payload["approval_id"],
                order_id=interrupt_payload["order_id"],
                amount=interrupt_payload["amount"],
                reason=interrupt_payload.get("reason", ""),
            ),
        )

    return ChatResponse(
        thread_id=thread_id,
        reply=result.get("final_response", ""),
        escalated=result.get("escalated", False),
    )


@app.get("/approvals", response_model=list[ApprovalSummary])
def list_approvals() -> list[ApprovalSummary]:
    approvals = agent_runtime.orders_client.list_pending_approvals()
    return [
        ApprovalSummary(
            id=approval.id,
            thread_id=approval.thread_id,
            order_id=approval.order_id,
            amount=approval.amount,
            status=approval.status,
            created_at=approval.created_at.isoformat(),
            expires_at=approval.expires_at.isoformat(),
        )
        for approval in approvals
    ]


@app.post("/approvals/{approval_id}", response_model=ApprovalDecisionResponse)
def decide_approval(approval_id: str, payload: ApprovalDecisionRequest) -> ApprovalDecisionResponse:
    approval = agent_runtime.orders_client.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"No approval found with id {approval_id}")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval {approval_id} already {approval.status}")

    agent_runtime.orders_client.decide_approval(approval_id, payload.approved)
    thread_config = {"configurable": {"thread_id": approval.thread_id}}
    decision = {"decision": "approved" if payload.approved else "rejected"}

    try:
        result = agent_runtime.graph.invoke(Command(resume=decision), config=thread_config)
    except Exception as exc:
        logger.exception("Resuming approval decision failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApprovalDecisionResponse(
        approval_id=approval_id,
        status="approved" if payload.approved else "rejected",
        reply=result.get("final_response", ""),
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="webui")

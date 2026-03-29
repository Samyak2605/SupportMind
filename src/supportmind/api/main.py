from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from supportmind.config import load_config, load_settings
from supportmind.generation import SupportMindService
from supportmind.logging_config import configure_logging
from supportmind.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

service: SupportMindService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    config = load_config()
    settings = load_settings()
    configure_logging(config.app.log_level)
    service = SupportMindService(config, settings)
    logger.info("SupportMind API initialized")
    yield


app = FastAPI(title="SupportMind API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.answer(payload)
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

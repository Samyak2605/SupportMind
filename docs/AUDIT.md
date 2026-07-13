# SupportMind 2.0 — Phase 0 Audit

Date: 2026-07-12
Scope: verified against the actual local repo (not the blueprint's description of it).

## 1. What exists today (verified by reading every module)

**Ingestion** (`src/supportmind/ingestion/`)
- `loader.py` — pandas load of `data/raw/bitext_support.csv`, drops nulls on `instruction/response/category/intent`.
- `chunker.py` — `RecursiveCharacterTextSplitter` (size 300 / overlap 50), builds `"Customer issue:\n...\n\nApproved support response:\n..."` blocks, ids like `row-{n}-chunk-{i}`.
- `vectorstore.py` — `SentenceTransformer` embedder (`BAAI/bge-small-en-v1.5`) + Chroma `PersistentClient`.
- `pipeline.py` — `build_vector_index()` orchestrates the above and writes `data/processed/chunks.jsonl`.

**Retrieval** (`src/supportmind/retrieval/`)
- `semantic.py` (Chroma query), `bm25.py` (rank_bm25 Okapi), `hybrid.py` (RRF fusion), `reranker.py` (cross-encoder `ms-marco-MiniLM-L-6-v2`).
- `pipeline.py` — `RetrievalService`: real logic beyond the README summary — row-level dedup, response-signature dedup (collapses near-duplicate template answers), dominant-intent filtering. This is more sophisticated than "hybrid + rerank," worth preserving carefully when `kb_search` wraps it.

**Generation** (`src/supportmind/generation/__init__.py`)
- `GroqGenerationService` using **`llama-3.1-8b-instant`** (config-driven, not 3.3-70b as the blueprint's diagram assumes).
- Non-trivial placeholder-handling logic: Bitext's `{{Placeholder}}` tokens get sanitized or trigger a canned "not available in KB" fallback answer when a chunk is placeholder-heavy with no actionable steps. Any future agent/tool layer that reuses generation needs to respect this, not bypass it.
- `SupportMindService.answer()` ties retrieval + generation + latency timing into `QueryResponse`.

**API** — FastAPI, only `GET /health` and `POST /query`. No auth, no other routes.

**UI** — single Gradio query tab (mode + reranker toggles, answer/chunks/citations/latency). No chat, no actions.

**Evaluation** (`src/supportmind/evaluation/`)
- `dataset.py` samples N rows from the raw CSV as eval ground truth.
- `runner.py` runs RAGAS (faithfulness, answer_relevance, context_precision, context_recall) across semantic / hybrid / hybrid+rerank.
- `scripts/run_evaluation.py` **prints a markdown table to stdout only** — confirms the blueprint's stated gap: results are never written to a `results/` file, so nothing is currently publishable.

**Config** — `config/supportmind.yaml` + pydantic `SupportMindConfig`/`EnvironmentSettings`. Clean, typed, single source of truth.

**Tests** — exactly one test (`tests/test_hybrid.py`, RRF ordering only). No API, generation, or ingestion tests.

**Data**
- `data/raw/bitext_support.csv` (18MB) present.
- `data/vectorstore/supportmind/chroma.sqlite3` (232MB) present — an index was built at some point.
- **`data/processed/` is empty** — `chunks.jsonl` is missing, even though `RetrievalService.__init__` raises `FileNotFoundError` without it. **`/query` will not run right now** until `scripts/build_index.py` is rerun. This is a pre-existing local state issue, not a blueprint gap.

**Dependencies** — `requirements.txt` includes `langchain`, `langchain-community`, `langchain_openai`, none of which are imported anywhere (only `langchain_text_splitters` is used). Dead weight. No `pyproject.toml`, no ruff/mypy config — `PYTHONPATH=src` is the only mechanism making the package importable.

**Docker** — single-stage, runs `scripts/run_api.py` only. Gradio is a separate process, not mounted into the FastAPI app.

**No CI** (no `.github/workflows/`). **No deployment config.** **No `docs/`** prior to this file — `docs/BLUEPRINT.md` referenced by the master prompt does not exist yet.

Git history (7 commits) matches the README's narrative exactly: scaffold → ingestion → hybrid retrieval → generation → API → UI → eval benchmarking.

## 2. Blueprint items that don't exist at all yet

Agent package, LangGraph dependency, orders/SQLAlchemy module, seed script, failure injection, approvals/HITL/checkpointing, idempotency keys, `/chat` endpoint, `evals/` scenario suite, simulated-customer runner, judge, `results/` directory, CI workflow, deploy config, `docs/PLAN.md`, `docs/notes/`.

## 3. Where the blueprint's assumptions don't match reality

- **Brain model**: blueprint diagram assumes `llama-3.3-70b-versatile`; config currently has `llama-3.1-8b-instant`. Need your call (below).
- **Deploy shape**: blueprint says "Dockerfile (FastAPI with Gradio mounted)" — today they're two separate entrypoints. Mounting Gradio into FastAPI (`gr.mount_gradio_app`) is a Week 2 deploy decision, not blocking Week 1.
- **Dead deps**: `langchain`/`langchain-community`/`langchain_openai` sit unused already; adding LangGraph on top is a good moment to prune them rather than compounding the clutter.

## 4. Immediate blocker

`data/processed/chunks.jsonl` is missing, so the existing `/query` path is currently broken locally. This needs `scripts/build_index.py` rerun before any end-to-end testing (existing or new) is possible.

## 5. Frozen this session (per master prompt's non-negotiables)

`/query` response contract, `retrieval/*`, `generation/*` (only reused, never rewritten), `ingestion/*`, existing test, UI query tab.

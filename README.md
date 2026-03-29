# SupportMind

**SupportMind** is a production-style GenAI customer support copilot built end to end with Retrieval-Augmented Generation (RAG), hybrid retrieval, reranking, grounded answer generation, API serving, UI delivery, and evaluation.

It is designed to look and behave like a real internal AI support assistant used by a SaaS company: customer query in, grounded support response out, with citations, retrieved evidence, latency breakdowns, and measurable retrieval quality.

## Why This Project Stands Out

- Built as a full-stack AI product, not just a notebook or toy demo
- Uses a real retrieval stack: semantic search + BM25 + hybrid RRF + cross-encoder reranking
- Adds grounded generation with source citations and hallucination control
- Ships with both a FastAPI backend and a Gradio frontend
- Includes evaluation using RAGAS across multiple retrieval strategies
- Structured like a maintainable production codebase with modular components, config-driven setup, and logging

This project demonstrates practical AI engineering across:

- LLM application design
- retrieval systems
- API engineering
- product-facing UI
- evaluation and benchmarking
- production-style code organization

## Product Overview

SupportMind helps answer customer support questions by retrieving relevant support knowledge from a local support dataset and generating a grounded answer using only retrieved evidence.

**User flow**

1. A user asks a support question like `How do I cancel my order?`
2. The system retrieves relevant support knowledge from the indexed knowledge base
3. Results are reranked using a cross-encoder
4. The LLM generates a grounded answer using the top evidence
5. The system returns:
   - final answer
   - retrieved evidence
   - source citations
   - retrieval / rerank / generation latency

## Architecture

```text
User Query
   |
   v
FastAPI / Gradio
   |
   v
Retrieval Layer
   |- Semantic Search (Chroma + BGE embeddings)
   |- BM25 Keyword Search
   |- Hybrid Fusion (RRF)
   |- Cross-Encoder Reranker
   |
   v
Grounded Generation Layer
   |- Groq LLM (llama-3.1-8b-instant)
   |- Citation-aware prompting
   |- Placeholder-aware sanitization
   |
   v
Response
   |- Answer
   |- Citations
   |- Retrieved chunks
   |- Latency metrics
```

## Tech Stack

- **Language:** Python
- **Backend:** FastAPI
- **Frontend:** Gradio
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Vector Store:** ChromaDB
- **Keyword Retrieval:** `rank-bm25`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Inference:** Groq `llama-3.1-8b-instant`
- **Chunking:** `RecursiveCharacterTextSplitter`
- **Evaluation:** RAGAS
- **Data Processing:** pandas

## Repository Structure

```text
SupportMind/
├── config/
│   └── supportmind.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
├── scripts/
│   ├── build_index.py
│   ├── run_api.py
│   ├── run_ui.py
│   └── run_evaluation.py
├── src/
│   └── supportmind/
│       ├── api/
│       ├── evaluation/
│       ├── generation/
│       ├── ingestion/
│       ├── retrieval/
│       ├── ui/
│       └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Core Features

### 1. Data Ingestion

- Loads `data/raw/bitext_support.csv` using pandas
- Drops null rows
- Uses:
  - `instruction` as query text
  - `response` as support knowledge
- Combines `instruction + response` into a single retrieval document

### 2. Chunking

- Uses `RecursiveCharacterTextSplitter`
- `chunk_size = 300`
- `chunk_overlap = 50`

### 3. Retrieval

- **Semantic search** over BGE embeddings stored in ChromaDB
- **BM25 search** for keyword-sensitive recall
- **Hybrid retrieval** using Reciprocal Rank Fusion
- **Cross-encoder reranking** for final ranking quality
- Deduplication and intent-aware filtering to reduce noisy results

### 4. Grounded Generation

- Uses Groq-hosted `llama-3.1-8b-instant`
- Prompting enforces:
  - grounded answers only
  - source-aware answering
  - citation output
  - irrelevant context suppression
- Placeholder-aware sanitization improves customer-facing readability

### 5. API

- FastAPI backend with:
  - `GET /health`
  - `POST /query`
- Returns:
  - answer
  - citations
  - retrieved chunks
  - latency metrics

### 6. UI

- Gradio app for interactive testing
- Supports:
  - support query input
  - semantic vs hybrid mode toggle
  - reranker toggle
  - grounded answer output
  - retrieved chunks viewer
  - citations
  - latency metrics

### 7. Evaluation

- Evaluation dataset generation from the source corpus
- RAGAS metrics:
  - faithfulness
  - answer_relevance
  - context_precision
  - context_recall
- Comparison modes:
  - semantic only
  - hybrid
  - hybrid + rerank

## What I Optimized For

This project was intentionally engineered to reflect how a serious AI application would be built in practice:

- modular architecture
- reusable services
- config-driven behavior
- logging over print debugging
- retrieval quality improvements over naive vector search
- latency visibility
- measurable evaluation rather than subjective demos only

In other words: this is not just “LLM wrapper code”; it is a retrieval system, application backend, frontend, and evaluation pipeline packaged as a deployable AI product.

## Local Run Guide

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
PYTHONPATH=src .venv/bin/pip install --no-cache-dir -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Add your Groq key in `.env`:

```env
GROQ_API_KEY=your_real_groq_api_key
SUPPORTMIND_CONFIG_PATH=config/supportmind.yaml
```

### 3. Build the vector index

```bash
PYTHONPATH=src .venv/bin/python scripts/build_index.py
```

### 4. Start backend

```bash
PYTHONPATH=src .venv/bin/python scripts/run_api.py
```

Backend endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 5. Start frontend

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ui.py
```

Frontend:

- `http://127.0.0.1:7860`

### 6. Run evaluation

```bash
PYTHONPATH=src .venv/bin/python scripts/run_evaluation.py
```

## Sample API Request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I cancel my order?",
    "mode": "hybrid",
    "use_reranker": true
  }'
```

## Example Product Behavior

Example queries you can test in the UI:

- `How do I cancel my order?`
- `How do I track my order?`
- `How do I reset my password?`
- `How can I unsubscribe from the newsletter?`

Expected product behaviors:

- answers stay on-topic
- answers are grounded in retrieved evidence
- citations reference source chunks
- latency metrics expose system timing
- retrieval mode changes can be inspected directly

## Recruiter Notes

If you are reviewing this project as a hiring manager or recruiter, the key signal here is breadth plus implementation depth.

This project demonstrates that I can:

- design and build LLM applications beyond simple prompting
- engineer retrieval systems that combine recall and precision techniques
- connect model systems to usable product interfaces
- evaluate AI quality systematically
- structure AI projects like maintainable software, not just experiments

This is the kind of project I’d build when the goal is not only to “make the model answer,” but to make the system observable, explainable, testable, and presentable.

## Future Improvements

- stronger response normalization for template-heavy datasets
- improved deduplication across semantically identical support responses
- structured tracing and metrics export
- auth, rate limiting, and deployment packaging
- richer evaluation dashboards

## License

This project is for educational, portfolio, and demonstration purposes unless otherwise specified.

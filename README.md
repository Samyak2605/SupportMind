# SupportMind

SupportMind is a production-grade GenAI customer support copilot built using Retrieval-Augmented Generation (RAG), hybrid retrieval, and reranking to deliver grounded, citation-backed responses.

It reflects how modern AI systems are engineered in production — combining retrieval pipelines, ranking optimization, and LLM reasoning into a reliable, explainable support assistant.

---

## Problem

Customer support systems today struggle with:

- scattered knowledge across FAQs and documentation
- inconsistent agent responses
- hallucinations from naive LLM-based chatbots
- lack of traceability in generated answers

---

## Solution

SupportMind solves this by building a **grounded GenAI system** that:

- retrieves relevant support knowledge using hybrid search
- reranks results for precision
- generates answers strictly based on retrieved evidence
- provides citations and system-level observability

This ensures answers are **accurate, explainable, and production-safe**

---

## Demo

A working demo showcasing query → retrieval → grounded answer generation is included.


https://github.com/user-attachments/assets/193999cd-1caa-4cf0-8e4d-04bf320097d1



## What Makes This Different

This is not a chatbot wrapper.

This is a **full retrieval system + LLM application stack**, including:

- hybrid retrieval (semantic + keyword)
- ranking optimization (RRF + cross-encoder)
- grounded generation with hallucination control
- evaluation using RAGAS
- API + UI delivery
- modular, production-style architecture

---

## System Architecture

```text
User Query
→
API / UI Layer
→
Retrieval System
├─ Semantic Search (Embeddings + ChromaDB)
├─ BM25 Keyword Search
├─ Hybrid Fusion (RRF)
├─ Cross-Encoder Reranking
→
Context Selection
→
LLM Generation (Groq - LLaMA 3)
→
Grounded Response + Citations + Metrics
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


## Core Capabilities

### Hybrid Retrieval Engine

- Embedding-based semantic search
- BM25 keyword recall
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for precision

---

### Grounded Answer Generation

- Uses Groq (`llama-3.1-8b-instant`)
- Strict grounding (no external hallucination)
- Citation-aware responses
- Context filtering for relevance

---

### Support Intelligence

- Answer real customer queries
- Show retrieved evidence
- Provide source attribution
- Expose latency metrics
- Compare retrieval strategies


## Dataset

- Source: Bitext Customer Support Dataset
- Size: ~27K rows
- Fields:
  - instruction (user query)
  - response (support answer)
  - intent / category

---

## Data Processing

- Combine:

```text
instruction + response
```
- Remove null and noisy rows
- Chunking:
- chunk_size = 300
- chunk_overlap = 50
- Metadata stored:
- intent
- category

---

## API

- `GET /health`
- `POST /query`

Returns:

- generated answer
- citations
- retrieved chunks
- latency breakdown

---

## UI

Interactive Gradio interface:

- query input
- hybrid / semantic toggle
- reranker toggle
- answer + citations
- retrieved context viewer
- latency metrics

---

## Evaluation

RAGAS-based evaluation pipeline:

- faithfulness
- answer_relevance
- context_precision
- context_recall

Compare:

- semantic retrieval
- hybrid retrieval
- hybrid + rerank

---

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

## Scalability Considerations

- Vector store can be replaced with FAISS / Pinecone for production-scale deployment
- Retrieval pipeline supports batching for large datasets
- Modular architecture allows easy swapping of embedding models and LLM providers
- API layer is stateless and deployable using Docker / Kubernetes

## License

This project is for educational, portfolio, and demonstration purposes unless otherwise specified.

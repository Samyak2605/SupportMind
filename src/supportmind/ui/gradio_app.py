from __future__ import annotations

import json

import gradio as gr

from supportmind.config import load_config, load_settings
from supportmind.generation import SupportMindService
from supportmind.logging_config import configure_logging
from supportmind.models import QueryRequest

config = load_config()
configure_logging(config.app.log_level)
service = SupportMindService(config, load_settings())


def run_query(query: str, mode: str, use_reranker: bool):
    response = service.answer(QueryRequest(query=query, mode=mode, use_reranker=use_reranker))
    citations = "\n".join(
        f"- {item.chunk_id} | category={item.category} | intent={item.intent} | row={item.row_id}" for item in response.citations
    )
    retrieved = json.dumps([chunk.model_dump() for chunk in response.retrieved_chunks], indent=2)
    latency = json.dumps(response.latency.model_dump(), indent=2)
    return response.answer, retrieved, citations, latency


def create_app() -> gr.Blocks:
    with gr.Blocks(title="SupportMind") as app:
        gr.Markdown(
            """
            # SupportMind
            Customer support copilot with semantic retrieval, BM25 hybrid search, reranking, grounded generation, and citations.
            """
        )
        with gr.Row():
            query = gr.Textbox(label="Support Query", lines=4, placeholder="How do I cancel my order if it has not shipped yet?")
        with gr.Row():
            mode = gr.Radio(choices=["semantic", "hybrid"], value="hybrid", label="Retrieval Mode")
            use_reranker = gr.Checkbox(value=True, label="Use Cross-Encoder Reranker")
        submit = gr.Button("Ask SupportMind", variant="primary")

        answer = gr.Markdown(label="Grounded Answer")
        retrieved_chunks = gr.Code(label="Retrieved Chunks", language="json")
        citations = gr.Markdown(label="Citations")
        latency = gr.Code(label="Latency Metrics", language="json")

        submit.click(run_query, inputs=[query, mode, use_reranker], outputs=[answer, retrieved_chunks, citations, latency])
    return app


app = create_app()

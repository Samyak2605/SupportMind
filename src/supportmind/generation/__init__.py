from __future__ import annotations

import re

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from supportmind.config import EnvironmentSettings, SupportMindConfig
from supportmind.models import Citation, QueryRequest, QueryResponse, RetrievedChunk
from supportmind.retrieval.pipeline import RetrievalService
from supportmind.utils.timing import TimerResult, timed

SYSTEM_PROMPT = """You are SupportMind, a production customer support copilot.
Answer only from the retrieved knowledge chunks.
If the context is insufficient, say that you do not have enough grounded information.
Do not invent policies, links, or steps.
Ignore context that is clearly off-topic for the user's request.
If multiple sources repeat the same answer, synthesize them once instead of listing duplicates.
Never output raw placeholder tokens like {{Placeholder}}. Replace them with generic phrases such as "your account", "the website", or "support team" only when clearly grounded.
If the available knowledge is mostly template placeholders without concrete customer-ready details, say that the exact operational details are not available in the knowledge base.
Every factual claim must be grounded in the provided sources.
End your answer with a short 'Sources:' section citing chunk ids like [row-1-chunk-0]."""


PLACEHOLDER_MAP = {
    "Order Number": "your order number",
    "Online Company Portal Info": "your account portal",
    "Online Order Interaction": "the order management section",
    "Customer Support Hours": "support hours",
    "Customer Support Phone Number": "the support phone number",
    "Website URL": "the website",
    "Login Page URL": "the login page",
    "Forgot Password": "the forgot password option",
    "Company Name": "the company",
}


def sanitize_placeholders(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return PLACEHOLDER_MAP.get(key, key.replace("_", " ").lower())

    return re.sub(r"\{\{([^}]+)\}\}", replace, text)


def chunk_placeholder_density(chunk: RetrievedChunk) -> int:
    return len(re.findall(r"\{\{[^}]+\}\}", chunk.text))


def has_actionable_steps(chunk: RetrievedChunk) -> bool:
    text = chunk.text.lower()
    return "please follow these steps" in text or "1." in text


def should_use_placeholder_fallback(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return True
    if any(has_actionable_steps(chunk) for chunk in chunks):
        return False
    placeholder_heavy = sum(1 for chunk in chunks if chunk_placeholder_density(chunk) >= 3)
    return placeholder_heavy == len(chunks)


def build_fallback_answer(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I do not have enough grounded information in the knowledge base to answer that confidently.\n\nSources: none"

    cited = ", ".join(f"[{chunk.chunk_id}]" for chunk in chunks[:2])
    return (
        "I found related support records, but they rely on unresolved internal placeholders rather than customer-ready operational details. "
        "I can confirm the general workflow, but the exact live portal labels, links, or contact details are not available in the knowledge base.\n\n"
        f"Sources: {cited}"
    )


def select_generation_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []
    return chunks[:1]


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            f"[{chunk.chunk_id}] "
            f"(category={chunk.metadata.get('category')}, intent={chunk.metadata.get('intent')})\n"
            f"{sanitize_placeholders(chunk.text)}"
        )

    joined_context = "\n\n".join(context_blocks)
    return f"""Customer query:
{query}

Retrieved knowledge:
{joined_context}

Instructions:
1. Provide a concise, support-ready answer.
2. Stay grounded in the retrieved knowledge only.
3. Mention uncertainty when the context does not fully answer the question.
4. Ignore snippets that are irrelevant to the exact support intent in the query.
5. Include inline citations using the chunk ids.
6. End with a 'Sources:' line listing the chunk ids used."""


class GroqGenerationService:
    def __init__(self, config: SupportMindConfig, settings: EnvironmentSettings):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")
        self.client = Groq(api_key=settings.groq_api_key)
        self.config = config

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        selected_chunks = select_generation_chunks(chunks)
        if should_use_placeholder_fallback(selected_chunks):
            return build_fallback_answer(selected_chunks)

        response = self.client.chat.completions.create(
            model=self.config.models.llm_model,
            temperature=self.config.generation.temperature,
            max_completion_tokens=self.config.generation.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, selected_chunks)},
            ],
            timeout=self.config.generation.request_timeout_seconds,
        )
        return response.choices[0].message.content or ""


class SupportMindService:
    def __init__(self, config: SupportMindConfig, settings: EnvironmentSettings):
        self.config = config
        self.retrieval = RetrievalService(config)
        self.generator = GroqGenerationService(config, settings)

    def answer(self, request: QueryRequest) -> QueryResponse:
        retrieval_timer = TimerResult()
        generation_timer = TimerResult()
        total_timer = TimerResult()

        with timed(total_timer):
            with timed(retrieval_timer):
                final_chunks, traces, rerank_ms = self.retrieval.retrieve(
                    query=request.query,
                    mode=request.mode,
                    use_reranker=request.use_reranker,
                    top_k=request.top_k or self.config.retrieval.final_top_k,
                )

            with timed(generation_timer):
                answer = self.generator.generate(request.query, final_chunks)

        generation_chunks = select_generation_chunks(final_chunks)

        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                category=str(chunk.metadata.get("category", "")),
                intent=str(chunk.metadata.get("intent", "")),
                row_id=int(chunk.metadata.get("row_id", -1)),
                excerpt=sanitize_placeholders(chunk.text[:240]),
            )
            for chunk in generation_chunks
        ]

        return QueryResponse(
            query=request.query,
            mode=request.mode,
            use_reranker=request.use_reranker,
            answer=answer,
            citations=citations,
            retrieved_chunks=final_chunks,
            latency={
                "retrieval_ms": retrieval_timer.elapsed_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": generation_timer.elapsed_ms,
                "total_ms": total_timer.elapsed_ms,
            },
        )

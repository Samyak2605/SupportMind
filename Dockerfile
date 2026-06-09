FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src

# Bake the retrieval index into the image at build time (downloads the
# embedding/reranker models and embeds the full Bitext corpus once) so
# container start is fast instead of re-embedding on every cold start.
# The orders DB is seeded automatically on first request (see AgentRuntime).
RUN python scripts/build_index.py

EXPOSE 8000
CMD ["python", "scripts/run_api.py"]

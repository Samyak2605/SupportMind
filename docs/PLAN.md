# SupportMind 2.0 — Build Plan

Tracking checkboxes for the blueprint in `docs/AUDIT.md`. Ticked only with evidence (test name, results
file, or artifact path) — see the linked evidence, not just the checkbox.

## Agent core

- [x] Orders module: SQLAlchemy models, seed script, 5% failure injection — `src/supportmind/orders/`,
      seeded 50 customers / 200 orders (`scripts/seed_orders.py` output)
- [x] Unified LLM client: Groq primary, Gemini fallback, JSON cache, failover logging —
      `src/supportmind/llm/client.py`
- [x] Tools: order_lookup, cancel_order, refund_initiate (idempotent), escalate_to_human, kb_search —
      `src/supportmind/agent/tools.py`, covered by `tests/test_agent_tools.py`
- [x] LangGraph agent: understand/ask/act/guard/respond/escalate, conditional edges, approval interrupt —
      `src/supportmind/agent/graph.py`, covered by `tests/test_agent_graph.py` (7 scenarios)
- [x] `/chat` and `/approvals` endpoints, `/query` unchanged — `src/supportmind/api/main.py`, verified
      live against a running server (order status, multi-turn ask, cancel, auto-refund, approval
      interrupt + approve, all exercised manually with curl)
- [x] Restart-resume approval test (mandatory) — `tests/test_agent_restart_resume.py::test_refund_approval_survives_restart_and_resumes_from_checkpoint`
- [x] Double-refund idempotency test (mandatory) — `tests/test_orders_idempotency.py`
- [x] Custom polished frontend (vanilla HTML/CSS/JS, no Gradio/Streamlit) — replaced Gradio entirely per
      explicit user request; `src/supportmind/webui/static/{index.html,styles.css,app.js}`, mounted at `/`
      in `src/supportmind/api/main.py`. Verified live: root/styles.css/app.js all 200, full chat +
      approval-interrupt + approve flow exercised via curl against the exact API calls the frontend makes.

## Evidence / evals

- [x] 30-scenario agent eval (simulated user + judge) published to `results/agent_eval.md` — 30/30 genuine
      coverage (added harness-error vs. task-failure distinction + Gemini call pacing after early runs hit
      partial rate-limit coverage), **97% task success rate**, 1 genuine failure case documented
- [ ] RAGAS re-run published to `results/ragas.md` — in progress (variant 1/3 near done at last check); hit
      Groq's 100k TPD quota on the 70b model repeatedly, ultimately fixed by: a dedicated `GROQ_EVAL_API_KEY`
      (separate from the live demo key) + judge model moved to `llama-3.1-8b-instant` (500k TPD, not shared
      with the 70b brain) + sequential execution + trimmed context per row

## Platform

- [x] CI: ruff + pytest, deterministic subset (`pytest -m "not requires_index"`), no secrets required —
      `.github/workflows/ci.yml`
- [x] Render deploy config: static UI + API in one FastAPI process, index baked at Docker build time —
      `Dockerfile`, `render.yaml`
- [x] README overhaul: hero, design decisions, failure analysis, limitations — results tables filled in
      once both eval runs are complete
- [ ] Live Render URL (requires the user's Render account — see handoff steps)
- [ ] Demo video (user's task)

## Notable deviations from the original blueprint, with reasons

- Brain model upgraded from `llama-3.1-8b-instant` (existing `/query` path, left untouched) to
  `llama-3.3-70b-versatile` for the *agent* specifically — user's choice, better structured-output
  reliability for routing/tool decisions.
- Gemini fallback model went through three revisions during build: `gemini-2.0-flash` → `gemini-flash-latest`
  (resolved to `gemini-3.5-flash`, 20 requests/day free quota) → `gemini-flash-lite-latest` (usable free
  quota). Documented in `docs/notes/agent-build.md`.
- RAGAS judge model deliberately kept separate from the agent's own brain model (`llama-3.1-8b-instant`
  instead of `llama-3.3-70b-versatile`) after the 70b model's 100k TPD free quota was exhausted mid-run —
  both because it's the pragmatic fix and because it avoids the eval competing with the live agent demo
  for the same shared quota.
- Gradio dropped entirely, replaced with a hand-built vanilla HTML/CSS/JS frontend — explicit user request
  mid-build ("I don't need any streamlit or gradio... create a highly refined and polished ui"). Not in the
  original blueprint at all; treated as a direct instruction that supersedes it.
- A second Groq API key (`GROQ_EVAL_API_KEY`) was introduced specifically to isolate eval-batch token usage
  from the live demo's quota — added to `EnvironmentSettings.for_eval()`, falls back to the main key if
  unset so a single-key setup still works.

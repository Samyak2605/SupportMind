# Build notes — SupportMind 2.0 agent layer

Per-component notes: what it does, the key decision, one hard interview question.

## Orders module (`src/supportmind/orders/`)

Mock order system on SQLite/SQLAlchemy: `Customer`, `Order`, `Refund`, `Approval`, `Escalation` tables,
a deterministic seed script (50 customers / 200 orders, seeded RNG), and an `OrdersClient` that injects
5% failure + latency on every call to simulate a flaky real OMS.

Key decision: `OrdersClient.get_or_create_refund` and `get_or_create_approval` are both keyed lookups
(idempotency key / refund_id) rather than blind inserts, because LangGraph replays a node's Python code
from the top on every resume-from-interrupt — anything before `interrupt()` runs again.

Hard interview question: *If two customers actually did submit the exact same order+amount+thread
combination legitimately (not a retry), would your idempotency key incorrectly merge them?*

## Unified LLM client (`src/supportmind/llm/`)

Single entry point for every agent LLM call: Groq primary, Gemini fallback on failure, JSON-file dev
cache, and every fallback recorded as a `FailoverEvent`.

Key decision: fallback is triggered on ANY Groq exception (not just rate limits) — a broad catch, because
in practice we hit both `RateLimitError` and generic timeouts, and the agent should degrade gracefully
either way rather than enumerate every Groq failure mode.

Hard interview question: *What happens to a JSON-mode call if Gemini's response isn't valid JSON — does
the caller crash, or is there a second layer of defense?* (Answer: `understand_node` catches
`json.JSONDecodeError`/`ValueError` and falls back to an `escalate` classification — the LLM client
itself doesn't validate JSON shape, only the caller does.)

## Agent graph (`src/supportmind/agent/graph.py`)

LangGraph `StateGraph`: understand → (ask | act | escalate) → guard → (escalate | respond). `act` is a
bounded while-loop (max 4 iterations) rather than per-iteration graph edges — simpler to reason about and
test, at the cost of not being able to visualize each tool call as its own graph node.

Key decision: only `messages` uses a LangGraph reducer (`operator.add`); every other state field
(intent, order_id, tool_log...) is fully recomputed each turn by `understand_node` reading the whole
message history. This avoids stale-state bugs across multi-turn conversations without needing to
manually reset fields.

Hard interview question: *Your act node calls `order_lookup` before `refund_initiate`. On a resume from
an approval interrupt, doesn't `order_lookup` run again?* Yes — deliberately accepted, since it's a
read-only, idempotent call; the tradeoff is documented, not hidden.

## HITL approvals (`refund_initiate` tool + `/approvals` endpoints)

Refunds over `agent.refund_approval_cap` call `langgraph.types.interrupt(...)`, which pauses the whole
graph run and checkpoints it via `SqliteSaver`. A human calls `POST /approvals/{id}`, which records the
decision then resumes the exact paused thread with `Command(resume=...)`.

Key decision: the approval row is looked up/created by `refund_id` (idempotent), not created blindly,
for the same replay-on-resume reason as above — verified directly by the restart-resume test, which
closes one `SqliteSaver` connection and opens a brand new one against the same file to prove persistence
survives a process restart, not just an in-memory pause.

Hard interview question: *What happens if the same approval_id is POSTed to twice (double-click)?*
Currently the second call gets a 409 (`Approval {id} already {status}`) because `decide_approval` is
called before the status check would... — actually check the code path: `decide_approval` fires only
after the `status != "pending"` guard, so the second call 409s cleanly.

## Eval harness (`evals/`)

30 scripted scenarios across 7 categories, a simulated customer (Gemini) that plays a persona against the
real agent, and a judge (a *third*, distinct Groq model) that scores reply quality. Task success itself is
never judged by an LLM — it's a direct SQLite assertion (order status, refund status, escalation row
present), because ground truth should come from the system of record, not a language model's opinion.

Key decision: brain=`llama-3.3-70b-versatile`, simulated user=Gemini, judge=`llama-3.1-8b-instant` — three
distinct model configurations so no single model is grading its own homework.

Hard interview question: *Groq's free tier has per-model daily token quotas (we hit the 70b model's 100k
TPD limit re-running RAGAS) — how would this eval harness behave under sustained production load, and
what's the actual mitigation path?* (Answer in Limitations: this is a demo-scale eval; production would
move to a paid tier or a self-hosted judge model.)

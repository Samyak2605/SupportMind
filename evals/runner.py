from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from google import genai
from groq import Groq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from evals.judge import score_reply
from evals.scenarios import SCENARIOS, Scenario
from evals.simulated_user import next_customer_message
from supportmind.agent.graph import AgentDeps, build_agent_graph
from supportmind.config import SupportMindConfig, load_config, load_settings
from supportmind.llm.client import LLMClient
from supportmind.orders.client import OrdersClient
from supportmind.orders.db import build_session_factory
from supportmind.orders.models import Escalation, Order, Refund
from supportmind.orders.seed import seed_database
from supportmind.retrieval.pipeline import RetrievalService

logger = logging.getLogger(__name__)

MAX_CUSTOMER_TURNS = 3
JUDGE_MODEL = "llama-3.1-8b-instant"

EXPECTED_TOOLS = {
    "order_status": {"order_lookup"},
    "cancel_eligible": {"order_lookup", "cancel_order"},
    "cancel_ineligible": {"order_lookup", "cancel_order"},
    "refund_auto": {"order_lookup", "refund_initiate"},
    "refund_over_cap": {"order_lookup", "refund_initiate"},
    "kb_question": {"kb_search"},
}


def pick_order_id(session_factory, status_filter: list[str] | None, amount_filter: str | None, cap: float, used_ids: set[int]) -> int:
    with session_factory() as session:
        query = session.query(Order)
        if status_filter:
            query = query.filter(Order.status.in_(status_filter))
        if amount_filter == "under_cap":
            query = query.filter(Order.amount <= cap)
        elif amount_filter == "over_cap":
            query = query.filter(Order.amount > cap)
        candidates = [order.id for order in query.all() if order.id not in used_ids]
        if not candidates:
            raise ValueError(f"No order matches status={status_filter} amount={amount_filter}")
        return candidates[0]


def run_scenario(
    scenario: Scenario,
    graph,
    orders_client: OrdersClient,
    session_factory,
    config: SupportMindConfig,
    gemini_client: genai.Client,
    used_order_ids: set[int],
) -> dict:
    order_needed = scenario["category"] not in ("kb_question", "escalate")
    order_id = None
    if order_needed:
        order_id = pick_order_id(session_factory, scenario.get("status_filter"), scenario.get("amount_filter"), config.agent.refund_approval_cap, used_order_ids)
        used_order_ids.add(order_id)

    goal_text = scenario["goal_template"].format(order_id=order_id if order_id is not None else "")
    persona = scenario["persona"]
    thread_id = f"eval-{scenario['id']}-{uuid.uuid4().hex[:8]}"
    graph_config = {"configurable": {"thread_id": thread_id}}
    transcript: list[dict] = []

    customer_message = next_customer_message(gemini_client, config.agent.fallback_model, persona, goal_text, transcript)
    transcript.append({"role": "user", "content": customer_message})
    result = graph.invoke({"thread_id": thread_id, "messages": [{"role": "user", "content": customer_message}]}, config=graph_config)

    for _ in range(MAX_CUSTOMER_TURNS - 1):
        if "__interrupt__" in result:
            break
        reply = result.get("final_response", "")

        needs_followup = (
            not scenario.get("provide_order_id_upfront", True)
            and order_id is not None
            and result.get("missing_field") == "order_id"
        )
        if not needs_followup:
            break

        transcript.append({"role": "assistant", "content": reply})
        followup_goal = f"{goal_text} If asked, the order number is {order_id}."
        customer_message = next_customer_message(gemini_client, config.agent.fallback_model, persona, followup_goal, transcript)
        transcript.append({"role": "user", "content": customer_message})
        result = graph.invoke({"thread_id": thread_id, "messages": [{"role": "user", "content": customer_message}]}, config=graph_config)

    pending_approval = None
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        pending_approval = payload
        decision = scenario.get("human_decision", "approved")
        orders_client.decide_approval(payload["approval_id"], decision == "approved")
        result = graph.invoke(Command(resume={"decision": decision}), config=graph_config)

    final_reply = result.get("final_response", "")
    transcript.append({"role": "assistant", "content": final_reply})

    return {
        "scenario": scenario,
        "order_id": order_id,
        "thread_id": thread_id,
        "transcript": transcript,
        "final_state": result,
        "pending_approval": pending_approval,
    }


def evaluate_result(bundle: dict, session_factory) -> dict:
    scenario = bundle["scenario"]
    category = scenario["category"]
    order_id = bundle["order_id"]
    final_state = bundle["final_state"]
    transcript = bundle["transcript"]
    tool_log = final_state.get("tool_log") or []
    final_reply = transcript[-1]["content"] if transcript else ""

    with session_factory() as session:
        order = session.get(Order, order_id) if order_id else None

        if category == "order_status":
            task_success = order is not None and str(order.id) in final_reply and order.status in final_reply.lower()
        elif category == "cancel_eligible":
            task_success = order is not None and order.status == "cancelled"
        elif category == "cancel_ineligible":
            task_success = order is not None and order.status != "cancelled"
        elif category == "refund_auto":
            refund = session.query(Refund).filter_by(order_id=order_id).order_by(Refund.id.desc()).first()
            task_success = bool(refund and refund.status == "completed" and order.status == "refunded")
        elif category == "refund_over_cap":
            refund = session.query(Refund).filter_by(order_id=order_id).order_by(Refund.id.desc()).first()
            expected_status = "completed" if scenario.get("human_decision") == "approved" else "rejected"
            task_success = bool(refund and refund.status == expected_status)
        elif category == "kb_question":
            task_success = "sources:" in final_reply.lower() and "row-" in final_reply.lower()
        elif category == "escalate":
            escalation_count = session.query(Escalation).filter_by(thread_id=bundle["thread_id"]).count()
            task_success = bool(final_state.get("escalated")) and escalation_count > 0
        else:
            task_success = False

    used_tools = {entry["tool"] for entry in tool_log}
    expected_tools = EXPECTED_TOOLS.get(category)
    if category == "escalate":
        tool_call_accuracy = 1.0 if "escalate_to_human" in used_tools else 0.0
    elif expected_tools:
        tool_call_accuracy = 1.0 if expected_tools.issubset(used_tools) else 0.0
    else:
        tool_call_accuracy = 1.0

    escalation_expected = category == "escalate"
    escalation_actual = bool(final_state.get("escalated"))
    escalation_correct = escalation_expected == escalation_actual

    turns_to_resolution = sum(1 for m in transcript if m["role"] == "user")
    citation_present = ("sources:" in final_reply.lower() and "[row-" in final_reply.lower()) if category == "kb_question" else None

    return {
        "scenario_id": scenario["id"],
        "category": category,
        "task_success": task_success,
        "tool_call_accuracy": tool_call_accuracy,
        "escalation_correct": escalation_correct,
        "turns_to_resolution": turns_to_resolution,
        "citation_present": citation_present,
        "final_reply": final_reply,
        "goal": scenario["goal_template"].format(order_id=bundle["order_id"] or ""),
        "persona": scenario["persona"],
    }


def run_all_scenarios() -> list[dict]:
    config = load_config()
    settings = load_settings().for_eval()

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="supportmind_eval_"))
    eval_config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={"orders_db": tmp_dir / "orders.db", "checkpoints_db": tmp_dir / "checkpoints.db"}
            )
        }
    )

    session_factory = build_session_factory(eval_config.paths.orders_db)
    seed_database(session_factory, eval_config)
    orders_client = OrdersClient(
        session_factory=session_factory,
        failure_rate=0.0,
        approval_timeout_hours=eval_config.agent.approval_timeout_hours,
    )

    retrieval_service = RetrievalService(eval_config)
    llm_client = LLMClient(eval_config, settings)
    deps = AgentDeps(llm_client=llm_client, orders_client=orders_client, retrieval_service=retrieval_service, config=eval_config)

    with SqliteSaver.from_conn_string(str(eval_config.paths.checkpoints_db)) as checkpointer:
        graph = build_agent_graph(deps, checkpointer)

        gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        if gemini_client is None:
            raise RuntimeError("GEMINI_API_KEY is required to run the agent eval suite (simulated customer).")
        groq_client = Groq(api_key=settings.groq_api_key)

        used_order_ids: set[int] = set()
        rows = []
        for index, scenario in enumerate(SCENARIOS):
            if index > 0:
                # Gemini's free tier enforces a per-minute request cap; pacing
                # calls avoids bursting past it and needing retries at all.
                time.sleep(5)
            try:
                bundle = run_scenario(scenario, graph, orders_client, session_factory, eval_config, gemini_client, used_order_ids)
                metrics = evaluate_result(bundle, session_factory)
                metrics["harness_error"] = False
            except Exception as exc:
                # Distinguish a harness/infra failure (e.g. the simulated-user
                # LLM call itself was rate-limited before the agent ever saw a
                # message) from a genuine agent task failure — conflating the
                # two would unfairly blame the agent for our test rig's own
                # API hiccups.
                logger.exception("Scenario %s could not be run (harness error)", scenario["id"])
                metrics = {
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "task_success": False,
                    "tool_call_accuracy": None,
                    "escalation_correct": None,
                    "turns_to_resolution": None,
                    "citation_present": None,
                    "final_reply": f"HARNESS ERROR (not an agent failure): {exc}",
                    "goal": scenario["goal_template"],
                    "persona": scenario["persona"],
                    "harness_error": True,
                }

            if metrics["harness_error"]:
                metrics["judge_score"] = None
                metrics["judge_justification"] = "skipped: harness error, not a genuine agent reply"
            else:
                judge_verdict = score_reply(groq_client, JUDGE_MODEL, metrics["goal"], metrics["persona"], metrics["final_reply"])
                metrics["judge_score"] = judge_verdict["score"]
                metrics["judge_justification"] = judge_verdict["justification"]
            rows.append(metrics)
            logger.info(
                "Scenario %s (%s): success=%s harness_error=%s judge=%s",
                scenario["id"],
                scenario["category"],
                metrics["task_success"],
                metrics["harness_error"],
                metrics["judge_score"],
            )

    return rows

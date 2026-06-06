from __future__ import annotations

from typing import Literal, TypedDict


class Scenario(TypedDict, total=False):
    id: str
    category: Literal["order_status", "cancel_eligible", "cancel_ineligible", "refund_auto", "refund_over_cap", "kb_question", "escalate"]
    persona: str
    goal_template: str
    status_filter: list[str] | None
    amount_filter: Literal["under_cap", "over_cap"] | None
    human_decision: Literal["approved", "rejected"] | None
    provide_order_id_upfront: bool


SCENARIOS: list[Scenario] = [
    # --- order_status (5) ---
    {"id": "s01", "category": "order_status", "persona": "polite and brief", "goal_template": "Ask what the status of order {order_id} is.", "status_filter": ["placed", "processing", "shipped", "delivered"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s02", "category": "order_status", "persona": "a bit anxious, wants reassurance", "goal_template": "Ask when order {order_id} will arrive.", "status_filter": ["shipped"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s03", "category": "order_status", "persona": "curt, one-line messages", "goal_template": "status of {order_id}?", "status_filter": ["delivered"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s04", "category": "order_status", "persona": "chatty, includes extra context", "goal_template": "Hi! I placed an order a while back, number {order_id}, could you check where it's at for me?", "status_filter": ["processing"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s05", "category": "order_status", "persona": "forgets the order number at first, then supplies it when asked", "goal_template": "Ask about your order without mentioning any order number.", "status_filter": ["placed"], "amount_filter": None, "provide_order_id_upfront": False},
    # --- cancel_eligible (5) ---
    {"id": "s06", "category": "cancel_eligible", "persona": "changed their mind, apologetic", "goal_template": "Ask to cancel order {order_id} because you changed your mind.", "status_filter": ["placed"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s07", "category": "cancel_eligible", "persona": "found a better price elsewhere", "goal_template": "Ask to cancel order {order_id} because you found it cheaper elsewhere.", "status_filter": ["processing"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s08", "category": "cancel_eligible", "persona": "ordered by mistake, slightly embarrassed", "goal_template": "Ask to cancel order {order_id}, explain it was ordered by accident.", "status_filter": ["placed"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s09", "category": "cancel_eligible", "persona": "direct and businesslike", "goal_template": "Cancel order {order_id}.", "status_filter": ["processing"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s10", "category": "cancel_eligible", "persona": "provides the order number only after being asked", "goal_template": "Ask to cancel your order without giving the order number up front.", "status_filter": ["placed"], "amount_filter": None, "provide_order_id_upfront": False},
    # --- cancel_ineligible (3) ---
    {"id": "s11", "category": "cancel_ineligible", "persona": "frustrated it already shipped", "goal_template": "Ask to cancel order {order_id} even though you suspect it may have already shipped.", "status_filter": ["shipped"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s12", "category": "cancel_ineligible", "persona": "confused about the policy", "goal_template": "Ask to cancel order {order_id}, which has already been delivered.", "status_filter": ["delivered"], "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s13", "category": "cancel_ineligible", "persona": "insistent", "goal_template": "Demand order {order_id} be cancelled right now.", "status_filter": ["shipped"], "amount_filter": None, "provide_order_id_upfront": True},
    # --- refund_auto (5) ---
    {"id": "s14", "category": "refund_auto", "persona": "item arrived damaged, mildly upset", "goal_template": "Ask for a refund on order {order_id} because it arrived damaged.", "status_filter": ["delivered", "shipped"], "amount_filter": "under_cap", "provide_order_id_upfront": True},
    {"id": "s15", "category": "refund_auto", "persona": "wrong item received, calm", "goal_template": "Request a refund for order {order_id}; you received the wrong item.", "status_filter": ["delivered"], "amount_filter": "under_cap", "provide_order_id_upfront": True},
    {"id": "s16", "category": "refund_auto", "persona": "changed their mind after delivery", "goal_template": "Ask for a refund on order {order_id}, no longer want the item.", "status_filter": ["delivered"], "amount_filter": "under_cap", "provide_order_id_upfront": True},
    {"id": "s17", "category": "refund_auto", "persona": "quality complaint, detailed", "goal_template": "Ask for a refund on order {order_id}, explain the quality was poor.", "status_filter": ["delivered", "shipped"], "amount_filter": "under_cap", "provide_order_id_upfront": True},
    {"id": "s18", "category": "refund_auto", "persona": "brief and to the point", "goal_template": "refund order {order_id} please", "status_filter": ["delivered"], "amount_filter": "under_cap", "provide_order_id_upfront": True},
    # --- refund_over_cap, approved (3) ---
    {"id": "s19", "category": "refund_over_cap", "persona": "expensive item never worked", "goal_template": "Ask for a refund on order {order_id}, the item never worked at all.", "status_filter": ["delivered", "shipped"], "amount_filter": "over_cap", "human_decision": "approved", "provide_order_id_upfront": True},
    {"id": "s20", "category": "refund_over_cap", "persona": "polite but firm about a large purchase", "goal_template": "Request a refund on order {order_id}; it's expensive and arrived broken.", "status_filter": ["delivered"], "amount_filter": "over_cap", "human_decision": "approved", "provide_order_id_upfront": True},
    {"id": "s21", "category": "refund_over_cap", "persona": "worried about the amount involved", "goal_template": "Ask for a refund on order {order_id}, mention it's a large amount so you understand it may take review.", "status_filter": ["shipped", "delivered"], "amount_filter": "over_cap", "human_decision": "approved", "provide_order_id_upfront": True},
    # --- refund_over_cap, rejected (2) ---
    {"id": "s22", "category": "refund_over_cap", "persona": "vague reason, pushy", "goal_template": "Demand a refund on order {order_id} without giving much reason.", "status_filter": ["delivered", "shipped"], "amount_filter": "over_cap", "human_decision": "rejected", "provide_order_id_upfront": True},
    {"id": "s23", "category": "refund_over_cap", "persona": "calm, testing the process", "goal_template": "Ask for a refund on order {order_id}.", "status_filter": ["delivered"], "amount_filter": "over_cap", "human_decision": "rejected", "provide_order_id_upfront": True},
    # --- kb_question (4) ---
    {"id": "s24", "category": "kb_question", "persona": "first-time visitor", "goal_template": "Ask how to reset your account password.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s25", "category": "kb_question", "persona": "wants to stop marketing emails", "goal_template": "Ask how to unsubscribe from the newsletter.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s26", "category": "kb_question", "persona": "curious about company policy", "goal_template": "Ask what the shipping options are.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s27", "category": "kb_question", "persona": "wants contact info", "goal_template": "Ask how to reach customer support directly.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    # --- escalate (3) ---
    {"id": "s28", "category": "escalate", "persona": "openly angry, wants a manager", "goal_template": "Say you're furious and demand to speak to a human manager immediately.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s29", "category": "escalate", "persona": "has a complex multi-part legal question", "goal_template": "Explain you have a complicated legal dispute about a purchase and need a specialist, not a bot.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
    {"id": "s30", "category": "escalate", "persona": "explicitly distrusts automated agents", "goal_template": "Say plainly that you do not want to talk to a bot and need a real person.", "status_filter": None, "amount_filter": None, "provide_order_id_upfront": True},
]

assert len(SCENARIOS) == 30
assert len({s["id"] for s in SCENARIOS}) == 30

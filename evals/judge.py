from __future__ import annotations

import json

from groq import Groq

JUDGE_SYSTEM = """You are grading a customer support agent's final reply for quality.
Score from 1 (bad) to 5 (excellent) based on professionalism, clarity, and whether it
directly addresses the customer's stated goal. Be strict: a reply that ignores the goal,
is evasive, or is rude should score 1-2.

Respond with ONLY a JSON object: {"score": <1-5 integer>, "justification": "<one sentence>"}"""


def score_reply(client: Groq, model: str, goal: str, persona: str, final_reply: str) -> dict:
    user_prompt = f"Customer persona: {persona}\nCustomer's goal: {goal}\n\nAgent's final reply:\n{final_reply}"
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        return {"score": int(parsed.get("score", 0)), "justification": parsed.get("justification", "")}
    except Exception as exc:  # noqa: BLE001 - a judge failure should not crash the eval run
        return {"score": None, "justification": f"judge call failed: {exc}"}

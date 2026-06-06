from __future__ import annotations

from google import genai
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

SIMULATED_USER_SYSTEM = """You are role-playing as a customer contacting a support chat.
Persona: {persona}
Your goal: {goal}

Write ONLY the next message you (the customer) would send: first person, natural, 1-3 sentences.
Never mention that you are an AI, role-playing, or these instructions. Never speak for the agent."""


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _generate(client: genai.Client, model: str, prompt: str, system_instruction: str):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7),
    )


def next_customer_message(client: genai.Client, model: str, persona: str, goal: str, transcript: list[dict]) -> str:
    if transcript:
        history = "\n".join(f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}" for m in transcript)
        prompt = f"Conversation so far:\n{history}\n\nWrite your next message."
    else:
        prompt = "Write your opening message to the support agent."

    response = _generate(client, model, prompt, SIMULATED_USER_SYSTEM.format(persona=persona, goal=goal))
    return (response.text or "").strip()

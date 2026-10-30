"""Generate one-line descriptions for new execution agents.

Uses a small/cheap LLM (default: Claude Haiku via OpenRouter) via a sync HTTP
call so this can be invoked from the sync tool-call path in `tools.py`
without needing to await.
"""

from __future__ import annotations

import httpx

from ...config import get_settings
from ...logging_config import logger

_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM = (
    "You write one-sentence summaries that describe the ongoing purpose of a "
    "persistent background agent. Output ONLY the sentence - no quotes, no "
    "preamble, no trailing period cleanup. Aim for 10-20 words. Describe the "
    "ongoing task the agent owns, not a single action."
)


def generate_description(agent_name: str, first_instructions: str) -> str:
    """Return a one-sentence purpose description, or an empty string on failure."""
    settings = get_settings()
    key = settings.openrouter_api_key
    if not key:
        return ""

    user_prompt = (
        f"Agent name: {agent_name}\n"
        f"First task given to this agent:\n{first_instructions}\n\n"
        "Write the one-sentence description now."
    )

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                _URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.description_generator_model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(f"Description generation failed for '{agent_name}': {exc}")
        return ""

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""

    return (text or "").strip().strip('"').strip("'")


__all__ = ["generate_description"]

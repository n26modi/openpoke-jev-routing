"""Stage 1 - Jev Query Router.

One batched Jev call classifies the incoming user turn into:
  - intent:   follow_up | new_task | chat
  - domain:   email | scheduling | research | draft | other

Downstream stages read these to decide whether to retrieve, which domain to
restrict retrieval to, and whether to even show the Interaction Agent any
roster at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...logging_config import logger
from .jev_client import JevNotConfigured, ask_jev, is_jev_available


Intent = Literal["follow_up", "new_task", "chat"]
Domain = Literal["email", "scheduling", "research", "draft", "other"]


_INTENT_CRITERIA = {
    "follow_up": (
        "The user is continuing a thread, referencing prior work, or asking "
        "about something an existing agent has handled before. Keywords like "
        "'reply to...', 'what did Alice say', 'the Tokyo trip', 'that email'."
    ),
    "new_task": (
        "The user is starting a wholly new task that no existing agent is "
        "likely to own. Fresh work: new email thread, new research topic, "
        "new scheduling request, new draft."
    ),
    "chat": (
        "The user is just chatting, greeting, reacting, or asking a meta "
        "question. No external action is required and no execution agent "
        "needs to be involved."
    ),
}


_DOMAIN_CRITERIA = {
    "email": "Reading, drafting, replying, forwarding, or searching email.",
    "scheduling": "Calendar, meetings, reminders, timing, availability.",
    "research": "Looking up information, finding something, answering a factual question.",
    "draft": "Composing a message or document for the user to review.",
    "other": "None of the above or ambiguous.",
}


@dataclass
class RouteDecision:
    """Structured output from the router stage."""

    intent: Intent
    intent_confidence: float
    domain: Domain
    domain_confidence: float
    used_jev: bool

    @property
    def should_retrieve(self) -> bool:
        """When True, Stage 2 (semantic recall + rerank) will run.

        We retrieve on every non-chat turn - even confident `new_task` - because
        the router sees only the query, not the roster. A query that looks net-new
        in isolation ("book my cleaning") often has a perfect existing-agent match
        the router can't see. The reranker's threshold is what actually filters
        the results, not the router's intent.
        """
        return self.intent != "chat"

    @property
    def show_roster(self) -> bool:
        """When False, the Interaction Agent sees an empty roster block."""
        return self.intent != "chat"

    def to_log_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 3),
            "domain": self.domain,
            "domain_confidence": round(self.domain_confidence, 3),
            "used_jev": self.used_jev,
        }


def _fallback_decision() -> RouteDecision:
    """Degraded fallback when Jev is unavailable - behave like OpenPoke baseline."""
    return RouteDecision(
        intent="follow_up",
        intent_confidence=0.0,
        domain="other",
        domain_confidence=0.0,
        used_jev=False,
    )


def _build_state(query: str, recent_history: str) -> str:
    history = (recent_history or "").strip()
    if history:
        return (
            f"<recent_conversation>\n{history}\n</recent_conversation>\n\n"
            f"<new_user_message>\n{query.strip()}\n</new_user_message>"
        )
    return f"<new_user_message>\n{query.strip()}\n</new_user_message>"


def route_query(query: str, recent_history: str = "") -> RouteDecision:
    """Classify a user turn in one Jev call. Returns a baseline fallback when Jev is unavailable."""
    if not is_jev_available():
        return _fallback_decision()

    try:
        from typesafe_sdk import Choice  # imported lazily so missing SDK doesn't poison import time
    except ImportError:
        return _fallback_decision()

    state = _build_state(query, recent_history)
    questions = {
        "intent": Choice(
            instructions="Classify what the user is asking the assistant to do.",
            criteria=_INTENT_CRITERIA,
        ),
        "domain": Choice(
            instructions="Which domain does this request fall into?",
            criteria=_DOMAIN_CRITERIA,
        ),
    }

    try:
        response = ask_jev(state, questions)
    except JevNotConfigured:
        return _fallback_decision()
    except Exception as exc:
        logger.warning(f"Jev router call failed, falling back: {exc}")
        return _fallback_decision()

    try:
        intent_answer = response.choices["intent"]
        domain_answer = response.choices["domain"]
    except (AttributeError, KeyError) as exc:
        logger.warning(f"Unexpected Jev response shape in router: {exc}")
        return _fallback_decision()

    decision = RouteDecision(
        intent=intent_answer.choice,  # type: ignore[arg-type]
        intent_confidence=float(getattr(intent_answer, "confidence", 0.0) or 0.0),
        domain=domain_answer.choice,  # type: ignore[arg-type]
        domain_confidence=float(getattr(domain_answer, "confidence", 0.0) or 0.0),
        used_jev=True,
    )
    logger.debug("Router decision", extra=decision.to_log_dict())
    return decision


__all__ = ["Domain", "Intent", "RouteDecision", "route_query"]

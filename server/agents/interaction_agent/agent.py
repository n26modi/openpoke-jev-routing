"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...config import get_settings
from ...logging_config import logger
from ...services.execution import AgentRecord, get_agent_roster
from ...services.retrieval import RouteDecision, retrieve_and_rerank

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    *,
    route: Optional[RouteDecision] = None,
    query_for_retrieval: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn.

    `route` is a Stage 1 output. When present on a user turn, the roster is
    filtered down to hot cache + reranked matches instead of dumping every
    agent. For agent-authored turns we skip filtering and show hot cache only
    - the Interaction Agent is reacting to an existing agent's response.
    """
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(
        f"<active_agents>\n{_render_active_agents(route, message_type, query_for_retrieval or latest_text)}\n</active_agents>"
    )
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _render_agent(record: AgentRecord) -> str:
    name = escape(record.name or "agent", quote=True)
    if record.description:
        return f'<agent name="{name}" description="{escape(record.description, quote=True)}" />'
    return f'<agent name="{name}" />'


def _merge_bounded(
    hot: List[AgentRecord],
    reranked: List[AgentRecord],
    max_total: int,
) -> List[AgentRecord]:
    """Hot cache first (recency wins), then reranked candidates. Dedup by name."""
    seen: set[str] = set()
    merged: List[AgentRecord] = []
    for rec in [*hot, *reranked]:
        if rec.name in seen:
            continue
        seen.add(rec.name)
        merged.append(rec)
        if len(merged) >= max_total:
            break
    return merged


# Format currently active execution agents into XML tags for LLM awareness
def _render_active_agents(
    route: Optional[RouteDecision],
    message_type: str,
    query_for_retrieval: str,
) -> str:
    """Build the <active_agents> block.

    Three code paths:
      1. Chat-only turn - show nothing.
      2. User turn with router available - hot cache ∪ reranked, bounded.
      3. Fallback (no router, or agent-authored turn) - full roster. This
         matches OpenPoke's original behaviour.
    """
    settings = get_settings()
    roster = get_agent_roster()
    roster.load()

    all_records = roster.all_records()
    if not all_records:
        return "None"

    # Chat-only: user is just texting. No agents needed.
    if route is not None and not route.show_roster:
        return "None (conversational turn)"

    # No router decision available, or it's an agent-authored message:
    # fall back to the baseline (show the whole roster, name-only).
    if route is None or message_type == "agent":
        return "\n".join(_render_agent(r) for r in all_records)

    hot = roster.get_hot_cache(settings.hot_cache_size)
    reranked_records: List[AgentRecord] = []
    if route.should_retrieve:
        try:
            ranked = retrieve_and_rerank(
                query_for_retrieval,
                domain_hint=route.domain,
            )
            reranked_records = [r.record for r in ranked]
        except Exception as exc:
            logger.warning(f"Rerank stage failed, using hot cache only: {exc}")

    merged = _merge_bounded(hot, reranked_records, settings.max_visible_agents)
    if not merged:
        # Nothing survived filtering. Fall back to the full roster rather than
        # strand the Interaction Agent with no options to pick from.
        return "\n".join(_render_agent(r) for r in all_records)

    logger.info(
        "Filtered roster",
        extra={
            "total_agents": len(all_records),
            "shown": len(merged),
            "route": route.to_log_dict(),
        },
    )
    return "\n".join(_render_agent(r) for r in merged)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"

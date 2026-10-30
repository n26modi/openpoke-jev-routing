"""Stage 2 - semantic recall + Jev rerank.

Pipeline for a single user turn:
  1. `embeddings.search` returns top-recall_k roster records by cosine
  2. One batched Jev call asks a Noul per candidate: "Is this agent relevant
     to the query?"
  3. Filter by `threshold`, sort by Noul probability, trim to precision_k

Degraded modes:
  - If Jev is unavailable, we fall through with the cosine ranking directly.
  - If the roster has no agents or embedding fails, we return [].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from ...logging_config import logger
from ..execution import AgentRecord
from .embeddings import search as embedding_search
from .jev_client import JevNotConfigured, ask_jev, is_jev_available


@dataclass
class RankedAgent:
    record: AgentRecord
    relevance: float   # Noul probability from Jev, or cosine when Jev unavailable
    cosine: float
    used_jev: bool

    def to_log_dict(self) -> dict:
        return {
            "name": self.record.name,
            "relevance": round(self.relevance, 3),
            "cosine": round(self.cosine, 3),
            "used_jev": self.used_jev,
        }


def _question_key(index: int) -> str:
    """Jev question keys need to be stable strings. One per candidate."""
    return f"candidate_{index}"


def _build_rerank_state(query: str, candidates: list[AgentRecord]) -> str:
    """State for the rerank call lists every candidate in one block so Jev sees them all."""
    lines = ["<user_query>", query.strip(), "</user_query>", "", "<candidate_agents>"]
    for i, rec in enumerate(candidates):
        desc = rec.description or "(no description)"
        lines.append(f'<candidate index="{i}" name="{rec.name}">{desc}</candidate>')
    lines.append("</candidate_agents>")
    return "\n".join(lines)


def retrieve_and_rerank(
    query: str,
    *,
    domain_hint: Optional[str] = None,
    recall_k: Optional[int] = None,
    precision_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> list[RankedAgent]:
    """Full Stage 2 pipeline. All thresholds default to settings values."""
    settings = get_settings()
    recall_k = recall_k if recall_k is not None else settings.retrieval_recall_k
    precision_k = precision_k if precision_k is not None else settings.retrieval_precision_k
    threshold = threshold if threshold is not None else settings.rerank_threshold

    hits = embedding_search(query, top_k=recall_k, domain_filter=domain_hint)
    if not hits:
        return []

    # Degraded path: no Jev. Return cosine-ranked candidates as-is, trimmed to precision_k.
    if not is_jev_available():
        return [
            RankedAgent(record=h.record, relevance=h.cosine, cosine=h.cosine, used_jev=False)
            for h in hits[:precision_k]
        ]

    try:
        from typesafe_sdk import Noul
    except ImportError:
        return [
            RankedAgent(record=h.record, relevance=h.cosine, cosine=h.cosine, used_jev=False)
            for h in hits[:precision_k]
        ]

    candidates = [h.record for h in hits]
    state = _build_rerank_state(query, candidates)
    questions = {
        _question_key(i): Noul(
            instructions=(
                f"Would the agent at index {i} be a sensible choice to handle the user's "
                "query? Answer yes if this agent plausibly owns the task or has relevant "
                "history; answer no if it's clearly unrelated."
            ),
        )
        for i in range(len(candidates))
    }

    try:
        response = ask_jev(state, questions)
    except JevNotConfigured:
        return [
            RankedAgent(record=h.record, relevance=h.cosine, cosine=h.cosine, used_jev=False)
            for h in hits[:precision_k]
        ]
    except Exception as exc:
        logger.warning(f"Jev rerank call failed, using cosine order: {exc}")
        return [
            RankedAgent(record=h.record, relevance=h.cosine, cosine=h.cosine, used_jev=False)
            for h in hits[:precision_k]
        ]

    ranked: list[RankedAgent] = []
    for i, hit in enumerate(hits):
        try:
            noul = float(response.nouls[_question_key(i)].noul)
        except (AttributeError, KeyError, TypeError) as exc:
            logger.debug(f"Missing Noul for candidate {i}: {exc}")
            noul = hit.cosine
        if noul < threshold:
            continue
        ranked.append(
            RankedAgent(record=hit.record, relevance=noul, cosine=hit.cosine, used_jev=True)
        )

    ranked.sort(key=lambda r: r.relevance, reverse=True)
    return ranked[:precision_k]


__all__ = ["RankedAgent", "retrieve_and_rerank"]

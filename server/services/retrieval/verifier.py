"""Stage 5 - Jev-powered spawn verifier.

Before a new execution agent is created, we ask Jev whether the proposed
agent would duplicate any of a small candidate set (hot cache + top semantic
matches). One batched call, one Noul per candidate. If any noul crosses the
duplicate threshold with confidence, we redirect to that existing agent
instead of inflating the roster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from ...logging_config import logger
from ..execution import AgentRecord
from .jev_client import JevNotConfigured, ask_jev, is_jev_available


@dataclass
class SpawnDecision:
    allow_spawn: bool
    redirect_to: Optional[str]
    best_match_score: float
    used_jev: bool

    def to_log_dict(self) -> dict:
        return {
            "allow_spawn": self.allow_spawn,
            "redirect_to": self.redirect_to,
            "best_match_score": round(self.best_match_score, 3),
            "used_jev": self.used_jev,
        }


def _question_key(index: int) -> str:
    return f"dup_{index}"


def _build_state(
    proposed_name: str,
    instructions: str,
    candidates: list[AgentRecord],
) -> str:
    lines = [
        "<proposed_new_agent>",
        f'name: {proposed_name}',
        f'first_task: {instructions.strip()}',
        "</proposed_new_agent>",
        "",
        "<existing_candidates>",
    ]
    for i, rec in enumerate(candidates):
        desc = rec.description or "(no description)"
        lines.append(f'<candidate index="{i}" name="{rec.name}">{desc}</candidate>')
    lines.append("</existing_candidates>")
    return "\n".join(lines)


def verify_spawn(
    proposed_name: str,
    instructions: str,
    candidates: list[AgentRecord],
    *,
    threshold: Optional[float] = None,
) -> SpawnDecision:
    """Return a SpawnDecision - redirect when a close duplicate exists, else allow."""
    if not candidates:
        return SpawnDecision(True, None, 0.0, used_jev=False)

    thresh = threshold if threshold is not None else get_settings().spawn_duplicate_threshold

    if not is_jev_available():
        return SpawnDecision(True, None, 0.0, used_jev=False)

    try:
        from typesafe_sdk import Noul
    except ImportError:
        return SpawnDecision(True, None, 0.0, used_jev=False)

    state = _build_state(proposed_name, instructions, candidates)
    questions = {
        _question_key(i): Noul(
            instructions=(
                f"Would the proposed new agent duplicate the responsibilities of the "
                f"existing candidate at index {i}? Answer yes if the existing agent "
                "already owns this kind of work (same person, same thread, same "
                "project, same recurring job)."
            ),
        )
        for i in range(len(candidates))
    }

    try:
        response = ask_jev(state, questions)
    except JevNotConfigured:
        return SpawnDecision(True, None, 0.0, used_jev=False)
    except Exception as exc:
        logger.warning(f"Jev spawn verifier failed, allowing spawn: {exc}")
        return SpawnDecision(True, None, 0.0, used_jev=False)

    best_idx = -1
    best_score = 0.0
    for i in range(len(candidates)):
        try:
            noul = float(response.nouls[_question_key(i)].noul)
        except (AttributeError, KeyError, TypeError):
            continue
        if noul > best_score:
            best_score = noul
            best_idx = i

    if best_idx >= 0 and best_score >= thresh:
        decision = SpawnDecision(
            allow_spawn=False,
            redirect_to=candidates[best_idx].name,
            best_match_score=best_score,
            used_jev=True,
        )
    else:
        decision = SpawnDecision(
            allow_spawn=True,
            redirect_to=None,
            best_match_score=best_score,
            used_jev=True,
        )

    logger.info("Spawn verifier decision", extra=decision.to_log_dict())
    return decision


__all__ = ["SpawnDecision", "verify_spawn"]

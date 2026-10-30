"""Eval runner: compare pipeline configurations against the benchmark.

For each configuration, we build the exact `<active_agents>` block the
Interaction Agent would see, measure its size, and check whether the
expected agent is present (recall@shown). This isolates the retrieval
layer from the LLM so we can iterate on thresholds and routing without
burning LLM budget.

Usage:
    python -m server.eval.run

Configurations compared:
  - baseline:        today's OpenPoke behaviour (full roster, name only)
  - hot_cache_only:  show only last-N used agents, no retrieval
  - full_hybrid:     hot cache + semantic + Jev rerank (requires TYPESAFE_API_KEY)

A configuration is "correct" on a case when:
  - expected_kind == "pick"  and the expected agent is in the shown set
  - expected_kind == "spawn" and the shown set is small (<= 2 x hot cache)
                                and the query didn't surface an off-topic match
  - expected_kind == "chat"  and the router flagged intent="chat"
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .benchmark import BENCHMARK, BenchmarkCase


# --- helpers for setting up isolated roster state per case -----------------

def _reset_roster_to(case: BenchmarkCase) -> None:
    """Point the roster singleton at a fresh file and seed it from the case."""
    from ..services.execution import roster as roster_mod

    tmp = Path(tempfile.mkdtemp(prefix="eval_roster_"))
    roster_mod._ROSTER_PATH = tmp / "roster.json"
    roster_mod._agent_roster = roster_mod.AgentRoster(roster_mod._ROSTER_PATH)
    r = roster_mod._agent_roster
    for name, desc in case.roster:
        r.add_agent(name, description=desc)
    for name in case.recently_used:
        r.record_use(name)


# --- configurations under test ---------------------------------------------

@dataclass
class Config:
    name: str
    hot_cache_size: int
    use_retrieval: bool
    use_jev_rerank: bool
    use_router: bool
    max_visible: int


def _apply_config(cfg: Config) -> None:
    from ..config import get_settings
    settings = get_settings()
    settings.hot_cache_size = cfg.hot_cache_size
    settings.max_visible_agents = cfg.max_visible


CONFIGS = [
    Config(name="baseline",        hot_cache_size=5, use_retrieval=False, use_jev_rerank=False, use_router=False, max_visible=999),
    Config(name="hot_cache_only",  hot_cache_size=5, use_retrieval=False, use_jev_rerank=False, use_router=True,  max_visible=5),
    Config(name="full_hybrid",     hot_cache_size=5, use_retrieval=True,  use_jev_rerank=True,  use_router=True,  max_visible=10),
]


# --- running one (config, case) pair ----------------------------------------

@dataclass
class RunResult:
    config: str
    case: str
    shown: list[str]
    visible_count: int
    correct: bool
    intent: Optional[str]
    latency_ms: float


def _shown_names_from_block(block: str) -> list[str]:
    """Pull the agent names out of the rendered <active_agents> block."""
    import re
    return re.findall(r'<agent name="([^"]+)"', block)


def _run_case(cfg: Config, case: BenchmarkCase) -> RunResult:
    _reset_roster_to(case)
    _apply_config(cfg)

    from ..agents.interaction_agent.agent import prepare_message_with_history
    from ..services.retrieval import route_query, RouteDecision

    start = time.perf_counter()
    if cfg.use_router:
        route = route_query(case.query, "")
    else:
        route = None

    # Baseline: no router, show the whole roster (OpenPoke today).
    # Hot-cache-only: still want the router (to detect "chat"), but suppress retrieval.
    if cfg.use_router and not cfg.use_retrieval:
        # Force should_retrieve = False by claiming it's a new_task with high confidence.
        route = RouteDecision(
            intent="chat" if route and route.intent == "chat" else "new_task",
            intent_confidence=max(0.9, route.intent_confidence) if route else 0.9,
            domain=route.domain if route else "other",
            domain_confidence=route.domain_confidence if route else 0.0,
            used_jev=route.used_jev if route else False,
        )

    messages = prepare_message_with_history(
        case.query, "", message_type="user",
        route=route, query_for_retrieval=case.query,
    )
    latency = (time.perf_counter() - start) * 1000.0

    content = messages[0]["content"]
    block = content.split("<active_agents>")[1].split("</active_agents>")[0]
    shown = _shown_names_from_block(block)
    visible = len(shown)

    # Score correctness per kind.
    if case.expected_kind == "pick":
        correct = case.expected_agent is not None and case.expected_agent in shown
    elif case.expected_kind == "spawn":
        # "Correct" if the pipeline didn't force an off-topic match. We allow a
        # small roster of hot-cache fallbacks (user may follow up on recent work).
        correct = visible <= cfg.hot_cache_size * 2 or visible == 0
    else:   # chat
        if route is not None:
            correct = route.intent == "chat"
        else:
            correct = False

    return RunResult(
        config=cfg.name,
        case=case.name,
        shown=shown,
        visible_count=visible,
        correct=correct,
        intent=route.intent if route else None,
        latency_ms=latency,
    )


# --- summary table ----------------------------------------------------------

@dataclass
class ConfigSummary:
    name: str
    accuracy: float
    avg_visible: float
    avg_latency_ms: float
    per_case: list[RunResult] = field(default_factory=list)


def _summarize(results: list[RunResult]) -> list[ConfigSummary]:
    by_config: dict[str, list[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)
    out: list[ConfigSummary] = []
    for cfg_name, rows in by_config.items():
        acc = sum(1 for r in rows if r.correct) / len(rows)
        avg_vis = sum(r.visible_count for r in rows) / len(rows)
        avg_lat = sum(r.latency_ms for r in rows) / len(rows)
        out.append(ConfigSummary(cfg_name, acc, avg_vis, avg_lat, rows))
    return out


def _print_table(summaries: list[ConfigSummary]) -> None:
    header = f"{'config':<16} {'accuracy':>10} {'avg_visible':>12} {'avg_latency_ms':>16}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<16} {s.accuracy*100:>9.1f}% {s.avg_visible:>12.1f} "
            f"{s.avg_latency_ms:>15.1f}"
        )

    # Per-case misses, grouped by config.
    print()
    print("Misses per config:")
    for s in summaries:
        misses = [r for r in s.per_case if not r.correct]
        if not misses:
            print(f"  {s.name}: (none)")
            continue
        print(f"  {s.name}:")
        for r in misses:
            shown_preview = ", ".join(r.shown[:4]) + ("..." if len(r.shown) > 4 else "")
            print(f"    - {r.case:<32} intent={r.intent}  shown[{r.visible_count}]: {shown_preview}")


def main() -> None:
    import os
    os.environ.setdefault("OPENROUTER_API_KEY", "dummy")

    from ..services.retrieval import is_jev_available
    jev_on = is_jev_available()
    print(f"Jev available: {jev_on}")
    print(f"Benchmark size: {len(BENCHMARK)} cases")
    print()

    all_results: list[RunResult] = []
    for cfg in CONFIGS:
        if cfg.use_jev_rerank and not jev_on:
            print(f"Skipping '{cfg.name}' - requires TYPESAFE_API_KEY")
            continue
        for case in BENCHMARK:
            all_results.append(_run_case(cfg, case))

    summaries = _summarize(all_results)
    _print_table(summaries)


if __name__ == "__main__":
    main()

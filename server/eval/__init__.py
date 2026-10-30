"""Offline eval harness for the Execution Agent Overload pipeline.

Run with:
    python -m server.eval.run

Compares retrieval configurations against a hand-written benchmark and
prints an ablation table (visible roster size, recall@K of the expected
agent, latency). Measures what the Interaction Agent *sees* - a direct
test of the overload fix, without needing to run the full LLM loop.
"""

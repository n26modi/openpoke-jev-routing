"""Retrieval pipeline: Jev-powered routing, reranking, and spawn verification."""

from .descriptions import generate_description
from .embeddings import SearchHit, embed_text, search
from .jev_client import JevNotConfigured, ask_jev, is_jev_available
from .reranker import RankedAgent, retrieve_and_rerank
from .router import RouteDecision, route_query
from .verifier import SpawnDecision, verify_spawn

__all__ = [
    "JevNotConfigured",
    "RankedAgent",
    "RouteDecision",
    "SearchHit",
    "SpawnDecision",
    "ask_jev",
    "embed_text",
    "generate_description",
    "is_jev_available",
    "retrieve_and_rerank",
    "route_query",
    "search",
    "verify_spawn",
]

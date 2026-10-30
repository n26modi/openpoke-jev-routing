"""Embedding + cosine search over the agent roster.

Picks a backend based on config:
  - `openai`: text-embedding-3-small (1536 dims) via direct HTTP. Cached on
    AgentRecord.embedding.
  - `lexical`: hashed bag-of-words into 256-dim L2-normalized vector. Zero
    external deps, used as a fallback so retrieval still returns something
    sensible without an OpenAI key.

Both backends go through the same interface so downstream code is backend-
agnostic. If the cached vector's dim doesn't match the current backend, it's
treated as missing and recomputed.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal, Optional

import httpx
import numpy as np

from ...config import get_settings
from ...logging_config import logger
from ..execution import AgentRecord, get_agent_roster


Backend = Literal["openai", "lexical"]


_OPENAI_DIM = 1536
_LEXICAL_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _current_backend() -> Backend:
    return "openai" if get_settings().openai_api_key else "lexical"


def _expected_dim(backend: Backend) -> int:
    return _OPENAI_DIM if backend == "openai" else _LEXICAL_DIM


# ---- lexical ----

def _lexical_embed(text: str) -> list[float]:
    """Hash tokens into a fixed-size vector and L2-normalize."""
    vec = np.zeros(_LEXICAL_DIM, dtype=np.float32)
    tokens = _TOKEN_RE.findall((text or "").lower())
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % _LEXICAL_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---- openai ----

_OPENAI_URL = "https://api.openai.com/v1/embeddings"


def _openai_embed(text: str) -> list[float]:
    settings = get_settings()
    key = settings.openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            _OPENAI_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": settings.embedding_model, "input": text or " "},
        )
        resp.raise_for_status()
        payload = resp.json()

    vector = payload["data"][0]["embedding"]
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


# ---- public API ----

@dataclass
class SearchHit:
    record: AgentRecord
    cosine: float


def embed_text(text: str) -> list[float]:
    """Embed a single string using the configured backend."""
    backend = _current_backend()
    if backend == "openai":
        try:
            return _openai_embed(text)
        except Exception as exc:
            logger.warning(f"OpenAI embed failed, falling back to lexical: {exc}")
            return _lexical_embed(text)
    return _lexical_embed(text)


def _embedding_text_for(record: AgentRecord) -> str:
    """What we embed per agent. Name alone is weak; name + description is much better."""
    if record.description:
        return f"{record.name}. {record.description}"
    return record.name


def _ensure_embedding(record: AgentRecord, backend: Backend) -> Optional[np.ndarray]:
    """Return the record's cached embedding, (re)computing if missing or wrong-dim."""
    expected_dim = _expected_dim(backend)
    cached = record.embedding
    if cached and len(cached) == expected_dim:
        return np.asarray(cached, dtype=np.float32)

    try:
        vector = embed_text(_embedding_text_for(record))
    except Exception as exc:
        logger.warning(f"Could not embed agent '{record.name}': {exc}")
        return None

    if len(vector) == expected_dim:
        get_agent_roster().set_embedding(record.name, vector)
    return np.asarray(vector, dtype=np.float32)


def search(
    query: str,
    top_k: int = 20,
    *,
    domain_filter: Optional[str] = None,
) -> list[SearchHit]:
    """Return the top_k roster records by cosine similarity to `query`.

    `domain_filter` is accepted for API symmetry but not yet used - agent
    records don't carry a domain tag today. Hook kept so Stage 1's domain
    hint can be wired in without a signature change later.
    """
    _ = domain_filter  # reserved
    backend = _current_backend()

    query_vec = np.asarray(embed_text(query), dtype=np.float32)
    if np.linalg.norm(query_vec) == 0:
        return []

    records = get_agent_roster().all_records()
    if not records:
        return []

    hits: list[SearchHit] = []
    for record in records:
        vec = _ensure_embedding(record, backend)
        if vec is None or vec.shape != query_vec.shape:
            continue
        cosine = float(np.dot(query_vec, vec))
        if not math.isfinite(cosine):
            continue
        hits.append(SearchHit(record=record, cosine=cosine))

    hits.sort(key=lambda h: h.cosine, reverse=True)
    return hits[:top_k]


__all__ = ["Backend", "SearchHit", "embed_text", "search"]

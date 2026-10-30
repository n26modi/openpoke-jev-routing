"""Thin singleton wrapper around TypeSafe's Jev SDK.

Jev is a classification model: give it a state string plus a dict of typed
questions (Choice / Noul / Score) and it returns calibrated decisions in one
batched call. Batching 13 questions into one call is reportedly 12.2x cheaper
and 10x faster than issuing them separately, so every caller in this package
builds the biggest batch it can.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ...config import get_settings
from ...logging_config import logger


class JevNotConfigured(RuntimeError):
    """Raised when a Jev call is attempted without TYPESAFE_API_KEY set."""


_client: Any = None
_client_init_failed = False


def is_jev_available() -> bool:
    """Return True when a Jev call could succeed (key + SDK present)."""
    settings = get_settings()
    if not settings.typesafe_api_key:
        return False
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client() -> Any:
    """Singleton TypeSafeClient; raises JevNotConfigured when unusable."""
    global _client, _client_init_failed

    if _client is not None:
        return _client
    if _client_init_failed:
        raise JevNotConfigured("Jev client previously failed to initialize")

    settings = get_settings()
    if not settings.typesafe_api_key:
        raise JevNotConfigured("TYPESAFE_API_KEY is not set")

    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError as exc:
        _client_init_failed = True
        raise JevNotConfigured(f"typesafe-sdk is not installed: {exc}") from exc

    try:
        _client = TypeSafeClient(
            api_key=settings.typesafe_api_key,
            model=settings.jev_model,
        )
    except TypeError:
        # Older/newer SDKs may read the key from env instead of constructor arg.
        _client = TypeSafeClient(model=settings.jev_model)
    except Exception as exc:
        _client_init_failed = True
        raise JevNotConfigured(f"Failed to initialize TypeSafeClient: {exc}") from exc

    return _client


def ask_jev(state: str, questions: Mapping[str, Any]) -> Any:
    """Submit one batched Jev request.

    `questions` is a dict keyed by caller-defined names, with Choice / Noul /
    Score values from `typesafe_sdk`. The returned object exposes `.choices`,
    `.nouls`, and `.scores` attribute dicts keyed by the same names.
    """
    if not questions:
        raise ValueError("ask_jev requires at least one question")

    client = _get_client()
    logger.debug(
        "Jev call",
        extra={"question_count": len(questions), "state_chars": len(state or "")},
    )
    return client.system_one(state=state, questions=dict(questions))


__all__ = ["JevNotConfigured", "ask_jev", "is_jev_available"]

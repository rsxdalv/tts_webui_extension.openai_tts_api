"""
Voice and model management service.
"""

import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Mutable registry: other services can call register_voice_getter(...) to add models.
_VOICE_GETTERS: dict[str, Callable[[], list]] = {}


def register_voice_getter(model: str, getter: Callable[[], list]) -> None:
    """Register a voice getter function for a model.

    Args:
        model:  Unique model identifier (e.g. "kokoro", "chatterbox").
        getter: Callable that returns a list[dict] with "label" and "value" keys.
    """
    _VOICE_GETTERS[model] = getter


def unregister_voice_getter(model: str) -> None:
    """Remove a voice getter (e.g., if a plugin is unloaded).

    Args:
        model: Model identifier to remove. No-op if not present.
    """
    _VOICE_GETTERS.pop(model, None)


def get_voices_by_model(model: str) -> list[dict]:
    """Get available voices for a specific model.

    Args:
        model: Model identifier registered via :func:`register_voice_getter`.

    Returns:
        List of voice dicts, each with ``label`` and ``value`` keys.
        Returns ``[]`` if ``model`` is empty or not registered.
    """
    if not model:
        return []
    getter = _VOICE_GETTERS.get(model)
    if getter is None:
        return []
    return getter()


def get_available_models() -> list[dict]:
    """Get list of available TTS models derived from registered voice getters.

    Returns:
        List of dicts with an ``id`` key per registered model.
    """
    return [{"id": k} for k in _VOICE_GETTERS]


# ---------------------------------------------------------------------------
# Built-in voice getters
# ---------------------------------------------------------------------------

from . import voice_getters  # noqa: F401


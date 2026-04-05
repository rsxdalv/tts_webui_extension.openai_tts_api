"""
TTS adapter registry for managing TTS model adapters.

This module provides registration and lookup for both streaming and non-streaming
TTS adapters used by the text-to-speech service.
"""

from typing import Callable, Iterator

# TTS Adapter registry (non-streaming) ----------------------------------------
_TTS_ADAPTERS: dict[
    str, Callable[[str, str, str | None, float | None, dict], dict]
] = {}


def register_tts_adapter(
    model: str,
    adapter: Callable[[str, str, str | None, float | None, dict], dict],
) -> None:
    """Register a TTS adapter for ``model``.

    Args:
        model: Model name key (e.g. ``"kokoro"``, ``"chatterbox"``).
        adapter: Callable with signature ``(model, text, voice, speed, params) -> audio_out dict``.
    """
    _TTS_ADAPTERS[model] = adapter


def unregister_tts_adapter(model: str) -> None:
    """Remove ``model`` from the TTS adapter registry."""
    _TTS_ADAPTERS.pop(model, None)


def _call_tts_adapter(
    model: str, text: str, voice: str | None, speed: float | None, params: dict
) -> dict:
    """Look up ``model`` in ``_TTS_ADAPTERS`` and call it; raises ``ValueError`` if not found."""
    adapter = _TTS_ADAPTERS.get(model)
    if adapter is None:
        raise ValueError(f"Model {model} not found in TTS adapter registry.")
    return adapter(model, text, voice, speed, params)


# TTS Streaming adapter registry -----------------------------------------------
_TTS_STREAMING_ADAPTERS: dict[
    str, Callable[[str, str, str | None, float | None, dict], Iterator[bytes]]
] = {}


def register_tts_streaming_adapter(
    model: str,
    adapter: Callable[[str, str, str | None, float | None, dict], Iterator[bytes]],
) -> None:
    """Register a streaming TTS adapter for ``model``.

    Args:
        model: Model name key.
        adapter: Callable with signature ``(model, text, voice, speed, params) -> Iterator[bytes]``.
    """
    _TTS_STREAMING_ADAPTERS[model] = adapter


def unregister_tts_streaming_adapter(model: str) -> None:
    """Remove ``model`` from the streaming TTS adapter registry."""
    _TTS_STREAMING_ADAPTERS.pop(model, None)


def _call_tts_streaming_adapter(
    model: str, text: str, voice: str | None, speed: float | None, params: dict
) -> Iterator[bytes]:
    """Look up ``model`` in ``_TTS_STREAMING_ADAPTERS`` and call it; raises ``ValueError`` if not found."""
    adapter = _TTS_STREAMING_ADAPTERS.get(model)
    if adapter is None:
        raise ValueError(f"Model {model} not found in TTS streaming adapter registry.")
    return adapter(model, text, voice, speed, params)

"""In-memory registry for HTTP-proxied TTS adapters.

Each registered proxy creates a TTS adapter (blocking) and a streaming adapter
that forward requests to a downstream OpenAI-compatible TTS endpoint.

Registration is ephemeral — it is lost when the process restarts.
"""

import io
import logging
from typing import Iterator
from urllib.parse import urlparse

import requests as _http
from pydantic import BaseModel, field_validator

from .tts_adapter_registry import (
    register_tts_adapter,
    register_tts_streaming_adapter,
    unregister_tts_adapter,
    unregister_tts_streaming_adapter,
)
from .voice_service import register_voice_getter, unregister_voice_getter

logger = logging.getLogger(__name__)

_PROXY_REGISTRATIONS: dict[str, "ProxyRegistration"] = {}


class ProxyRegistration(BaseModel):
    """Describes a single proxied downstream TTS server."""

    model: str
    url: str
    api_key: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"URL scheme '{parsed.scheme}' is not allowed. Use 'http' or 'https'."
            )
        return v.rstrip("/")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auth_headers(api_key: str | None) -> dict:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _wav_bytes_to_audio_out(wav_bytes: bytes) -> dict:
    """Parse raw WAV bytes and return the standard adapter dict."""
    from scipy.io import wavfile

    buf = io.BytesIO(wav_bytes)
    sample_rate, audio_data = wavfile.read(buf)
    return {"audio_out": (sample_rate, audio_data)}


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------

def _make_proxy_tts_adapter(reg: "ProxyRegistration"):
    """Return a blocking TTS adapter that calls the downstream /v1/audio/speech."""

    def _adapter(
        model: str,
        text: str,
        voice: str | None,
        speed: float | None,
        params: dict,
    ) -> dict:
        payload = {
            "model": model,
            "input": text,
            "voice": voice or "default",
            "speed": speed if speed is not None else 1.0,
            "response_format": "wav",
            "stream": False,
            "params": params or {},
        }
        headers = {**_auth_headers(reg.api_key), "Content-Type": "application/json"}
        response = _http.post(
            f"{reg.url}/v1/audio/speech",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return _wav_bytes_to_audio_out(response.content)

    return _adapter


def _make_proxy_streaming_adapter(reg: "ProxyRegistration"):
    """Return a streaming TTS adapter that proxies chunked bytes from the downstream."""

    def _streaming_adapter(
        model: str,
        text: str,
        voice: str | None,
        speed: float | None,
        params: dict,
    ) -> Iterator[bytes]:
        payload = {
            "model": model,
            "input": text,
            "voice": voice or "default",
            "speed": speed if speed is not None else 1.0,
            "response_format": "wav",
            "stream": True,
            "params": params or {},
        }
        headers = {**_auth_headers(reg.api_key), "Content-Type": "application/json"}
        with _http.post(
            f"{reg.url}/v1/audio/speech",
            json=payload,
            headers=headers,
            timeout=120,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

    return _streaming_adapter


def _make_proxy_voice_getter(reg: "ProxyRegistration"):
    """Return a voice getter that fetches voices live from the downstream."""

    def _voice_getter() -> list[dict]:
        headers = _auth_headers(reg.api_key)
        try:
            response = _http.get(
                f"{reg.url}/v1/audio/voices/{reg.model}",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("voices", [])
        except Exception as exc:
            logger.warning(
                "[proxy] Could not fetch voices for '%s' from %s: %s",
                reg.model,
                reg.url,
                exc,
            )
            return []

    return _voice_getter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_proxy(reg: ProxyRegistration) -> None:
    """Register a proxied TTS adapter (overwrites any previous registration for the model)."""
    _PROXY_REGISTRATIONS[reg.model] = reg
    register_tts_adapter(reg.model, _make_proxy_tts_adapter(reg))
    register_tts_streaming_adapter(reg.model, _make_proxy_streaming_adapter(reg))
    register_voice_getter(reg.model, _make_proxy_voice_getter(reg))
    logger.info("[proxy] Registered '%s' -> %s", reg.model, reg.url)


def unregister_proxy(model: str) -> bool:
    """Unregister a proxied model. Returns True if a registration existed."""
    if model not in _PROXY_REGISTRATIONS:
        return False
    del _PROXY_REGISTRATIONS[model]
    unregister_tts_adapter(model)
    unregister_tts_streaming_adapter(model)
    unregister_voice_getter(model)
    logger.info("[proxy] Unregistered '%s'", model)
    return True


def list_proxies() -> list[ProxyRegistration]:
    """Return a snapshot of all current proxy registrations."""
    return list(_PROXY_REGISTRATIONS.values())

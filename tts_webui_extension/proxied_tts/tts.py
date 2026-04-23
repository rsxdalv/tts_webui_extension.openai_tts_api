"""tts.py — Kokoro-backed TTS implementation for the proxied_tts reference server.

These two functions satisfy the ``harness.create_app`` contract:

* ``tts(model, text, voice, speed, params) -> bytes``  — returns WAV bytes
* ``get_voices(model) -> list[dict]``                  — returns voice list
"""

import io
import logging

logger = logging.getLogger(__name__)

# Known kokoro voices (used as a fallback when the kokoro package cannot be
# imported or does not expose a ``get_voices`` helper).
_FALLBACK_VOICES = [
    "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


def tts(model: str, text: str, voice: str, speed: float, params: dict) -> bytes:
    """Generate speech with kokoro and return WAV bytes.

    The *model* parameter is the proxy-side name given at registration time
    and is ignored here — the kokoro backend always uses ``_KOKORO_REPO_ID``.
    """
    from tts_webui_extension.kokoro.main import tts as kokoro_tts

    result = kokoro_tts(text=text, voice=voice, speed=speed, model_name=_KOKORO_REPO_ID, **params)
    return _result_to_wav(result)


def get_voices(model: str) -> list[dict]:
    """Return available kokoro voices."""
    try:
        from tts_webui_extension.kokoro.main import get_voices as _get  # type: ignore

        return [{"value": v, "label": v} for v in _get()]
    except Exception:
        return [{"value": v, "label": v} for v in _FALLBACK_VOICES]


def _result_to_wav(result: dict) -> bytes:
    """Convert a ``{"audio_out": (sample_rate, data)}`` dict to WAV bytes."""
    import numpy as np
    from scipy.io import wavfile

    sample_rate, audio_data = result["audio_out"]
    buf = io.BytesIO()
    if audio_data.dtype in (np.float32, np.float64):
        if abs(audio_data).max() > 1.0:
            audio_data = audio_data / abs(audio_data).max()
        wavfile.write(buf, sample_rate, audio_data.astype(np.float32))
    else:
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        wavfile.write(buf, sample_rate, audio_data)
    buf.seek(0)
    return buf.read()

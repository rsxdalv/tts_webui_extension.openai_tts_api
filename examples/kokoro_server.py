"""Example: run a Kokoro TTS server registered with the main tts-webui instance.

This is a reference for how a TTS engine provider package would expose its
model through tts-webui by calling ``setup_oai_server`` from the harness API.

Usage
-----
::

    python examples/kokoro_server.py

Environment variables
---------------------
GRADIO_SERVER_PORT
    The Kokoro server binds to ``GRADIO_SERVER_PORT + 2000`` (default 29770).
"""

import os
import sys

# Allow running directly from the repo root without installing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tts_webui_extension.openai_tts_api.harness import result_to_wav, setup_oai_server

_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"

_FALLBACK_VOICES = [
    "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


def kokoro_tts(model: str, text: str, voice: str, speed: float, params: dict) -> bytes:
    from tts_webui_extension.kokoro.main import tts as _tts

    result = _tts(text=text, voice=voice, speed=speed, model_name=_KOKORO_REPO_ID, **params)
    return result_to_wav(result)


def kokoro_voices(model: str) -> list:
    try:
        from tts_webui_extension.kokoro.main import get_voices as _get

        return [{"value": v, "label": v} for v in _get()]
    except Exception:
        return [{"value": v, "label": v} for v in _FALLBACK_VOICES]


thread = setup_oai_server(
    tts_fn=kokoro_tts,
    get_voices_fn=kokoro_voices,
    model="my-proxied-kokoro",
    register_with="http://localhost:7778",
)

print("Kokoro server is running.  Press Ctrl-C to stop.")
try:
    thread.join()
except KeyboardInterrupt:
    print("Shutting down.")

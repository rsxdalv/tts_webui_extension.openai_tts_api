"""Example: run a proxied Piper TTS server.

Start this script, then speak to it through the main tts-webui OpenAI-TTS
API using the model name ``"my-proxied-piper"``.

Usage
-----
::

    python examples/piper_server.py

Environment variables
---------------------
GRADIO_SERVER_PORT
    Port the main tts-webui server listens on (default 7778, used only for
    registration).  The Piper proxy server itself binds to
    ``GRADIO_SERVER_PORT + 2000`` (default 29770).

Notes
-----
This example assumes ``tts_webui_extension.piper_tts`` is installed and its
models are available.  Voices must be configured separately in the Piper
extension; ``piper_voices`` returns an empty list by default because Piper
voice enumeration depends on local model files.
"""

import os
import sys

# Allow running directly from the repo root without installing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tts_webui_extension.openai_tts_api.harness import result_to_wav, setup_oai_server


def piper_tts(model: str, text: str, voice: str, speed: float, params: dict) -> bytes:
    """Invoke the Piper TTS backend and return WAV bytes."""
    from tts_webui_extension.piper_tts.api import tts

    length_scale = (1.0 / speed) if speed else 1.0
    result = tts(
        text=text,
        voice_name=voice,
        length_scale=length_scale,
        noise_scale=params.get("noise_scale", 0.667),
        noise_w=params.get("noise_w", 0.8),
        sentence_silence=params.get("sentence_silence", 0.2),
    )
    return result_to_wav(result)


def piper_voices(model: str) -> list:
    """Return available Piper voices (empty by default — populate as needed)."""
    return []


thread = setup_oai_server(
    tts_fn=piper_tts,
    get_voices_fn=piper_voices,
    model="my-proxied-piper",
    register_with="http://localhost:7778",
)

print("Piper proxy server is running.  Press Ctrl-C to stop.")
try:
    thread.join()
except KeyboardInterrupt:
    print("Shutting down.")

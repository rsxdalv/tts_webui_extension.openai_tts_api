#!/usr/bin/env python3
"""
Smoke test for the proxied_tts harness with a real kokoro backend.

This test:
  1. Builds a downstream server using harness.create_app + tts.tts / tts.get_voices
  2. Starts it on a background thread (with uvicorn)
  3. Registers the downstream with the main OpenAI TTS API (via TestClient)
  4. Makes a real speech request routed through the proxy
  5. Saves the output WAV to tests/output_proxy_kokoro.wav for manual validation
  6. Verifies the WAV is structurally valid

Requires:
  - tts_webui_extension.kokoro to be installed and models to be available
  - scipy / numpy (already in deps)

Run::

    python tests/smoke_test_proxy_kokoro.py
"""

import io
import os
import sys
import threading
import time

import numpy as np
import requests as _req
import uvicorn
from fastapi.testclient import TestClient
from scipy.io import wavfile

# Ensure local source takes precedence over any installed version
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

_DOWNSTREAM_PORT = 19878
_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output_proxy_kokoro.wav")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ok(msg: str):
    print(f"  \u2713 {msg}")


def _fail(msg: str):
    print(f"  \u2717 {msg}")
    sys.exit(1)


# ─── Start downstream server ─────────────────────────────────────────────────

def _start_downstream():
    """Build and start the kokoro-backed server on _DOWNSTREAM_PORT."""
    from tts_webui_extension.proxied_tts.harness import create_app
    from tts_webui_extension.proxied_tts.tts import get_voices, tts

    app = create_app(tts, get_voices)
    config = uvicorn.Config(app, host="127.0.0.1", port=_DOWNSTREAM_PORT, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait until the server responds to a health probe
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = _req.get(f"http://127.0.0.1:{_DOWNSTREAM_PORT}/v1/audio/voices/kokoro", timeout=1)
            if r.status_code == 200:
                return server
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("Downstream server did not start in time")


# ─── Main test ───────────────────────────────────────────────────────────────

def run():
    print("Building downstream kokoro server...")
    server = _start_downstream()
    _ok(f"Downstream ready on port {_DOWNSTREAM_PORT}")

    from tts_webui_extension.openai_tts_api.router import app as main_app

    with TestClient(main_app, raise_server_exceptions=True) as client:
        # 1. Register
        print("[1] Register kokoro proxy")
        r = client.post("/tts_webui/proxy/register", json={
            "model": "test-kokoro-proxy",
            "url": f"http://localhost:{_DOWNSTREAM_PORT}",
            "api_key": None,
        })
        if r.status_code != 201:
            _fail(f"Registration returned {r.status_code}: {r.text}")
        _ok(f"Registered (status {r.status_code})")

        # 2. Voices — verify kokoro voices come through
        print("[2] Get voices via proxy")
        r = client.get("/v1/audio/voices/test-kokoro-proxy")
        if r.status_code != 200:
            _fail(f"Voices returned {r.status_code}: {r.text}")
        voices = r.json().get("voices", [])
        if not voices:
            _fail("No voices returned")
        _ok(f"Got {len(voices)} voice(s): first={voices[0].get('value', '?')}")

        # 3. Real TTS request
        print("[3] Generate speech via proxy (kokoro)")
        r = client.post("/v1/audio/speech", json={
            "model": "test-kokoro-proxy",
            "input": "Hello from the proxy smoke test.",
            "voice": "af_bella",
            "response_format": "wav",
            "stream": False,
        })
        if r.status_code != 200:
            _fail(f"Speech returned {r.status_code}: {r.text[:400]}")

        audio_bytes = r.content
        if len(audio_bytes) < 44:
            _fail(f"Audio too small: {len(audio_bytes)} bytes")
        _ok(f"Received {len(audio_bytes)} bytes of audio")

        # 4. Validate WAV structure
        print("[4] Validate WAV")
        try:
            buf = io.BytesIO(audio_bytes)
            sample_rate, audio_data = wavfile.read(buf)
            duration = len(audio_data) / sample_rate
            _ok(f"Valid WAV — sample_rate={sample_rate}, samples={len(audio_data)}, duration={duration:.2f}s")
        except Exception as exc:
            _fail(f"WAV parse failed: {exc}")

        # 5. Save output
        with open(_OUTPUT_FILE, "wb") as f:
            f.write(audio_bytes)
        _ok(f"Saved output to {os.path.relpath(_OUTPUT_FILE)}")

        # 6. Unregister
        print("[5] Unregister")
        r = client.delete("/tts_webui/proxy/register/test-kokoro-proxy")
        if r.status_code != 200:
            _fail(f"Unregister returned {r.status_code}: {r.text}")
        _ok("Unregistered")

    print("\nAll kokoro proxy smoke tests passed.")


if __name__ == "__main__":
    run()

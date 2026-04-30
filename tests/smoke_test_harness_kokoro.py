"""smoke_test_harness_kokoro.py — integration test for setup_oai_server + Kokoro.

Requires Kokoro and its dependencies to be installed.
Saves generated audio to tests/output_harness_kokoro.wav.

Run with::

    python tests/smoke_test_harness_kokoro.py
"""

import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _WORKSPACE_ROOT)

os.environ.setdefault("PYTHONUTF8", "1")

import requests

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #
_PORT = 29880  # isolated port, away from any production defaults
_BASE = f"http://localhost:{_PORT}"
_PASS = "PASS"
_FAIL = "FAIL"


def _wait_ready(base_url: str, timeout: float = 30.0) -> bool:
    """Poll until the server is up and returns a non-5xx response."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/audio/voices/kokoro", timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    results = []

    # ------------------------------------------------------------------ #
    #  STEP 1 — start server via setup_oai_server                         #
    # ------------------------------------------------------------------ #
    print("STEP 1 — start setup_oai_server (kokoro backend, no registration)")
    from tts_webui_extension.openai_tts_api.harness import result_to_wav, setup_oai_server

    _KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
    _FALLBACK_VOICES = [
        "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael",
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis",
    ]

    def _tts(model, text, voice, speed, params):
        from tts_webui_extension.kokoro.main import tts as kokoro_tts
        result = kokoro_tts(text=text, voice=voice, speed=speed, model_name=_KOKORO_REPO_ID, **params)
        return result_to_wav(result)

    def _get_voices(model):
        try:
            from tts_webui_extension.kokoro.main import get_voices as _get
            return [{"value": v, "label": v} for v in _get()]
        except Exception:
            return [{"value": v, "label": v} for v in _FALLBACK_VOICES]

    thread = setup_oai_server(
        tts_fn=_tts,
        get_voices_fn=_get_voices,
        port=_PORT,
        # Intentionally skip registration — testing the server itself only.
    )

    ok = _wait_ready(_BASE)
    print(f"  {_PASS if ok else _FAIL}: server ready at {_BASE}")
    results.append(ok)
    if not ok:
        print("Server did not start in time — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  STEP 2 — GET voices                                                #
    # ------------------------------------------------------------------ #
    print("STEP 2 — GET /v1/audio/voices/kokoro")
    r = requests.get(f"{_BASE}/v1/audio/voices/kokoro")
    body = r.json()
    ok = r.status_code == 200 and "voices" in body
    n_voices = len(body.get("voices", []))
    print(f"  {_PASS if ok else _FAIL}: status={r.status_code}, voices={n_voices}")
    results.append(ok)

    # ------------------------------------------------------------------ #
    #  STEP 3 — POST /v1/audio/speech                                     #
    # ------------------------------------------------------------------ #
    print("STEP 3 — POST /v1/audio/speech")
    r = requests.post(
        f"{_BASE}/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello from the harness test.",
            "voice": "af_sky",
            "speed": 1.0,
        },
        timeout=60,
    )
    ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("audio")
    print(f"  {_PASS if ok else _FAIL}: status={r.status_code}, bytes={len(r.content)}")
    results.append(ok)

    # ------------------------------------------------------------------ #
    #  STEP 4 — validate WAV                                              #
    # ------------------------------------------------------------------ #
    print("STEP 4 — validate WAV")
    import numpy as np
    from scipy.io import wavfile

    buf = io.BytesIO(r.content)
    sr, data = wavfile.read(buf)
    duration = len(data) / sr
    ok = sr > 0 and len(data) > 0 and duration > 0.1
    print(f"  {_PASS if ok else _FAIL}: sample_rate={sr}, samples={len(data)}, duration={duration:.2f}s")
    results.append(ok)

    # ------------------------------------------------------------------ #
    #  STEP 5 — save WAV                                                  #
    # ------------------------------------------------------------------ #
    print("STEP 5 — save WAV")
    out_path = os.path.join(_HERE, "output_harness_kokoro.wav")
    with open(out_path, "wb") as f:
        f.write(r.content)
    ok = os.path.getsize(out_path) > 0
    print(f"  {_PASS if ok else _FAIL}: saved {os.path.getsize(out_path):,} bytes to {out_path}")
    results.append(ok)

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smoke test for the HTTP proxy TTS registration system.

The test spins up a *mock* downstream server (no ML dependencies required)
and exercises the full proxy flow:

  1. POST /tts_webui/proxy/register   — register the mock downstream
  2. GET  /tts_webui/proxy/register   — list registrations
  3. GET  /v1/audio/voices/{model}    — voices fetched live from downstream
  4. POST /v1/audio/speech            — blocking speech proxied to downstream
  5. POST /v1/audio/speech (stream)   — streaming speech proxied to downstream
  6. DELETE /tts_webui/proxy/register/{model} — unregister

The test can run in two modes:

  Standalone (no main server required):
      python tests/smoke_test_proxy.py

  Against a live main server (port 7778):
      python tests/smoke_test_proxy.py --live

In standalone mode the test imports the FastAPI app directly and uses
FastAPI's TestClient so nothing needs to be running.
"""

import argparse
import io
import sys
import os

# Ensure the workspace source tree takes precedence over any installed version
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import threading
import time

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from scipy.io import wavfile

# ─── Mock downstream app ─────────────────────────────────────────────────────

_MOCK_PORT = 19877  # Arbitrary free port for the standalone mock server
_MOCK_VOICES = [{"value": "af_bella", "label": "af_bella"}]

_downstream = FastAPI(title="Mock downstream TTS")


def _make_silence_wav(sample_rate: int = 22050, duration_secs: float = 0.5) -> bytes:
    """Return a minimal valid WAV (silence) — no ML required."""
    n = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, np.zeros(n, dtype=np.int16))
    buf.seek(0)
    return buf.read()


@_downstream.post("/v1/audio/speech")
async def _mock_speech():
    return Response(content=_make_silence_wav(), media_type="audio/wav")


@_downstream.get("/v1/audio/voices/{model}")
async def _mock_voices(model: str):
    return {"voices": _MOCK_VOICES}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ok(msg: str):
    print(f"  ✓ {msg}")


def _fail(msg: str):
    print(f"  ✗ {msg}")
    sys.exit(1)


# ─── Test suite ──────────────────────────────────────────────────────────────

def run_tests(client, base_url: str, model_name: str):
    """
    Execute all proxy smoke tests.

    *client* is either a requests.Session (live mode) or a TestClient (standalone).
    *base_url* is the API root (e.g. "http://localhost:7778" or "" for TestClient).
    """

    # 1. Register
    print("[1] Register proxy model")
    reg_body = {
        "model": model_name,
        "url": f"http://localhost:{_MOCK_PORT}",
        "api_key": None,
    }
    r = client.post(f"{base_url}/tts_webui/proxy/register", json=reg_body)
    if r.status_code != 201:
        _fail(f"Expected 201, got {r.status_code}: {r.text}")
    _ok(f"Registered '{model_name}' (status {r.status_code})")

    # 2. List registrations
    print("[2] List registrations")
    r = client.get(f"{base_url}/tts_webui/proxy/register")
    if r.status_code != 200:
        _fail(f"Expected 200, got {r.status_code}: {r.text}")
    data = r.json()
    names = [p["model"] for p in data["registrations"]]
    if model_name not in names:
        _fail(f"'{model_name}' not found in registrations: {names}")
    _ok(f"Listed registrations: {names}")

    # 3. Voices (proxied live from downstream)
    print("[3] Get voices (proxied)")
    r = client.get(f"{base_url}/v1/audio/voices/{model_name}")
    if r.status_code != 200:
        _fail(f"Expected 200, got {r.status_code}: {r.text}")
    voices = r.json().get("voices", [])
    if not voices:
        _fail("No voices returned from proxy")
    _ok(f"Voices: {[v.get('value') or v.get('id') for v in voices]}")

    # 4. Blocking speech
    print("[4] POST /v1/audio/speech (blocking)")
    speech_body = {
        "model": model_name,
        "input": "Hello from the smoke test.",
        "voice": "af_bella",
        "response_format": "wav",
        "stream": False,
    }
    r = client.post(f"{base_url}/v1/audio/speech", json=speech_body)
    if r.status_code != 200:
        _fail(f"Expected 200, got {r.status_code}: {r.text}")
    if len(r.content) < 44:  # WAV header is 44 bytes at minimum
        _fail(f"Audio response too small ({len(r.content)} bytes)")
    _ok(f"Received {len(r.content)} bytes of audio")

    # 5. Streaming speech
    print("[5] POST /v1/audio/speech (stream=True)")
    speech_body_stream = {**speech_body, "stream": True}
    # requests.Session supports stream=True; TestClient (httpx) does not.
    is_requests_session = hasattr(client, "mount")  # requests.Session-specific method
    if is_requests_session:
        r = client.post(f"{base_url}/v1/audio/speech", json=speech_body_stream, stream=True)
        if r.status_code != 200:
            _fail(f"Expected 200, got {r.status_code}: {r.text}")
        chunks = list(r.iter_content(chunk_size=4096))
        total = sum(len(c) for c in chunks)
    else:
        r = client.post(f"{base_url}/v1/audio/speech", json=speech_body_stream)
        if r.status_code != 200:
            _fail(f"Expected 200, got {r.status_code}: {r.text}")
        total = len(r.content)
    if total < 44:
        _fail(f"Streaming audio too small ({total} bytes)")
    _ok(f"Received {total} bytes via streaming")

    # 6. Unregister
    print("[6] DELETE /tts_webui/proxy/register/{model}")
    r = client.delete(f"{base_url}/tts_webui/proxy/register/{model_name}")
    if r.status_code != 200:
        _fail(f"Expected 200, got {r.status_code}: {r.text}")
    _ok(f"Unregistered '{model_name}'")

    # 7. Verify 404 after unregister
    print("[7] Verify 404 on second DELETE")
    r = client.delete(f"{base_url}/tts_webui/proxy/register/{model_name}")
    if r.status_code != 404:
        _fail(f"Expected 404 after unregister, got {r.status_code}")
    _ok("Second DELETE returned 404 as expected")

    print("\nAll proxy smoke tests passed.")


# ─── Standalone mode: start mock downstream + use TestClient for main app ────

def _start_mock_downstream():
    """Run the mock downstream server in a background thread."""
    config = uvicorn.Config(_downstream, host="127.0.0.1", port=_MOCK_PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait until the server is ready
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{_MOCK_PORT}/v1/audio/voices/test", timeout=1)
            return server
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Mock downstream did not start in time")


def run_standalone():
    """Run tests using the FastAPI TestClient — no external process needed."""
    from tts_webui_extension.openai_tts_api.router import app as main_app

    print(f"Starting mock downstream on :{_MOCK_PORT} ...")
    server = _start_mock_downstream()
    print("Mock downstream ready.")

    # TestClient wraps the ASGI app synchronously.  stream=True requests go
    # through iter_content normally.
    with TestClient(main_app, raise_server_exceptions=True) as client:
        run_tests(client, base_url="", model_name="test-proxied-tts")


# ─── Live mode: main server must already be running on :7778 ─────────────────

def run_live(host: str = "http://localhost:7778"):
    print(f"Starting mock downstream on :{_MOCK_PORT} ...")
    _start_mock_downstream()
    print(f"Testing against live server at {host}")

    session = requests.Session()

    # Tiny wrapper so run_tests can call .post(..., stream=True)
    class _LiveClient:
        def post(self, url, **kwargs):
            return session.post(host + url, **kwargs)

        def get(self, url, **kwargs):
            return session.get(host + url, **kwargs)

        def delete(self, url, **kwargs):
            return session.delete(host + url, **kwargs)

    run_tests(_LiveClient(), base_url="", model_name="test-proxied-tts")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Test against a running server on localhost:7778 instead of TestClient",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:7778",
        help="Base URL when --live is used",
    )
    args = parser.parse_args()

    if args.live:
        run_live(args.host)
    else:
        run_standalone()

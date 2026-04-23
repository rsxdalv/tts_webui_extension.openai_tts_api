"""harness.py — turn any TTS function + voice getter into an OpenAI-compatible server.

This is the core reusable piece.  Everything else in this package is either a
concrete backend (``tts.py``) or wiring (``server.py``, ``register.py``).

Usage
-----
::

    from tts_webui_extension.proxied_tts.harness import create_app, serve

    def my_tts(model: str, text: str, voice: str, speed: float, params: dict) -> bytes:
        '''Return WAV bytes.'''
        ...

    def my_voices(model: str) -> list[dict]:
        '''Return [{"value": "...", "label": "..."}, ...].'''
        return [{"value": "default", "label": "Default"}]

    app = create_app(my_tts, my_voices)

    if __name__ == "__main__":
        serve(app)

API surface exposed by the returned app
----------------------------------------
``POST /v1/audio/speech``
    Body: ``{"model": str, "input": str, "voice": str, "speed": float,
    "response_format": str, "stream": bool, "params": dict}``

    Response: ``audio/wav`` bytes.

``GET /v1/audio/voices/{model}``
    Response: ``{"voices": [{"value": str, "label": str}, ...]}``
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    model: str = Field(default="kokoro")
    input: str
    voice: str = Field(default="default")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = Field(default="wav")
    params: Optional[dict] = None
    stream: bool = Field(default=False)


def create_app(
    tts_fn: Callable[[str, str, str, float, dict], bytes],
    get_voices_fn: Callable[[str], list],
    api_key: Optional[str] = None,
    on_startup: Optional[Callable] = None,
    on_shutdown: Optional[Callable] = None,
) -> FastAPI:
    """Create and return a minimal OpenAI-compatible TTS FastAPI app.

    Parameters
    ----------
    tts_fn:
        A callable with signature ``(model, text, voice, speed, params) -> bytes``
        that returns raw WAV audio bytes.
    get_voices_fn:
        A callable with signature ``(model) -> list[dict]`` that returns a list
        of ``{"value": ..., "label": ...}`` dicts.
    api_key:
        Optional bearer token.  If *None*, no authentication is required.
        Can also be set or overridden later via the ``PROXIED_TTS_API_KEY``
        environment variable.
    on_startup:
        Optional zero-argument callable invoked (in a thread) after the server
        starts accepting connections.  Useful for self-registration.
    on_shutdown:
        Optional zero-argument callable invoked (in a thread) before the server
        shuts down.  Useful for self-unregistration.
    """
    _key = api_key or os.environ.get("PROXIED_TTS_API_KEY") or None

    @asynccontextmanager
    async def _lifespan(app):
        if on_startup:
            await asyncio.to_thread(on_startup)
        yield
        if on_shutdown:
            await asyncio.to_thread(on_shutdown)

    app = FastAPI(
        title="Proxied TTS Server",
        description="Minimal OpenAI-compatible TTS endpoint.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if not _key:
            return await call_next(request)
        if request.url.path in ("/", "/docs", "/openapi.json"):
            return await call_next(request)
        auth = request.headers.get("authorization") or ""
        provided: Optional[str] = None
        if auth.lower().startswith("bearer "):
            provided = auth.split(" ", 1)[1].strip()
        if not provided:
            provided = request.query_params.get("api_key")
        if provided == _key:
            return await call_next(request)
        return Response(status_code=401, content=b"Unauthorized")

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        try:
            wav = tts_fn(req.model, req.input, req.voice, req.speed, req.params or {})
            return Response(content=wav, media_type="audio/wav")
        except Exception as exc:
            logger.exception("tts_fn raised an error")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/v1/audio/voices/{model}")
    async def voices(model: str):
        try:
            return {"voices": get_voices_fn(model)}
        except Exception as exc:
            logger.exception("get_voices_fn raised an error")
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def serve(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 12345,
) -> None:
    """Run *app* with uvicorn.  Blocks until the server exits.

    CLI args ``--host`` and ``--port`` override the defaults when the module
    is run as ``__main__``.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Proxied TTS server")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)

"""harness — wrap any TTS backend in a non-blocking OpenAI-compatible HTTP server.

This package is the public hook that TTS engine providers import to expose
their model through the tts-webui OpenAI-compatible API.

    from tts_webui_extension.openai_tts_api.harness import setup_oai_server

    setup_oai_server(
        tts_fn=my_tts,
        get_voices_fn=my_voices,
        model="my-model",
        register_with="http://localhost:7778",
    )

The server starts in a background daemon thread and returns immediately.
It exposes the standard OpenAI-compatible TTS endpoints::

    POST /v1/audio/speech
    GET  /v1/audio/voices/{model}

If *model* and *register_with* are provided the server auto-registers on
startup and auto-unregisters on shutdown.

Port selection
--------------
The server port defaults to ``GRADIO_SERVER_PORT`` env var (default 27770)
plus 2000, giving 29770 by default.  Pass *port* explicitly to override.

Sub-modules
-----------
``harness.app``
    ``create_app`` factory and blocking ``serve`` helper.
``harness.register``
    ``register`` / ``unregister`` HTTP helpers for the main server.
"""

import logging
import os
import threading
from typing import Callable, Optional

from .app import create_app, serve
from .helpers import result_to_wav
from .register import register, unregister

__all__ = ["setup_oai_server", "create_app", "serve", "register", "unregister", "result_to_wav"]

logger = logging.getLogger(__name__)

_DEFAULT_GRADIO_PORT = 27770
_PORT_OFFSET = 2000
_DEFAULT_UPSTREAM = "http://localhost:7778"


def setup_oai_server(
    tts_fn: Callable,
    get_voices_fn: Callable,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    register_with: Optional[str] = None,
    main_api_key: Optional[str] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
) -> threading.Thread:
    """Start a non-blocking OpenAI-compatible TTS server in a daemon thread.

    Parameters
    ----------
    tts_fn:
        ``(model, text, voice, speed, params) -> bytes`` — returns WAV bytes.
    get_voices_fn:
        ``(model) -> list[dict]`` — returns ``[{"value": ..., "label": ...}]``.
    api_key:
        Optional bearer token required by this server.
    model:
        Model name to register under on the main server.  If *None* or
        *register_with* is *None*, auto-registration is skipped.
    register_with:
        Base URL of the main tts-webui server, e.g. ``"http://localhost:7778"``.
    main_api_key:
        Optional bearer token for the main server's registration endpoint.
    host:
        Host to bind to (default ``"0.0.0.0"``).
    port:
        Port to listen on.  Defaults to ``GRADIO_SERVER_PORT + 2000``
        (i.e. 29770 when ``GRADIO_SERVER_PORT`` is not set).

    Returns
    -------
    threading.Thread
        The daemon thread running the uvicorn server.  Call
        ``thread.join()`` if you need the process to block until exit.
    """
    gradio_port = int(os.environ.get("GRADIO_SERVER_PORT", _DEFAULT_GRADIO_PORT))
    port = port if port is not None else (gradio_port + _PORT_OFFSET)
    register_with = register_with or os.environ.get("OPENAI_PROXY_HOST", _DEFAULT_UPSTREAM)

    on_startup: Optional[Callable] = None
    on_shutdown: Optional[Callable] = None

    if model and register_with:
        _server_url = f"http://localhost:{port}"

        def on_startup():
            register(
                model=model,
                url=_server_url,
                api_key=api_key,
                host=register_with,
                main_api_key=main_api_key,
            )
            logger.info("Registered model '%s' at %s", model, register_with)

        def on_shutdown():
            unregister(model=model, host=register_with, main_api_key=main_api_key)
            logger.info("Unregistered model '%s' from %s", model, register_with)

    import uvicorn

    app = create_app(
        tts_fn,
        get_voices_fn,
        api_key=api_key,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        daemon=True,
        name=f"uvicorn-{model or 'tts'}-{port}",
    )
    thread.start()
    logger.info("TTS server starting on http://%s:%d", host, port)
    return thread

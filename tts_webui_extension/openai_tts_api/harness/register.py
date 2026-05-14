"""register.py — HTTP helpers for registering with the main tts-webui server.

Usage
-----
::

    from tts_webui_extension.openai_tts_api.harness.register import register, unregister

    register(
        model="my-model",
        url="http://localhost:29770",
        api_key="optional-downstream-key",
        host="http://localhost:7778",
        main_api_key="optional-main-key",
    )

    unregister(model="my-model", host="http://localhost:7778")
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def register(
    model: str,
    url: str,
    api_key: Optional[str] = None,
    host: str = "http://localhost:7778",
    main_api_key: Optional[str] = None,
) -> dict:
    """Register a downstream TTS server with the main OpenAI TTS API server.

    Parameters
    ----------
    model:
        The model name callers will use in ``POST /v1/audio/speech``.
    url:
        Base URL of the downstream server (e.g. ``http://localhost:29770``).
    api_key:
        Bearer token required by the downstream server (forwarded by the proxy).
    host:
        Base URL of the main API server (default: ``http://localhost:7778``).
    main_api_key:
        Bearer token for the main API server (if auth is configured there).

    Returns
    -------
    dict
        The JSON response from the main server.
    """
    headers: dict = {"Content-Type": "application/json"}
    if main_api_key:
        headers["Authorization"] = f"Bearer {main_api_key}"

    payload = {"model": model, "url": url, "api_key": api_key}
    try:
        response = requests.post(
            f"{host}/tts_webui/proxy/register",
            json=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        logger.info("Registered '%s' -> %s: %s", model, url, result)
        print(f"Registered '{model}' -> {url}: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error("Failed to register '%s': %s", model, e)
        raise


def unregister(
    model: str,
    host: str = "http://localhost:7778",
    main_api_key: Optional[str] = None,
) -> dict:
    """Unregister a previously registered proxy model.

    Parameters
    ----------
    model:
        The model name that was registered.
    host:
        Base URL of the main API server.
    main_api_key:
        Bearer token for the main API server (if auth is configured there).

    Returns
    -------
    dict
        The JSON response from the main server.
    """
    headers: dict = {}
    if main_api_key:
        headers["Authorization"] = f"Bearer {main_api_key}"

    response = requests.delete(
        f"{host}/tts_webui/proxy/register/{model}",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    logger.info("Unregistered '%s': %s", model, result)
    return result

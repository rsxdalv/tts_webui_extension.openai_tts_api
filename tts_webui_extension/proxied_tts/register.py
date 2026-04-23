"""register.py — helper for registering a proxied TTS server with the main API.

Usage (Python)
--------------
::

    from tts_webui_extension.proxied_tts.register import register

    register(
        model="my-proxied-kokoro",
        url="http://localhost:12345",
        api_key="optional-downstream-key",      # sent to the downstream
        host="http://localhost:7778",            # main API server
        main_api_key="optional-main-key",       # main API bearer token
    )

Usage (CLI)
-----------
::

    python -m tts_webui_extension.proxied_tts.register \\
        --model my-proxied-kokoro \\
        --url http://localhost:12345 \\
        --api-key optional-downstream-key \\
        --host http://localhost:7778 \\
        --main-api-key optional-main-key
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
        Base URL of the downstream server (e.g. ``http://localhost:12345``).
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
    response = requests.post(
        f"{host}/tts_webui/proxy/register",
        json=payload,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    logger.info("Registered '%s' -> %s: %s", model, url, result)
    return result


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register a proxied TTS server")
    parser.add_argument("--model", required=True, help="Model name to register")
    parser.add_argument("--url", required=True, help="Downstream server base URL")
    parser.add_argument("--api-key", default=None, help="Downstream bearer token")
    parser.add_argument("--host", default="http://localhost:7778", help="Main API server URL")
    parser.add_argument("--main-api-key", default=None, help="Main API bearer token")
    parser.add_argument("--unregister", action="store_true", help="Unregister instead of register")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.unregister:
        result = unregister(args.model, host=args.host, main_api_key=args.main_api_key)
    else:
        result = register(
            args.model,
            args.url,
            api_key=args.api_key,
            host=args.host,
            main_api_key=args.main_api_key,
        )
    print(result)

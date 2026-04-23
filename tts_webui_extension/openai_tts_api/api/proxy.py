"""Proxy registration endpoints.

Routes
------
POST   /tts_webui/proxy/register          — register a downstream TTS server
GET    /tts_webui/proxy/register          — list all registered proxies
DELETE /tts_webui/proxy/register/{model}  — unregister a proxy

Authentication is enforced by the global middleware already mounted on the
FastAPI app, so no per-route auth code is needed here.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.proxy_registry import (
    ProxyRegistration,
    list_proxies,
    register_proxy,
    unregister_proxy,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts_webui/proxy", tags=["proxy"])


class RegisterRequest(BaseModel):
    """Body for POST /tts_webui/proxy/register."""

    model: str
    url: str
    api_key: Optional[str] = None


@router.post("/register", status_code=201)
async def register_proxy_endpoint(body: RegisterRequest):
    """Register a downstream HTTP TTS server as a named model.

    The *model* name is used as the ``model`` field in subsequent
    ``POST /v1/audio/speech`` requests routed through the proxy.
    """
    try:
        reg = ProxyRegistration(model=body.model, url=body.url, api_key=body.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    register_proxy(reg)
    return {"registered": body.model, "url": reg.url}


@router.get("/register")
async def list_registrations():
    """List all currently registered proxy models (API keys are not exposed)."""
    return {
        "registrations": [
            {
                "model": p.model,
                "url": p.url,
                "has_api_key": p.api_key is not None,
            }
            for p in list_proxies()
        ]
    }


@router.delete("/register/{model}")
async def unregister_proxy_endpoint(model: str):
    """Unregister a proxied model by name."""
    if not unregister_proxy(model):
        raise HTTPException(
            status_code=404, detail=f"No proxy registered for model '{model}'"
        )
    return {"unregistered": model}

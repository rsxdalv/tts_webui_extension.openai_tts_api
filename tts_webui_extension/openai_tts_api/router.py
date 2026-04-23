import logging
import os
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from tts_webui.config.config_utils import get_config_value

# Import route modules
from .api import audio, models, info, proxy

logger = logging.getLogger(__name__)


# Create the FastAPI app
def _get_expected_api_key() -> Optional[str]:
    # Priority: env var, then config value (saved via UI), else None (no auth)
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    cfg_key = get_config_value("extension_openai_tts_api", "api_key", None)
    if cfg_key:
        try:
            return str(cfg_key).strip() if cfg_key else None
        except Exception:
            return None
    return None


async def _auth_middleware(request: Request, call_next):
    # Allow CORS preflight and public docs without auth
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path in ("/", "/docs", "/openapi.json"):
        return await call_next(request)

    expected = _get_expected_api_key()
    # If no key configured anywhere, allow all (backward compatible)
    if not expected:
        return await call_next(request)

    # Check standard OpenAI style Authorization: Bearer <key>
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    provided: Optional[str] = None
    if auth_header and auth_header.lower().startswith("bearer "):
        provided = auth_header.split(" ", 1)[1].strip()

    # Also allow query param ?api_key=... for simple tests
    if not provided:
        provided = request.query_params.get("api_key")

    if provided == expected:
        return await call_next(request)

    # Return 401; CORS headers will be added by CORSMiddleware when possible
    return Response(status_code=401, content=b"Unauthorized: invalid API key")


app = FastAPI(
    title="OpenAI-Compatible TTS API",
    description="A FastAPI implementation of an OpenAI-compatible Text-to-Speech endpoint",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key auth middleware (env var OPENAI_API_KEY or config api_key)
@app.middleware("http")
async def _auth_middleware_entry(request: Request, call_next):
    return await _auth_middleware(request, call_next)

# Include route modules
app.include_router(audio.router)
app.include_router(models.router)
app.include_router(info.router)
app.include_router(proxy.router)


# OpenAI-compatible error response model
class ErrorResponse(BaseModel):
    error: dict


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch

    apply_torch_load_patch()
    uvicorn.run(app, host="0.0.0.0", port=7778)
class ErrorResponse(BaseModel):
    error: dict


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch

    apply_torch_load_patch()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7778)

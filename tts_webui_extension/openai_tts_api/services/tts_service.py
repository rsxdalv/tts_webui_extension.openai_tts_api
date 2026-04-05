"""
Text-to-speech service for handling speech generation.
"""
import logging
import tempfile
from typing import Callable, Iterator

import ffmpeg

from ..models import CreateSpeechRequest, ResponseFormatEnum

logger = logging.getLogger(__name__)


def generate_speech_stream(request: CreateSpeechRequest) -> Iterator[bytes]:
    """Generate speech as a stream of audio chunks."""
    text = request.input
    model = request.model
    params = request.params or {}

    logger.info(f"[generate_speech_stream] model={model} voice={request.voice} speed={request.speed} params={params}")

    return _call_tts_streaming_adapter(model, text, request.voice, request.speed, params)


def generate_speech(request: CreateSpeechRequest) -> bytes:
    """Generate speech as a single audio file (non-streaming)."""
    text = request.input
    model = request.model
    params = request.params or {}

    logger.info(f"[generate_speech] model={model} voice={request.voice} speed={request.speed} params={params}")

    if model == "global_preset":
        result = _preset_adapter(request, text)
    else:
        result = _call_tts_adapter(model, text, request.voice, request.speed, params)

    if params.get("rvc_params"):
        result = rvc_adapter(result, params["rvc_params"])

    result = webui_to_wav(result)
    return result


def get_content_type(format: ResponseFormatEnum) -> str:
    """Get the MIME type for the specified audio format"""
    content_types = {
        ResponseFormatEnum.MP3: "audio/mpeg",
        ResponseFormatEnum.OPUS: "audio/opus",
        ResponseFormatEnum.AAC: "audio/aac",
        ResponseFormatEnum.FLAC: "audio/flac",
        ResponseFormatEnum.WAV: "audio/wav",
        ResponseFormatEnum.PCM: "audio/pcm",
    }
    return content_types.get(format, "application/octet-stream")


# TTS Adapters -----------------------------------------------------------------------
# Signature: (model: str, text: str, voice: str | None, speed: float | None, params: dict) -> audio_out dict
# Adapters handle their own param translation; ``params`` carries model-specific extras only.
# All adapters live in adapters.py and auto-register on import.

from . import adapters  # noqa: F401 — triggers adapter + streaming-adapter registration
from .tts_adapter_registry import (
    register_tts_adapter,
    unregister_tts_adapter,
    _call_tts_adapter,
    register_tts_streaming_adapter,
    unregister_tts_streaming_adapter,
    _call_tts_streaming_adapter,
)

def _generic_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    """Proxy to ``_call_tts_adapter`` for backward compatibility with ``_preset_adapter``."""
    return _call_tts_adapter(model, text, voice, speed, params)


def _preset_adapter(request: CreateSpeechRequest, text):
    from ..utils import preset_manager
    
    params_preset = preset_manager.get_preset(request.model, request.voice)
    params = params_preset.get("params", {})
    model = params_preset.get("model", None)
    rvc_params = params_preset.get("rvc_params", {})

    audio_result = _generic_tts_adapter(model, text, request.voice, request.speed, params)

    if rvc_params:
        return rvc_adapter(audio_result, rvc_params)
    else:
        return audio_result


def rvc_adapter(audio_result, rvc_params):
    import tempfile
    import os

    try:
        from tts_webui_extension.rvc.rvc_tab import run_rvc
    except ImportError:
        raise ImportError(
            "RVC extension is not installed. Please install it to use RVC voice conversion features."
        )

    audio = webui_to_wav(audio_result)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        temp_file.write(audio)
        temp_file.flush()
        temp_file.close()

        audio_file = temp_file.name
        return run_rvc(original_audio_path=audio_file, **rvc_params)
    finally:
        try:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {temp_file.name}: {e}")


def webui_to_wav(result):
    sample_rate, audio_data = result["audio_out"]
    return to_wav(sample_rate, audio_data)


def to_wav(sample_rate, audio_data):
    from scipy.io import wavfile
    import numpy as np
    import io

    buffer = io.BytesIO()

    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        wavfile.write(buffer, sample_rate, audio_data.astype(np.float32))
    else:
        if audio_data.dtype != np.int16:
            if np.max(np.abs(audio_data)) > 32767:
                audio_data = audio_data * (32767 / np.max(np.abs(audio_data)))
            audio_data = audio_data.astype(np.int16)
        wavfile.write(buffer, sample_rate, audio_data)

    buffer.seek(0)
    return buffer.read()


def convert_audio_format(audio_data: bytes, format: ResponseFormatEnum) -> bytes:
    """Convert audio data to the specified format"""
    if format == ResponseFormatEnum.WAV:
        return audio_data

    format_str = format.value.lower()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
        temp_in_path = temp_in.name
        temp_in.write(audio_data)
        temp_in.flush()

    temp_out_path = temp_in_path.replace(".wav", f".{format_str}")

    try:
        (
            ffmpeg.input(temp_in_path)
            .output(temp_out_path, format=format_str, strict=-2)
            .run(quiet=False, overwrite_output=True)
        )

        with open(temp_out_path, "rb") as f:
            converted_data = f.read()

        return converted_data
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        return audio_data
    finally:
        try:
            import os
            if os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if os.path.exists(temp_out_path):
                os.unlink(temp_out_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary files: {e}")

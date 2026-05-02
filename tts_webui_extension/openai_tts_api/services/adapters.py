"""TTS adapter implementations. Each adapter auto-registers on import."""
from typing import Iterator

from tts_webui_extension.openai_tts_api.services.tts_adapter_registry import (
    register_tts_adapter,
    register_tts_streaming_adapter,
)


def _kokoro_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.kokoro.main import tts

    return tts(text=text, voice=voice, speed=speed, model_name=model, **params)


register_tts_adapter("kokoro", _kokoro_adapter)
register_tts_adapter("hexgrad/Kokoro-82M", _kokoro_adapter)


def _chatterbox_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.chatterbox.api import tts

    audio_prompt_path = None if voice == "random" else voice

    if "chunked" not in params:
        params["chunked"] = True
    return tts(text, audio_prompt_path=audio_prompt_path, **params)

register_tts_adapter("chatterbox", _chatterbox_adapter)


def _styletts2_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.styletts2.main import tts

    return tts(text, voice=voice, speed=speed, **params)


register_tts_adapter("styletts2", _styletts2_adapter)


def _f5_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.f5_tts.gradio_app import infer_decorated as tts

    return tts(text, voice=voice, speed=speed, **params)


register_tts_adapter("f5-tts", _f5_tts_adapter)


def _vall_e_x_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.vall_e_x.api import tts

    return tts(
        text=text,
        prompt=params.get("prompt", ""),
        language=params.get("language", "English"),
        accent=params.get("accent", "no-accent"),
        mode=params.get("mode", "short"),
        **params,
    )


register_tts_adapter("vall-e-x", _vall_e_x_adapter)


def _chatterbox_streaming_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> Iterator[bytes]:
    from tts_webui_extension.chatterbox.api import tts_stream
    from ..utils import to_wav_streaming_header

    header_sent = False
    sample_rate = None

    stream_params = dict(params)
    if voice is not None:
        stream_params["audio_prompt_path"] = None if voice == "random" else voice
    if speed is not None and "speed" not in stream_params:
        stream_params["speed"] = speed

    for partial_result in tts_stream(text, **stream_params):
        if partial_result and "audio_out" in partial_result:
            current_sample_rate, audio_data = partial_result["audio_out"]

            if not header_sent:
                sample_rate = current_sample_rate
                header = to_wav_streaming_header(sample_rate)
                yield header
                header_sent = True

            import numpy as np

            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype != np.int16:
                if np.max(np.abs(audio_data)) > 32767:
                    audio_data = audio_data * (32767 / np.max(np.abs(audio_data)))
                audio_data = audio_data.astype(np.int16)

            yield audio_data.tobytes()


register_tts_streaming_adapter("chatterbox", _chatterbox_streaming_adapter)

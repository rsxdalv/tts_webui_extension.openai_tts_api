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


def _kitten_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.kitten_tts.api import tts

    return tts(model_name="KittenML/kitten-tts-mini-0.1", text=text, voice=voice, speed=speed, **params)


register_tts_adapter("kitten-tts", _kitten_tts_adapter)


def _chatterbox_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.chatterbox.api import tts

    audio_prompt_path = None if voice == "random" else voice
    return tts(text, audio_prompt_path=audio_prompt_path, chunked=True, **params)


register_tts_adapter("chatterbox", _chatterbox_adapter)


def _styletts2_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.styletts2.main import tts

    return tts(text, voice=voice, speed=speed, **params)


register_tts_adapter("styletts2", _styletts2_adapter)


def _f5_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.f5_tts.gradio_app import infer_decorated as tts

    return tts(text, voice=voice, speed=speed, **params)


register_tts_adapter("f5-tts", _f5_tts_adapter)


def _piper_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.piper_tts.api import tts

    length_scale = (1.0 / speed) if speed else 1.0
    return tts(
        text=text,
        voice_name=voice,
        length_scale=length_scale,
        noise_scale=params.get("noise_scale", 0.667),
        noise_w=params.get("noise_w", 0.8),
        sentence_silence=params.get("sentence_silence", 0.2),
        **params,
    )


register_tts_adapter("piper-tts", _piper_tts_adapter)


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


def _parler_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.parler_tts.api import tts

    return tts(
        text=text,
        description=params.get("description", "A neutral voice."),
        model_name=params.get("model_name", "parler-tts/parler-tts-mini-v1"),
        attn_implementation=params.get("attn_implementation", "eager"),
        compile_mode=params.get("compile_mode", None),
        **params,
    )


register_tts_adapter("parler-tts", _parler_tts_adapter)


def _megatts3_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.megatts3.tts import tts

    return tts(
        target_text=text,
        reference_audio_path=params.get("reference_audio_path", ""),
        latent_npy_path=params.get("latent_npy_path", ""),
        inference_steps=params.get("inference_steps", 32),
        intelligibility_weight=params.get("intelligibility_weight", 0.8),
        similarity_weight=params.get("similarity_weight", 0.8),
    )


register_tts_adapter("megatts3", _megatts3_adapter)


def _fireredtts2_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.fireredtts2.api import tts

    return tts(
        text=text,
        temperature=params.get("temperature", 0.9),
        topk=params.get("topk", 30),
        prompt_wav=params.get("prompt_wav"),
        prompt_text=params.get("prompt_text"),
        model_name=params.get("model_name", "monologue"),
        device=params.get("device", "cuda"),
    )


register_tts_adapter("fireredtts2", _fireredtts2_adapter)


def _higgs_v2_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.higgs_v2.api import tts

    return tts(
        text=text,
        temperature=params.get("temperature", 0.8),
        audio_prompt_path=params.get("audio_prompt_path"),
        seed=params.get("seed", -1),
        scene_description=params.get("scene_description"),
    )


register_tts_adapter("higgs_v2", _higgs_v2_adapter)


def _mms_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.mms.api import tts

    return tts(
        text=text,
        language=params.get("language", "eng"),
        speaking_rate=speed,
        noise_scale=params.get("noise_scale", 0.667),
        noise_scale_duration=params.get("noise_scale_duration", 0.8),
        **params,
    )


register_tts_adapter("mms", _mms_adapter)


def _maha_tts_adapter(model: str, text: str, voice: str | None, speed: float | None, params: dict) -> dict:
    from tts_webui_extension.maha_tts.api import tts

    return tts(
        text=text,
        model_name=params.get("model_name", "Smolie-in"),
        text_language=params.get("text_language", "english"),
        speaker_name=voice,
        device=params.get("device", "auto"),
    )


register_tts_adapter("maha_tts", _maha_tts_adapter)


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

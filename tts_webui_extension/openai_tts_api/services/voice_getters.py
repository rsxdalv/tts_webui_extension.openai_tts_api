"""Built-in voice getter functions. Each registers on import."""

import logging
import os

from tts_webui_extension.openai_tts_api.services.voice_service import (
    register_voice_getter,
)

logger = logging.getLogger(__name__)


def _get_kokoro_voices() -> list[dict]:
    """Get available Kokoro voices."""
    try:
        from tts_webui_extension.kokoro.CHOICES import CHOICES

        return [{"label": key, "value": value} for key, value in CHOICES.items()]
    except ImportError:
        logger.warning("Kokoro extension not available")
        return []


register_voice_getter("kokoro", _get_kokoro_voices)
register_voice_getter("hexgrad/Kokoro-82M", _get_kokoro_voices)


def _get_chatterbox_voices() -> list[dict]:
    """Get available Chatterbox voices."""
    default = [{"label": "Random", "value": "random"}]
    try:
        chatterbox_dir = "voices/chatterbox"
        if os.path.exists(chatterbox_dir):
            default.extend(
                [
                    {
                        "label": file.replace(".wav", ""),
                        "value": f"voices/chatterbox/{file}",
                    }
                    for file in os.listdir(chatterbox_dir)
                    if file.endswith(".wav")
                ]
            )
        return default
    except Exception as e:
        logger.warning(f"Could not get chatterbox voices: {e}")
        return default


register_voice_getter("chatterbox", _get_chatterbox_voices)


def _get_global_preset_voices() -> list[dict]:
    try:
        from ..utils import preset_manager

        return preset_manager.get_all_presets()
    except Exception as e:
        logger.warning(f"Could not get global preset voices: {e}")
        return []


register_voice_getter("global_preset", _get_global_preset_voices)


def _get_f5_tts_voices() -> list[dict]:
    try:
        f5_dir = "voices/f5-tts"
        if os.path.exists(f5_dir):
            return [
                {"label": file.replace(".wav", ""), "value": f"{f5_dir}/{file}"}
                for file in os.listdir(f5_dir)
                if file.endswith(".wav")
            ]
        return []
    except Exception as e:
        logger.warning(f"Could not get F5-TTS voices: {e}")
        return []


register_voice_getter("f5-tts", _get_f5_tts_voices)


def _get_styletts2_voices() -> list[dict]:
    try:
        styletts2_dir = "voices/styletts2"
        if os.path.exists(styletts2_dir):
            return [
                {"label": file.replace(".wav", ""), "value": f"{styletts2_dir}/{file}"}
                for file in os.listdir(styletts2_dir)
                if file.endswith(".wav")
            ]
        return []
    except Exception as e:
        logger.warning(f"Could not get StyleTTS2 voices: {e}")
        return []


register_voice_getter("styletts2", _get_styletts2_voices)


def _get_vall_e_x_voices() -> list[dict]:
    return []


register_voice_getter("vall-e-x", _get_vall_e_x_voices)


def _get_higgs_v2_voices() -> list[dict]:
    return []


register_voice_getter("higgs_v2", _get_higgs_v2_voices)


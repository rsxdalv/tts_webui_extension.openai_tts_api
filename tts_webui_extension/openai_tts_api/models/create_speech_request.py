from enum import Enum
from typing import Any, Dict, Union, Optional, Union

from pydantic import BaseModel, Field, validator
from .response_format import ResponseFormatEnum


class ModelEnum(str, Enum):
    KOKORO = "hexgrad/Kokoro-82M"
    CHATTERBOX = "chatterbox"
    GLOBAL_PRESET = "global_preset"
    STYLETTS2 = "styletts2"
    F5_TTS = "f5-tts"
    PIPER_TTS = "piper-tts"
    VALL_E_X = "vall-e-x"
    PARLER_TTS = "parler-tts"
    MEGATTS3 = "megatts3"
    HIGGS_V2 = "higgs_v2"
    MMS = "mms"
    MAHA_TTS = "maha_tts"


class CreateSpeechRequest(BaseModel):
    model: Union[ModelEnum, str] = Field(
        ..., description="One of the available TTS models"
    )
    input: str = Field(
        ..., description="The text to generate audio for", max_length=4096
    )
    voice: str = Field(..., description="The voice to use when generating the audio")
    response_format: ResponseFormatEnum = Field(
        default=ResponseFormatEnum.WAV, description="The format to audio in"
    )
    speed: float = Field(
        default=1.0, description="The speed of the generated audio", ge=0.25, le=4.0
    )
    instructions: Optional[str] = Field(
        None,
        description="Control the voice of your generated audio with additional instructions",
        max_length=4096,
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional parameters for the TTS engine"
    )
    stream: bool = Field(
        default=True, description="Whether to stream the audio response"
    )

    @validator("instructions")
    def validate_instructions(cls, v, values):
        if v is not None and values.get("model") in [
            ModelEnum.TTS_1,
            ModelEnum.TTS_1_HD,
        ]:
            raise ValueError("Instructions do not work with 'tts-1' or 'tts-1-hd'")
        return v

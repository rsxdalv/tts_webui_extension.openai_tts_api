import gradio as gr

from .models.create_speech_request import CreateSpeechRequest
from .services.tts_service import _preset_adapter


def extra_functions_ui():
    def get_chatterbox_voices():
        import os

        voices = []
        for file in os.listdir("voices/chatterbox"):
            if file.endswith(".wav"):
                voices.append(file)
        return voices

    gr.Button("Get Chatterbox Voices").click(
        fn=get_chatterbox_voices,
        outputs=[gr.JSON()],
        api_name="get_chatterbox_voices",
    )

    def test_api_with_open_ai(params):
        request = CreateSpeechRequest(**params)
        text = request.input
        result = _preset_adapter(request, text)
        return result["audio_out"]

    gr.Button("Test Voice").click(
        fn=test_api_with_open_ai,
        inputs=[gr.JSON()],
        outputs=[gr.Audio()],
        api_name="open_ai_api_test_voice_preset",
    )

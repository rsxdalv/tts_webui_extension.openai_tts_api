"""api_caller_ui.py — Gradio tab for calling the TTS API interactively."""

import json
import logging

import gradio as gr

logger = logging.getLogger(__name__)


def _call(model, text, voice, speed, response_format, extra_json):
    from .services.tts_service import generate_speech
    from .models.create_speech_request import CreateSpeechRequest

    try:
        extra = json.loads(extra_json) if extra_json and extra_json.strip() else {}
    except json.JSONDecodeError as e:
        raise gr.Error(f"Extra parameters JSON is invalid: {e}")

    # Normalise empty string from Gradio to None
    voice_value = voice.strip() if voice else None
    if not voice_value or voice_value.lower() == "none":
        voice_value = None

    logger.info(
        "[api_caller_ui] model=%s voice=%r speed=%s format=%s extra=%s",
        model, voice_value, speed, response_format, extra,
    )

    request = CreateSpeechRequest(
        model=model,
        input=text,
        voice=voice_value,
        speed=speed,
        response_format=response_format,
        params=extra if extra else None,
    )

    logger.info("[api_caller_ui] resolved request.voice=%r", request.voice)

    try:
        wav_bytes = generate_speech(request)
    except Exception as e:
        logger.exception("[api_caller_ui] generate_speech failed")
        raise gr.Error(str(e))

    import io
    return io.BytesIO(wav_bytes)


def render_api_caller_ui():
    gr.Markdown("## API Caller")
    gr.Markdown(
        "Call the TTS backend directly. Bypasses the HTTP server — works even if the API is not started."
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Standard Parameters")
            model = gr.Textbox(
                label="Model",
                placeholder="e.g. hexgrad/Kokoro-82M, chatterbox, piper-tts …",
            )
            text = gr.Textbox(
                label="Input text",
                lines=4,
                placeholder="Enter the text to synthesise.",
            )
            with gr.Row():
                voice = gr.Textbox(
                    label="Voice (optional)",
                    placeholder="e.g. af_sky",
                )
                speed = gr.Slider(
                    label="Speed",
                    minimum=0.25,
                    maximum=4.0,
                    step=0.05,
                    value=1.0,
                )
            response_format = gr.Dropdown(
                label="Response format",
                choices=["wav", "mp3", "opus", "aac", "flac", "pcm"],
                value="wav",
            )

            gr.Markdown("### Extra Parameters (JSON)")
            extra_json = gr.Code(
                label="params (JSON object)",
                language="json",
                value="{}",
                lines=6,
            )

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            audio_out = gr.Audio(label="Generated audio", type="filepath")

    generate_btn.click(
        fn=_call,
        inputs=[model, text, voice, speed, response_format, extra_json],
        outputs=[audio_out],
        api_name="api_caller_generate",
    )

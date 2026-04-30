"""api_caller_ui.py — Gradio tab for calling the live HTTP API."""

import io
import json
import logging
import os

import gradio as gr

logger = logging.getLogger(__name__)


def _call(host, port, api_key, model, text, voice, speed, response_format, extra_json):
    import requests
    from tts_webui.config.config_utils import get_config_value

    try:
        extra = json.loads(extra_json) if extra_json and extra_json.strip() else {}
    except json.JSONDecodeError as e:
        raise gr.Error(f"Extra parameters JSON is invalid: {e}")

    voice_value = voice.strip() if voice else None
    if not voice_value or voice_value.lower() == "none":
        voice_value = None

    host_val = host.strip() or "localhost"
    if host_val == "0.0.0.0":
        host_val = "localhost"
    port_val = int(port) if port else get_config_value("extension_openai_tts_api", "port", 7778)

    url = f"http://{host_val}:{port_val}/v1/audio/speech"

    key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY") or get_config_value("extension_openai_tts_api", "api_key", None)
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    payload = {
        "model": model,
        "input": text,
        "voice": voice_value,
        "speed": speed,
        "response_format": response_format,
        "params": extra if extra else None,
    }

    logger.info("[api_caller_ui] POST %s  model=%s voice=%r speed=%s", url, model, voice_value, speed)

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise gr.Error(f"Could not connect to {url} — is the API server running?")
    except requests.exceptions.HTTPError as e:
        logger.error("[api_caller_ui] HTTP %s: %s", response.status_code, response.text)
        raise gr.Error(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        logger.exception("[api_caller_ui] request failed")
        raise gr.Error(str(e))

    logger.info("[api_caller_ui] response %s bytes", len(response.content))
    return io.BytesIO(response.content)


def render_api_caller_ui():
    from tts_webui.config.config_utils import get_config_value

    gr.Markdown("## API Caller")
    gr.Markdown("Send a request to the running HTTP API server.")

    with gr.Row():
        host = gr.Textbox(
            label="Host",
            value=lambda: get_config_value("extension_openai_tts_api", "host", "0.0.0.0"),
            scale=2,
        )
        port = gr.Number(
            label="Port",
            value=lambda: get_config_value("extension_openai_tts_api", "port", 7778),
            scale=1,
        )
        api_key = gr.Textbox(
            label="API Key (optional)",
            type="password",
            placeholder="Leave blank to use saved key or OPENAI_API_KEY env var",
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Request")
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
        inputs=[host, port, api_key, model, text, voice, speed, response_format, extra_json],
        outputs=[audio_out],
        api_name="api_caller_generate",
    )

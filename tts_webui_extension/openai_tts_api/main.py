import os
import gradio as gr

from tts_webui.config.config_utils import get_config_value, set_config_value

from .migrate_config import migrate_config
from .extension_metadata import get_metadata


# from .threader import (
#     activate_api,
#     deactivate_api,
#     get_api_status,
# )


def activate_api(host=None, port=None):
    host = host or get_config_value("extension_openai_tts_api", "host", "0.0.0.0")
    port = port or get_config_value("extension_openai_tts_api", "port", 7778)
    from .router import app

    import uvicorn

    import threading

    threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port},
        daemon=True,
    ).start()
    return {"status": "success"}


def deactivate_api():
    return {"status": "not implemented"}


def get_api_status():
    return {"status": "not implemented"}


def test_api(host=None, port=None):
    host = host or get_config_value("extension_openai_tts_api", "host", "0.0.0.0")
    port = port or get_config_value("extension_openai_tts_api", "port", 7778)
    import requests
    # Include Authorization header if API key is configured (env has priority)
    api_key = os.environ.get("OPENAI_API_KEY") or get_config_value(
        "extension_openai_tts_api", "api_key", None
    )

    if host == "0.0.0.0":
        host = "localhost"

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    response = requests.post(
        f"http://{host}:{port}/v1/audio/speech",
        headers=headers,
        json={
            "model": "hexgrad/Kokoro-82M",
            "input": "Hello world with custom parameters.",
            "voice": "af_heart",
            "speed": 1.0,
            # "params": {
            #     "pitch_up_key": "2",
            #     "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
            # },
        },
    )
    audio = response.content
    return audio


def test_api_with_open_ai(host=None, port=None):
    host = host or get_config_value("extension_openai_tts_api", "host", "0.0.0.0")
    port = port or get_config_value("extension_openai_tts_api", "port", 7778)
    from openai import OpenAI

    if host == "0.0.0.0":
        host = "localhost"

    import os
    api_key = os.environ.get("OPENAI_API_KEY") or get_config_value(
        "extension_openai_tts_api", "api_key", "sk-1234567890"
    )
    client = OpenAI(api_key=api_key, base_url=f"http://{host}:{port}/v1")

    with client.audio.speech.with_streaming_response.create(
        model="hexgrad/Kokoro-82M",
        voice="af_heart",
        input="Today is a wonderful day to build something people love!",
        extra_body={
            "params": {
                "use_gpu": True,
                # "rvc_params": {
                #     "pitch_up_key": "0",
                #     "index_path": "CaitArcane\\added_IVF65_Flat_nprobe_1_CaitArcane_v2",
                #     "pitch_collection_method": "harvest",
                #     "model_path": "CaitArcane\\CaitArcane",
                #     "index_rate": 0.66,
                #     "filter_radius": 3,
                #     "resample_sr": 0,
                #     "rms_mix_rate": 1,
                #     "protect": 0.33,
                # },
            },
        },
    ) as response:
        import io

        audio = io.BytesIO(response.read())
        return audio.getvalue()


def presets_ui():
    import json

    from .utils.presets import preset_manager

    presets = preset_manager.get_presets()
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Presets")
            presets_json = gr.JSON(value=presets, label="Presets")
            presets_input = gr.TextArea(
                value=json.dumps(presets, indent=4),
                lines=12,
                label="Edit Presets",
                interactive=True,
            )
            gr.Button("Save Presets").click(
                fn=lambda x: preset_manager.set_presets(json.loads(x)),
                inputs=[presets_input],
                outputs=[presets_json],
                api_name="open_ai_api_save_presets",
            )
            gr.Button("Load Presets").click(
                fn=lambda: preset_manager.get_presets(),
                outputs=[presets_json],
                api_name="open_ai_api_load_presets",
            )


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
        from .services.tts_service import _preset_adapter
        from .models.create_speech_request import CreateSpeechRequest

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


def ui():
    gr.Markdown("# OpenAI TTS API")
    with gr.Tabs():
        with gr.Tab("Startup"):
            startup_ui()
        with gr.Tab("Presets"):
            presets_ui()
        with gr.Tab("Extra Functions (Backend)"):
            extra_functions_ui()


def startup_ui():
    with gr.Row():
        with gr.Column():
            url = f"""localhost:{get_config_value("extension_openai_tts_api", "port", 7778)}"""
            gr.Markdown(
                f"""
                This extension adds an OpenAI compatible API endpoint for TTS models. You can use this to generate audio from text.
                
                To use the API, you need to activate it. This will start the API server and you can then use the API endpoint.
                
                The default API endpoint is http://{url}/v1/audio/speech
                
                The default OpenAI API Base_url is http://{url}/v1/
                
                Authentication: You can protect the API with an API key. Set the OPENAI_API_KEY environment variable or save a key below. If no key is set, the API is open.
                """
            )

            host = gr.Textbox(
                label="Host",
                value=lambda: get_config_value(
                    "extension_openai_tts_api", "host", "0.0.0.0"
                ),
            )
            port = gr.Number(
                label="Port",
                value=lambda: get_config_value(
                    "extension_openai_tts_api", "port", 7778
                ),
            )
            host.change(
                fn=lambda x: set_config_value("extension_openai_tts_api", "host", x),
                inputs=[host],
                outputs=[],
            )
            port.change(
                fn=lambda x: set_config_value("extension_openai_tts_api", "port", x),
                inputs=[port],
                outputs=[],
            )

            api_key = gr.Textbox(
                label="API Key (OpenAI-compatible)",
                type="password",
                value=lambda: get_config_value("extension_openai_tts_api", "api_key", ""),
                placeholder="Leave blank for no auth. Env var OPENAI_API_KEY overrides.",
            )
            api_key.change(
                fn=lambda x: set_config_value("extension_openai_tts_api", "api_key", x or None),
                inputs=[api_key],
                outputs=[],
            )

            activate_api_btn = gr.Button("Activate API")
            activate_api_btn.click(
                fn=activate_api,
                inputs=[host, port],
                outputs=[gr.JSON()],
                api_name="activate_api",
            )

            auto_start_api = gr.Checkbox(
                label="Auto start API",
                value=lambda: get_config_value(
                    "extension_openai_tts_api", "auto_start", False
                ),
            )
            auto_start_api.change(
                fn=lambda x: set_config_value(
                    "extension_openai_tts_api", "auto_start", x
                ),
                inputs=[auto_start_api],
                outputs=[],
            )

            # deactivate_api_btn = gr.Button("Deactivate API")
            # deactivate_api_btn.click(
            #     fn=deactivate_api,
            #     inputs=[],
            #     outputs=[gr.JSON()],
            #     api_name="deactivate_api",
            # )

            # get_api_status_btn = gr.Button("Get API Status")
            # get_api_status_btn.click(
            #     fn=get_api_status,
            #     inputs=[],
            #     outputs=[gr.JSON()],
            #     api_name="get_api_status",
            # )
        with gr.Column():
            test_button = gr.Button("Test API with Python requests")
            test_button.click(
                fn=test_api,
                inputs=[host, port],
                outputs=[gr.Audio()],
                api_name="test_api",
            )

            with gr.Accordion("Code Snippets", open=False):
                gr.Markdown(
                    f"""
            ```python
            import requests

            response = requests.post(
                "http://{url}/v1/audio/speech",
                headers={{"Authorization": f"Bearer YOUR_API_KEY"}},  # remove if no key
                json={{
                    "model": "hexgrad/Kokoro-82M",
                    "input": "Hello world with custom parameters.",
                    "voice": "af_heart",
                    "speed": 1.0,
                    "params": {{
                        "pitch_up_key": "2",
                        "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
                    }},
                }},
            )
            audio = response.content
            with open("audio.mp3", "wb") as f:
                f.write(audio)
            ```
            """
                )

            test_with_open_ai = gr.Button("Test with Python OpenAI client")
            test_with_open_ai.click(
                fn=test_api_with_open_ai,
                inputs=[host, port],
                outputs=[gr.Audio()],
                api_name="test_api_with_open_ai",
            )

            with gr.Accordion("Code Snippets", open=False):
                gr.Markdown(
                    f"""
            ```python
            from openai import OpenAI

            client = OpenAI(api_key="YOUR_API_KEY", base_url="http://{url}/v1")

            with client.audio.speech.with_streaming_response.create(
                model="hexgrad/Kokoro-82M",
                voice="af_heart",
                input="Today is a wonderful day to build something people love!",
                extra_body={{
                    "params": {{
                        "use_gpu": True,
                    }},
                }},
            ) as response:
                audio = response.read()
                with open("audio.mp3", "wb") as f:
                    f.write(audio)
            ```
            """
                )


def extension__tts_generation_webui():
    """Extension entry point."""
    migrate_config()
    ui()

    ENV_AUTO_ACTIVATE_OPENAI_API = os.environ.get("AUTO_ACTIVATE_OPENAI_API", "0")
    config_auto_activate_openai_api = get_config_value(
        "extension_openai_tts_api", "auto_start", False
    )

    if ENV_AUTO_ACTIVATE_OPENAI_API == "1" or config_auto_activate_openai_api:
        activate_api()

    return get_metadata()


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch
    apply_torch_load_patch()

    if "demo" in locals():
        locals()["demo"].close()  # type: ignore
    with gr.Blocks() as demo:
        extension__tts_generation_webui()

    demo.launch(
        server_port=7770,
    )

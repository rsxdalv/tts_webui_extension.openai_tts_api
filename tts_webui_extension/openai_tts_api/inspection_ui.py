"""inspection_ui.py — Gradio tab for inspecting the live backend state."""

import gradio as gr


def _get_backend_state() -> dict:
    from .services.tts_adapter_registry import _TTS_ADAPTERS, _TTS_STREAMING_ADAPTERS
    from .services.voice_service import _VOICE_GETTERS
    from .services.proxy_registry import _PROXY_REGISTRATIONS

    models_inprocess = sorted(set(_TTS_ADAPTERS) | set(_TTS_STREAMING_ADAPTERS))
    models_proxied = sorted(_PROXY_REGISTRATIONS)

    adapters = []
    for model in models_inprocess:
        adapters.append({
            "model": model,
            "type": "in-process",
            "blocking": model in _TTS_ADAPTERS,
            "streaming": model in _TTS_STREAMING_ADAPTERS,
        })
    for model in models_proxied:
        reg = _PROXY_REGISTRATIONS[model]
        adapters.append({
            "model": model,
            "type": "proxy",
            "url": reg.url,
            "auth": bool(reg.api_key),
        })

    voice_getters = sorted(_VOICE_GETTERS)

    return {
        "adapters": adapters,
        "voice_getters": voice_getters,
    }


def _get_voices_for_model(model: str) -> list[dict]:
    if not model:
        return []
    from .services.voice_service import get_voices_by_model
    try:
        return get_voices_by_model(model)
    except Exception as e:
        return [{"error": str(e)}]


def render_inspection_ui():
    gr.Markdown("## Backend Inspection")
    gr.Markdown(
        "Live view of registered TTS adapters, proxy registrations, and available voices."
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh", variant="secondary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Registered Adapters")
            adapters_json = gr.JSON(label="Adapters", value=lambda: _get_backend_state()["adapters"])

        with gr.Column():
            gr.Markdown("### Voice Getters")
            voice_getters_json = gr.JSON(label="Voice getter models", value=lambda: _get_backend_state()["voice_getters"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Voices per Model")
            model_input = gr.Textbox(
                label="Model",
                placeholder="Type a model name and click Check Voices",
            )
            check_voices_btn = gr.Button("Check Voices", variant="secondary")
            voices_json = gr.JSON(label="Voices")
            check_voices_btn.click(
                fn=_get_voices_for_model,
                inputs=[model_input],
                outputs=[voices_json],
            )

    def _refresh():
        state = _get_backend_state()
        return state["adapters"], state["voice_getters"]

    refresh_btn.click(
        fn=_refresh,
        outputs=[adapters_json, voice_getters_json],
    )

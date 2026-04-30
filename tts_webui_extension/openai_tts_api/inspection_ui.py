"""inspection_ui.py — Gradio tab for inspecting the live backend state."""

import gradio as gr


def _format_adapters(adapters: list[dict]) -> str:
    if not adapters:
        return "No adapters registered."
    lines = ["| Model | Type | Blocking | Streaming | URL | Auth |", "|---|---|---|---|---|---|"]
    for a in adapters:
        model = a.get("model", "")
        adapter_type = a.get("type", "")
        blocking = "✓" if a.get("blocking") else ""
        streaming = "✓" if a.get("streaming") else ""
        url = a.get("url", "")
        auth = "✓" if a.get("auth") else ""
        lines.append(f"| {model} | {adapter_type} | {blocking} | {streaming} | {url} | {auth} |")
    return "\n".join(lines)


def _format_voice_getters(voice_getters: list[str]) -> str:
    if not voice_getters:
        return "No voice getters registered."
    lines = ["| Voice Getter Model |", "|---|"]
    for v in voice_getters:
        lines.append(f"| {v} |")
    return "\n".join(lines)


def _format_voices(voices: list[dict]) -> str:
    if not voices:
        return "No voices found."
    if any(v.get("error") for v in voices):
        return f"Error: {voices[0]['error']}"
    lines = ["| Voice ID | Name | Language | Gender |", "|---|---|---|---|"]
    for v in voices:
        lines.append(f"| {v.get('voice_id', '')} | {v.get('name', '')} | {v.get('language', '')} | {v.get('gender', '')} |")
    return "\n".join(lines)


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


def _get_voices_for_model(model: str) -> tuple[list[dict], str]:
    if not model:
        return [], ""
    from .services.voice_service import get_voices_by_model
    try:
        voices = get_voices_by_model(model)
        return voices, _format_voices(voices)
    except Exception as e:
        return [{"error": str(e)}], f"Error: {str(e)}"


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
            adapters_md = gr.Markdown(label="Adapters (formatted)")

        with gr.Column():
            gr.Markdown("### Voice Getters")
            voice_getters_json = gr.JSON(label="Voice getter models", value=lambda: _get_backend_state()["voice_getters"])
            voice_getters_md = gr.Markdown(label="Voice getter models (formatted)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Voices per Model")
            model_input = gr.Textbox(
                label="Model",
                placeholder="Type a model name and click Check Voices",
            )
            check_voices_btn = gr.Button("Check Voices", variant="secondary")
            voices_json = gr.JSON(label="Voices")
            voices_md = gr.Markdown(label="Voices (formatted)")
            check_voices_btn.click(
                fn=_get_voices_for_model,
                inputs=[model_input],
                outputs=[voices_json, voices_md],
            )

    def _refresh():
        state = _get_backend_state()
        adapters = state["adapters"]
        voice_getters = state["voice_getters"]
        return (
            adapters, _format_adapters(adapters),
            voice_getters, _format_voice_getters(voice_getters),
        )

    refresh_btn.click(
        fn=_refresh,
        outputs=[adapters_json, adapters_md, voice_getters_json, voice_getters_md],
    )

"""helpers.py — utilities for TTS engine providers using the harness.

The main helper is ``result_to_wav``, which converts the standard tts-webui
result dict ``{"audio_out": (sample_rate, numpy_array)}`` into WAV bytes
suitable for returning from a ``tts_fn`` passed to ``setup_oai_server``.
"""

import io


def result_to_wav(result: dict) -> bytes:
    """Convert a tts-webui result dict to WAV bytes.

    Parameters
    ----------
    result:
        Dict with key ``"audio_out"`` containing a ``(sample_rate, data)``
        tuple, where *data* is a NumPy array (float32, float64, or int16).

    Returns
    -------
    bytes
        Raw WAV file bytes.
    """
    import numpy as np
    from scipy.io import wavfile

    sample_rate, audio_data = result["audio_out"]
    # Normalise layout: scipy expects (samples,) or (samples, channels).
    # Some engines return (channels, samples) — detect by a plausible channel count.
    if audio_data.ndim == 2 and audio_data.shape[0] <= 32:
        audio_data = audio_data.T  # (channels, samples) -> (samples, channels)
    buf = io.BytesIO()
    if audio_data.dtype in (np.float32, np.float64):
        if abs(audio_data).max() > 1.0:
            audio_data = audio_data / abs(audio_data).max()
        wavfile.write(buf, sample_rate, audio_data.astype(np.float32))
    else:
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        wavfile.write(buf, sample_rate, audio_data)
    buf.seek(0)
    return buf.read()

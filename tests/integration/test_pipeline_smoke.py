"""Integration smoke tests — skipped unless real audio fixtures exist.

These tests require heavy ML backends and are marked 'slow'.
Run with: pytest -m slow
"""
import os
import pytest

FIXTURE_WAV = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_16k_mono.wav")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(FIXTURE_WAV), reason="No fixture WAV file present")
def test_enrich_audio_smoke():
    from speechtelemetry import enrich_audio, PipelineConfig

    cfg = PipelineConfig(
        asr_backend="faster-whisper",
        asr_model_size="tiny",
        vad_backend="silero",
        alignment_backend="whisperx",
        diarization_backend=None,
        prosody_backend=[],
        emotion_backend=None,
        device="cpu",
    )
    doc = enrich_audio(FIXTURE_WAV, config=cfg)
    assert doc is not None
    assert doc.duration_s > 0
    assert isinstance(doc.segments, list)

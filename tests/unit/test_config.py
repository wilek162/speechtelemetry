"""Unit tests for speechtelemetry.config — PipelineConfig validation and defaults."""

import pytest
from pydantic import ValidationError

from speechtelemetry.config import PipelineConfig


def test_default_config():
    cfg = PipelineConfig()
    assert cfg.device == "auto"
    assert cfg.asr_backend == "faster-whisper"
    assert cfg.vad_backend == "silero"
    assert cfg.alignment_backend == "whisperx"
    assert cfg.diarization_backend is None
    assert cfg.prosody_backend == []
    assert cfg.emotion_backend == "speechbrain"
    assert cfg.export_formats == ["json"]
    assert cfg.asr_beam_size == 5
    assert cfg.chunk_audio is True


def test_config_is_frozen():
    cfg = PipelineConfig()
    with pytest.raises(ValidationError):
        cfg.device = "cpu"  # type: ignore[misc]


def test_config_custom_device():
    cfg = PipelineConfig(device="cpu")
    assert cfg.device == "cpu"


def test_config_cuda_device():
    cfg = PipelineConfig(device="cuda")
    assert cfg.device == "cuda"


def test_config_invalid_device():
    with pytest.raises(ValidationError):
        PipelineConfig(device="tpu")  # type: ignore[arg-type]


def test_config_auto_device_is_valid():
    cfg = PipelineConfig(device="auto")
    assert cfg.device == "auto"


def test_config_beam_size_bounds():
    with pytest.raises(ValidationError):
        PipelineConfig(asr_beam_size=0)
    with pytest.raises(ValidationError):
        PipelineConfig(asr_beam_size=11)


def test_config_vad_threshold_bounds():
    with pytest.raises(ValidationError):
        PipelineConfig(vad_threshold=-0.1)
    with pytest.raises(ValidationError):
        PipelineConfig(vad_threshold=1.1)


def test_iter_backends_no_diarize_no_emotion():
    cfg = PipelineConfig(diarization_backend=None, emotion_backend=None, prosody_backend=[])
    pairs = cfg.iter_backends()
    stages = [p[0] for p in pairs]
    assert "asr" in stages
    assert "vad" in stages
    assert "alignment" in stages
    assert "diarization" not in stages
    assert "emotion" not in stages
    assert "prosody" not in stages


def test_iter_backends_with_diarize():
    cfg = PipelineConfig(diarization_backend="pyannote")
    pairs = cfg.iter_backends()
    stages = [p[0] for p in pairs]
    assert "diarization" in stages


def test_iter_backends_with_prosody():
    cfg = PipelineConfig(prosody_backend=["parselmouth"])
    pairs = cfg.iter_backends()
    prosody_pairs = [(s, n) for s, n in pairs if s == "prosody"]
    assert ("prosody", "parselmouth") in prosody_pairs


def test_iter_backends_with_emotion():
    cfg = PipelineConfig(emotion_backend="speechbrain")
    pairs = cfg.iter_backends()
    assert ("emotion", "speechbrain") in pairs


def test_config_no_prosody():
    cfg = PipelineConfig(prosody_backend=[])
    assert cfg.prosody_backend == []

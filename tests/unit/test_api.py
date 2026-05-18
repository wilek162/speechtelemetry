"""Unit tests for api.py stage-level functions — no real backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speechtelemetry.config import PipelineConfig
from speechtelemetry.types import Segment


def test_enrich_media_raises_file_not_found():
    from speechtelemetry.api import enrich_media

    with pytest.raises(FileNotFoundError):
        enrich_media("/nonexistent/__speechtelemetry_test__.mp4")


def test_enrich_audio_raises_file_not_found():
    from speechtelemetry.api import enrich_audio

    with pytest.raises(FileNotFoundError):
        enrich_audio("/nonexistent/__speechtelemetry_test__.wav")


def test_run_vad_calls_backend_and_returns_result():
    mock_backend = MagicMock()
    mock_backend.get_speech_intervals.return_value = [{"start": 0.0, "end": 1.5}]
    config = PipelineConfig()

    with patch("speechtelemetry.registry.get_backend", return_value=mock_backend) as mock_get:
        from speechtelemetry.api import run_vad

        result = run_vad("/audio.wav", config)

    mock_get.assert_called_once_with("vad", config.vad_backend)
    mock_backend.get_speech_intervals.assert_called_once_with("/audio.wav")
    assert result == [{"start": 0.0, "end": 1.5}]


def test_run_asr_calls_backend_and_returns_tuple():
    mock_backend = MagicMock()
    mock_segments = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    mock_info = MagicMock()
    mock_backend.transcribe.return_value = (mock_segments, mock_info)
    config = PipelineConfig()

    with patch("speechtelemetry.registry.get_backend", return_value=mock_backend):
        from speechtelemetry.api import run_asr

        segments, info = run_asr("/audio.wav", config)

    assert segments is mock_segments
    assert info is mock_info
    mock_backend.transcribe.assert_called_once_with(
        "/audio.wav",
        language=config.asr_language,
        beam_size=config.asr_beam_size,
    )


def test_run_alignment_calls_backend_and_returns_result():
    mock_backend = MagicMock()
    mock_aligned = [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]
    mock_backend.align.return_value = mock_aligned
    config = PipelineConfig()
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]

    with patch("speechtelemetry.registry.get_backend", return_value=mock_backend):
        from speechtelemetry.api import run_alignment

        result = run_alignment(raw_segments, "/audio.wav", "en", config)

    assert result is mock_aligned
    mock_backend.align.assert_called_once_with(raw_segments, "/audio.wav", "en")


def test_run_diarization_returns_empty_list_when_no_backend():
    config = PipelineConfig(diarization_backend=None)

    from speechtelemetry.api import run_diarization

    result = run_diarization("/audio.wav", config)
    assert result == []


def test_run_diarization_calls_backend_when_configured():
    mock_backend = MagicMock()
    mock_turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    mock_backend.diarize.return_value = mock_turns
    config = PipelineConfig(diarization_backend="pyannote")

    with patch("speechtelemetry.registry.get_backend", return_value=mock_backend):
        from speechtelemetry.api import run_diarization

        result = run_diarization("/audio.wav", config)

    assert result is mock_turns
    mock_backend.diarize.assert_called_once_with(
        "/audio.wav",
        min_speakers=config.diarization_min_speakers,
        max_speakers=config.diarization_max_speakers,
    )


def test_run_prosody_returns_segments_unchanged_when_no_backend():
    config = PipelineConfig(prosody_backend=[])
    segs = [Segment(start=0.0, end=1.0, text="hi", confidence=0.9)]

    from speechtelemetry.api import run_prosody

    result = run_prosody(segs, "/audio.wav", config)
    assert result is segs


def test_run_emotion_returns_segments_unchanged_when_no_backend():
    config = PipelineConfig(emotion_backend=None)
    segs = [Segment(start=0.0, end=1.0, text="hi", confidence=0.9)]

    from speechtelemetry.api import run_emotion

    result = run_emotion(segs, "/audio.wav", config)
    assert result is segs

"""Unit tests for pipeline orchestration — G4 (silence gaps), G8 (provenance), G10 (chunking).

TDD: Tests written before implementation fixes. All use mock backends — no ML, no I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from speechtelemetry.config import PipelineConfig
from speechtelemetry.core.pipeline import (
    _build_segments,
    _compute_silence_spans,
)
from speechtelemetry.types import Segment, SilenceSpan

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_silence_spans() -> list[SilenceSpan]:
    return [
        SilenceSpan(start=0.0, end=0.5, duration_ms=500.0, reason="non_speech"),
        SilenceSpan(start=1.5, end=1.6, duration_ms=100.0, reason="speech_gap"),
        SilenceSpan(start=2.5, end=3.0, duration_ms=500.0, reason="non_speech"),
    ]


# ── _build_segments ───────────────────────────────────────────────────────────


def test_build_segments_creates_word_objects():
    raw = [
        {
            "start": 0.5,
            "end": 1.5,
            "text": "hello world",
            "confidence": 0.95,
            "words": [
                {"word": "hello", "start": 0.5, "end": 0.9, "score": 0.97},
                {"word": "world", "start": 0.9, "end": 1.5, "score": 0.93},
            ],
        }
    ]
    segments = _build_segments(raw, [])
    assert len(segments) == 1
    assert segments[0].words is not None
    assert len(segments[0].words) == 2
    assert segments[0].words[0].text == "hello"


def test_build_segments_assigns_speaker_from_diarization():
    raw = [{"start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
    diarize = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    segments = _build_segments(raw, diarize)
    assert segments[0].speaker == "SPEAKER_00"


def test_build_segments_no_speaker_without_diarization():
    raw = [{"start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
    segments = _build_segments(raw, [])
    assert segments[0].speaker is None


# ── _compute_silence_spans ────────────────────────────────────────────────────


def test_compute_silence_spans_produces_gaps_between_speech():
    speech = [{"start": 0.5, "end": 1.5}, {"start": 1.6, "end": 2.5}]
    spans = _compute_silence_spans(speech, total_duration_s=3.0)
    assert len(spans) == 3
    assert abs(spans[0].start - 0.0) < 0.001
    assert abs(spans[0].end - 0.5) < 0.001
    assert spans[0].reason == "speech_gap"  # leading silence: gap before first speech
    assert abs(spans[1].start - 1.5) < 0.001
    assert abs(spans[1].end - 1.6) < 0.001
    assert spans[1].reason == "speech_gap"


def test_compute_silence_spans_no_trailing_when_speech_reaches_end():
    speech = [{"start": 0.0, "end": 3.0}]
    spans = _compute_silence_spans(speech, total_duration_s=3.0)
    assert len(spans) == 0


# ── G4: _attach_silence_gaps ──────────────────────────────────────────────────


def test_attach_silence_gaps_exists():
    """_attach_silence_gaps must be importable from core.pipeline (G4)."""
    from speechtelemetry.core.pipeline import _attach_silence_gaps  # noqa: F401


def test_attach_silence_gaps_populates_silence_before_ms():
    from speechtelemetry.core.pipeline import _attach_silence_gaps

    segments = [
        Segment(start=0.5, end=1.5, text="hello world", confidence=0.95),
        Segment(start=1.6, end=2.5, text="test audio", confidence=0.88),
    ]
    result = _attach_silence_gaps(segments, _make_silence_spans())
    assert result[0].silence_before_ms is not None
    assert abs(result[0].silence_before_ms - 500.0) < 1.0


def test_attach_silence_gaps_populates_silence_after_ms():
    from speechtelemetry.core.pipeline import _attach_silence_gaps

    segments = [
        Segment(start=0.5, end=1.5, text="hello world", confidence=0.95),
        Segment(start=1.6, end=2.5, text="test audio", confidence=0.88),
    ]
    result = _attach_silence_gaps(segments, _make_silence_spans())
    assert result[-1].silence_after_ms is not None
    assert abs(result[-1].silence_after_ms - 500.0) < 1.0


def test_attach_silence_gaps_returns_same_length():
    from speechtelemetry.core.pipeline import _attach_silence_gaps

    segments = [Segment(start=0.0, end=1.0, text="only", confidence=0.9)]
    result = _attach_silence_gaps(segments, [])
    assert len(result) == 1


def test_attach_silence_gaps_handles_empty_segments():
    from speechtelemetry.core.pipeline import _attach_silence_gaps

    result = _attach_silence_gaps([], _make_silence_spans())
    assert result == []


# ── G8: provenance field on TranscriptDocument ────────────────────────────────


def test_transcript_document_has_provenance_field():
    """TranscriptDocument must expose a provenance attribute (G8)."""
    from speechtelemetry.types import ProcessingReport, TranscriptDocument

    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=3.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )
    assert hasattr(doc, "provenance"), "TranscriptDocument is missing provenance field"


def test_transcript_document_provenance_accepts_pipeline_provenance():
    """provenance field must accept a PipelineProvenance instance."""
    from speechtelemetry.provenance import PipelineProvenance
    from speechtelemetry.types import ProcessingReport, TranscriptDocument

    prov = PipelineProvenance()
    prov.record(stage="vad", backend_name="silero")
    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=3.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
        provenance=prov,
    )
    assert doc.provenance is prov
    assert len(doc.provenance.stages) == 1
    assert doc.provenance.stages[0].stage == "vad"


# ── G10: chunking config not dead ─────────────────────────────────────────────


def test_pipeline_config_chunk_audio_default_true():
    """chunk_audio must default to True (config exists and is meaningful)."""
    cfg = PipelineConfig()
    assert cfg.chunk_audio is True


def test_pipeline_config_max_chunk_duration_default():
    """max_chunk_duration_s must default to 30.0."""
    cfg = PipelineConfig()
    assert cfg.max_chunk_duration_s == 30.0


def test_asr_transcribe_receives_chunk_size_when_chunk_audio_true():
    """When chunk_audio=True, pipeline must pass chunk_size_s to ASR transcribe()."""
    from speechtelemetry.core.pipeline import run_pipeline

    config = PipelineConfig(
        vad_backend="silero",
        asr_backend="faster-whisper",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
        chunk_audio=True,
        max_chunk_duration_s=30.0,
    )

    mock_vad = MagicMock()
    mock_vad.get_speech_intervals.return_value = [{"start": 0.0, "end": 3.0}]

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ([], type("Info", (), {"language": "en"})())

    mock_aligner = MagicMock()
    mock_aligner.align.return_value = []

    def fake_get_backend(stage, name, **kwargs):
        return {"vad": mock_vad, "asr": mock_asr, "alignment": mock_aligner}[stage]

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=fake_get_backend),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        run_pipeline("fake.wav", config, skip_decode=True)

    call_kwargs = mock_asr.transcribe.call_args
    assert call_kwargs is not None, "ASR transcribe() was never called"
    _, kwargs = call_kwargs
    assert (
        "chunk_size_s" in kwargs
    ), f"chunk_size_s not passed to ASR transcribe(). Got kwargs: {list(kwargs.keys())}"

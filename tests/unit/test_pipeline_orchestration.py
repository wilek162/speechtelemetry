"""Unit tests for pipeline orchestration — G4 (silence gaps), G8 (provenance), G10 (chunking).

TDD: Tests written before implementation fixes. All use mock backends — no ML, no I/O.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from speechtelemetry.config import PipelineConfig
from speechtelemetry.core.pipeline import (
    _build_segments,
    _compute_silence_spans,
    preflight_check,
)
from speechtelemetry.exceptions import EnvironmentCheckError
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


def test_build_segments_confidence_from_avg_logprob():
    """avg_logprob (ASR log-prob) must be converted to [0,1] via exp()."""
    import math

    raw = [{"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.3}]
    segments = _build_segments(raw, [])
    expected = math.exp(-0.3)
    assert abs(segments[0].confidence - expected) < 1e-6


def test_build_segments_confidence_from_word_scores_when_no_logprob():
    """When avg_logprob is absent (post-alignment), mean word score is used."""
    raw = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hi there",
            "words": [
                {"word": "hi", "start": 0.0, "end": 0.4, "score": 0.8},
                {"word": "there", "start": 0.4, "end": 1.0, "score": 0.9},
            ],
        }
    ]
    segments = _build_segments(raw, [])
    assert abs(segments[0].confidence - 0.85) < 1e-6


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
    assert "chunk_size_s" in kwargs, (
        f"chunk_size_s not passed to ASR transcribe(). Got kwargs: {list(kwargs.keys())}"
    )


# ── Config kwargs wired to backend constructors ───────────────────────────────


def _run_pipeline_capturing_get_backend_calls(config: PipelineConfig) -> list:
    """Helper: run pipeline with skip_decode, capture all get_backend(stage, name, **kwargs) calls."""
    from speechtelemetry.core.pipeline import run_pipeline

    calls: list[tuple[str, str, dict]] = []

    def capturing_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        calls.append((stage, name, dict(kwargs)))
        m = MagicMock()
        _seg = [
            {"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.5, "no_speech_prob": 0.1}
        ]
        m.get_speech_intervals.return_value = [{"start": 0.0, "end": 3.0}]
        m.transcribe.return_value = (_seg, type("Info", (), {"language": "en"})())
        m.align.return_value = _seg
        return m

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=capturing_get_backend),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        run_pipeline("fake.wav", config, skip_decode=True)

    return calls


def test_pipeline_passes_model_size_and_device_to_asr_backend():
    """Pipeline must pass asr_model_size, device, and compute_type to the ASR backend constructor."""
    config = PipelineConfig(
        asr_backend="faster-whisper",
        asr_model_size="small",
        device="cpu",
        asr_compute_type="int8",
        vad_backend="silero",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
    )
    calls = _run_pipeline_capturing_get_backend_calls(config)
    asr_calls = [(s, n, kw) for s, n, kw in calls if s == "asr"]
    assert asr_calls, "get_backend was never called for stage 'asr'"
    _, _, kwargs = asr_calls[0]
    assert kwargs.get("model_size") == "small", f"model_size not passed. Got: {kwargs}"
    assert kwargs.get("device") == "cpu", f"device not passed. Got: {kwargs}"
    assert kwargs.get("compute_type") == "int8", f"compute_type not passed. Got: {kwargs}"


def test_pipeline_passes_threshold_to_vad_backend():
    """Pipeline must pass vad_threshold, vad_min_silence_ms, vad_min_speech_ms to the VAD backend."""
    config = PipelineConfig(
        vad_backend="silero",
        vad_threshold=0.7,
        vad_min_silence_ms=500,
        vad_min_speech_ms=150,
        asr_backend="faster-whisper",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
    )
    calls = _run_pipeline_capturing_get_backend_calls(config)
    vad_calls = [(s, n, kw) for s, n, kw in calls if s == "vad"]
    assert vad_calls, "get_backend was never called for stage 'vad'"
    _, _, kwargs = vad_calls[0]
    assert kwargs.get("threshold") == 0.7, f"threshold not passed. Got: {kwargs}"
    assert kwargs.get("min_silence_duration_ms") == 500, (
        f"min_silence_duration_ms not passed. Got: {kwargs}"
    )
    assert kwargs.get("min_speech_duration_ms") == 150, (
        f"min_speech_duration_ms not passed. Got: {kwargs}"
    )


def test_pipeline_passes_device_to_alignment_backend():
    """Pipeline must pass device to the alignment backend constructor."""
    config = PipelineConfig(
        device="cpu",
        alignment_backend="whisperx",
        asr_backend="faster-whisper",
        vad_backend="silero",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
    )
    calls = _run_pipeline_capturing_get_backend_calls(config)
    align_calls = [(s, n, kw) for s, n, kw in calls if s == "alignment"]
    assert align_calls, "get_backend was never called for stage 'alignment'"
    _, _, kwargs = align_calls[0]
    assert kwargs.get("device") == "cpu", f"device not passed to alignment. Got: {kwargs}"


def test_pipeline_passes_device_to_emotion_backend():
    """Pipeline must pass device to the emotion backend constructor."""
    from speechtelemetry.core.pipeline import _attach_emotion

    config = PipelineConfig(
        device="cpu",
        emotion_backend="speechbrain",
        asr_backend="faster-whisper",
        vad_backend="silero",
        alignment_backend="whisperx",
        prosody_backend=[],
        diarization_backend=None,
    )

    captured_kwargs: dict = {}

    def capturing_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        m = MagicMock()
        m.predict_segment.return_value = {
            "label_distribution": {"neutral": 1.0},
            "confidence": 0.9,
            "backend_name": "test",
        }
        return m

    with patch("speechtelemetry.core.pipeline.get_backend", side_effect=capturing_get_backend):
        from speechtelemetry.types import Segment

        _attach_emotion(
            [Segment(start=0.0, end=2.0, text="hi", confidence=0.9)],
            "fake.wav",
            config,
        )

    assert captured_kwargs.get("device") == "cpu", (
        f"device not passed to emotion backend. Got: {captured_kwargs}"
    )


# ── Word.alignment_backend provenance ─────────────────────────────────────────


def test_build_segments_sets_alignment_backend_on_words():
    """Word.alignment_backend must be set when alignment_backend is passed."""
    raw = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "confidence": 0.9,
            "words": [{"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9}],
        }
    ]
    segments = _build_segments(raw, [], alignment_backend="whisperx")
    assert segments[0].words is not None
    assert segments[0].words[0].alignment_backend == "whisperx"


def test_build_segments_alignment_backend_none_by_default():
    """Word.alignment_backend must be None when alignment did not run."""
    raw = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "confidence": 0.9,
            "words": [{"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9}],
        }
    ]
    segments = _build_segments(raw, [], alignment_backend=None)
    assert segments[0].words is not None
    assert segments[0].words[0].alignment_backend is None


# ── Fail-soft: alignment failure ──────────────────────────────────────────────


def test_pipeline_continues_when_alignment_fails():
    """When alignment raises, pipeline must not crash — segments retain ASR data."""
    from speechtelemetry.core.pipeline import run_pipeline

    config = PipelineConfig(
        vad_backend="silero",
        asr_backend="faster-whisper",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
        chunk_audio=False,
    )

    _seg = [{"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": -0.3, "no_speech_prob": 0.1}]

    def fake_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.get_speech_intervals.return_value = [{"start": 0.0, "end": 3.0}]
        m.transcribe.return_value = (_seg, type("Info", (), {"language": "en"})())
        if stage == "alignment":
            m.align.side_effect = RuntimeError("alignment model unavailable")
        return m

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=fake_get_backend),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        doc = run_pipeline("fake.wav", config, skip_decode=True)

    assert len(doc.segments) == 1, "Segments must still be produced after alignment failure"
    assert doc.segments[0].text == "hello"
    alignment_errors = [e for e in doc.processing_report.errors if e.stage == "alignment"]
    assert len(alignment_errors) == 1, "Alignment failure must be recorded in errors"


# ── Fail-soft: emotion failure ────────────────────────────────────────────────


def test_pipeline_continues_when_emotion_fails():
    """When emotion backend raises, pipeline must not crash — segments get no emotion score."""
    from speechtelemetry.core.pipeline import _attach_emotion

    config = PipelineConfig(
        device="cpu",
        emotion_backend="speechbrain",
        asr_backend="faster-whisper",
        vad_backend="silero",
        alignment_backend="whisperx",
        prosody_backend=[],
        diarization_backend=None,
    )

    from speechtelemetry.types import ProcessingReport

    report = ProcessingReport()

    def failing_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.predict_segment.side_effect = RuntimeError("model load failed")
        return m

    with patch("speechtelemetry.core.pipeline.get_backend", side_effect=failing_get_backend):
        from speechtelemetry.types import Segment

        segments = _attach_emotion(
            [Segment(start=0.0, end=2.0, text="hi", confidence=0.9)],
            "fake.wav",
            config,
            report=report,
        )

    assert segments[0].emotion is None, "Emotion must be None after per-segment failure"
    emotion_errors = [e for e in report.errors if e.stage == "emotion"]
    assert len(emotion_errors) == 1, "Emotion failure must be recorded in ProcessingReport.errors"


# ── Provenance: compute_type recorded for ASR ─────────────────────────────────


def test_pipeline_records_compute_type_in_asr_provenance():
    """Provenance must record compute_type when it is set in config."""
    from speechtelemetry.core.pipeline import run_pipeline
    from speechtelemetry.provenance import PipelineProvenance

    config = PipelineConfig(
        asr_backend="faster-whisper",
        asr_model_size="small",
        asr_compute_type="int8",
        device="cpu",
        vad_backend="silero",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
        chunk_audio=False,
    )

    _seg = [{"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.2, "no_speech_prob": 0.1}]

    def fake_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.get_speech_intervals.return_value = [{"start": 0.0, "end": 3.0}]
        m.transcribe.return_value = (_seg, type("Info", (), {"language": "en"})())
        m.align.return_value = _seg
        return m

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=fake_get_backend),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        doc = run_pipeline("fake.wav", config, skip_decode=True)

    assert doc.provenance is not None
    assert isinstance(doc.provenance, PipelineProvenance)
    asr_stages = [s for s in doc.provenance.stages if s.stage == "asr"]
    assert len(asr_stages) == 1, "ASR provenance must be recorded"
    assert asr_stages[0].compute_type == "int8", (
        f"compute_type not recorded. Got: {asr_stages[0].compute_type!r}"
    )


# ── Speaker assignment: overlap-based ─────────────────────────────────────────


def test_build_segments_assigns_speaker_by_highest_overlap():
    """When two turns overlap a segment, the one with the largest overlap wins."""
    raw = [{"start": 0.0, "end": 2.0, "text": "hi", "confidence": 0.9}]
    diarize = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    segments = _build_segments(raw, diarize)
    assert segments[0].speaker == "SPEAKER_00"


def test_build_segments_assigns_speaker_even_when_midpoint_outside_turn():
    """Overlap-based assignment must work even when the segment midpoint is strictly outside the turn.

    Segment [0.5, 1.4] → midpoint = 0.95, turn ends at 0.9 → midpoint is outside.
    But there is 0.4s of overlap [0.5, 0.9] → speaker must be assigned.
    """
    raw = [{"start": 0.5, "end": 1.4, "text": "hey", "confidence": 0.9}]
    diarize = [{"start": 0.0, "end": 0.9, "speaker": "SPEAKER_00"}]
    segments = _build_segments(raw, diarize)
    assert segments[0].speaker == "SPEAKER_00"


def test_build_segments_returns_none_speaker_when_segment_outside_all_turns():
    """Speaker must be None when the segment has no overlap with any diarization turn."""
    raw = [{"start": 5.0, "end": 6.0, "text": "echo", "confidence": 0.5}]
    diarize = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    segments = _build_segments(raw, diarize)
    assert segments[0].speaker is None


# ── Diarization provenance recorded ───────────────────────────────────────────


def test_pipeline_records_diarization_in_provenance():
    """Provenance must record the diarization stage backend name after a successful run."""
    from speechtelemetry.core.pipeline import run_pipeline
    from speechtelemetry.provenance import PipelineProvenance

    config = PipelineConfig(
        asr_backend="faster-whisper",
        asr_model_size="small",
        device="cpu",
        vad_backend="silero",
        alignment_backend="whisperx",
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend="pyannote",
        chunk_audio=False,
    )

    _seg = [{"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.2}]

    def fake_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.get_speech_intervals.return_value = [{"start": 0.0, "end": 3.0}]
        m.transcribe.return_value = (_seg, type("Info", (), {"language": "en"})())
        m.align.return_value = _seg
        m.diarize.return_value = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        return m

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=fake_get_backend),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        doc = run_pipeline("fake.wav", config, skip_decode=True)

    assert doc.provenance is not None
    assert isinstance(doc.provenance, PipelineProvenance)
    diar_stages = [s for s in doc.provenance.stages if s.stage == "diarization"]
    assert len(diar_stages) == 1, "Diarization provenance must be recorded"
    assert diar_stages[0].backend_name == "pyannote"


# ── preflight_check ───────────────────────────────────────────────────────────


def _mock_cls() -> MagicMock:
    """Return a MagicMock that passes _check_available() without raising."""
    cls = MagicMock()
    cls._check_available = MagicMock(return_value=None)
    return cls


def test_preflight_raises_when_ffmpeg_missing():
    config = PipelineConfig(diarization_backend=None, prosody_backend=[], emotion_backend=None)
    with (
        patch("speechtelemetry.core.pipeline.shutil.which", return_value=None),
        patch("speechtelemetry.core.pipeline.resolve_backend", return_value=_mock_cls()),
        pytest.raises(EnvironmentCheckError) as exc_info,
    ):
        preflight_check(config)
    assert "FFmpeg" in str(exc_info.value)


def test_preflight_raises_when_cuda_not_available():
    config = PipelineConfig(
        device="cuda",
        diarization_backend=None,
        prosody_backend=[],
        emotion_backend=None,
    )
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with (
        patch("speechtelemetry.core.pipeline.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.core.pipeline.resolve_backend", return_value=_mock_cls()),
        patch.dict(sys.modules, {"torch": mock_torch}),
        pytest.raises(EnvironmentCheckError) as exc_info,
    ):
        preflight_check(config)
    assert "cuda" in str(exc_info.value).lower()


def test_preflight_raises_when_hf_token_missing_for_diarization():
    config = PipelineConfig(
        diarization_backend="pyannote",
        prosody_backend=[],
        emotion_backend=None,
    )
    env_clean = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGINGFACE_TOKEN")}

    with (
        patch("speechtelemetry.core.pipeline.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.core.pipeline.resolve_backend", return_value=_mock_cls()),
        patch.dict(os.environ, env_clean, clear=True),
        pytest.raises(EnvironmentCheckError) as exc_info,
    ):
        preflight_check(config)
    assert "HF_TOKEN" in str(exc_info.value)


def test_preflight_passes_when_all_checks_satisfied():
    config = PipelineConfig(diarization_backend=None, prosody_backend=[], emotion_backend=None)
    with (
        patch("speechtelemetry.core.pipeline.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.core.pipeline.resolve_backend", return_value=_mock_cls()),
    ):
        preflight_check(config)  # must not raise


# ── run_pipeline export path ──────────────────────────────────────────────────


def _make_minimal_fake_get_backend(exporter: MagicMock) -> object:
    """Return a fake_get_backend side-effect that injects the given exporter."""

    def fake_get_backend(stage: str, name: str, **kwargs: object) -> MagicMock:
        if stage == "exporter":
            return exporter
        m = MagicMock()
        m.get_speech_intervals.return_value = []
        m.transcribe.return_value = ([], None)
        m.align.return_value = []
        return m

    return fake_get_backend


def test_run_pipeline_calls_exporter_when_output_dir_set(tmp_path: object) -> None:
    """Pipeline must call exporter.export() for each export_format when output_dir is given."""
    from speechtelemetry.core.pipeline import run_pipeline

    config = PipelineConfig(
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
        export_formats=["json"],
    )
    mock_exporter = MagicMock()

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch(
            "speechtelemetry.core.pipeline.get_backend",
            side_effect=_make_minimal_fake_get_backend(mock_exporter),
        ),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        run_pipeline("fake.wav", config, skip_decode=True, output_dir=str(tmp_path))

    mock_exporter.export.assert_called_once()


def test_export_failure_recorded_as_stage_error_and_pipeline_continues(
    tmp_path: object,
) -> None:
    """Export failure must be caught, recorded in ProcessingReport, and must not raise."""
    from speechtelemetry.core.pipeline import run_pipeline

    config = PipelineConfig(
        emotion_backend=None,
        prosody_backend=[],
        diarization_backend=None,
        export_formats=["json"],
    )
    mock_exporter = MagicMock()
    mock_exporter.export.side_effect = RuntimeError("disk full")

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch(
            "speechtelemetry.core.pipeline.get_backend",
            side_effect=_make_minimal_fake_get_backend(mock_exporter),
        ),
        patch("speechtelemetry.core.pipeline._get_duration", return_value=3.0),
    ):
        doc = run_pipeline("fake.wav", config, skip_decode=True, output_dir=str(tmp_path))

    export_errors = [e for e in doc.processing_report.errors if "export" in e.stage]
    assert len(export_errors) == 1
    assert "disk full" in export_errors[0].message

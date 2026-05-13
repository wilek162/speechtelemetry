"""Unit tests for speechtelemetry.types — pure data model, zero external deps."""

from speechtelemetry.types import (
    EmotionScore,
    ProcessingReport,
    ProsodyWindow,
    Segment,
    SilenceSpan,
    StageError,
    TranscriptDocument,
    Word,
)


def test_word_instantiation():
    w = Word(text="hello", start=0.0, end=0.5, confidence=0.95)
    assert w.text == "hello"
    assert w.token_id is None
    assert w.alignment_backend is None


def test_prosody_window_defaults():
    pw = ProsodyWindow(f0_mean=120.0, f0_variance=5.0, energy_mean=-20.0, energy_variance=2.0)
    assert pw.speech_rate_sps is None
    assert pw.pause_density is None
    assert pw.backend_name is None


def test_emotion_score_top_label():
    es = EmotionScore(
        label_distribution={"happy": 0.7, "neutral": 0.2, "sad": 0.1},
        confidence=0.85,
        backend_name="speechbrain/test",
    )
    assert es.top_label == "happy"


def test_emotion_score_top_label_single():
    es = EmotionScore(
        label_distribution={"neutral": 1.0},
        confidence=1.0,
        backend_name="test",
    )
    assert es.top_label == "neutral"


def test_silence_span_default_reason():
    ss = SilenceSpan(start=1.0, end=2.0, duration_ms=1000.0)
    assert ss.reason == "non_speech"


def test_silence_span_custom_reason():
    ss = SilenceSpan(start=0.0, end=0.5, duration_ms=500.0, reason="speech_gap")
    assert ss.reason == "speech_gap"


def test_segment_minimal():
    seg = Segment(start=0.0, end=1.0, text="hello", confidence=0.9)
    assert seg.speaker is None
    assert seg.words is None
    assert seg.prosody is None
    assert seg.emotion is None


def test_stage_error_no_segment():
    err = StageError(stage="prosody", message="failed", exception_type="ValueError")
    assert err.segment_index is None


def test_processing_report_defaults():
    report = ProcessingReport()
    assert report.errors == []
    assert report.real_time_factor == 0.0
    assert report.peak_ram_mb == 0.0
    assert isinstance(report.stage_timings, dict)


def test_processing_report_add_error():
    report = ProcessingReport()
    report.add_error("vad", ValueError("test error"))
    assert len(report.errors) == 1
    assert report.errors[0].stage == "vad"
    assert report.errors[0].exception_type == "ValueError"
    assert report.errors[0].message == "test error"


def test_processing_report_add_error_with_segment():
    report = ProcessingReport()
    report.add_error("prosody", RuntimeError("boom"), segment_index=3)
    assert report.errors[0].segment_index == 3


def test_transcript_document_instantiation():
    report = ProcessingReport()
    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=5.0,
        segments=[],
        silence_spans=[],
        processing_report=report,
    )
    assert doc.source_path == "test.wav"
    assert doc.language == "en"
    assert doc.segments == []


def test_transcript_document_null_language():
    doc = TranscriptDocument(
        source_path="x.mp3",
        language=None,
        duration_s=0.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )
    assert doc.language is None

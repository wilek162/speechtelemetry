"""Unit tests for speechtelemetry.types — pure data model, zero external deps."""

from speechtelemetry.types import (
    SCHEMA_VERSION,
    EmotionScore,
    ProcessingReport,
    ProsodyWindow,
    Segment,
    SilenceSpan,
    SpeakerProfile,
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


def test_transcript_document_schema_version_default():
    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=1.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )
    assert doc.schema_version == SCHEMA_VERSION
    assert isinstance(doc.schema_version, str)


def test_transcript_document_speakers_none_by_default():
    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=1.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )
    assert doc.speakers is None


def test_transcript_document_speakers_field_accepts_list():
    profile = SpeakerProfile(
        speaker_id="SPEAKER_00",
        speaking_time_s=10.0,
        turn_count=3,
        word_count=25,
        mean_segment_confidence=0.75,
    )
    doc = TranscriptDocument(
        source_path="test.wav",
        language="en",
        duration_s=15.0,
        segments=[],
        silence_spans=[],
        processing_report=ProcessingReport(),
        speakers=[profile],
    )
    assert doc.speakers is not None
    assert len(doc.speakers) == 1
    assert doc.speakers[0].speaker_id == "SPEAKER_00"


def test_speaker_profile_minimal():
    sp = SpeakerProfile(
        speaker_id="SPEAKER_01",
        speaking_time_s=5.5,
        turn_count=2,
        word_count=12,
        mean_segment_confidence=0.82,
    )
    assert sp.speaker_id == "SPEAKER_01"
    assert sp.dominant_emotion is None
    assert sp.emotion_distribution is None


def test_speaker_profile_with_emotion():
    sp = SpeakerProfile(
        speaker_id="SPEAKER_00",
        speaking_time_s=20.0,
        turn_count=8,
        word_count=85,
        mean_segment_confidence=0.68,
        dominant_emotion="ang",
        emotion_distribution={"neu": 0.1, "ang": 0.7, "hap": 0.1, "sad": 0.1},
    )
    assert sp.dominant_emotion == "ang"
    assert abs(sum(sp.emotion_distribution.values()) - 1.0) < 0.01


def test_schema_version_constant_is_string():
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0

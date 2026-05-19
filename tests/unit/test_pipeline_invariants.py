"""Pipeline invariant tests — properties that must hold for any correct pipeline run.

Runs the full pipeline (mock backends + real 3s WAV fixture) and asserts
structural invariants that must be true regardless of which backends are used.

All tests run with zero ML dependencies. The real WAV is used so that duration,
silence-span computation, and Job lifecycle are exercised against actual file I/O.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speechtelemetry import PipelineConfig
from speechtelemetry.types import ProcessingReport, TranscriptDocument

_FIXTURE_WAV = Path(__file__).parent.parent / "fixtures" / "sample_16k_mono.wav"

# ── Shared fixture ────────────────────────────────────────────────────────────


def _make_backend(stage: str, name: str, **kwargs: object) -> object:
    from tests.fixtures.mock_backends import (
        MockAlignmentBackend,
        MockASRBackend,
        MockEmotionBackend,
        MockVADBackend,
    )

    return {
        "vad": MockVADBackend(),
        "asr": MockASRBackend(),
        "alignment": MockAlignmentBackend(),
        "emotion": MockEmotionBackend(),
    }.get(stage, MagicMock())


_CONFIG_WITH_EMOTION = PipelineConfig(
    device="cpu",
    asr_backend="faster-whisper",
    asr_model_size="tiny",
    asr_language="en",
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,
    prosody_backend=[],
    emotion_backend="speechbrain",
)

_CONFIG_NO_EMOTION = PipelineConfig(
    device="cpu",
    asr_backend="faster-whisper",
    asr_model_size="tiny",
    asr_language="en",
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,
    prosody_backend=[],
    emotion_backend=None,
)


def _run(config: PipelineConfig) -> TranscriptDocument:
    from speechtelemetry import enrich_audio

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=_make_backend),
    ):
        return enrich_audio(str(_FIXTURE_WAV), config=config)


@pytest.fixture(scope="module")
def doc_with_emotion():
    return _run(_CONFIG_WITH_EMOTION)


@pytest.fixture(scope="module")
def doc_no_emotion():
    return _run(_CONFIG_NO_EMOTION)


# ── TranscriptDocument structure ──────────────────────────────────────────────


def test_returns_transcript_document(doc_with_emotion):
    assert isinstance(doc_with_emotion, TranscriptDocument)


def test_source_path_set(doc_with_emotion):
    assert doc_with_emotion.source_path == str(_FIXTURE_WAV)


def test_duration_positive_and_finite(doc_with_emotion):
    d = doc_with_emotion.duration_s
    assert d > 0, f"duration_s={d} must be positive"
    assert math.isfinite(d), "duration_s must be finite"


def test_segments_is_list(doc_with_emotion):
    assert isinstance(doc_with_emotion.segments, list)


def test_silence_spans_is_list(doc_with_emotion):
    assert isinstance(doc_with_emotion.silence_spans, list)


def test_processing_report_present(doc_with_emotion):
    assert isinstance(doc_with_emotion.processing_report, ProcessingReport)


# ── Processing report invariants ──────────────────────────────────────────────


def test_errors_always_a_list(doc_with_emotion):
    """ProcessingReport.errors must always be a list — never None."""
    assert doc_with_emotion.processing_report.errors is not None
    assert isinstance(doc_with_emotion.processing_report.errors, list)


def test_errors_empty_when_mocks_succeed(doc_with_emotion):
    assert doc_with_emotion.processing_report.errors == []


def test_stage_timings_vad_present(doc_with_emotion):
    assert "vad" in doc_with_emotion.processing_report.stage_timings


def test_stage_timings_asr_present(doc_with_emotion):
    assert "asr" in doc_with_emotion.processing_report.stage_timings


def test_stage_timings_all_positive(doc_with_emotion):
    for stage, t in doc_with_emotion.processing_report.stage_timings.items():
        assert t >= 0, f"Stage timing for '{stage}' is negative: {t}"


def test_rtf_is_positive(doc_with_emotion):
    assert doc_with_emotion.processing_report.real_time_factor > 0


# ── Segment timestamp invariants ──────────────────────────────────────────────


def test_segments_ordered_by_start(doc_with_emotion):
    starts = [s.start for s in doc_with_emotion.segments]
    assert starts == sorted(starts), f"Segments are not sorted by start: {starts}"


def test_all_segments_end_after_start(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        assert s.end > s.start, f"seg{i}: end={s.end} <= start={s.start}"


def test_all_segments_start_non_negative(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        assert s.start >= 0, f"seg{i}: start={s.start} < 0"


def test_all_segments_within_audio_duration(doc_with_emotion):
    duration = doc_with_emotion.duration_s
    for i, s in enumerate(doc_with_emotion.segments):
        assert s.end <= duration + 0.5, (
            f"seg{i}: end={s.end:.3f} exceeds duration={duration:.3f} + 0.5s tolerance"
        )


# ── Segment confidence invariants ─────────────────────────────────────────────


def test_segment_confidence_in_range(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        assert 0.0 <= s.confidence <= 1.0, f"seg{i}: confidence={s.confidence} out of [0.0, 1.0]"


# ── Word-level invariants ──────────────────────────────────────────────────────


def test_word_timestamps_valid(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        for j, w in enumerate(s.words or []):
            assert w.end > w.start, f"seg{i} word{j}: end={w.end} <= start={w.start}"


def test_word_confidence_in_range(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        for j, w in enumerate(s.words or []):
            assert 0.0 <= w.confidence <= 1.0, (
                f"seg{i} word{j}: confidence={w.confidence} out of [0,1]"
            )


def test_word_alignment_backend_set_when_alignment_ran(doc_with_emotion):
    """All words must have alignment_backend set (MockAlignmentBackend ran)."""
    all_words = [w for s in doc_with_emotion.segments for w in (s.words or [])]
    if not all_words:
        pytest.skip("No words in output")
    missing = [w for w in all_words if not w.alignment_backend]
    assert not missing, f"{len(missing)} words missing alignment_backend"


# ── Emotion invariants ────────────────────────────────────────────────────────


def test_emotion_none_when_disabled(doc_no_emotion):
    for i, s in enumerate(doc_no_emotion.segments):
        assert s.emotion is None, f"seg{i}: emotion should be None when disabled"


def test_emotion_present_on_eligible_segments(doc_with_emotion):
    """Segments >=0.5s must have emotion when emotion backend is configured."""
    eligible = [s for s in doc_with_emotion.segments if (s.end - s.start) >= 0.5]
    if not eligible:
        pytest.skip("No segments >= 0.5s in mock output")
    missing = [s for s in eligible if s.emotion is None]
    assert not missing, f"{len(missing)}/{len(eligible)} eligible segments missing emotion score"


def test_emotion_distribution_sums_to_one(doc_with_emotion):
    """EmotionScore.label_distribution must sum to approximately 1.0 (softmax output)."""
    for i, s in enumerate(doc_with_emotion.segments):
        if s.emotion is None:
            continue
        total = sum(s.emotion.label_distribution.values())
        assert abs(total - 1.0) < 0.02, (
            f"seg{i}: label_distribution sums to {total:.6f} (expected ~1.0)"
        )


def test_emotion_labels_are_strings(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        if s.emotion is None:
            continue
        for lbl in s.emotion.label_distribution:
            assert isinstance(lbl, str), f"seg{i}: emotion label {lbl!r} is not a string"


def test_emotion_confidence_in_range(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        if s.emotion is None:
            continue
        assert 0.0 <= s.emotion.confidence <= 1.0, (
            f"seg{i}: emotion confidence={s.emotion.confidence} out of [0,1]"
        )


def test_emotion_backend_name_set(doc_with_emotion):
    for i, s in enumerate(doc_with_emotion.segments):
        if s.emotion is None:
            continue
        assert s.emotion.backend_name, f"seg{i}: emotion backend_name is empty"


def test_emotion_mock_has_four_iemocap_labels(doc_with_emotion):
    """Mock emotion backend must return all 4 IEMOCAP labels to match real model."""
    expected = {"neu", "ang", "hap", "sad"}
    for i, s in enumerate(doc_with_emotion.segments):
        if s.emotion is None:
            continue
        labels = set(s.emotion.label_distribution.keys())
        assert labels == expected, (
            f"seg{i}: emotion labels {labels} != expected IEMOCAP labels {expected}"
        )


# ── Provenance invariants ─────────────────────────────────────────────────────


def test_provenance_present(doc_with_emotion):
    assert doc_with_emotion.provenance is not None


def test_provenance_records_vad(doc_with_emotion):
    from speechtelemetry.provenance import PipelineProvenance

    prov = doc_with_emotion.provenance
    assert isinstance(prov, PipelineProvenance)
    stages = [s.stage for s in prov.stages]
    assert "vad" in stages, f"Provenance missing 'vad'. Got: {stages}"


def test_provenance_records_asr(doc_with_emotion):
    from speechtelemetry.provenance import PipelineProvenance

    prov = doc_with_emotion.provenance
    assert isinstance(prov, PipelineProvenance)
    stages = [s.stage for s in prov.stages]
    assert "asr" in stages, f"Provenance missing 'asr'. Got: {stages}"


def test_provenance_records_emotion_when_enabled(doc_with_emotion):
    """Provenance must include the emotion stage when emotion_backend is configured."""
    from speechtelemetry.provenance import PipelineProvenance

    prov = doc_with_emotion.provenance
    assert isinstance(prov, PipelineProvenance)
    stages = [s.stage for s in prov.stages]
    assert "emotion" in stages, (
        f"Provenance missing 'emotion' stage. Got: {stages}\n"
        "This indicates the emotion stage completed but was not recorded in provenance."
    )


def test_provenance_emotion_absent_when_disabled(doc_no_emotion):
    """Provenance must NOT include emotion when emotion_backend=None."""
    from speechtelemetry.provenance import PipelineProvenance

    prov = doc_no_emotion.provenance
    if prov is None:
        return
    assert isinstance(prov, PipelineProvenance)
    stages = [s.stage for s in prov.stages]
    assert "emotion" not in stages, (
        f"Provenance unexpectedly contains 'emotion' when emotion_backend=None: {stages}"
    )


def test_provenance_backend_names_set(doc_with_emotion):
    from speechtelemetry.provenance import PipelineProvenance

    prov = doc_with_emotion.provenance
    assert isinstance(prov, PipelineProvenance)
    for entry in prov.stages:
        assert entry.backend_name, (
            f"Provenance entry for stage '{entry.stage}' has empty backend_name"
        )


# ── Silence span invariants ───────────────────────────────────────────────────


def test_silence_spans_non_negative_duration(doc_with_emotion):
    for i, sp in enumerate(doc_with_emotion.silence_spans):
        assert sp.duration_ms > 0, f"span{i}: duration_ms={sp.duration_ms} <= 0"


def test_silence_spans_end_after_start(doc_with_emotion):
    for i, sp in enumerate(doc_with_emotion.silence_spans):
        assert sp.end > sp.start, f"span{i}: end={sp.end} <= start={sp.start}"


# ── Speaker profile invariants ────────────────────────────────────────────────


def test_speakers_none_when_no_diarization(doc_with_emotion):
    """When diarization is disabled, speakers must be None."""
    assert doc_with_emotion.speakers is None, (
        "speakers should be None when diarization_backend=None"
    )


def test_speakers_none_on_doc_no_emotion(doc_no_emotion):
    assert doc_no_emotion.speakers is None


# ── Schema version invariants ─────────────────────────────────────────────────


def test_schema_version_present(doc_with_emotion):
    from speechtelemetry.types import SCHEMA_VERSION

    assert doc_with_emotion.schema_version == SCHEMA_VERSION


def test_json_output_has_schema_version(doc_with_emotion, tmp_path):
    """JSON export must include schema_version as the first key."""
    import json

    from speechtelemetry.exporters.json_exporter import JsonExporter

    out = str(tmp_path / "schema_test.json")
    JsonExporter().export(doc_with_emotion, out)
    data = json.loads(Path(out).read_text())
    assert "schema_version" in data
    assert list(data.keys())[0] == "schema_version"


def test_json_output_floats_clean(doc_with_emotion, tmp_path):
    """JSON output must not contain floating-point noise."""

    from speechtelemetry.exporters.json_exporter import JsonExporter

    out = str(tmp_path / "float_test.json")
    JsonExporter().export(doc_with_emotion, out)
    raw = Path(out).read_text()
    # No floating-point trailing 9s or 0s noise
    assert "9999999" not in raw
    assert "0000001" not in raw
    # No scientific notation from near-zero float values (e.g. 2.14e-12 → 0.0)
    # Use regex to avoid matching legitimate strings like "large-v3" or "e-mail"
    import re

    assert not re.search(r"\d[eE][+-]\d", raw), "Scientific notation in float values detected"


def test_srt_no_overlaps_on_real_wav_output(doc_with_emotion, tmp_path):
    """SRT produced from mock pipeline output must have non-overlapping cues."""
    import re

    from speechtelemetry.exporters.srt import SRTExporter

    out = str(tmp_path / "pipeline_test.srt")
    SRTExporter().export(doc_with_emotion, out)
    content = Path(out).read_text()

    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
    matches = ts_pattern.findall(content)

    def to_ms(ts: str) -> int:
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)

    prev_end = 0
    for start_str, end_str in matches:
        start_ms = to_ms(start_str)
        assert start_ms >= prev_end, (
            f"SRT overlap: cue start={start_str} ({start_ms}ms) < prev_end={prev_end}ms"
        )
        prev_end = to_ms(end_str)


def test_silence_spans_reason_is_valid(doc_with_emotion):
    valid_reasons = {"speech_gap", "non_speech", "overlap"}
    for i, sp in enumerate(doc_with_emotion.silence_spans):
        assert sp.reason in valid_reasons, f"span{i}: reason='{sp.reason}' not in {valid_reasons}"

"""Structural golden output test for sample_16k_mono.wav (B7).

Runs enrich_audio() on the committed 3-second WAV fixture with ML-free mock
backends. Verifies structural correctness of the output document and JSON
round-trip integrity. No ML dependencies — runs in CI without GPU or model
downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from speechtelemetry import PipelineConfig

_FIXTURE_WAV = Path(__file__).parent.parent / "fixtures" / "sample_16k_mono.wav"


def _make_fake_get_backend(stage: str, name: str, **kwargs: object) -> object:
    """Return deterministic ML-free mocks for every pipeline stage."""
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


_GOLDEN_CONFIG = PipelineConfig(
    device="cpu",
    asr_backend="faster-whisper",
    asr_model_size="tiny",
    asr_language="en",
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,
    prosody_backend=[],
    emotion_backend="speechbrain",
    export_formats=["json"],
)


@pytest.fixture(scope="module")
def golden_doc():
    """Run enrich_audio() once with mock backends on the real 3s WAV fixture."""
    from speechtelemetry import enrich_audio

    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=_make_fake_get_backend),
    ):
        return enrich_audio(str(_FIXTURE_WAV), config=_GOLDEN_CONFIG)


def test_golden_fixture_wav_exists():
    """The committed WAV fixture must be present in the repository."""
    assert _FIXTURE_WAV.exists(), f"Fixture WAV missing: {_FIXTURE_WAV}"


def test_golden_duration_within_five_percent(golden_doc):
    """duration_s must be within 5% of the actual fixture length (3.0 seconds)."""
    assert abs(golden_doc.duration_s - 3.0) <= 0.15, (
        f"duration_s={golden_doc.duration_s:.3f} — expected 2.85..3.15"
    )


def test_golden_segments_is_list(golden_doc):
    """segments must be a list (possibly empty for silence-only audio)."""
    assert isinstance(golden_doc.segments, list)


def test_golden_no_pipeline_errors(golden_doc):
    """processing_report.errors must be empty when all mock backends succeed."""
    errors = golden_doc.processing_report.errors
    assert errors == [], f"Unexpected pipeline errors: {errors}"


def test_golden_stage_timings_populated(golden_doc):
    """VAD and ASR stage timings must be recorded in processing_report."""
    timings = golden_doc.processing_report.stage_timings
    assert "vad" in timings, f"Missing vad timing. Recorded: {list(timings.keys())}"
    assert "asr" in timings, f"Missing asr timing. Recorded: {list(timings.keys())}"


def test_golden_silence_spans_is_list(golden_doc):
    """silence_spans must be a list (the mock VAD returns one speech interval)."""
    assert isinstance(golden_doc.silence_spans, list)
    assert len(golden_doc.silence_spans) > 0, (
        "Expected silence spans — mock VAD covers 0.5-2.5s leaving leading and trailing silence"
    )


def test_golden_segments_have_emotion_scores(golden_doc):
    """Segments longer than 0.5s must have emotion scores from MockEmotionBackend."""
    long_segs = [s for s in golden_doc.segments if (s.end - s.start) >= 0.5]
    if not long_segs:
        pytest.skip("No segments long enough for emotion scoring in mock output")
    emotion_segs = [s for s in long_segs if s.emotion is not None]
    assert len(emotion_segs) == len(long_segs), (
        f"Only {len(emotion_segs)}/{len(long_segs)} eligible segments have emotion scores"
    )


def test_golden_emotion_distribution_is_full_dict(golden_doc):
    """EmotionScore.label_distribution must be a dict with float values (never a single label)."""
    emotion_segs = [s for s in golden_doc.segments if s.emotion is not None]
    if not emotion_segs:
        pytest.skip("No segments with emotion scores in mock output")
    for seg in emotion_segs:
        dist = seg.emotion.label_distribution  # type: ignore[union-attr]
        assert isinstance(dist, dict), f"label_distribution is {type(dist)}, expected dict"
        assert all(isinstance(v, float) for v in dist.values()), (
            f"label_distribution values must be floats: {dist}"
        )


def test_golden_json_round_trips(golden_doc, tmp_path):
    """The JSON export must be valid JSON that round-trips through json.loads."""
    from speechtelemetry.exporters.json_exporter import JsonExporter

    out_path = str(tmp_path / "golden_output.json")
    JsonExporter().export(golden_doc, out_path)

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "source_path" in data
    assert "duration_s" in data
    assert "segments" in data
    assert "silence_spans" in data
    assert "processing_report" in data
    assert isinstance(data["segments"], list)
    assert isinstance(data["silence_spans"], list)
    assert isinstance(data["processing_report"]["errors"], list)


def test_golden_json_segments_have_correct_structure(golden_doc, tmp_path):
    """Each JSON segment must contain the canonical fields defined in types.py."""
    from speechtelemetry.exporters.json_exporter import JsonExporter

    out_path = str(tmp_path / "golden_segments.json")
    JsonExporter().export(golden_doc, out_path)

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    required_keys = {"start", "end", "text", "confidence"}
    for i, seg in enumerate(data["segments"]):
        missing = required_keys - seg.keys()
        assert not missing, f"Segment {i} missing keys: {missing}"
        assert isinstance(seg["start"], (int, float))
        assert isinstance(seg["end"], (int, float))
        assert seg["end"] > seg["start"], f"Segment {i}: end <= start"
        assert isinstance(seg["text"], str)

"""End-to-end integration test using a real media file and mock ML backends.

Validates the full pipeline path: FFmpeg decode → WAV validate → mock VAD/ASR/alignment → export.
FFmpeg and soundfile run for real; the ML backends are deterministic mocks.

Skipped automatically when:
  - FFmpeg is not on PATH
  - mock_data/ directory is absent (developer-only fixture, not tracked in git)

Run explicitly: pytest tests/integration/test_cli_e2e.py -v -m slow
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from speechtelemetry import PipelineConfig, enrich_media
from tests.fixtures.mock_backends import (
    MockAlignmentBackend,
    MockASRBackend,
    MockVADBackend,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_MOCK_MEDIA = str(_REPO_ROOT / "mock_data" / "BLOOD_liquidity_Captions_V2.mp4")
_MOCK_OUTPUT_DIR = str(_REPO_ROOT / "mock_output")

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
_MEDIA_AVAILABLE = os.path.isfile(_MOCK_MEDIA)

_SKIP_REASONS = []
if not _FFMPEG_AVAILABLE:
    _SKIP_REASONS.append("FFmpeg not on PATH")
if not _MEDIA_AVAILABLE:
    _SKIP_REASONS.append(f"mock media not found: {_MOCK_MEDIA}")

skip_unless_ready = pytest.mark.skipif(
    not (_FFMPEG_AVAILABLE and _MEDIA_AVAILABLE),
    reason="; ".join(_SKIP_REASONS) or "",
)

_MOCK_VAD = MockVADBackend()
_MOCK_ASR = MockASRBackend()
_MOCK_ALIGN = MockAlignmentBackend()


def _mock_get_backend(stage: str, name: str, **kwargs: object) -> object:
    """Return deterministic mock backends for ML stages; real implementations for exporters."""
    ml_mocks = {"vad": _MOCK_VAD, "asr": _MOCK_ASR, "alignment": _MOCK_ALIGN}
    if stage in ml_mocks:
        return ml_mocks[stage]
    from speechtelemetry.registry import get_backend as _real_get_backend

    return _real_get_backend(stage, name, **kwargs)


_BASE_CONFIG = PipelineConfig(
    device="cpu",
    asr_backend="faster-whisper",
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,
    prosody_backend=[],
    emotion_backend=None,
    export_formats=["json"],
)


@pytest.fixture()
def mock_pipeline():
    """Patch preflight and ML backends with deterministic mocks for the test duration."""
    with (
        patch("speechtelemetry.core.pipeline.preflight_check"),
        patch("speechtelemetry.core.pipeline.get_backend", side_effect=_mock_get_backend),
    ):
        yield


@pytest.mark.slow
@skip_unless_ready
def test_enrich_media_returns_transcript_document(mock_pipeline):
    from speechtelemetry.types import TranscriptDocument

    doc = enrich_media(_MOCK_MEDIA, config=_BASE_CONFIG)

    assert isinstance(doc, TranscriptDocument)
    assert doc.source_path == _MOCK_MEDIA
    assert doc.duration_s > 0
    assert isinstance(doc.segments, list)
    assert len(doc.segments) > 0


@pytest.mark.slow
@skip_unless_ready
def test_enrich_media_segments_have_text(mock_pipeline):
    doc = enrich_media(_MOCK_MEDIA, config=_BASE_CONFIG)

    for seg in doc.segments:
        assert isinstance(seg.text, str)
        assert len(seg.text.strip()) > 0


@pytest.mark.slow
@skip_unless_ready
def test_enrich_media_json_export(mock_pipeline, tmp_path):
    out_json = str(tmp_path / "result.json")
    enrich_media(_MOCK_MEDIA, config=_BASE_CONFIG, output_path=out_json)

    assert os.path.exists(out_json), "JSON output was not created"
    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("source_path", "language", "duration_s", "segments", "processing_report"):
        assert key in data, f"Missing key in JSON: {key}"
    assert data["duration_s"] > 0
    assert isinstance(data["segments"], list)


@pytest.mark.slow
@skip_unless_ready
def test_enrich_media_report_has_stage_timings(mock_pipeline):
    doc = enrich_media(_MOCK_MEDIA, config=_BASE_CONFIG)

    timings = doc.processing_report.stage_timings
    assert "decode" in timings, f"No decode timing. Got: {list(timings.keys())}"
    assert "vad" in timings, f"No vad timing. Got: {list(timings.keys())}"
    assert "asr" in timings, f"No asr timing. Got: {list(timings.keys())}"
    assert timings["decode"] > 0


@pytest.mark.slow
@skip_unless_ready
def test_enrich_media_no_pipeline_errors(mock_pipeline):
    doc = enrich_media(_MOCK_MEDIA, config=_BASE_CONFIG)

    assert (
        doc.processing_report.errors == []
    ), f"Expected no errors; got: {doc.processing_report.errors}"


@pytest.mark.slow
@skip_unless_ready
def test_mock_output_all_formats(mock_pipeline):
    """Run pipeline and write all four export formats to mock_output/.

    Inspect mock_output/ after this test to verify the full pipeline output structure.
    """
    os.makedirs(_MOCK_OUTPUT_DIR, exist_ok=True)
    config = _BASE_CONFIG.model_copy(update={"export_formats": ["json", "srt", "vtt", "textgrid"]})

    doc = enrich_media(_MOCK_MEDIA, config=config, output_dir=_MOCK_OUTPUT_DIR)

    stem = "BLOOD_liquidity_Captions_V2"
    out = Path(_MOCK_OUTPUT_DIR)
    expected = {
        "json": out / f"{stem}.json",
        "srt": out / f"{stem}.srt",
        "vtt": out / f"{stem}.vtt",
        "textgrid": out / f"{stem}.TextGrid",
    }

    for fmt, path in expected.items():
        assert path.exists(), f"Missing {fmt} output: {path}"

    data = json.loads(expected["json"].read_text(encoding="utf-8"))
    assert data["duration_s"] > 0
    assert len(data["segments"]) > 0

    srt = expected["srt"].read_text(encoding="utf-8")
    assert "1\n" in srt
    assert "hello world" in srt
    assert expected["vtt"].read_text(encoding="utf-8").startswith("WEBVTT")
    assert 'File type = "ooTextFile"' in expected["textgrid"].read_text(encoding="utf-8")

    assert doc.processing_report.errors == [], f"Pipeline errors: {doc.processing_report.errors}"

    timings = doc.processing_report.stage_timings
    print(f"\nmock_output/ written to: {_MOCK_OUTPUT_DIR}")
    print(f"  Duration: {doc.duration_s:.1f}s  Segments: {len(doc.segments)}")
    print(f"  Timings: {timings}")

"""Integration test: full pipeline with speaker diarization on ABSOLUTELYNOT_Sequence02.mp4.

Validates:
  - Multiple distinct speakers are identified in the clip
  - Segments carry speaker labels (SPEAKER_00, SPEAKER_01, …)
  - Provenance records the diarization stage
  - All four export formats contain speaker information
  - JSON output is structurally correct and complete

The pipeline runs once via a session-scoped fixture and all tests share the result.

Requirements:
  - FFmpeg on PATH
  - HF_TOKEN set (pyannote.audio model license acceptance)
  - pyannote.audio installed
  - faster-whisper + silero-vad installed
  - mock_data/ABSOLUTELYNOT_Sequence02.mp4 present

Run: pytest tests/integration/test_diarization_speaker_detection.py -v -m slow -s
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_MOCK_MEDIA = str(_REPO_ROOT / "mock_data" / "ABSOLUTELYNOT_Sequence02.mp4")
_MOCK_OUTPUT_DIR = str(_REPO_ROOT / "mock_output" / "diarization")

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
_MEDIA_AVAILABLE = os.path.isfile(_MOCK_MEDIA)
_HF_TOKEN_AVAILABLE = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))

try:
    from pyannote.audio import Pipeline as _PyannoteP  # noqa: F401

    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False

try:
    import faster_whisper  # noqa: F401

    _ASR_AVAILABLE = True
except ImportError:
    _ASR_AVAILABLE = False

try:
    import silero_vad  # noqa: F401

    _VAD_AVAILABLE = True
except ImportError:
    _VAD_AVAILABLE = False

_SKIP_REASONS = []
if not _FFMPEG_AVAILABLE:
    _SKIP_REASONS.append("FFmpeg not on PATH")
if not _MEDIA_AVAILABLE:
    _SKIP_REASONS.append(f"mock media not found: {_MOCK_MEDIA}")
if not _HF_TOKEN_AVAILABLE:
    _SKIP_REASONS.append("HF_TOKEN not set (required for pyannote diarization)")
if not _PYANNOTE_AVAILABLE:
    _SKIP_REASONS.append("pyannote.audio not installed")
if not _ASR_AVAILABLE:
    _SKIP_REASONS.append("faster-whisper not installed")
if not _VAD_AVAILABLE:
    _SKIP_REASONS.append("silero-vad not installed")

skip_unless_ready = pytest.mark.skipif(
    bool(_SKIP_REASONS),
    reason="; ".join(_SKIP_REASONS) or "",
)


def _diarization_config(export_formats: list[str] | None = None):
    from speechtelemetry import PipelineConfig

    return PipelineConfig(
        device="cpu",
        asr_backend="faster-whisper",
        asr_model_size="small",
        asr_compute_type="int8",
        asr_language="en",
        asr_beam_size=5,
        vad_backend="silero",
        vad_threshold=0.5,
        alignment_backend="whisperx",
        diarization_backend="pyannote",
        diarization_min_speakers=None,
        diarization_max_speakers=None,
        prosody_backend=[],
        emotion_backend="speechbrain",
        export_formats=export_formats or ["json", "srt", "vtt", "textgrid"],
        chunk_audio=True,
        max_chunk_duration_s=30.0,
    )


@pytest.fixture(scope="session")
def diarized_doc():
    """Run the full diarization pipeline once and share the result across all tests.

    Writes all four export formats to mock_output/diarization/.
    """
    if _SKIP_REASONS:
        pytest.skip("; ".join(_SKIP_REASONS))

    from speechtelemetry import enrich_media

    os.makedirs(_MOCK_OUTPUT_DIR, exist_ok=True)
    config = _diarization_config(export_formats=["json", "srt", "vtt", "textgrid"])
    doc = enrich_media(_MOCK_MEDIA, config=config, output_dir=_MOCK_OUTPUT_DIR)

    print(f"\n=== Diarized pipeline → {_MOCK_OUTPUT_DIR} ===")
    print(f"  File:     {_MOCK_MEDIA}")
    print(f"  Duration: {doc.duration_s:.1f}s  Language: {doc.language}")
    print(f"  Segments: {len(doc.segments)}")
    speakers = sorted({s.speaker for s in doc.segments if s.speaker})
    print(f"  Speakers: {speakers}")
    print(f"  RTF: {doc.processing_report.real_time_factor:.3f}")
    print(f"  Peak RAM: {doc.processing_report.peak_ram_mb:.0f} MB")
    print(f"  Stage timings: {doc.processing_report.stage_timings}")

    print("\n  Transcript with speakers:")
    for seg in doc.segments:
        print(
            f"    [{seg.start:.2f}-{seg.end:.2f}] {seg.speaker or 'NO_SPEAKER'}: {seg.text.strip()}"
        )

    if doc.processing_report.errors:
        print("\n  Soft errors:")
        for err in doc.processing_report.errors:
            print(f"    [{err.stage}] {err.message[:120]}")

    return doc


@pytest.mark.slow
@skip_unless_ready
def test_diarization_pipeline_produces_transcript(diarized_doc):
    """Pipeline must return a TranscriptDocument with real transcribed text."""
    from speechtelemetry.types import TranscriptDocument

    doc = diarized_doc
    assert isinstance(doc, TranscriptDocument)
    assert doc.source_path == _MOCK_MEDIA
    assert doc.duration_s > 0, "Audio duration must be positive"
    assert len(doc.segments) > 0, "Pipeline must produce at least one segment"

    all_text = " ".join(seg.text.strip() for seg in doc.segments)
    assert len(all_text) > 10, f"Transcript is suspiciously short: {all_text!r}"


@pytest.mark.slow
@skip_unless_ready
def test_diarization_detects_multiple_speakers(diarized_doc):
    """Pipeline must identify at least two distinct speakers in the clip."""
    speakers = {seg.speaker for seg in diarized_doc.segments if seg.speaker is not None}
    assert len(speakers) >= 2, (
        f"Expected ≥2 distinct speakers; got {len(speakers)}: {sorted(speakers)}\n"
        f"Segments with speakers: "
        f"{sum(1 for s in diarized_doc.segments if s.speaker)}/{len(diarized_doc.segments)}"
    )

    for speaker in sorted(speakers):
        segs = [s for s in diarized_doc.segments if s.speaker == speaker]
        total_time = sum(s.end - s.start for s in segs)
        print(f"\n  {speaker}: {len(segs)} segments, {total_time:.1f}s total speech")


@pytest.mark.slow
@skip_unless_ready
def test_diarization_segments_have_speaker_labels(diarized_doc):
    """Most segments must have a speaker label — coverage must be > 70%."""
    labeled = sum(1 for s in diarized_doc.segments if s.speaker is not None)
    total = len(diarized_doc.segments)
    coverage = labeled / total if total > 0 else 0.0

    assert coverage >= 0.70, (
        f"Speaker label coverage too low: {labeled}/{total} ({coverage:.0%}). "
        f"Diarization may have failed."
    )
    print(f"\n  Speaker coverage: {labeled}/{total} ({coverage:.0%})")


@pytest.mark.slow
@skip_unless_ready
def test_diarization_speaker_labels_match_format(diarized_doc):
    """All speaker labels must follow the 'SPEAKER_NN' format."""
    for seg in diarized_doc.segments:
        if seg.speaker is not None:
            assert seg.speaker.startswith("SPEAKER_"), (
                f"Speaker label '{seg.speaker}' does not match SPEAKER_NN format"
            )


@pytest.mark.slow
@skip_unless_ready
def test_diarization_provenance_is_recorded(diarized_doc):
    """Provenance must include the diarization stage after a successful run."""
    from speechtelemetry.provenance import PipelineProvenance

    doc = diarized_doc
    assert doc.provenance is not None
    assert isinstance(doc.provenance, PipelineProvenance)

    stage_names = [s.stage for s in doc.provenance.stages]
    assert "vad" in stage_names, "Provenance missing vad stage"
    assert "asr" in stage_names, "Provenance missing asr stage"
    assert "diarization" in stage_names, f"Provenance missing diarization stage. Got: {stage_names}"

    diar_prov = next(s for s in doc.provenance.stages if s.stage == "diarization")
    assert diar_prov.backend_name == "pyannote"
    print(f"\n  Provenance stages: {stage_names}")


@pytest.mark.slow
@skip_unless_ready
def test_diarization_pipeline_writes_all_formats(diarized_doc):
    """All four export files must exist and contain speaker information."""
    stem = "ABSOLUTELYNOT_Sequence02"
    out = Path(_MOCK_OUTPUT_DIR)
    expected = {
        "json": out / f"{stem}.json",
        "srt": out / f"{stem}.srt",
        "vtt": out / f"{stem}.vtt",
        "textgrid": out / f"{stem}.TextGrid",
    }

    for fmt, path in expected.items():
        assert path.exists(), f"Missing {fmt} output: {path}"
        assert path.stat().st_size > 0, f"{fmt} output is empty"

    data = json.loads(expected["json"].read_text(encoding="utf-8"))
    assert data["duration_s"] > 0
    assert data["language"] is not None, "Language must be detected"
    assert len(data["segments"]) > 0

    labeled_segs = [s for s in data["segments"] if s.get("speaker") is not None]
    assert len(labeled_segs) > 0, "JSON output must contain segments with speaker labels"

    speakers_in_json = {s["speaker"] for s in labeled_segs}
    assert len(speakers_in_json) >= 2, (
        f"JSON must show ≥2 speakers; got {len(speakers_in_json)}: {sorted(speakers_in_json)}"
    )

    prov_stages = {s["stage"]: s for s in data["provenance"]["stages"]}
    assert "diarization" in prov_stages, "JSON provenance missing diarization stage"
    assert prov_stages["diarization"]["backend_name"] == "pyannote"

    srt_content = expected["srt"].read_text(encoding="utf-8")
    assert "1\n" in srt_content, "SRT must contain at least one subtitle entry"
    assert "SPEAKER_" in srt_content, "SRT must contain speaker labels"

    vtt_content = expected["vtt"].read_text(encoding="utf-8")
    assert vtt_content.startswith("WEBVTT"), "VTT must start with WEBVTT header"
    assert "<v SPEAKER_" in vtt_content, "VTT must contain speaker voice tags"

    tg_content = expected["textgrid"].read_text(encoding="utf-8")
    assert 'File type = "ooTextFile"' in tg_content

    print(f"\n  Files in {_MOCK_OUTPUT_DIR}:")
    for fmt, path in expected.items():
        print(f"    {fmt}: {path.stat().st_size:,} bytes")

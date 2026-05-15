"""Real-backend E2E integration test using actual ML models.

Runs the full pipeline on mock_data/BLOOD_liquidity_Captions_V2.mp4 with:
  - Silero VAD (real model, MIT)
  - faster-whisper ASR (real model, MIT, small/tiny for speed)
  - WhisperX alignment (real model, fail-soft if unavailable)
  - SpeechBrain emotion (real model, Apache 2.0, fail-soft)
  - No diarization (requires HF_TOKEN)
  - All 4 export formats written to mock_output/

Output reflects the actual audio content of the MP4, not mock fixtures.

Run: pytest tests/integration/test_real_pipeline.py -v -m slow -s
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_MOCK_MEDIA = str(_REPO_ROOT / "mock_data" / "BLOOD_liquidity_Captions_V2.mp4")
_MOCK_OUTPUT_DIR = str(_REPO_ROOT / "mock_output")

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
_MEDIA_AVAILABLE = os.path.isfile(_MOCK_MEDIA)

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
if not _ASR_AVAILABLE:
    _SKIP_REASONS.append("faster-whisper not installed")
if not _VAD_AVAILABLE:
    _SKIP_REASONS.append("silero-vad not installed")

skip_unless_ready = pytest.mark.skipif(
    bool(_SKIP_REASONS),
    reason="; ".join(_SKIP_REASONS) or "",
)


def _real_config(export_formats: list[str] | None = None):
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
        diarization_backend=None,
        prosody_backend=[],
        emotion_backend="speechbrain",
        export_formats=export_formats or ["json"],
        chunk_audio=True,
        max_chunk_duration_s=30.0,
    )


@pytest.mark.slow
@skip_unless_ready
def test_real_pipeline_produces_transcript():
    """Pipeline must return a TranscriptDocument with real transcribed text."""
    from speechtelemetry import enrich_media
    from speechtelemetry.types import TranscriptDocument

    doc = enrich_media(_MOCK_MEDIA, config=_real_config())

    assert isinstance(doc, TranscriptDocument)
    assert doc.source_path == _MOCK_MEDIA
    assert doc.duration_s > 0, "Audio duration must be positive"
    assert len(doc.segments) > 0, "Pipeline must produce at least one segment"

    all_text = " ".join(seg.text.strip() for seg in doc.segments)
    assert len(all_text) > 10, f"Transcript is suspiciously short: {all_text!r}"

    print(f"\n--- Real transcript ({len(doc.segments)} segments, {doc.duration_s:.1f}s) ---")
    for seg in doc.segments:
        print(f"  [{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}")
    if doc.processing_report.errors:
        print("\nSoft errors (expected for optional backends):")
        for err in doc.processing_report.errors:
            print(f"  [{err.stage}] {err.exception_type}: {err.message[:80]}")


@pytest.mark.slow
@skip_unless_ready
def test_real_pipeline_segments_have_valid_timestamps():
    """All segments must have non-negative start < end timestamps within audio duration."""
    from speechtelemetry import enrich_media

    doc = enrich_media(_MOCK_MEDIA, config=_real_config())

    for i, seg in enumerate(doc.segments):
        assert seg.start >= 0, f"Segment {i} has negative start: {seg.start}"
        assert seg.end > seg.start, f"Segment {i}: end {seg.end} <= start {seg.start}"
        assert seg.end <= doc.duration_s + 0.5, (
            f"Segment {i} end {seg.end} exceeds audio duration {doc.duration_s}"
        )


@pytest.mark.slow
@skip_unless_ready
def test_real_pipeline_report_has_timings():
    """ProcessingReport must record all mandatory stage timings: decode, vad, asr, alignment."""
    from speechtelemetry import enrich_media

    doc = enrich_media(_MOCK_MEDIA, config=_real_config())
    timings = doc.processing_report.stage_timings

    assert "decode" in timings and timings["decode"] > 0, "Missing decode timing"
    assert "vad" in timings and timings["vad"] > 0, "Missing vad timing"
    assert "asr" in timings and timings["asr"] > 0, "Missing asr timing"
    assert "alignment" in timings and timings["alignment"] > 0, "Missing alignment timing"

    assert doc.processing_report.real_time_factor > 0, "RTF must be positive"

    print(f"\nStage timings: {timings}")
    print(f"RTF: {doc.processing_report.real_time_factor:.3f}")
    print(f"Peak RAM: {doc.processing_report.peak_ram_mb:.0f} MB")


@pytest.mark.slow
@skip_unless_ready
def test_real_pipeline_writes_all_formats_to_mock_output():
    """Run real pipeline and write all 4 export formats to mock_output/.

    This is the primary output verification test. Inspect mock_output/ after
    this test to see the actual transcription of BLOOD_liquidity_Captions_V2.mp4.
    """
    from speechtelemetry import enrich_media

    os.makedirs(_MOCK_OUTPUT_DIR, exist_ok=True)
    config = _real_config(export_formats=["json", "srt", "vtt", "textgrid"])

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
        assert path.stat().st_size > 0, f"{fmt} output is empty"

    data = json.loads(expected["json"].read_text(encoding="utf-8"))
    assert data["duration_s"] > 0
    assert data["language"] is not None, "Language must be detected"
    assert len(data["segments"]) > 0

    # Word-level timestamps must be present — alignment stage must have run.
    segs_with_words = [s for s in data["segments"] if s.get("words")]
    assert len(segs_with_words) > 0, "No segments have word-level timestamps (alignment failed?)"
    first_word = segs_with_words[0]["words"][0]
    assert first_word["alignment_backend"] == "whisperx", (
        f"Word alignment_backend must be 'whisperx', got: {first_word['alignment_backend']!r}"
    )
    assert isinstance(first_word["start"], float) and first_word["start"] >= 0
    assert isinstance(first_word["end"], float) and first_word["end"] > first_word["start"]

    # Silence spans must be populated from VAD output.
    assert len(data["silence_spans"]) > 0, "silence_spans must be non-empty for real audio"
    ss = data["silence_spans"][0]
    assert ss["duration_ms"] > 0, "Silence span duration must be positive"

    # Provenance must record VAD, ASR, and alignment backends.
    prov_stages = {s["stage"]: s for s in data["provenance"]["stages"]}
    assert "vad" in prov_stages, "Provenance missing vad stage"
    assert "asr" in prov_stages, "Provenance missing asr stage"
    assert "alignment" in prov_stages, "Provenance missing alignment stage"
    assert prov_stages["asr"]["backend_name"] == "faster-whisper"
    assert prov_stages["asr"]["model_id"] == "small"

    srt = expected["srt"].read_text(encoding="utf-8")
    assert "1\n" in srt, "SRT must contain at least one subtitle entry"

    vtt = expected["vtt"].read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT"), "VTT must start with WEBVTT header"

    tg = expected["textgrid"].read_text(encoding="utf-8")
    assert 'File type = "ooTextFile"' in tg

    print(f"\n=== Real pipeline output written to: {_MOCK_OUTPUT_DIR} ===")
    print(f"  Duration: {doc.duration_s:.1f}s")
    print(f"  Language: {doc.language}")
    print(f"  Segments: {len(doc.segments)}")
    print(f"  RTF: {doc.processing_report.real_time_factor:.3f}")
    print(f"  Peak RAM: {doc.processing_report.peak_ram_mb:.0f} MB")
    print("\n  Transcript:")
    for seg in doc.segments:
        emotion_info = ""
        if seg.emotion:
            top = seg.emotion.top_label
            conf = seg.emotion.confidence
            emotion_info = f"  [emotion: {top} {conf:.2f}]"
        print(f"    [{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}{emotion_info}")
    if doc.processing_report.errors:
        print("\n  Soft errors:")
        for err in doc.processing_report.errors:
            print(f"    [{err.stage}] {err.message[:100]}")

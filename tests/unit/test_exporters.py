"""Unit tests for all exporters — SRT, VTT, JSON, TextGrid.

Uses in-memory TranscriptDocument with no external deps beyond soundfile/json.
"""

import json
import os

import pytest

from speechtelemetry.exporters.json_exporter import JsonExporter
from speechtelemetry.exporters.srt import SRTExporter
from speechtelemetry.exporters.textgrid import TextGridExporter
from speechtelemetry.exporters.vtt import VTTExporter
from speechtelemetry.types import (
    ProcessingReport,
    Segment,
    SilenceSpan,
    TranscriptDocument,
)


def _minimal_doc(**overrides) -> TranscriptDocument:
    defaults = dict(
        source_path="test.wav",
        language="en",
        duration_s=5.0,
        segments=[
            Segment(start=0.5, end=2.0, text="Hello world", confidence=0.9),
            Segment(start=2.5, end=4.5, text="Goodbye", confidence=0.85, speaker="SPEAKER_00"),
        ],
        silence_spans=[SilenceSpan(start=0.0, end=0.5, duration_ms=500.0)],
        processing_report=ProcessingReport(),
    )
    defaults.update(overrides)
    return TranscriptDocument(**defaults)


@pytest.fixture()
def doc():
    return _minimal_doc()


@pytest.fixture()
def tmp_path_str(tmp_path):
    return str(tmp_path)


# ── JSON Exporter ──────────────────────────────────────────────────────────────


def test_json_export_creates_file(doc, tmp_path):
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    assert os.path.exists(out)


def test_json_export_valid_json(doc, tmp_path):
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    with open(out) as f:
        data = json.load(f)
    assert data["source_path"] == "test.wav"
    assert data["language"] == "en"
    assert len(data["segments"]) == 2


def test_json_export_segments_have_text(doc, tmp_path):
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    with open(out) as f:
        data = json.load(f)
    assert data["segments"][0]["text"] == "Hello world"


def test_json_load_roundtrip(doc, tmp_path):
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    loaded = JsonExporter.load(out)
    assert loaded["language"] == "en"
    assert loaded["duration_s"] == 5.0


def test_json_export_null_language(tmp_path):
    doc = _minimal_doc(language=None)
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    with open(out) as f:
        data = json.load(f)
    assert data["language"] is None


# ── SRT Exporter ──────────────────────────────────────────────────────────────


def test_srt_export_creates_file(doc, tmp_path):
    out = str(tmp_path / "out.srt")
    SRTExporter().export(doc, out)
    assert os.path.exists(out)


def test_srt_export_content(doc, tmp_path):
    out = str(tmp_path / "out.srt")
    SRTExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert "Hello world" in content
    assert "Goodbye" in content
    assert "00:00:00,500" in content


def test_srt_export_speaker_label(doc, tmp_path):
    out = str(tmp_path / "out.srt")
    SRTExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert "[SPEAKER_00]" in content


def test_srt_format_two_digits_hours():
    from speechtelemetry.exporters.srt import _format_srt_time

    assert _format_srt_time(3661.5) == "01:01:01,500"


def test_srt_format_zero():
    from speechtelemetry.exporters.srt import _format_srt_time

    assert _format_srt_time(0.0) == "00:00:00,000"


def test_srt_format_float_precision():
    """10.18s must produce 180ms, not 179ms — float truncation bug guard."""
    from speechtelemetry.exporters.srt import _format_srt_time

    assert _format_srt_time(10.18) == "00:00:10,180"


# ── VTT Exporter ──────────────────────────────────────────────────────────────


def test_vtt_export_creates_file(doc, tmp_path):
    out = str(tmp_path / "out.vtt")
    VTTExporter().export(doc, out)
    assert os.path.exists(out)


def test_vtt_export_starts_with_webvtt(doc, tmp_path):
    out = str(tmp_path / "out.vtt")
    VTTExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert content.startswith("WEBVTT")


def test_vtt_export_has_speaker_tag(doc, tmp_path):
    out = str(tmp_path / "out.vtt")
    VTTExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert "<v SPEAKER_00>" in content


def test_vtt_format_time():
    from speechtelemetry.exporters.vtt import _format_vtt_time

    assert _format_vtt_time(3661.5) == "01:01:01.500"
    assert _format_vtt_time(0.0) == "00:00:00.000"


def test_vtt_format_float_precision():
    """10.18s must produce 180ms, not 179ms — float truncation bug guard."""
    from speechtelemetry.exporters.vtt import _format_vtt_time

    assert _format_vtt_time(10.18) == "00:00:10.180"


# ── TextGrid Exporter ──────────────────────────────────────────────────────────


def test_textgrid_export_creates_file(doc, tmp_path):
    out = str(tmp_path / "out.TextGrid")
    TextGridExporter().export(doc, out)
    assert os.path.exists(out)


def test_textgrid_export_has_header(doc, tmp_path):
    out = str(tmp_path / "out.TextGrid")
    TextGridExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert 'File type = "ooTextFile"' in content
    assert 'Object class = "TextGrid"' in content


def test_textgrid_export_has_utterances_tier(doc, tmp_path):
    out = str(tmp_path / "out.TextGrid")
    TextGridExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert '"utterances"' in content
    assert "Hello world" in content


def test_textgrid_export_has_speakers_tier(doc, tmp_path):
    out = str(tmp_path / "out.TextGrid")
    TextGridExporter().export(doc, out)
    with open(out) as f:
        content = f.read()
    assert '"speakers"' in content

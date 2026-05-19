"""Unit tests for all exporters â€” SRT, VTT, JSON, TextGrid.

Uses in-memory TranscriptDocument with no external deps beyond soundfile/json.
"""

import json
import os
from pathlib import Path

import pytest

from speechtelemetry.exporters.json_exporter import JsonExporter
from speechtelemetry.exporters.srt import SRTExporter
from speechtelemetry.exporters.textgrid import TextGridExporter
from speechtelemetry.exporters.vtt import VTTExporter
from speechtelemetry.types import (
    EmotionScore,
    ProcessingReport,
    Segment,
    SilenceSpan,
    SpeakerProfile,
    TranscriptDocument,
    Word,
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


def _overlapping_doc() -> TranscriptDocument:
    """Doc where consecutive segments overlap at boundaries (real-world ASR output)."""
    return TranscriptDocument(
        source_path="overlap_test.wav",
        language="en",
        duration_s=6.0,
        segments=[
            Segment(start=0.0, end=1.021, text="You're pointy.", confidence=0.68),
            Segment(start=1.0, end=5.62, text="Elon Musk has deserved.", confidence=0.85),
            Segment(start=5.6, end=6.0, text="Jeff Bezos.", confidence=0.83),
        ],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )


def _diarized_doc() -> TranscriptDocument:
    """Doc with two speakers and emotion scores."""
    seg0 = Segment(
        start=0.5,
        end=2.0,
        text="Hello world",
        confidence=0.9,
        speaker="SPEAKER_00",
        emotion=EmotionScore(
            label_distribution={"neu": 0.7, "ang": 0.1, "hap": 0.15, "sad": 0.05},
            confidence=0.7,
            backend_name="mock",
        ),
    )
    seg1 = Segment(
        start=2.5,
        end=4.5,
        text="Goodbye",
        confidence=0.85,
        speaker="SPEAKER_01",
        emotion=EmotionScore(
            label_distribution={"neu": 0.1, "ang": 0.8, "hap": 0.05, "sad": 0.05},
            confidence=0.8,
            backend_name="mock",
        ),
    )
    speakers = [
        SpeakerProfile(
            speaker_id="SPEAKER_00",
            speaking_time_s=1.5,
            turn_count=1,
            word_count=2,
            mean_segment_confidence=0.9,
            dominant_emotion="neu",
            emotion_distribution={"neu": 0.7, "ang": 0.1, "hap": 0.15, "sad": 0.05},
        ),
        SpeakerProfile(
            speaker_id="SPEAKER_01",
            speaking_time_s=2.0,
            turn_count=1,
            word_count=1,
            mean_segment_confidence=0.85,
            dominant_emotion="ang",
            emotion_distribution={"neu": 0.1, "ang": 0.8, "hap": 0.05, "sad": 0.05},
        ),
    ]
    return TranscriptDocument(
        source_path="diarized.wav",
        language="en",
        duration_s=5.0,
        segments=[seg0, seg1],
        silence_spans=[],
        processing_report=ProcessingReport(),
        speakers=speakers,
    )


def _words_doc() -> TranscriptDocument:
    """Doc with word-level timestamps."""
    seg = Segment(
        start=0.5,
        end=2.0,
        text="Hello world",
        confidence=0.9,
        words=[
            Word(text="Hello", start=0.5, end=0.9, confidence=0.95, alignment_backend="whisperx"),
            Word(text="world", start=1.0, end=2.0, confidence=0.88, alignment_backend="whisperx"),
        ],
    )
    return TranscriptDocument(
        source_path="words.wav",
        language="en",
        duration_s=3.0,
        segments=[seg],
        silence_spans=[],
        processing_report=ProcessingReport(),
    )


@pytest.fixture()
def doc():
    return _minimal_doc()


@pytest.fixture()
def tmp_path_str(tmp_path):
    return str(tmp_path)


# â”€â”€ JSON Exporter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


def test_json_has_schema_version(doc, tmp_path):
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    assert "schema_version" in data
    assert isinstance(data["schema_version"], str)


def test_json_schema_version_is_first_key(doc, tmp_path):
    """schema_version must be the first key for quick tooling identification."""
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    assert list(data.keys())[0] == "schema_version"


def test_json_floats_rounded(tmp_path):
    """Floating-point noise must be eliminated â€” no more 0.6819999999999999."""
    doc = _minimal_doc(
        duration_s=57.8773125,
        segments=[Segment(start=0.0, end=1.021, text="test", confidence=0.6819999999999999)],
    )
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    raw = Path(out).read_text()
    assert "0.6819999999999999" not in raw
    data = json.loads(raw)
    assert data["segments"][0]["confidence"] == round(0.6819999999999999, 4)


def test_json_emotion_near_zero_rounded(tmp_path):
    """Near-zero emotion probabilities like 2.14e-12 must round to 0.0."""
    seg = Segment(
        start=0.0,
        end=2.0,
        text="x",
        confidence=0.9,
        emotion=EmotionScore(
            label_distribution={
                "neu": 1.0,
                "ang": 2.1379740425958582e-12,
                "hap": 9.471409767591865e-11,
                "sad": 1.751637029867581e-11,
            },
            confidence=1.0,
            backend_name="speechbrain",
        ),
    )
    doc = _minimal_doc(segments=[seg])
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    raw = Path(out).read_text()
    # Scientific notation should be gone
    assert "e-12" not in raw
    assert "e-11" not in raw
    data = json.loads(raw)
    dist = data["segments"][0]["emotion"]["label_distribution"]
    assert dist["ang"] == 0.0
    assert dist["neu"] == 1.0


def test_json_source_path_forward_slashes(tmp_path):
    doc = _minimal_doc(source_path="C:\\Users\\Greg\\interview.mp4")
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    assert "\\" not in data["source_path"]
    assert data["source_path"] == "C:/Users/Greg/interview.mp4"


def test_json_speakers_absent_without_diarization(doc, tmp_path):
    """speakers block must be omitted when no diarization ran."""
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    assert "speakers" not in data


def test_json_speakers_present_with_diarization(tmp_path):
    """speakers block must appear before segments when diarization ran."""
    doc = _diarized_doc()
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    assert "speakers" in data
    keys = list(data.keys())
    assert keys.index("speakers") < keys.index("segments"), "speakers must appear before segments"


def test_json_speakers_fields(tmp_path):
    doc = _diarized_doc()
    out = str(tmp_path / "out.json")
    JsonExporter().export(doc, out)
    data = json.loads(Path(out).read_text())
    sp = data["speakers"][0]
    assert sp["speaker_id"] == "SPEAKER_00"
    assert "speaking_time_s" in sp
    assert "turn_count" in sp
    assert "word_count" in sp
    assert "mean_segment_confidence" in sp
    assert sp["dominant_emotion"] == "neu"
    assert isinstance(sp["emotion_distribution"], dict)


# â”€â”€ SRT Exporter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
    """10.18s must produce 180ms, not 179ms â€” float truncation bug guard."""
    from speechtelemetry.exporters.srt import _format_srt_time

    assert _format_srt_time(10.18) == "00:00:10,180"


def test_srt_no_overlapping_timestamps(tmp_path):
    """Consecutive SRT cues must not overlap (seg boundary overlap fix)."""
    doc = _overlapping_doc()
    out = str(tmp_path / "out.srt")
    SRTExporter().export(doc, out)
    content = Path(out).read_text()

    # Parse all timestamp lines
    import re

    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
    matches = ts_pattern.findall(content)
    assert len(matches) == 3

    def to_ms(ts: str) -> int:
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)

    prev_end = 0
    for start_str, end_str in matches:
        start_ms = to_ms(start_str)
        end_ms = to_ms(end_str)
        assert start_ms >= prev_end, (
            f"SRT cue start {start_str} ({start_ms}ms) < previous end ({prev_end}ms)"
        )
        assert end_ms > start_ms
        prev_end = end_ms


# â”€â”€ VTT Exporter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
    """10.18s must produce 180ms, not 179ms â€” float truncation bug guard."""
    from speechtelemetry.exporters.vtt import _format_vtt_time

    assert _format_vtt_time(10.18) == "00:00:10.180"


def test_vtt_has_note_header(doc, tmp_path):
    """VTT must contain a NOTE metadata block after WEBVTT."""
    out = str(tmp_path / "out.vtt")
    VTTExporter().export(doc, out)
    content = Path(out).read_text()
    assert "NOTE speechtelemetry" in content


def test_vtt_no_overlapping_timestamps(tmp_path):
    """Consecutive VTT cues must not overlap."""
    doc = _overlapping_doc()
    out = str(tmp_path / "out.vtt")
    VTTExporter().export(doc, out)
    content = Path(out).read_text()

    import re

    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})")
    matches = ts_pattern.findall(content)
    assert len(matches) == 3

    def to_ms(ts: str) -> int:
        h, m, rest = ts.split(":")
        s, ms = rest.split(".")
        return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)

    prev_end = 0
    for start_str, end_str in matches:
        start_ms = to_ms(start_str)
        assert start_ms >= prev_end, (
            f"VTT cue start {start_str} ({start_ms}ms) < previous end ({prev_end}ms)"
        )
        prev_end = to_ms(end_str)


# â”€â”€ TextGrid Exporter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


def test_textgrid_has_emotion_tier(tmp_path):
    """TextGrid must include an emotion tier when segments have emotion scores."""
    doc = _diarized_doc()
    out = str(tmp_path / "emotion.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()
    assert '"emotion"' in content
    assert "neu" in content or "ang" in content


def test_textgrid_has_word_tier_when_words_present(tmp_path):
    """TextGrid must include a words tier when word-level timestamps are present."""
    doc = _words_doc()
    out = str(tmp_path / "words.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()
    assert '"words"' in content
    assert "Hello" in content


def test_textgrid_no_word_tier_without_words(doc, tmp_path):
    """TextGrid must NOT include a words tier when no word timestamps exist."""
    out = str(tmp_path / "no_words.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()
    assert '"words"' not in content


def test_textgrid_intervals_contiguous_and_cover_full_range(tmp_path):
    """Every tier must cover exactly [0, xmax] with no gaps or overlaps."""
    doc = _overlapping_doc()
    out = str(tmp_path / "contiguous.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()

    import re

    # Extract all xmin/xmax pairs per tier from the utterances tier
    # Tier 1 (utterances) intervals must be contiguous from 0 to duration
    xmin_vals = [float(m) for m in re.findall(r"xmin = ([\d.]+)", content)]
    xmax_vals = [float(m) for m in re.findall(r"xmax = ([\d.]+)", content)]

    # The file-level xmax must match doc duration
    assert abs(xmax_vals[0] - doc.duration_s) < 0.001

    # Every interval's xmax must be â‰¥ its xmin
    for xmin, xmax in zip(xmin_vals[2:], xmax_vals[2:], strict=False):  # skip file and tier headers
        assert xmax >= xmin, f"xmax={xmax} < xmin={xmin}"


def test_textgrid_overlap_doc_produces_valid_tier(tmp_path):
    """When ASR boundaries overlap, utterances tier must have non-overlapping intervals."""
    doc = _overlapping_doc()
    out = str(tmp_path / "overlap.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()

    import re

    # Extract only the utterances tier block (between first item [1]: and item [2]:)
    tier1_match = re.search(r'name = "utterances".*?(?=\n    item \[2\]|\Z)', content, re.DOTALL)
    assert tier1_match, "utterances tier not found"
    tier1_content = tier1_match.group(0)

    interval_blocks = re.findall(r"xmin = ([\d.]+)\s+xmax = ([\d.]+)", tier1_content)
    # First match is the tier header (xmin=0, xmax=duration), skip it
    interval_blocks = interval_blocks[1:]
    assert len(interval_blocks) >= 3, f"Expected >=3 intervals; got {len(interval_blocks)}"

    prev_end = 0.0
    for start_str, end_str in interval_blocks:
        start = float(start_str)
        end = float(end_str)
        assert start >= prev_end - 1e-6, (
            f"Overlap in utterances tier: start={start} < prev_end={prev_end}"
        )
        assert end >= start
        prev_end = end

    # Verify the seg1.end=1.021 / seg2.start=1.000 overlap was resolved
    starts = [float(s) for s, _ in interval_blocks]
    assert starts[1] >= 1.021 - 1e-6, (
        f"seg2 start should have been clamped to >=1.021; got {starts[1]}"
    )


def test_textgrid_tier_count_with_diarization_and_emotion(tmp_path):
    """With diarization + emotion, TextGrid must have at least 3 tiers."""
    doc = _diarized_doc()
    out = str(tmp_path / "full.TextGrid")
    TextGridExporter().export(doc, out)
    content = Path(out).read_text()

    import re

    size_match = re.search(r"^size = (\d+)", content, re.MULTILINE)
    assert size_match is not None
    tier_count = int(size_match.group(1))
    assert tier_count >= 3, f"Expected >=3 tiers; got {tier_count}"

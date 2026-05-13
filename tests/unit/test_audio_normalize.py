"""Unit tests for audio_normalize — validate_wav and iter_chunks.

TDD: These tests are written before or alongside the implementation fixes.
No ML deps. No I/O except temporary files created by the tests themselves.
"""

from __future__ import annotations

import wave

import pytest

from speechtelemetry.io.audio_normalize import iter_chunks, validate_wav

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_wav(path: str, sample_rate: int, channels: int, duration_s: float = 1.0) -> None:
    """Write a minimal silent WAV file for testing."""
    n_frames = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * n_frames)


# ── validate_wav ──────────────────────────────────────────────────────────────


def test_validate_wav_accepts_canonical_format(tmp_path):
    wav = str(tmp_path / "valid.wav")
    _write_wav(wav, sample_rate=16000, channels=1, duration_s=3.0)
    info = validate_wav(wav)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert abs(info["duration_s"] - 3.0) < 0.01


def test_validate_wav_rejects_wrong_sample_rate(tmp_path):
    wav = str(tmp_path / "bad_rate.wav")
    _write_wav(wav, sample_rate=44100, channels=1)
    with pytest.raises(ValueError, match="44100"):
        validate_wav(wav)


def test_validate_wav_rejects_stereo(tmp_path):
    wav = str(tmp_path / "stereo.wav")
    _write_wav(wav, sample_rate=16000, channels=2)
    with pytest.raises(ValueError, match="[Cc]hannel"):
        validate_wav(wav)


def test_validate_wav_rejects_wrong_rate_and_channels(tmp_path):
    wav = str(tmp_path / "both_wrong.wav")
    _write_wav(wav, sample_rate=8000, channels=2)
    with pytest.raises(ValueError):
        validate_wav(wav)


def test_validate_wav_returns_frames_count(tmp_path):
    wav = str(tmp_path / "frames.wav")
    _write_wav(wav, sample_rate=16000, channels=1, duration_s=2.0)
    info = validate_wav(wav)
    assert info["frames"] == 32000


# ── iter_chunks ───────────────────────────────────────────────────────────────


def test_iter_chunks_single_chunk_for_short_audio(tmp_path):
    wav = str(tmp_path / "short.wav")
    _write_wav(wav, sample_rate=16000, channels=1, duration_s=3.0)
    chunks = list(iter_chunks(wav, chunk_duration_s=30.0))
    assert len(chunks) == 1
    start, end = chunks[0]
    assert start == 0.0
    assert abs(end - 3.0) < 0.01


def test_iter_chunks_multiple_for_long_audio(tmp_path):
    wav = str(tmp_path / "long.wav")
    _write_wav(wav, sample_rate=16000, channels=1, duration_s=65.0)
    chunks = list(iter_chunks(wav, chunk_duration_s=30.0))
    assert len(chunks) == 3
    assert chunks[0] == (0.0, 30.0)
    assert chunks[1] == (30.0, 60.0)
    start, end = chunks[2]
    assert start == 60.0
    assert abs(end - 65.0) < 0.01


def test_iter_chunks_no_overlap(tmp_path):
    wav = str(tmp_path / "overlap_check.wav")
    _write_wav(wav, sample_rate=16000, channels=1, duration_s=90.0)
    chunks = list(iter_chunks(wav, chunk_duration_s=30.0))
    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0]

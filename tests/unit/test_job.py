"""Unit tests for Job lifecycle and decode behaviour (G1: validate_wav wiring).

TDD: Tests written to drive the G1 fix — Job.decode() must call validate_wav()
after normalize_to_wav() so invalid WAV is caught before entering the pipeline.
"""

from __future__ import annotations

import os
import wave
from unittest.mock import patch

import pytest

from speechtelemetry.config import PipelineConfig
from speechtelemetry.core.job import Job

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_wav(path: str, sample_rate: int, channels: int, duration_s: float = 1.0) -> None:
    n_frames = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * n_frames)


# ── Lifecycle tests ───────────────────────────────────────────────────────────


def test_job_creates_temp_dir_on_enter():
    with Job("input.wav") as job:
        assert job._temp_dir is not None
        assert os.path.isdir(job._temp_dir)


def test_job_removes_temp_dir_on_exit():
    with Job("input.wav") as job:
        temp_dir = job._temp_dir
    assert not os.path.exists(temp_dir)


def test_job_temp_wav_path_is_inside_temp_dir():
    with Job("input.wav") as job:
        assert job.temp_wav.startswith(job._temp_dir)
        assert job.temp_wav.endswith("audio.wav")


def test_job_temp_wav_raises_if_not_entered():
    job = Job("input.wav")
    with pytest.raises(AssertionError):
        _ = job.temp_wav


def test_job_output_path_uses_input_stem():
    with Job("/some/path/interview.mp4") as job:
        p = job.output_path("json")
    assert p.endswith("interview.json")


def test_job_output_path_uses_output_dir_when_set():
    with Job("/some/path/interview.mp4") as job:
        job.output_dir = "/out"
        p = job.output_path("srt")
    assert p.startswith("/out")
    assert p.endswith("interview.srt")


# ── G1: validate_wav wiring ───────────────────────────────────────────────────


def test_job_decode_calls_validate_wav_after_normalize(tmp_path):
    """Job.decode() must call validate_wav() after normalize_to_wav()."""
    wav_file = str(tmp_path / "audio.wav")
    _write_wav(wav_file, sample_rate=16000, channels=1, duration_s=1.0)

    with (
        Job(wav_file) as job,
        patch("speechtelemetry.io.ffmpeg.normalize_to_wav") as mock_norm,
        patch("speechtelemetry.io.audio_normalize.validate_wav") as mock_val,
    ):
        mock_norm.return_value = None
        mock_val.return_value = {
            "sample_rate": 16000,
            "channels": 1,
            "duration_s": 1.0,
            "frames": 16000,
        }
        job.decode(wav_file, PipelineConfig())
        mock_norm.assert_called_once()
        mock_val.assert_called_once_with(job.temp_wav)


def test_job_decode_raises_on_invalid_wav(tmp_path):
    """Job.decode() must propagate ValueError from validate_wav when WAV is invalid."""
    bad_wav = str(tmp_path / "bad.wav")
    _write_wav(bad_wav, sample_rate=44100, channels=2, duration_s=1.0)

    with Job(bad_wav) as job, patch("speechtelemetry.io.ffmpeg.normalize_to_wav") as mock_norm:
        mock_norm.side_effect = lambda src, dst: _write_wav(dst, 44100, 2)
        with pytest.raises(ValueError):
            job.decode(bad_wav, PipelineConfig())

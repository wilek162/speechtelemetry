"""Unit tests for io/ffmpeg.py — error paths and canonical flags."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speechtelemetry.io.ffmpeg import normalize_to_wav


def test_raises_os_error_when_ffmpeg_not_on_path():
    with (
        patch("speechtelemetry.io.ffmpeg.shutil.which", return_value=None),
        pytest.raises(OSError, match="FFmpeg not found on PATH"),
    ):
        normalize_to_wav("/input.mp4", "/output.wav")


def test_raises_runtime_error_on_nonzero_exit_code():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"Error: codec not found"

    with (
        patch("speechtelemetry.io.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.io.ffmpeg.subprocess.run", return_value=mock_result),
        pytest.raises(RuntimeError, match="FFmpeg failed"),
    ):
        normalize_to_wav("/input.mp4", "/output.wav")


def test_runtime_error_includes_exit_code():
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stderr = b"some error"

    with (
        patch("speechtelemetry.io.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.io.ffmpeg.subprocess.run", return_value=mock_result),
        pytest.raises(RuntimeError, match="exit 2"),
    ):
        normalize_to_wav("/input.mp4", "/output.wav")


def test_passes_canonical_normalization_flags():
    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("speechtelemetry.io.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.io.ffmpeg.subprocess.run", return_value=mock_result) as mock_run,
    ):
        normalize_to_wav("/input.mp4", "/output.wav")

    cmd = mock_run.call_args[0][0]
    assert "-ac" in cmd
    assert "1" in cmd  # mono
    assert "-ar" in cmd
    assert "16000" in cmd  # 16 kHz
    assert "-acodec" in cmd
    assert "pcm_s16le" in cmd


def test_succeeds_with_zero_exit_code():
    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("speechtelemetry.io.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("speechtelemetry.io.ffmpeg.subprocess.run", return_value=mock_result),
    ):
        normalize_to_wav("/input.mp4", "/output.wav")  # must not raise

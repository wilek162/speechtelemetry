"""FFmpeg media decode — Stage 1 of the pipeline.

Single responsibility: decode any media file to canonical mono 16 kHz PCM WAV.
No ML, no alignment, no domain knowledge. Pure I/O.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Canonical audio format required by all downstream backends
_TARGET_SAMPLE_RATE = 16000
_TARGET_CHANNELS = 1
_TARGET_CODEC = "pcm_s16le"


def normalize_to_wav(input_path: str, output_path: str) -> None:
    """Decode any media file to mono 16 kHz 16-bit PCM WAV.

    Args:
        input_path: Path to any audio/video file supported by FFmpeg.
        output_path: Destination .wav path (will be overwritten).

    Raises:
        EnvironmentError: if FFmpeg is not on PATH.
        RuntimeError: if FFmpeg exits with a non-zero return code.
    """
    if not shutil.which("ffmpeg"):
        raise OSError(
            "FFmpeg not found on PATH.\n"
            "  Windows: winget install --id=Gyan.FFmpeg -e\n"
            "  Ubuntu:  sudo apt-get install ffmpeg\n"
            "  Verify:  ffmpeg -version"
        )

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output without prompting
        "-i",
        input_path,
        "-vn",  # strip all video streams
        "-acodec",
        _TARGET_CODEC,
        "-ac",
        str(_TARGET_CHANNELS),  # mono
        "-ar",
        str(_TARGET_SAMPLE_RATE),  # 16 kHz
        output_path,
        "-loglevel",
        "error",
    ]

    logger.debug("FFmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, timeout=3600)  # 1hr timeout

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n{stderr}\n" f"Input: {input_path}"
        )

    logger.debug("FFmpeg: decoded to %s", output_path)

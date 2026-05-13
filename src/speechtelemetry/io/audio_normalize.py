"""Post-FFmpeg WAV validation and chunking — Stage 2 of the pipeline.

Validates the WAV file produced by ffmpeg.py and provides chunking utilities.
No ML, no decode logic. Pure I/O validation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


def validate_wav(wav_path: str) -> dict[str, object]:
    """Validate that a WAV file matches the canonical format.

    Returns:
        Dict with "sample_rate", "channels", "duration_s", "frames".

    Raises:
        ValueError: if the file does not match the expected format.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for WAV validation.\n" "pip install soundfile"
        ) from exc

    info = sf.info(wav_path)

    errors = []
    if info.samplerate != EXPECTED_SAMPLE_RATE:
        errors.append(f"Sample rate {info.samplerate} Hz != expected {EXPECTED_SAMPLE_RATE} Hz")
    if info.channels != EXPECTED_CHANNELS:
        errors.append(f"Channels {info.channels} != expected {EXPECTED_CHANNELS}")

    if errors:
        raise ValueError(
            "WAV file does not match canonical format:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\nFile: {wav_path}"
        )

    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_s": info.duration,
        "frames": info.frames,
    }


def iter_chunks(
    wav_path: str,
    chunk_duration_s: float = 30.0,
) -> Iterator[tuple[float, float]]:
    """Yield (start_s, end_s) tuples for non-overlapping audio chunks.

    Used by backends that need to process long audio in manageable pieces.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("soundfile required for chunking. pip install soundfile") from exc

    info = sf.info(wav_path)
    duration = info.duration
    start = 0.0
    while start < duration:
        end = min(start + chunk_duration_s, duration)
        yield start, end
        start = end

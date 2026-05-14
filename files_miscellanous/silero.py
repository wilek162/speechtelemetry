"""Silero VAD backend — default VAD stage.

License: MIT
No token required. ~2 MB model auto-downloads from HuggingFace Hub.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import VADBackend

logger = logging.getLogger(__name__)

try:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class SileroVADBackend(VADBackend):
    """Silero VAD — lightweight, MIT-licensed, CPU-first speech activity detection."""

    STAGE: ClassVar[str] = "vad"
    NAME: ClassVar[str] = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 30,
    ) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "silero-vad is not installed.\nRun: pip install silero-vad"
            )
        torch.set_num_threads(1)  # important for deterministic CPU results
        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_pad_ms = speech_pad_ms

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "silero-vad is not installed.\nRun: pip install silero-vad"
            )

    def get_speech_intervals(self, wav_path: str) -> list[dict[str, float]]:
        """Detect speech regions. Returns [{"start": float, "end": float}] in seconds."""
        wav = read_audio(wav_path)  # returns torch.Tensor, expects mono 16 kHz
        timestamps = get_speech_timestamps(
            wav,
            self.model,
            return_seconds=True,  # always work in seconds
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            min_speech_duration_ms=self.min_speech_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        logger.debug("Silero VAD: %d speech intervals detected", len(timestamps))
        return timestamps

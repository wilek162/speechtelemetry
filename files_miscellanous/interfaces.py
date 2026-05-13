"""Abstract Base Classes for all speechtelemetry backend stages.

Every backend must inherit from the appropriate ABC here.
Python raises TypeError at instantiation if any abstract method is not implemented —
giving developers immediate feedback rather than a runtime AttributeError deep in the stack.

Rules:
  - Zero external imports (no torch, no ML libs).
  - Only imports from speechtelemetry.types (the domain core).
  - Method signatures here ARE the backend contract. Do not deviate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from speechtelemetry.types import TranscriptDocument


class VADBackend(ABC):
    """Voice Activity Detection backend contract."""

    STAGE: str = "vad"

    @abstractmethod
    def get_speech_intervals(self, wav_path: str) -> list[dict[str, float]]:
        """Detect speech regions in a mono 16 kHz WAV file.

        Returns:
            List of {"start": float, "end": float} dicts, in seconds.
        """

    @classmethod
    def _check_available(cls) -> None:
        """Raise BackendNotAvailableError if the backend's dependencies are missing.

        Called by the pre-flight check. Default implementation does nothing.
        Backends with optional deps must override this.
        """


class ASRBackend(ABC):
    """Automatic Speech Recognition backend contract."""

    STAGE: str = "asr"

    @abstractmethod
    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        beam_size: int = 5,
    ) -> tuple[list[dict], object]:
        """Transcribe a mono 16 kHz WAV file.

        Returns:
            (segments, info) where segments is a list of dicts with at minimum:
            {"start": float, "end": float, "text": str}
            info is a backend-specific object (may be None).
        """

    @classmethod
    def _check_available(cls) -> None: ...


class AlignmentBackend(ABC):
    """Forced alignment backend contract."""

    STAGE: str = "alignment"

    @abstractmethod
    def align(
        self,
        segments: list[dict],
        audio_path: str,
        language: str,
    ) -> list[dict]:
        """Refine segment timestamps to word level via forced alignment.

        Returns:
            Updated segments list; each segment has a "words" key:
            [{"word": str, "start": float, "end": float, "score": float}, ...]
        """

    @classmethod
    def _check_available(cls) -> None: ...


class DiarizationBackend(ABC):
    """Speaker diarization backend contract."""

    STAGE: str = "diarization"

    @abstractmethod
    def diarize(
        self,
        wav_path: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict[str, object]]:
        """Assign speaker labels to time intervals.

        Returns:
            List of {"start": float, "end": float, "speaker": str} dicts.
            Speaker format: "SPEAKER_00", "SPEAKER_01", etc.
        """

    @classmethod
    def _check_available(cls) -> None: ...


class ProsodyBackend(ABC):
    """Prosody feature extraction backend contract."""

    STAGE: str = "prosody"

    @abstractmethod
    def extract_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict[str, float | str | None]:
        """Extract prosody features for a single segment.

        Returns:
            Dict compatible with ProsodyWindow. Must include at minimum:
            {"f0_mean": float, "f0_variance": float,
             "energy_mean": float, "energy_variance": float}
        """

    @classmethod
    def _check_available(cls) -> None: ...


class EmotionBackend(ABC):
    """Speech emotion recognition backend contract."""

    STAGE: str = "emotion"

    @abstractmethod
    def predict_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict[str, object]:
        """Estimate emotion probabilities for a single segment.

        Returns:
            Dict compatible with EmotionScore. Must include:
            {"label_distribution": dict[str, float],
             "confidence": float,
             "backend_name": str}
        """

    @classmethod
    def _check_available(cls) -> None: ...


class Exporter(ABC):
    """Output serialization backend contract."""

    STAGE: str = "exporter"

    @abstractmethod
    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        """Serialize a TranscriptDocument to a file at output_path."""

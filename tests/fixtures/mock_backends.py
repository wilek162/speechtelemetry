"""Deterministic ML-free mock backends for unit testing.

All mocks implement the exact interface ABCs from speechtelemetry.interfaces.
No ML deps, no I/O, deterministic outputs only.
"""

from __future__ import annotations

from speechtelemetry.interfaces import (
    AlignmentBackend,
    ASRBackend,
    EmotionBackend,
    VADBackend,
)


class MockVADBackend(VADBackend):
    """Returns a single speech interval covering 0.5s–2.5s of a 3s file."""

    def get_speech_intervals(self, wav_path: str) -> list[dict[str, float]]:
        return [{"start": 0.5, "end": 2.5}]


class MockASRBackend(ASRBackend):
    """Returns two deterministic segments with word-level data."""

    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        beam_size: int = 5,
        chunk_size_s: float | None = None,
    ) -> tuple[list[dict], object]:
        segments = [
            {
                "start": 0.5,
                "end": 1.5,
                "text": "hello world",
                "confidence": 0.95,
                "words": [
                    {"word": "hello", "start": 0.5, "end": 0.9, "score": 0.97},
                    {"word": "world", "start": 0.9, "end": 1.5, "score": 0.93},
                ],
            },
            {
                "start": 1.6,
                "end": 2.5,
                "text": "test audio",
                "confidence": 0.88,
                "words": [
                    {"word": "test", "start": 1.6, "end": 2.0, "score": 0.90},
                    {"word": "audio", "start": 2.0, "end": 2.5, "score": 0.86},
                ],
            },
        ]

        class _Info:
            language = "en"

        return segments, _Info()


class MockAlignmentBackend(AlignmentBackend):
    """Returns segments unchanged — word timestamps already present from ASR mock."""

    def align(
        self,
        segments: list[dict],
        audio_path: str,
        language: str,
    ) -> list[dict]:
        return segments


class MockEmotionBackend(EmotionBackend):
    """Returns a fixed IEMOCAP-format emotion score for every segment.

    Uses the same abbreviated label names as the real SpeechBrain IEMOCAP model
    (neu, ang, hap, sad) so mock-based tests accurately reflect real output structure.
    Probabilities sum to 1.0.
    """

    def predict_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict[str, object]:
        return {
            "label_distribution": {"neu": 0.70, "ang": 0.10, "hap": 0.15, "sad": 0.05},
            "confidence": 0.70,
            "backend_name": "mock-emotion-iemocap",
        }

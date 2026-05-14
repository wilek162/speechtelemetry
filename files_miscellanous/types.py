"""Canonical data model for speechtelemetry.

This file is the ONLY source of truth for all output types.
Rules enforced here:
  - Zero imports from any other speechtelemetry module.
  - Zero processing logic. Pure data only.
  - Every ML-derived field carries backend_name provenance.
  - Missing values use Optional[T] = None, never sentinel strings.
  - ProcessingReport.errors is always a list (never None).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Word ──────────────────────────────────────────────────────────────────────


@dataclass
class Word:
    """A single aligned word with timing and provenance."""

    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float  # [0.0, 1.0]
    token_id: int | None = None
    alignment_backend: str | None = None  # e.g. "whisperx", "none" on failure


# ── ProsodyWindow ─────────────────────────────────────────────────────────────


@dataclass
class ProsodyWindow:
    """Acoustic prosody features for a segment or sliding window."""

    f0_mean: float  # Hz; 0.0 if unvoiced
    f0_variance: float  # Hz²
    energy_mean: float  # dB
    energy_variance: float  # dB²
    speech_rate_sps: float | None = None  # syllables per second
    pause_density: float | None = None  # ratio of pause time to total time
    voice_quality_hnr: float | None = None  # harmonics-to-noise ratio (dB)
    jitter: float | None = None  # local jitter ratio
    shimmer: float | None = None  # local shimmer ratio
    backend_name: str | None = None  # e.g. "parselmouth"


# ── EmotionScore ──────────────────────────────────────────────────────────────


@dataclass
class EmotionScore:
    """Probabilistic emotion estimate for a speech segment.

    CRITICAL: label_distribution is ALWAYS a full dict — never collapsed
    to a single string. Downstream consumers decide how to present it.
    """

    label_distribution: dict[str, float]  # e.g. {"happy": 0.7, "neutral": 0.3}
    confidence: float  # [0.0, 1.0]
    backend_name: str  # e.g. "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    valence: float | None = None  # [-1.0, 1.0] if supported
    arousal: float | None = None  # [-1.0, 1.0] if supported

    @property
    def top_label(self) -> str:
        """Convenience: the emotion label with the highest probability."""
        return max(self.label_distribution, key=self.label_distribution.__getitem__)


# ── SilenceSpan ───────────────────────────────────────────────────────────────

SilenceReason = Literal["speech_gap", "non_speech", "overlap"]


@dataclass
class SilenceSpan:
    """A contiguous silent or non-speech interval."""

    start: float  # seconds
    end: float  # seconds
    duration_ms: float  # milliseconds
    reason: SilenceReason = "non_speech"


# ── Segment ───────────────────────────────────────────────────────────────────


@dataclass
class Segment:
    """A single speech segment (utterance or sentence)."""

    start: float  # seconds
    end: float  # seconds
    text: str
    confidence: float  # [0.0, 1.0]; ASR-level confidence
    speaker: str | None = None  # e.g. "SPEAKER_00"; None if diarization disabled
    silence_before_ms: float | None = None
    silence_after_ms: float | None = None
    words: list[Word] | None = None
    prosody: ProsodyWindow | None = None
    emotion: EmotionScore | None = None


# ── StageError ────────────────────────────────────────────────────────────────


@dataclass
class StageError:
    """A non-fatal error recorded during pipeline execution."""

    stage: str  # e.g. "prosody", "emotion", "alignment"
    message: str
    exception_type: str  # e.g. "RuntimeError"
    segment_index: int | None = None  # None = stage-level failure


# ── ProcessingReport ──────────────────────────────────────────────────────────


@dataclass
class ProcessingReport:
    """Telemetry and error record for a single pipeline run.

    Always present on TranscriptDocument.
    errors is always a list (may be empty), never None.
    """

    real_time_factor: float = 0.0  # wall_time / audio_duration
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)
    errors: list[StageError] = field(default_factory=list)

    def add_error(
        self,
        stage: str,
        exc: Exception,
        segment_index: int | None = None,
    ) -> None:
        """Convenience method to append a StageError."""
        self.errors.append(
            StageError(
                stage=stage,
                message=str(exc),
                exception_type=type(exc).__name__,
                segment_index=segment_index,
            )
        )


# ── TranscriptDocument ────────────────────────────────────────────────────────


@dataclass
class TranscriptDocument:
    """The canonical root output object of the speechtelemetry pipeline.

    Every field that can be absent uses Optional[T] = None.
    processing_report is always present.
    """

    source_path: str
    language: str | None  # ISO 639-1 code; None if not detected
    duration_s: float
    segments: list[Segment]
    silence_spans: list[SilenceSpan]
    processing_report: ProcessingReport

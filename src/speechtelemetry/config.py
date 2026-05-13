"""PipelineConfig — all user-tunable settings with sensible defaults.

Rules:
  - This is the ONLY place where pipeline settings are declared.
  - Every field has a default. Users can call PipelineConfig() with zero arguments.
  - No business logic here. Validation only.
  - Immutable after instantiation (frozen=True).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineConfig(BaseModel):
    """Configuration for the speechtelemetry pipeline.

    Defaults are chosen to work on any CPU-capable machine with no tokens required.
    All ML-heavy or token-gated features are disabled by default.
    """

    model_config = ConfigDict(frozen=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description=(
            "Inference device. 'cpu' is the default. "
            "Set 'cuda' explicitly after installing a CUDA-enabled torch: "
            "pip install speechtelemetry[cuda]"
        ),
    )

    # ── ASR ───────────────────────────────────────────────────────────────────
    asr_backend: Literal["faster-whisper", "whisperx", "whisper.cpp", "sensevoice"] = Field(
        default="faster-whisper",
        description="ASR backend. faster-whisper is the MIT-licensed default.",
    )
    asr_model_size: str = Field(
        default="large-v3",
        description="Model size for the ASR backend.",
    )
    asr_compute_type: str | None = Field(
        default=None,
        description="Quantization type. None = auto (float16 on GPU, int8 on CPU).",
    )
    asr_language: str | None = Field(
        default=None,
        description="ISO 639-1 language code. None = auto-detect.",
    )
    asr_beam_size: int = Field(default=5, ge=1, le=10)

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_backend: Literal["silero", "funasr"] = Field(
        default="silero",
        description="VAD backend. silero is MIT-licensed, fast, and CPU-first.",
    )
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_silence_ms: int = Field(default=300, ge=0)
    vad_min_speech_ms: int = Field(default=250, ge=0)

    # ── Alignment ─────────────────────────────────────────────────────────────
    alignment_backend: Literal["whisperx", "forcealign"] = Field(
        default="whisperx",
        description="Word-level forced alignment backend.",
    )

    # ── Diarization ───────────────────────────────────────────────────────────
    diarization_backend: Literal["pyannote", "funasr"] | None = Field(
        default=None,
        description=(
            "Speaker diarization backend. None disables diarization. "
            "pyannote requires HF_TOKEN and model license acceptance."
        ),
    )
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)

    # ── Prosody ───────────────────────────────────────────────────────────────
    prosody_backend: list[Literal["parselmouth", "copasul", "librosa", "audioflux"]] = Field(
        default_factory=list,
        description=(
            "Prosody extraction backends (ordered list). Empty by default — prosody is opt-in. "
            "parselmouth is GPL-3.0; prefer librosa (MIT) for permissive projects. "
            "See docs/license_policy.md."
        ),
    )

    # ── Emotion ───────────────────────────────────────────────────────────────
    emotion_backend: Literal["speechbrain", "emotion2vec", "emobox"] | None = Field(
        default="speechbrain",
        description="Emotion recognition backend. Apache 2.0 licensed.",
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_formats: list[Literal["json", "srt", "vtt", "textgrid"]] = Field(
        default_factory=lambda: ["json"],
        description="Output formats. JSON is the authoritative lossless format.",
    )

    # ── Performance ───────────────────────────────────────────────────────────
    chunk_audio: bool = Field(
        default=True,
        description="Process audio in chunks to reduce peak memory usage.",
    )
    max_chunk_duration_s: float = Field(
        default=30.0,
        ge=5.0,
        description="Maximum chunk size for chunked inference (seconds).",
    )

    def iter_backends(self) -> list[tuple[str, str]]:
        """Return (stage, backend_name) pairs for all enabled backends.

        Used by the pre-flight check to verify all requested backends are importable.
        """
        pairs: list[tuple[str, str]] = [
            ("asr", self.asr_backend),
            ("vad", self.vad_backend),
            ("alignment", self.alignment_backend),
        ]
        if self.diarization_backend:
            pairs.append(("diarization", self.diarization_backend))
        for pb in self.prosody_backend:
            pairs.append(("prosody", pb))
        if self.emotion_backend:
            pairs.append(("emotion", self.emotion_backend))
        return pairs

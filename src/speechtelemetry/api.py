"""Public entry points for speechtelemetry.

Only enrich_media() and enrich_audio() are public.
Stage-level functions (run_vad, run_asr, etc.) are available for advanced use.

Rules:
  - This file only validates inputs, builds config, and delegates to core/pipeline.py.
  - Zero domain logic here.
  - All returned objects are TranscriptDocument or subtypes from types.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from speechtelemetry.config import PipelineConfig
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


def enrich_media(
    input_path: str | Path,
    config: PipelineConfig | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TranscriptDocument:
    """Process any audio or video file through the full speechtelemetry pipeline.

    This is the primary public entry point. Runs all configured stages and
    returns a fully-typed TranscriptDocument.

    Args:
        input_path: Path to any audio or video file supported by FFmpeg.
        config: Pipeline configuration. Uses PipelineConfig() defaults if None.
        output_path: Optional path for a single JSON export (side effect, any location).
        output_dir: Optional directory to write all config.export_formats outputs.
            Output file names are derived from the input file stem.
            Example: output_dir="out/", export_formats=["json","srt"] →
                     out/<stem>.json, out/<stem>.srt

    Returns:
        TranscriptDocument containing all pipeline outputs.

    Raises:
        EnvironmentCheckError: if FFmpeg is missing, HF_TOKEN is not set when
            diarization is enabled, or a requested backend is not installed.
        FileNotFoundError: if input_path does not exist.

    Example::

        from speechtelemetry import enrich_media, PipelineConfig

        doc = enrich_media(
            "interview.mp4",
            config=PipelineConfig(device="cpu", export_formats=["json", "srt"]),
            output_dir="transcripts/",
        )
    """
    from speechtelemetry.core.pipeline import run_pipeline

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if config is None:
        config = PipelineConfig()

    doc = run_pipeline(str(input_path), config, output_dir=output_dir)

    if output_path is not None:
        from speechtelemetry.exporters.json_exporter import JsonExporter

        JsonExporter().export(doc, str(output_path))
        logger.info("JSON export written to %s", output_path)

    return doc


def enrich_audio(
    wav_path: str | Path,
    config: PipelineConfig | None = None,
    output_dir: str | Path | None = None,
) -> TranscriptDocument:
    """Process a pre-normalized mono 16 kHz WAV file, skipping FFmpeg decode.

    For users who manage their own audio pipeline and have already produced
    a clean mono 16 kHz WAV. Skips Stage 1 (Decode) and Stage 2 (Normalize).

    Args:
        wav_path: Path to a mono 16 kHz PCM WAV file.
        config: Pipeline configuration. Uses PipelineConfig() defaults if None.
        output_dir: Optional directory to write all config.export_formats outputs.

    Returns:
        TranscriptDocument.
    """
    from speechtelemetry.core.pipeline import run_pipeline

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if config is None:
        config = PipelineConfig()

    return run_pipeline(str(wav_path), config, skip_decode=True, output_dir=output_dir)


# ── Stage-level APIs (advanced use) ───────────────────────────────────────────
# These allow building custom pipelines from individual stages.
# Documented as advanced/unstable — signature may change in minor versions.


def run_vad(wav_path: str, config: PipelineConfig) -> list[dict[str, float]]:
    """Run VAD only. Returns list of {"start": float, "end": float} dicts."""
    from speechtelemetry.registry import get_backend

    backend = get_backend("vad", config.vad_backend)
    return backend.get_speech_intervals(wav_path)  # type: ignore[no-any-return]


def run_asr(wav_path: str, config: PipelineConfig) -> tuple[list[dict[str, Any]], object]:
    """Run ASR only. Returns (segments, info)."""
    from speechtelemetry.registry import get_backend

    backend = get_backend("asr", config.asr_backend)
    return backend.transcribe(  # type: ignore[no-any-return]
        wav_path,
        language=config.asr_language,
        beam_size=config.asr_beam_size,
    )


def run_alignment(
    segments: list[dict[str, Any]],
    wav_path: str,
    language: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Run forced alignment only. Returns segments with word-level timestamps."""
    from speechtelemetry.registry import get_backend

    backend = get_backend("alignment", config.alignment_backend)
    return backend.align(segments, wav_path, language)  # type: ignore[no-any-return]


def run_diarization(wav_path: str, config: PipelineConfig) -> list[dict[str, Any]]:
    """Run diarization only. Returns speaker-labeled time intervals."""
    if not config.diarization_backend:
        return []
    from speechtelemetry.registry import get_backend

    backend = get_backend("diarization", config.diarization_backend)
    return backend.diarize(  # type: ignore[no-any-return]
        wav_path,
        min_speakers=config.diarization_min_speakers,
        max_speakers=config.diarization_max_speakers,
    )


def run_prosody(segments: list[Any], wav_path: str, config: PipelineConfig) -> list[Any]:
    """Run prosody extraction on a list of segments. Returns updated segments."""
    from speechtelemetry.core.pipeline import _attach_prosody

    return _attach_prosody(segments, wav_path, config)


def run_emotion(segments: list[Any], wav_path: str, config: PipelineConfig) -> list[Any]:
    """Run emotion recognition on a list of segments. Returns updated segments."""
    from speechtelemetry.core.pipeline import _attach_emotion

    return _attach_emotion(segments, wav_path, config)

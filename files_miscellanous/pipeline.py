"""Pipeline orchestration — the only module that knows stage ordering.

Rules:
  - Never import a concrete backend class directly. Always use registry.
  - All stage calls are wrapped in try/except (fail-soft policy).
  - Failures append to ProcessingReport; they never propagate to the caller.
  - Hard failures (FFmpeg missing, HF_TOKEN missing) raise before any compute.
  - This module owns: stage ordering, fail-soft logic, timing, nothing else.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import tracemalloc

from speechtelemetry.config import PipelineConfig
from speechtelemetry.exceptions import EnvironmentCheckError
from speechtelemetry.registry import get_backend, resolve_backend
from speechtelemetry.types import (
    EmotionScore,
    ProcessingReport,
    ProsodyWindow,
    Segment,
    SilenceSpan,
    TranscriptDocument,
    Word,
)

logger = logging.getLogger(__name__)


# ── Pre-flight check ──────────────────────────────────────────────────────────


def preflight_check(config: PipelineConfig) -> None:
    """Verify all prerequisites before any compute starts.

    Raises:
        EnvironmentCheckError: with a list of ALL problems found, not just the first.
    """
    errors: list[str] = []

    # 1. FFmpeg — always required
    if not shutil.which("ffmpeg"):
        errors.append(
            "FFmpeg not found on PATH.\n"
            "  Windows: winget install --id=Gyan.FFmpeg -e\n"
            "  Ubuntu:  sudo apt-get install ffmpeg\n"
            "  Verify:  ffmpeg -version"
        )

    # 2. HF_TOKEN — required only when diarization is enabled
    if config.diarization_backend is not None:
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_TOKEN"):
            errors.append(
                "HF_TOKEN environment variable not set.\n"
                "  Required for diarization_backend='pyannote'.\n"
                "  Steps:\n"
                "    1. Create account at https://hf.co\n"
                "    2. Accept model license at https://hf.co/pyannote/speaker-diarization-community-1\n"
                "    3. Generate token at https://hf.co/settings/tokens\n"
                "    4. Set: export HF_TOKEN=hf_YOUR_TOKEN  (Linux/macOS)\n"
                "       Or:  $env:HF_TOKEN = 'hf_YOUR_TOKEN'  (PowerShell)"
            )

    # 3. CUDA — only if explicitly requested
    if config.device == "cuda":
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                errors.append(
                    "device='cuda' requested but torch.cuda.is_available() returned False.\n"
                    "  Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads\n"
                    "  Or use device='cpu' or device='auto'."
                )
        except ImportError:
            errors.append(
                "device='cuda' requested but torch is not installed.\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )

    # 4. Backend availability — check all requested backends
    for stage, name in config.iter_backends():
        try:
            cls = resolve_backend(stage, name)
            cls._check_available()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Backend '{name}' (stage: {stage}) is not available:\n  {exc}")

    if errors:
        msg = "Pre-flight checks failed:\n\n" + "\n\n".join(f"• {e}" for e in errors)
        raise EnvironmentCheckError(msg)


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run_pipeline(
    input_path: str,
    config: PipelineConfig,
    skip_decode: bool = False,
) -> TranscriptDocument:
    """Run the full pipeline and return a TranscriptDocument.

    Args:
        input_path: Path to media file (any format) or pre-decoded WAV.
        config: Pipeline configuration.
        skip_decode: If True, treat input_path as a ready mono 16 kHz WAV.

    Returns:
        TranscriptDocument with all requested stages populated.
    """
    from speechtelemetry.core.job import Job

    preflight_check(config)

    report = ProcessingReport()
    tracemalloc.start()

    with Job(input_path) as job:
        audio_start = time.perf_counter()

        # ── Stage 1+2: Decode + Normalize ─────────────────────────────────
        if skip_decode:
            wav_path = input_path
            logger.debug("Skipping decode stage (skip_decode=True)")
        else:
            t0 = time.perf_counter()
            wav_path = job.decode(input_path, config)
            report.stage_timings["decode"] = time.perf_counter() - t0

        audio_duration_s = _get_duration(wav_path)

        # ── Stage 3: VAD ──────────────────────────────────────────────────
        speech_intervals: list[dict] = []
        silence_spans: list[SilenceSpan] = []
        try:
            t0 = time.perf_counter()
            vad = get_backend("vad", config.vad_backend)
            speech_intervals = vad.get_speech_intervals(wav_path)
            silence_spans = _compute_silence_spans(speech_intervals, audio_duration_s)
            report.stage_timings["vad"] = time.perf_counter() - t0
            logger.info("VAD: %d speech intervals found", len(speech_intervals))
        except Exception as exc:
            report.add_error("vad", exc)
            logger.warning("VAD failed, using full audio: %s", exc)
            speech_intervals = [{"start": 0.0, "end": audio_duration_s}]

        # ── Stage 4: ASR ──────────────────────────────────────────────────
        raw_segments: list[dict] = []
        detected_language: str | None = None
        try:
            t0 = time.perf_counter()
            asr = get_backend("asr", config.asr_backend)
            raw_segments, info = asr.transcribe(
                wav_path,
                language=config.asr_language,
                beam_size=config.asr_beam_size,
            )
            detected_language = getattr(info, "language", None) or config.asr_language
            report.stage_timings["asr"] = time.perf_counter() - t0
            logger.info(
                "ASR: %d segments transcribed (lang=%s)", len(raw_segments), detected_language
            )
        except Exception as exc:
            report.add_error("asr", exc)
            logger.error("ASR failed: %s", exc)
            raw_segments = []

        # ── Stage 5: Alignment ────────────────────────────────────────────
        if raw_segments and detected_language:
            try:
                t0 = time.perf_counter()
                aligner = get_backend("alignment", config.alignment_backend)
                raw_segments = aligner.align(raw_segments, wav_path, detected_language)
                report.stage_timings["alignment"] = time.perf_counter() - t0
                logger.info("Alignment: word timestamps added to %d segments", len(raw_segments))
            except Exception as exc:
                report.add_error("alignment", exc)
                logger.warning("Alignment failed, keeping segment-level timestamps: %s", exc)

        # ── Stage 6: Diarization ──────────────────────────────────────────
        diarize_output: list[dict] = []
        if config.diarization_backend:
            try:
                t0 = time.perf_counter()
                diarizer = get_backend("diarization", config.diarization_backend)
                diarize_output = diarizer.diarize(
                    wav_path,
                    min_speakers=config.diarization_min_speakers,
                    max_speakers=config.diarization_max_speakers,
                )
                report.stage_timings["diarization"] = time.perf_counter() - t0
                logger.info("Diarization: %d speaker turns detected", len(diarize_output))
            except Exception as exc:
                report.add_error("diarization", exc)
                logger.warning("Diarization failed, returning no speaker labels: %s", exc)

        # ── Build Segment objects ─────────────────────────────────────────
        segments = _build_segments(raw_segments, diarize_output)

        # ── Stage 7: Prosody ──────────────────────────────────────────────
        segments = _attach_prosody(segments, wav_path, config, report)

        # ── Stage 8: Emotion ──────────────────────────────────────────────
        segments = _attach_emotion(segments, wav_path, config, report)

        # ── Stage 9: Export ───────────────────────────────────────────────
        if job.output_dir:
            for fmt in config.export_formats:
                try:
                    exporter = get_backend("exporter", fmt)
                    out_path = job.output_path(fmt)
                    exporter.export(
                        TranscriptDocument(
                            source_path=input_path,
                            language=detected_language,
                            duration_s=audio_duration_s,
                            segments=segments,
                            silence_spans=silence_spans,
                            processing_report=report,
                        ),
                        out_path,
                    )
                    logger.info("Exported %s to %s", fmt, out_path)
                except Exception as exc:
                    report.add_error(f"export:{fmt}", exc)
                    logger.error("Export to %s failed: %s", fmt, exc)

        # ── Finalize report ───────────────────────────────────────────────
        wall_time = time.perf_counter() - audio_start
        report.real_time_factor = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report.peak_ram_mb = peak / 1_048_576

        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                report.peak_vram_mb = torch.cuda.max_memory_allocated() / 1_048_576
        except ImportError:
            pass

        return TranscriptDocument(
            source_path=input_path,
            language=detected_language,
            duration_s=audio_duration_s,
            segments=segments,
            silence_spans=silence_spans,
            processing_report=report,
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_duration(wav_path: str) -> float:
    """Return audio duration in seconds using soundfile."""
    try:
        import soundfile as sf  # noqa: PLC0415

        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return 0.0


def _compute_silence_spans(
    speech_intervals: list[dict],
    total_duration_s: float,
) -> list[SilenceSpan]:
    spans: list[SilenceSpan] = []
    prev_end = 0.0
    for interval in speech_intervals:
        gap = interval["start"] - prev_end
        if gap > 0.01:
            spans.append(
                SilenceSpan(
                    start=prev_end,
                    end=interval["start"],
                    duration_ms=gap * 1000,
                    reason="speech_gap",
                )
            )
        prev_end = interval["end"]
    trailing = total_duration_s - prev_end
    if trailing > 0.01:
        spans.append(
            SilenceSpan(
                start=prev_end,
                end=total_duration_s,
                duration_ms=trailing * 1000,
                reason="non_speech",
            )
        )
    return spans


def _build_segments(
    raw_segments: list[dict],
    diarize_output: list[dict],
) -> list[Segment]:
    """Convert raw dicts from ASR/alignment into typed Segment objects."""
    segments: list[Segment] = []
    for raw in raw_segments:
        words = None
        if "words" in raw and raw["words"]:
            words = [
                Word(
                    text=w.get("word", w.get("text", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=float(w.get("score", w.get("confidence", 0.0))),
                )
                for w in raw["words"]
            ]

        # Assign speaker from diarization output using midpoint overlap
        speaker: str | None = None
        if diarize_output:
            seg_mid = (raw.get("start", 0.0) + raw.get("end", 0.0)) / 2
            for turn in diarize_output:
                if turn["start"] <= seg_mid <= turn["end"]:
                    speaker = str(turn["speaker"])
                    break

        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")).strip(),
                confidence=float(raw.get("confidence", raw.get("avg_logprob", 0.0))),
                speaker=speaker,
                words=words,
            )
        )
    return segments


def _attach_prosody(
    segments: list[Segment],
    wav_path: str,
    config: PipelineConfig,
    report: ProcessingReport | None = None,
) -> list[Segment]:
    """Attach prosody features to each segment. Fail-soft per segment."""
    if not config.prosody_backend:
        return segments

    for pb_name in config.prosody_backend:
        t0 = time.perf_counter()
        try:
            prosody_backend = get_backend("prosody", pb_name)
        except Exception as exc:
            if report:
                report.add_error("prosody", exc)
            logger.error("Prosody backend '%s' unavailable: %s", pb_name, exc)
            continue

        for i, seg in enumerate(segments):
            duration = seg.end - seg.start
            if duration < 0.04:  # parselmouth minimum
                if report:
                    report.add_error(
                        "prosody",
                        ValueError(f"Segment {i} too short ({duration:.3f}s < 40ms), skipping"),
                        segment_index=i,
                    )
                continue
            try:
                prosody_dict = prosody_backend.extract_segment(wav_path, seg.start, seg.end)
                seg.prosody = ProsodyWindow(
                    f0_mean=prosody_dict.get("f0_mean", 0.0),
                    f0_variance=prosody_dict.get("f0_variance", 0.0),
                    energy_mean=prosody_dict.get("energy_mean", 0.0),
                    energy_variance=prosody_dict.get("energy_variance", 0.0),
                    voice_quality_hnr=prosody_dict.get("hnr"),
                    jitter=prosody_dict.get("jitter"),
                    shimmer=prosody_dict.get("shimmer"),
                    backend_name=pb_name,
                )
            except Exception as exc:
                if report:
                    report.add_error("prosody", exc, segment_index=i)
                logger.debug("Prosody failed for segment %d: %s", i, exc)

        if report:
            report.stage_timings[f"prosody:{pb_name}"] = time.perf_counter() - t0

    return segments


def _attach_emotion(
    segments: list[Segment],
    wav_path: str,
    config: PipelineConfig,
    report: ProcessingReport | None = None,
) -> list[Segment]:
    """Attach emotion scores to each segment. Fail-soft per segment."""
    if not config.emotion_backend:
        return segments

    t0 = time.perf_counter()
    try:
        emotion_backend = get_backend("emotion", config.emotion_backend)
    except Exception as exc:
        if report:
            report.add_error("emotion", exc)
        logger.error("Emotion backend unavailable: %s", exc)
        return segments

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < 0.5:  # SpeechBrain minimum reliable duration
            logger.debug("Skipping emotion for segment %d (too short: %.3fs)", i, duration)
            continue
        try:
            emotion_dict = emotion_backend.predict_segment(wav_path, seg.start, seg.end)
            seg.emotion = EmotionScore(
                label_distribution=emotion_dict["label_distribution"],
                confidence=float(emotion_dict.get("confidence", 0.0)),
                backend_name=str(emotion_dict.get("backend_name", config.emotion_backend)),
                valence=emotion_dict.get("valence"),
                arousal=emotion_dict.get("arousal"),
            )
        except Exception as exc:
            if report:
                report.add_error("emotion", exc, segment_index=i)
            logger.debug("Emotion failed for segment %d: %s", i, exc)

    if report:
        report.stage_timings["emotion"] = time.perf_counter() - t0

    return segments

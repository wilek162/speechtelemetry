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
import math
import os
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any

from speechtelemetry.config import PipelineConfig
from speechtelemetry.exceptions import EnvironmentCheckError
from speechtelemetry.provenance import PipelineProvenance
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


def _patch_speechbrain_lazy_modules() -> None:
    """Make SpeechBrain 1.x LazyModule safe during CPython frame inspection.

    SpeechBrain uses LazyModule objects in sys.modules for optional integrations
    (k2_fsa, flair, encodec, …). CPython's inspect.getmodule() calls
    hasattr(module, '__file__') on every sys.modules entry while building a
    traceback. That triggers SpeechBrain's lazy import, which raises ImportError
    if the optional package (k2, flair, etc.) is not installed — replacing the
    real exception with an unrelated import error in our try/except blocks.

    Fix: intercept __file__ and other pure-metadata attributes on LazyModule and
    return None instead of raising, leaving all other attribute access unchanged.
    """
    try:
        from speechbrain.utils.importutils import LazyModule  # noqa: PLC0415

        _orig = LazyModule.__getattr__
        _METADATA_ATTRS = frozenset(
            ("__file__", "__spec__", "__loader__", "__package__", "__path__", "__name__")
        )

        def _safe_getattr(self: LazyModule, attr: str) -> object:
            if attr in _METADATA_ATTRS:
                try:
                    return _orig(self, attr)
                except Exception:
                    return None
            return _orig(self, attr)

        LazyModule.__getattr__ = _safe_getattr
        logger.debug("SpeechBrain LazyModule patched for safe frame inspection")
    except Exception:
        pass


_patch_speechbrain_lazy_modules()


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
    if (
        config.diarization_backend is not None
        and not os.environ.get("HF_TOKEN")
        and not os.environ.get("HUGGINGFACE_TOKEN")
    ):
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
                    "  Then reinstall torch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
                    "  Or set device='cpu' to run without a GPU."
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
    output_dir: str | Path | None = None,
) -> TranscriptDocument:
    """Run the full pipeline and return a TranscriptDocument.

    Args:
        input_path: Path to media file (any format) or pre-decoded WAV.
        config: Pipeline configuration.
        skip_decode: If True, treat input_path as a ready mono 16 kHz WAV.
        output_dir: Directory to write all configured export_formats. If None,
            exports are skipped (caller handles export via the returned doc).

    Returns:
        TranscriptDocument with all requested stages populated.
    """
    from speechtelemetry.core.job import Job

    preflight_check(config)

    report = ProcessingReport()
    provenance = PipelineProvenance()
    tracemalloc.start()

    logger.debug(
        "Pipeline start: input=%s skip_decode=%s device=%s asr=%s/%s vad=%s align=%s emotion=%s",
        input_path,
        skip_decode,
        config.device,
        config.asr_backend,
        config.asr_model_size,
        config.vad_backend,
        config.alignment_backend,
        config.emotion_backend,
    )

    with Job(input_path) as job:
        if output_dir is not None:
            job.output_dir = str(output_dir)
        audio_start = time.perf_counter()

        # ── Stage 1+2: Decode + Normalize ─────────────────────────────────
        if skip_decode:
            wav_path = input_path
            logger.debug("[decode] Skipped (skip_decode=True) — using %s directly", input_path)
        else:
            logger.debug("[decode] Starting FFmpeg decode: %s", input_path)
            t0 = time.perf_counter()
            wav_path = job.decode(input_path, config)
            elapsed = time.perf_counter() - t0
            report.stage_timings["decode"] = elapsed
            logger.debug("[decode] Done in %.3fs → %s", elapsed, wav_path)

        audio_duration_s = _get_duration(wav_path)
        logger.debug("[decode] Audio duration: %.3fs", audio_duration_s)

        # ── Stage 3: VAD ──────────────────────────────────────────────────
        speech_intervals: list[dict[str, Any]] = []
        silence_spans: list[SilenceSpan] = []
        try:
            logger.debug(
                "[vad] Starting %s (threshold=%.2f, min_silence=%dms, min_speech=%dms)",
                config.vad_backend,
                config.vad_threshold,
                config.vad_min_silence_ms,
                config.vad_min_speech_ms,
            )
            t0 = time.perf_counter()
            vad = get_backend(
                "vad",
                config.vad_backend,
                threshold=config.vad_threshold,
                min_silence_duration_ms=config.vad_min_silence_ms,
                min_speech_duration_ms=config.vad_min_speech_ms,
            )
            speech_intervals = vad.get_speech_intervals(wav_path)
            silence_spans = _compute_silence_spans(speech_intervals, audio_duration_s)
            elapsed = time.perf_counter() - t0
            report.stage_timings["vad"] = elapsed
            provenance.record("vad", config.vad_backend)
            logger.info("VAD: %d speech intervals found", len(speech_intervals))
            logger.debug(
                "[vad] Done in %.3fs → %d intervals, %d silence spans",
                elapsed,
                len(speech_intervals),
                len(silence_spans),
            )
        except Exception as exc:
            report.add_error("vad", exc)
            logger.warning("VAD failed, using full audio: %s", exc)
            speech_intervals = [{"start": 0.0, "end": audio_duration_s}]

        # ── Stage 4: ASR ──────────────────────────────────────────────────
        raw_segments: list[dict[str, Any]] = []
        detected_language: str | None = None
        try:
            logger.debug(
                "[asr] Starting %s (model=%s, lang=%s, beam=%d, chunk=%s)",
                config.asr_backend,
                config.asr_model_size,
                config.asr_language or "auto-detect",
                config.asr_beam_size,
                f"{config.max_chunk_duration_s}s" if config.chunk_audio else "disabled",
            )
            t0 = time.perf_counter()
            asr = get_backend(
                "asr",
                config.asr_backend,
                model_size=config.asr_model_size,
                device=config.device,
                compute_type=config.asr_compute_type,
            )
            asr_kwargs: dict[str, Any] = {
                "language": config.asr_language,
                "beam_size": config.asr_beam_size,
            }
            if config.chunk_audio:
                asr_kwargs["chunk_size_s"] = config.max_chunk_duration_s
            raw_segments, info = asr.transcribe(wav_path, **asr_kwargs)
            detected_language = getattr(info, "language", None) or config.asr_language
            elapsed = time.perf_counter() - t0
            report.stage_timings["asr"] = elapsed
            provenance.record(
                "asr",
                config.asr_backend,
                model_id=config.asr_model_size,
                device=config.device,
                compute_type=config.asr_compute_type,
            )
            logger.info(
                "ASR: %d segments transcribed (lang=%s)", len(raw_segments), detected_language
            )
            logger.debug(
                "[asr] Done in %.3fs → %d segments (lang=%s, RTF=%.2f)",
                elapsed,
                len(raw_segments),
                detected_language,
                elapsed / audio_duration_s if audio_duration_s > 0 else 0.0,
            )
        except Exception as exc:
            report.add_error("asr", exc)
            logger.error("ASR failed: %s", exc)
            raw_segments = []

        # ── Stage 5: Alignment ────────────────────────────────────────────
        _alignment_ran = False
        if raw_segments and detected_language:
            try:
                logger.debug(
                    "[alignment] Starting %s (%d segments, lang=%s)",
                    config.alignment_backend,
                    len(raw_segments),
                    detected_language,
                )
                t0 = time.perf_counter()
                aligner = get_backend("alignment", config.alignment_backend, device=config.device)
                raw_segments = aligner.align(raw_segments, wav_path, detected_language)
                elapsed = time.perf_counter() - t0
                report.stage_timings["alignment"] = elapsed
                provenance.record("alignment", config.alignment_backend)
                _alignment_ran = True
                total_words = sum(len(s.get("words") or []) for s in raw_segments)
                logger.info("Alignment: word timestamps added to %d segments", len(raw_segments))
                logger.debug(
                    "[alignment] Done in %.3fs → %d segments, %d total words",
                    elapsed,
                    len(raw_segments),
                    total_words,
                )
            except Exception as exc:
                report.add_error("alignment", exc)
                logger.warning("Alignment failed, keeping segment-level timestamps: %s", exc)

        # ── Stage 6: Diarization ──────────────────────────────────────────
        diarize_output: list[dict[str, Any]] = []
        if config.diarization_backend:
            try:
                logger.debug("[diarization] Starting %s", config.diarization_backend)
                t0 = time.perf_counter()
                diarizer = get_backend(
                    "diarization", config.diarization_backend, device=config.device
                )
                diarize_output = diarizer.diarize(
                    wav_path,
                    min_speakers=config.diarization_min_speakers,
                    max_speakers=config.diarization_max_speakers,
                )
                elapsed = time.perf_counter() - t0
                report.stage_timings["diarization"] = elapsed
                provenance.record("diarization", config.diarization_backend)
                logger.info("Diarization: %d speaker turns detected", len(diarize_output))
                logger.debug(
                    "[diarization] Done in %.3fs → %d speaker turns", elapsed, len(diarize_output)
                )
            except Exception as exc:
                report.add_error("diarization", exc)
                logger.warning("Diarization failed, returning no speaker labels: %s", exc)

        # ── Build Segment objects ─────────────────────────────────────────
        logger.debug("[pipeline] Building %d typed Segment objects", len(raw_segments))
        _word_alignment_backend = config.alignment_backend if _alignment_ran else None
        segments = _build_segments(
            raw_segments, diarize_output, alignment_backend=_word_alignment_backend
        )
        segments = _attach_silence_gaps(segments, silence_spans)

        # ── Stage 7: Prosody ──────────────────────────────────────────────
        segments = _attach_prosody(segments, wav_path, config, report, provenance)

        # ── Stage 8: Emotion ──────────────────────────────────────────────
        segments = _attach_emotion(segments, wav_path, config, report, provenance)

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

        logger.debug(
            "[pipeline] Complete: RTF=%.3f, RAM=%.0fMB, VRAM=%.0fMB, errors=%d",
            report.real_time_factor,
            report.peak_ram_mb,
            report.peak_vram_mb,
            len(report.errors),
        )

        doc = TranscriptDocument(
            source_path=input_path,
            language=detected_language,
            duration_s=audio_duration_s,
            segments=segments,
            silence_spans=silence_spans,
            processing_report=report,
            provenance=provenance,
        )

        # ── Stage 9: Export ───────────────────────────────────────────────
        if job.output_dir:
            for fmt in config.export_formats:
                try:
                    logger.debug("[export] Writing %s → %s", fmt, job.output_path(fmt))
                    exporter = get_backend("exporter", fmt)
                    out_path = job.output_path(fmt)
                    exporter.export(doc, out_path)
                    logger.info("Exported %s to %s", fmt, out_path)
                except Exception as exc:
                    report.add_error(f"export:{fmt}", exc)
                    logger.error("Export to %s failed: %s", fmt, exc)

        return doc


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_duration(wav_path: str) -> float:
    """Return audio duration in seconds using soundfile."""
    try:
        import soundfile as sf  # noqa: PLC0415

        info = sf.info(wav_path)
        return float(info.duration)
    except Exception:
        return 0.0


def _compute_silence_spans(
    speech_intervals: list[dict[str, Any]],
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


def _segment_confidence(raw: dict[str, Any]) -> float:
    """Derive a [0.0, 1.0] confidence for a segment dict.

    Priority:
      1. 'confidence' key (already in [0,1]) — e.g. from custom backends
      2. 'avg_logprob' (ASR log-probability) — converted via exp()
      3. Mean word alignment score — present after WhisperX alignment
      4. 0.0 fallback
    """
    if (v := raw.get("confidence")) is not None:
        return float(max(0.0, min(1.0, v)))
    if (lp := raw.get("avg_logprob")) is not None:
        return float(max(0.0, min(1.0, math.exp(lp))))
    word_dicts = raw.get("words") or []
    scores = [float(w["score"]) for w in word_dicts if "score" in w]
    return sum(scores) / len(scores) if scores else 0.0


def _assign_speaker(
    seg_start: float,
    seg_end: float,
    diarize_output: list[dict[str, Any]],
) -> str | None:
    """Return the speaker whose turn has the maximum overlap with the segment.

    Uses overlap rather than midpoint so segments that straddle a turn boundary
    or have a midpoint outside the turn still receive a speaker label.
    Returns None only when there is zero overlap with every turn.
    """
    if not diarize_output:
        return None
    best_overlap = 0.0
    best_speaker: str | None = None
    for turn in diarize_output:
        overlap = max(0.0, min(seg_end, float(turn["end"])) - max(seg_start, float(turn["start"])))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(turn["speaker"])
    return best_speaker


def _build_segments(
    raw_segments: list[dict[str, Any]],
    diarize_output: list[dict[str, Any]],
    alignment_backend: str | None = None,
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
                    alignment_backend=alignment_backend,
                )
                for w in raw["words"]
            ]

        speaker = _assign_speaker(
            float(raw.get("start", 0.0)),
            float(raw.get("end", 0.0)),
            diarize_output,
        )

        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")).strip(),
                confidence=_segment_confidence(raw),
                speaker=speaker,
                words=words,
            )
        )
    return segments


def _attach_silence_gaps(
    segments: list[Segment],
    silence_spans: list[SilenceSpan],
) -> list[Segment]:
    """Populate silence_before_ms and silence_after_ms on each Segment."""
    for seg in segments:
        for span in silence_spans:
            if abs(span.end - seg.start) < 0.05:
                seg.silence_before_ms = span.duration_ms
            if abs(span.start - seg.end) < 0.05:
                seg.silence_after_ms = span.duration_ms
    return segments


def _attach_prosody(
    segments: list[Segment],
    wav_path: str,
    config: PipelineConfig,
    report: ProcessingReport | None = None,
    provenance: PipelineProvenance | None = None,
) -> list[Segment]:
    """Attach prosody features to each segment. Fail-soft per segment."""
    if not config.prosody_backend:
        return segments

    for pb_name in config.prosody_backend:
        t0 = time.perf_counter()
        logger.debug("[prosody] Starting %s for %d segments", pb_name, len(segments))
        try:
            prosody_backend = get_backend("prosody", pb_name)
        except Exception as exc:
            if report:
                report.add_error("prosody", exc)
            logger.error("Prosody backend '%s' unavailable: %s", pb_name, exc)
            continue

        if provenance:
            provenance.record("prosody", pb_name)

        for i, seg in enumerate(segments):
            duration = seg.end - seg.start
            if duration < 0.04:  # parselmouth minimum
                logger.debug(
                    "[prosody:%s] Segment %d skipped: %.3fs < 40ms minimum", pb_name, i, duration
                )
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
                logger.debug(
                    "[prosody:%s] Segment %d [%.2f-%.2f]: f0=%.1fHz energy=%.1fdB",
                    pb_name,
                    i,
                    seg.start,
                    seg.end,
                    seg.prosody.f0_mean,
                    seg.prosody.energy_mean,
                )
            except Exception as exc:
                if report:
                    report.add_error("prosody", exc, segment_index=i)
                logger.debug("[prosody:%s] Segment %d failed: %s", pb_name, i, exc)

        elapsed = time.perf_counter() - t0
        if report:
            report.stage_timings[f"prosody:{pb_name}"] = elapsed
        logger.debug("[prosody:%s] Done in %.3fs", pb_name, elapsed)

    return segments


def _attach_emotion(
    segments: list[Segment],
    wav_path: str,
    config: PipelineConfig,
    report: ProcessingReport | None = None,
    provenance: PipelineProvenance | None = None,
) -> list[Segment]:
    """Attach emotion scores to each segment. Fail-soft per segment."""
    if not config.emotion_backend:
        return segments

    logger.debug("[emotion] Starting %s for %d segments", config.emotion_backend, len(segments))
    t0 = time.perf_counter()
    try:
        emotion_backend = get_backend("emotion", config.emotion_backend, device=config.device)
    except Exception as exc:
        if report:
            report.add_error("emotion", exc)
        logger.error("[emotion] Backend unavailable: %s", exc)
        return segments

    if provenance:
        provenance.record("emotion", config.emotion_backend, device=config.device)

    scored = 0
    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < 0.5:  # SpeechBrain minimum reliable duration
            logger.debug(
                "[emotion] Segment %d [%.2f-%.2f] skipped: %.3fs < 500ms minimum",
                i,
                seg.start,
                seg.end,
                duration,
            )
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
            scored += 1
            logger.debug(
                "[emotion] Segment %d [%.2f-%.2f]: %s (conf=%.3f)",
                i,
                seg.start,
                seg.end,
                seg.emotion.top_label,
                seg.emotion.confidence,
            )
        except Exception as exc:
            if report:
                report.add_error("emotion", exc, segment_index=i)
            logger.debug("[emotion] Segment %d failed: %s", i, exc)

    elapsed = time.perf_counter() - t0
    if report:
        report.stage_timings["emotion"] = elapsed
    logger.debug("[emotion] Done in %.3fs → %d/%d segments scored", elapsed, scored, len(segments))

    return segments

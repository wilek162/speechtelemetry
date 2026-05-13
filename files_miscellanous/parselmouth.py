"""Parselmouth (praat-parselmouth) prosody backend — default prosody stage.

License: GPL-3.0 — OPTIONAL ADAPTER ONLY.
This backend MUST NOT be a required default. It is behind an optional adapter
because GPL is viral. See docs/license_policy.md.

Provides: F0/pitch, intensity/energy, jitter, shimmer, HNR via Praat algorithms.
No Praat installation required — binary wheels ship for Win/Linux/macOS.
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import ProsodyBackend

logger = logging.getLogger(__name__)

try:
    import parselmouth
    from parselmouth.praat import call
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class ParselmouthBackend(ProsodyBackend):
    """Praat prosody analysis via parselmouth. GPL-3.0 optional adapter."""

    STAGE: ClassVar[str] = "prosody"
    NAME: ClassVar[str] = "parselmouth"

    # Singleton Sound object — load WAV once per backend instance, extract_part per segment.
    # Avoids re-opening the file for every segment (major performance win on long audio).
    _sound_cache: dict[str, object] = {}

    def __init__(
        self,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        time_step: float = 0.01,
    ) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "praat-parselmouth is not installed.\n"
                "Run: pip install praat-parselmouth\n"
                "NOTE: parselmouth is GPL-3.0 licensed. "
                "Ensure your project license is compatible."
            )
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "praat-parselmouth is not installed.\n"
                "Run: pip install praat-parselmouth"
            )

    def _get_sound(self, wav_path: str) -> object:
        """Load and cache parselmouth.Sound object (loaded once per wav_path)."""
        if wav_path not in self._sound_cache:
            self._sound_cache[wav_path] = parselmouth.Sound(wav_path)
        return self._sound_cache[wav_path]

    def extract_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict:
        """Extract prosody features for a single segment."""
        duration = end_s - start_s
        if duration < 0.04:
            raise ValueError(
                f"Segment too short ({duration:.3f}s < 40ms). "
                "Parselmouth requires at least 40ms of audio."
            )

        full_sound = self._get_sound(wav_path)
        snd = full_sound.extract_part(  # type: ignore[attr-defined]
            from_time=start_s,
            to_time=end_s,
            window_shape=parselmouth.WindowShape.RECTANGULAR,
            relative_width=1.0,
            preserve_times=False,
        )

        # ── Pitch (F0) ───────────────────────────────────────────────────
        pitch = snd.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        pitch_values = pitch.selected_array["frequency"]
        voiced_frames = pitch_values[pitch_values > 0]  # exclude unvoiced frames
        f0_mean = float(np.mean(voiced_frames)) if len(voiced_frames) > 0 else 0.0
        f0_variance = float(np.var(voiced_frames)) if len(voiced_frames) > 0 else 0.0

        # ── Intensity (Energy) ───────────────────────────────────────────
        intensity = snd.to_intensity(
            minimum_pitch=self.pitch_floor,
            time_step=self.time_step,
        )
        energy_mean = float(call(intensity, "Get mean", 0, 0, "energy"))
        energy_std = float(call(intensity, "Get standard deviation", 0, 0))
        energy_variance = energy_std ** 2

        # ── Voice Quality ────────────────────────────────────────────────
        jitter: float | None = None
        shimmer: float | None = None
        hnr: float | None = None

        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", self.pitch_floor, self.pitch_ceiling)
            jitter = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shimmer = float(
                call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, self.pitch_floor, 0.1, 1.0)
            hnr = float(call(harmonicity, "Get mean", 0, 0))
        except Exception as exc:
            logger.debug("Voice quality extraction failed for segment: %s", exc)

        return {
            "f0_mean": f0_mean,
            "f0_variance": f0_variance,
            "energy_mean": energy_mean,
            "energy_variance": energy_variance,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "backend_name": "parselmouth",
        }

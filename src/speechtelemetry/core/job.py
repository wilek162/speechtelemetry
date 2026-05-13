"""Job lifecycle — temp file creation and cleanup.

A Job manages the lifecycle of a single pipeline run:
  - Creates a temp directory for intermediate files
  - Provides a canonical path for the decoded WAV
  - Cleans up temp files in __exit__ (even if pipeline fails)
  - Tracks output paths for export

Rules:
  - No domain logic here. Lifecycle and file paths only.
  - Always clean up in finally/__exit__, even on exceptions.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from speechtelemetry.config import PipelineConfig

logger = logging.getLogger(__name__)


class Job:
    """Context manager for a single speechtelemetry pipeline run."""

    def __init__(self, input_path: str) -> None:
        self.input_path = input_path
        self.job_id = str(uuid.uuid4())[:8]
        self._temp_dir: str | None = None
        self.output_dir: str | None = None

    def __enter__(self) -> Job:
        self._temp_dir = tempfile.mkdtemp(prefix=f"st_{self.job_id}_")
        logger.debug("Job %s: temp dir %s", self.job_id, self._temp_dir)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove temp directory and release GPU memory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                logger.debug("Job %s: cleaned up %s", self.job_id, self._temp_dir)
            except Exception as exc:
                logger.warning("Job %s: cleanup failed: %s", self.job_id, exc)

        # Release GPU memory if torch is available
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def temp_wav(self) -> str:
        """Path for the decoded canonical WAV file."""
        assert self._temp_dir is not None, "Job not started (use as context manager)"
        return os.path.join(self._temp_dir, "audio.wav")

    def output_path(self, fmt: str) -> str:
        """Derive an output file path for a given format."""
        stem = Path(self.input_path).stem
        ext_map = {"json": ".json", "srt": ".srt", "vtt": ".vtt", "textgrid": ".TextGrid"}
        ext = ext_map.get(fmt, f".{fmt}")
        base_dir = self.output_dir or str(Path(self.input_path).parent)
        return os.path.join(base_dir, f"{stem}{ext}")

    def decode(self, input_path: str, config: PipelineConfig) -> str:
        """Decode input media to canonical mono 16 kHz WAV.

        Returns the path to the decoded WAV file.
        """
        from speechtelemetry.io.audio_normalize import validate_wav  # noqa: PLC0415
        from speechtelemetry.io.ffmpeg import normalize_to_wav  # noqa: PLC0415

        normalize_to_wav(input_path, self.temp_wav)
        validate_wav(self.temp_wav)
        return self.temp_wav

"""WhisperX alignment backend — default word-level alignment stage.

License: BSD-4-Clause
Provides word-level timestamps via wav2vec2 forced alignment.
"""

from __future__ import annotations

import gc
import logging
from typing import ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import AlignmentBackend

logger = logging.getLogger(__name__)

try:
    import whisperx

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class WhisperXAlignmentBackend(AlignmentBackend):
    """WhisperX forced alignment — best word-level timestamps via wav2vec2."""

    STAGE: ClassVar[str] = "alignment"
    NAME: ClassVar[str] = "whisperx"

    def __init__(self, device: str = "auto") -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError("whisperx is not installed.\nRun: pip install whisperx")
        if device == "auto":
            try:
                import torch  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self._align_model: object | None = None
        self._metadata: object | None = None
        self._loaded_lang: str | None = None

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError("whisperx is not installed.\nRun: pip install whisperx")

    def align(
        self,
        segments: list[dict],
        audio_path: str,
        language: str,
    ) -> list[dict]:
        """Refine segment timestamps to word level.

        Returns segments with a "words" key on each segment containing
        [{"word": str, "start": float, "end": float, "score": float}].
        """
        # Reload alignment model only when language changes
        if self._loaded_lang != language:
            if self._align_model is not None:
                del self._align_model
                gc.collect()
                try:
                    import torch  # noqa: PLC0415

                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            logger.info("WhisperX: loading alignment model for language '%s'", language)
            self._align_model, self._metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            self._loaded_lang = language

        audio = whisperx.load_audio(audio_path)
        result = whisperx.align(
            segments,
            self._align_model,
            self._metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return result["segments"]

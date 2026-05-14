"""faster-whisper ASR backend — default ASR stage.

License: MIT
4x faster than original Whisper. int8 quantization on CPU. CTranslate2-based.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import ASRBackend

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class FasterWhisperBackend(ASRBackend):
    """faster-whisper — MIT-licensed, best speed/quality tradeoff for local batch ASR."""

    STAGE: ClassVar[str] = "asr"
    NAME: ClassVar[str] = "faster-whisper"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "faster-whisper is not installed.\nRun: pip install faster-whisper"
            )

        if device == "auto":
            try:
                import torch  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper model '%s' on %s (%s)", model_size, device, compute_type
        )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.device = device

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "faster-whisper is not installed.\nRun: pip install faster-whisper"
            )

    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
    ) -> tuple[list[dict], object]:
        """Transcribe a WAV file. Returns (segments_list, info).

        Note: word_timestamps=False — word-level timestamps come from the alignment stage.
        """
        segments_gen, info = self.model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=False,  # use WhisperX alignment for accuracy
            vad_filter=vad_filter,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # fallback chain
        )

        # Materialise the generator before returning — it's lazy and single-use
        segments = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in segments_gen
        ]

        logger.debug(
            "faster-whisper: %d segments, lang=%s, RTF=%.2f",
            len(segments),
            getattr(info, "language", "?"),
            getattr(info, "duration", 0) and 0,
        )
        return segments, info

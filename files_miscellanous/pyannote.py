"""pyannote.audio diarization backend — default diarization stage (optional).

License: MIT (library) + CC-BY-4.0 (model weights)
Requires: HF_TOKEN environment variable + model license acceptance.
"""
from __future__ import annotations

import logging
import os
from typing import ClassVar, Optional

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import DiarizationBackend

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class PyannoteBackend(DiarizationBackend):
    """pyannote.audio — best open-source speaker diarization ecosystem."""

    STAGE: ClassVar[str] = "diarization"
    NAME: ClassVar[str] = "pyannote"

    RECOMMENDED_MODEL = "pyannote/speaker-diarization-community-1"
    FALLBACK_MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        device: str = "auto",
        model_id: str = RECOMMENDED_MODEL,
    ) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "pyannote.audio is not installed.\n"
                "Run: pip install pyannote.audio\n"
                "Also required:\n"
                "  1. Accept model license at https://hf.co/pyannote/speaker-diarization-community-1\n"
                "  2. Set HF_TOKEN environment variable"
            )

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN environment variable not set.\n"
                "Required for pyannote.audio diarization.\n"
                "Steps:\n"
                "  1. Create account at https://hf.co\n"
                "  2. Accept model license at https://hf.co/pyannote/speaker-diarization-community-1\n"
                "  3. Generate token at https://hf.co/settings/tokens\n"
                "  4. export HF_TOKEN=hf_YOUR_TOKEN"
            )

        if device == "auto":
            try:
                import torch  # noqa: PLC0415
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Loading pyannote model '%s' on %s", model_id, device)
        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

        try:
            import torch  # noqa: PLC0415
            self.pipeline.to(torch.device(device))
        except Exception as exc:
            logger.warning("Could not move pyannote pipeline to %s: %s", device, exc)

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "pyannote.audio is not installed.\n"
                "Run: pip install pyannote.audio"
            )

    def diarize(
        self,
        wav_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[dict]:
        """Assign speaker labels to time intervals.

        Returns [{"start": float, "end": float, "speaker": str}].
        Speaker format: "SPEAKER_00", "SPEAKER_01", etc.
        """
        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        output = self.pipeline(wav_path, **kwargs)

        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in output.itertracks(yield_label=True)
        ]
        logger.debug("pyannote: %d speaker turns", len(turns))
        return turns

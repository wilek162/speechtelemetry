"""pyannote.audio diarization backend — default diarization stage (optional).

License: MIT (library) + CC-BY-4.0 (model weights)
Requires: HF_TOKEN environment variable + model license acceptance.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import DiarizationBackend

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def _patch_pyannote_compat() -> None:
    """Apply runtime compatibility patches for pyannote 3.x on newer library versions.

    Patches applied:
      1. hf_hub_download: pyannote 3.x passes use_auth_token= but huggingface_hub >= 0.23
         renamed the parameter to 'token'.
      2. lightning_fabric._load: PyTorch 2.6+ defaults weights_only=True which rejects
         pyannote's older checkpoint format. Default to False for trusted HF model files.
      3. SpeechBrain LazyModule: CPython's inspect.getmodule() triggers LazyModule.__getattr__
         for metadata attributes, causing ImportError for uninstalled optional deps (k2, flair).
         Return None for pure metadata attributes instead of triggering the lazy import.
    """
    try:
        from speechbrain.utils.importutils import LazyModule  # noqa: PLC0415

        _orig_la = LazyModule.__getattr__
        _METADATA = frozenset(
            ("__file__", "__spec__", "__loader__", "__package__", "__path__", "__name__")
        )

        def _safe_la(self: LazyModule, attr: str) -> object:
            if attr in _METADATA:
                try:
                    return _orig_la(self, attr)
                except Exception:
                    return None
            return _orig_la(self, attr)

        LazyModule.__getattr__ = _safe_la
        logger.debug("SpeechBrain LazyModule patched (diarization context)")
    except Exception as exc:
        logger.debug("SpeechBrain LazyModule patch skipped: %s", exc)
    try:
        import huggingface_hub as _hfh

        _orig_hf = _hfh.hf_hub_download

        def _compat_download(*args: Any, **kwargs: Any) -> Any:
            if "use_auth_token" in kwargs:
                auth = kwargs.pop("use_auth_token")
                if auth is not None and "token" not in kwargs:
                    kwargs["token"] = auth
            return _orig_hf(*args, **kwargs)

        # Patch the local hf_hub_download reference in every pyannote module.
        import importlib as _il

        _patched_hf = 0
        for _mod_name in (
            "pyannote.audio.core.pipeline",
            "pyannote.audio.core.model",
            "pyannote.audio.core.inference",
        ):
            try:
                _mod = _il.import_module(_mod_name)
                if hasattr(_mod, "hf_hub_download"):
                    _mod.hf_hub_download = _compat_download  # type: ignore[attr-defined]
                    _patched_hf += 1
            except Exception:
                pass

        logger.debug("hf_hub_download compat patch: %d pyannote modules", _patched_hf)
    except Exception as exc:
        logger.debug("hf_hub_download patch skipped: %s", exc)

    try:
        import lightning_fabric.utilities.cloud_io as _lf
        import pyannote.audio.core.model as _pam
        import pytorch_lightning.core.saving as _pls

        _orig_load = _lf._load

        def _compat_load(
            path_or_url: Any, map_location: Any = None, weights_only: Any = None
        ) -> Any:
            if weights_only is None:
                weights_only = False
            return _orig_load(path_or_url, map_location=map_location, weights_only=weights_only)

        _lf._load = _compat_load
        _pam.pl_load = _compat_load
        _pls.pl_load = _compat_load  # type: ignore[attr-defined]
        logger.debug("weights_only=False compat patch applied to pl_load in all locations")
    except Exception as exc:
        logger.debug("pl_load weights_only patch skipped: %s", exc)


if _AVAILABLE:
    _patch_pyannote_compat()


class PyannoteBackend(DiarizationBackend):
    """pyannote.audio — best open-source speaker diarization ecosystem."""

    STAGE: ClassVar[str] = "diarization"
    NAME: ClassVar[str] = "pyannote"

    RECOMMENDED_MODEL = "pyannote/speaker-diarization-community-1"
    FALLBACK_MODEL = "pyannote/speaker-diarization-3.1"
    OPEN_MODEL = "tensorlake/speaker-diarization-3.1"

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
            raise OSError(
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

        self.pipeline = self._load_pipeline(model_id, hf_token, device)
        if self.pipeline is None:
            raise BackendNotAvailableError(
                f"Could not load pyannote diarization model '{model_id}'.\n"
                f"If the model is gated, accept the license at https://hf.co/{model_id}\n"
                f"and ensure HF_TOKEN is set to a token with model access."
            )

        try:
            import torch  # noqa: PLC0415

            self.pipeline.to(torch.device(device))
        except Exception as exc:
            logger.warning("Could not move pyannote pipeline to %s: %s", device, exc)

    def _load_pipeline(self, model_id: str, hf_token: str, device: str) -> Any:
        """Try loading the pipeline, falling back through model alternatives.

        Priority:
          1. Requested model_id
          2. FALLBACK_MODEL (pyannote/speaker-diarization-3.1)
          3. OPEN_MODEL (tensorlake/speaker-diarization-3.1, MIT, no gating)
        """
        candidates = [model_id]
        if model_id != self.FALLBACK_MODEL:
            candidates.append(self.FALLBACK_MODEL)
        if self.OPEN_MODEL not in candidates:
            candidates.append(self.OPEN_MODEL)

        for candidate in candidates:
            logger.info("Loading pyannote model '%s' on %s", candidate, device)
            pipeline = Pipeline.from_pretrained(candidate, use_auth_token=hf_token)
            if pipeline is not None:
                logger.info("Loaded pyannote model '%s'", candidate)
                return pipeline
            logger.warning(
                "pyannote model '%s' not available (gated or download failed), trying next",
                candidate,
            )
        return None

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "pyannote.audio is not installed.\nRun: pip install pyannote.audio"
            )

    def diarize(
        self,
        wav_path: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict[str, object]]:
        """Assign speaker labels to time intervals.

        Returns [{"start": float, "end": float, "speaker": str}].
        Speaker format: "SPEAKER_00", "SPEAKER_01", etc.
        """
        kwargs: dict[str, int] = {}
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

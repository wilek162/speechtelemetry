"""Backend registry — the single place where backend names map to classes.

Rules:
  - This is the ONLY file in the codebase that imports concrete backend modules.
  - Imports are lazy (string paths resolved at call time) to avoid loading ML
    frameworks at import time.
  - Adding a new backend = one line in REGISTRY. Nothing else changes.
  - register() is the public extension point for external users and third-party packages.
  - _autodiscover() loads entry-point-registered backends at module import time.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import warnings
from typing import Any

from speechtelemetry.exceptions import BackendNotFoundError

logger = logging.getLogger(__name__)

# ── Registry storage ──────────────────────────────────────────────────────────
# Maps stage → backend_name → "module.path.ClassName" (lazy string imports).
# Concrete classes are imported only when get_backend() is called.

_REGISTRY: dict[str, dict[str, str]] = {
    "asr": {
        "faster-whisper": "speechtelemetry.backends.asr.faster_whisper.FasterWhisperBackend",
        "whisperx": "speechtelemetry.backends.asr.whisperx.WhisperXASRBackend",
        "whisper.cpp": "speechtelemetry.backends.asr.whisper_cpp.WhisperCppBackend",
        "sensevoice": "speechtelemetry.backends.asr.sensevoice.SenseVoiceBackend",
    },
    "vad": {
        "silero": "speechtelemetry.backends.vad.silero.SileroVADBackend",
        "funasr": "speechtelemetry.backends.vad.funasr.FunASRVADBackend",
    },
    "alignment": {
        "whisperx": "speechtelemetry.backends.alignment.whisperx.WhisperXAlignmentBackend",
        "forcealign": "speechtelemetry.backends.alignment.forcealign.ForceAlignBackend",
    },
    "diarization": {
        "pyannote": "speechtelemetry.backends.diarization.pyannote.PyannoteBackend",
        "funasr": "speechtelemetry.backends.diarization.funasr.FunASRDiarizationBackend",
    },
    "prosody": {
        "parselmouth": "speechtelemetry.backends.prosody.parselmouth.ParselmouthBackend",
        # GPL-3.0 — see docs/license_policy.md
        "copasul": "speechtelemetry.backends.prosody.copasul.CoPaSulBackend",
        "librosa": "speechtelemetry.backends.prosody.librosa.LibrosaBackend",
        "audioflux": "speechtelemetry.backends.prosody.audioflux.AudioFluxBackend",
    },
    "emotion": {
        "speechbrain": "speechtelemetry.backends.emotion.speechbrain.SpeechBrainEmotionBackend",
        "emotion2vec": "speechtelemetry.backends.emotion.emotion2vec.Emotion2VecBackend",
        "emobox": "speechtelemetry.backends.emotion.emobox.EmoBoxBackend",
    },
    "exporter": {
        "json": "speechtelemetry.exporters.json_exporter.JsonExporter",
        "srt": "speechtelemetry.exporters.srt.SRTExporter",
        "vtt": "speechtelemetry.exporters.vtt.VTTExporter",
        "textgrid": "speechtelemetry.exporters.textgrid.TextGridExporter",
    },
}

# License flags for backends that require runtime warnings.
# Maps "module.path.ClassName" → SPDX license identifier.
_LICENSE_FLAGS: dict[str, str] = {
    "speechtelemetry.backends.prosody.parselmouth.ParselmouthBackend": "GPL-3.0",
}


# ── Public API ────────────────────────────────────────────────────────────────


def register(stage: str, name: str, cls: type[Any]) -> None:
    """Register a backend class for a given stage and name.

    This is the primary extension point. Call before enrich_media() to add
    a custom backend without modifying library source code.

    Example::

        from speechtelemetry import registry
        from speechtelemetry.interfaces import EmotionBackend

        class MyModel(EmotionBackend):
            STAGE = "emotion"
            NAME  = "my-model"
            def predict_segment(self, wav_path, start_s, end_s): ...

        registry.register("emotion", "my-model", MyModel)
    """
    if stage not in _REGISTRY:
        _REGISTRY[stage] = {}
    _REGISTRY[stage][name] = f"{cls.__module__}.{cls.__qualname__}"
    # Also store the class directly to avoid re-import
    _CLASS_CACHE[f"{stage}:{name}"] = cls


_CLASS_CACHE: dict[str, type[Any]] = {}


def resolve_backend(stage: str, name: str) -> type[Any]:
    """Resolve a backend name to its class.

    Raises:
        BackendNotFoundError: if the name is not registered for the stage.
    """
    cache_key = f"{stage}:{name}"
    if cache_key in _CLASS_CACHE:
        return _CLASS_CACHE[cache_key]

    stage_registry = _REGISTRY.get(stage, {})
    if name not in stage_registry:
        available = list(stage_registry.keys())
        raise BackendNotFoundError(
            f"Unknown {stage} backend: '{name}'. "
            f"Available: {available}. "
            f"Register a custom backend with registry.register('{stage}', '{name}', MyClass)."
        )

    path = stage_registry[name]
    module_path, class_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Backend '{name}' for stage '{stage}' could not be imported.\n"
            f"This usually means an optional dependency is not installed.\n"
            f"Original error: {exc}"
        ) from exc

    cls: type[Any] = getattr(module, class_name)

    # Warn about GPL-licensed backends at resolution time
    if path in _LICENSE_FLAGS:
        license_id = _LICENSE_FLAGS[path]
        warnings.warn(
            f"Backend '{name}' ({class_name}) is licensed under {license_id}. "
            f"Ensure your project's license is compatible. "
            f"See docs/license_policy.md for details.",
            stacklevel=3,
        )

    _CLASS_CACHE[cache_key] = cls
    return cls


def get_backend(stage: str, name: str, **kwargs: Any) -> Any:
    """Resolve and instantiate a backend.

    Convenience wrapper for resolve_backend() + instantiation.
    Pass kwargs to the backend constructor.
    """
    cls = resolve_backend(stage, name)
    return cls(**kwargs)


def list_backends(stage: str) -> list[str]:
    """Return registered backend names for a stage."""
    return list(_REGISTRY.get(stage, {}).keys())


def list_stages() -> list[str]:
    """Return all registered stage names."""
    return list(_REGISTRY.keys())


# ── Entry-point auto-discovery ────────────────────────────────────────────────


def _autodiscover() -> None:
    """Load third-party backends registered via Python entry points.

    Third-party packages can ship backends without modifying this file by
    declaring in their pyproject.toml::

        [project.entry-points."speechtelemetry.backends"]
        my-asr = "my_package:MyASRBackend"

    The backend class must define class-level STAGE and NAME attributes.
    """
    try:
        eps = importlib.metadata.entry_points(group="speechtelemetry.backends")
    except Exception:
        return

    for ep in eps:
        try:
            cls = ep.load()
            stage = getattr(cls, "STAGE", None)
            name = getattr(cls, "NAME", None)
            if stage and name:
                register(stage, name, cls)
                logger.debug("Auto-discovered backend: %s/%s from %s", stage, name, ep.name)
            else:
                warnings.warn(
                    f"Entry-point backend '{ep.name}' is missing STAGE or NAME class attributes. "
                    f"Skipping auto-discovery.",
                    stacklevel=1,
                )
        except Exception as exc:
            warnings.warn(
                f"Failed to load speechtelemetry backend from entry point '{ep.name}': {exc}",
                stacklevel=1,
            )


_autodiscover()

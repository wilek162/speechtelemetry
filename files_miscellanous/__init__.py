"""speechtelemetry — local-first speech intelligence.

Public API surface. Everything re-exported here is stable and versioned.
Everything NOT re-exported here is internal and may change without notice.

Typical usage::

    from speechtelemetry import enrich_media, PipelineConfig

    doc = enrich_media("interview.mp4", config=PipelineConfig())
    for seg in doc.segments:
        print(seg.start, seg.text, seg.emotion.top_label if seg.emotion else "")
"""

from speechtelemetry.api import enrich_audio, enrich_media
from speechtelemetry.config import PipelineConfig
from speechtelemetry.exceptions import (
    BackendError,
    BackendNotAvailableError,
    BackendNotFoundError,
    EnvironmentCheckError,
    SpeechTelemetryError,
)
from speechtelemetry.interfaces import (
    AlignmentBackend,
    ASRBackend,
    DiarizationBackend,
    EmotionBackend,
    Exporter,
    ProsodyBackend,
    VADBackend,
)
from speechtelemetry.types import (
    EmotionScore,
    ProcessingReport,
    ProsodyWindow,
    Segment,
    SilenceSpan,
    StageError,
    TranscriptDocument,
    Word,
)
from speechtelemetry import registry  # noqa: F401 — re-exported for registry.register()

__version__ = "0.1.0"

__all__ = [
    # Entry points
    "enrich_media",
    "enrich_audio",
    # Configuration
    "PipelineConfig",
    # Data model
    "TranscriptDocument",
    "Segment",
    "Word",
    "SilenceSpan",
    "ProsodyWindow",
    "EmotionScore",
    "ProcessingReport",
    "StageError",
    # Interfaces (for custom backend development)
    "ASRBackend",
    "VADBackend",
    "AlignmentBackend",
    "DiarizationBackend",
    "ProsodyBackend",
    "EmotionBackend",
    "Exporter",
    # Registry (for custom backend registration)
    "registry",
    # Exceptions
    "SpeechTelemetryError",
    "BackendNotAvailableError",
    "BackendNotFoundError",
    "EnvironmentCheckError",
    "BackendError",
    # Version
    "__version__",
]

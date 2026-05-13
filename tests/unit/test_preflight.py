"""Setup and preflight tests — verify the package imports and environment basics."""
import importlib
import sys


def test_package_importable():
    import speechtelemetry
    assert speechtelemetry.__version__ == "0.1.0"


def test_public_api_exports():
    from speechtelemetry import (
        PipelineConfig,
        TranscriptDocument,
        enrich_media,
        enrich_audio,
        registry,
    )
    assert callable(enrich_media)
    assert callable(enrich_audio)
    assert PipelineConfig is not None
    assert TranscriptDocument is not None


def test_all_types_importable():
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


def test_all_interfaces_importable():
    from speechtelemetry.interfaces import (
        AlignmentBackend,
        ASRBackend,
        DiarizationBackend,
        EmotionBackend,
        Exporter,
        ProsodyBackend,
        VADBackend,
    )


def test_all_exceptions_importable():
    from speechtelemetry.exceptions import (
        BackendError,
        BackendNotAvailableError,
        BackendNotFoundError,
        EnvironmentCheckError,
        SpeechTelemetryError,
    )


def test_registry_importable():
    from speechtelemetry import registry
    assert hasattr(registry, "register")
    assert hasattr(registry, "get_backend")
    assert hasattr(registry, "list_stages")
    assert hasattr(registry, "list_backends")


def test_config_importable():
    from speechtelemetry.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg is not None


def test_exporters_importable():
    from speechtelemetry.exporters.json_exporter import JsonExporter
    from speechtelemetry.exporters.srt import SRTExporter
    from speechtelemetry.exporters.vtt import VTTExporter
    from speechtelemetry.exporters.textgrid import TextGridExporter


def test_io_modules_importable():
    from speechtelemetry.io import ffmpeg
    from speechtelemetry.io import audio_normalize


def test_core_modules_importable():
    from speechtelemetry.core import pipeline
    from speechtelemetry.core import job

"""Setup and preflight tests — verify the package imports and environment basics."""


def test_package_importable():
    import speechtelemetry

    assert speechtelemetry.__version__ == "0.1.0"


def test_public_api_exports():
    from speechtelemetry import (
        PipelineConfig,
        TranscriptDocument,
        enrich_audio,
        enrich_media,
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

    assert all(
        t is not None
        for t in [
            TranscriptDocument,
            Segment,
            Word,
            SilenceSpan,
            ProsodyWindow,
            EmotionScore,
            ProcessingReport,
            StageError,
        ]
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

    assert all(
        t is not None
        for t in [
            ASRBackend,
            VADBackend,
            AlignmentBackend,
            DiarizationBackend,
            ProsodyBackend,
            EmotionBackend,
            Exporter,
        ]
    )


def test_all_exceptions_importable():
    from speechtelemetry.exceptions import (
        BackendError,
        BackendNotAvailableError,
        BackendNotFoundError,
        EnvironmentCheckError,
        SpeechTelemetryError,
    )

    assert all(
        issubclass(e, Exception)
        for e in [
            SpeechTelemetryError,
            BackendNotAvailableError,
            BackendNotFoundError,
            EnvironmentCheckError,
            BackendError,
        ]
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
    from speechtelemetry.exporters.textgrid import TextGridExporter
    from speechtelemetry.exporters.vtt import VTTExporter

    assert all(t is not None for t in [JsonExporter, SRTExporter, VTTExporter, TextGridExporter])


def test_io_modules_importable():
    from speechtelemetry.io.audio_normalize import iter_chunks, validate_wav
    from speechtelemetry.io.ffmpeg import normalize_to_wav

    assert callable(validate_wav)
    assert callable(iter_chunks)
    assert callable(normalize_to_wav)


def test_core_modules_importable():
    from speechtelemetry.core.job import Job
    from speechtelemetry.core.pipeline import preflight_check, run_pipeline

    assert callable(run_pipeline)
    assert callable(preflight_check)
    assert Job is not None

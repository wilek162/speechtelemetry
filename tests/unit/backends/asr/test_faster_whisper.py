"""Contract tests for FasterWhisperBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend

    assert FasterWhisperBackend is not None


def test_inherits_from_asr_abc():
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend
    from speechtelemetry.interfaces import ASRBackend

    assert issubclass(FasterWhisperBackend, ASRBackend)


def test_stage_attribute():
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend

    assert FasterWhisperBackend.STAGE == "asr"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.asr.faster_whisper as mod
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        FasterWhisperBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.asr.faster_whisper as mod
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        FasterWhisperBackend._check_available()


def test_transcribe_signature_matches_abc():
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend

    sig = inspect.signature(FasterWhisperBackend.transcribe)
    params = list(sig.parameters)
    assert "self" in params
    assert "wav_path" in params
    assert "language" in params
    assert "beam_size" in params
    assert "chunk_size_s" in params


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.asr.faster_whisper as mod
    from speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        FasterWhisperBackend()

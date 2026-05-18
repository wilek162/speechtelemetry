"""Contract tests for SileroVADBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.vad.silero import SileroVADBackend

    assert SileroVADBackend is not None


def test_inherits_from_vad_abc():
    from speechtelemetry.backends.vad.silero import SileroVADBackend
    from speechtelemetry.interfaces import VADBackend

    assert issubclass(SileroVADBackend, VADBackend)


def test_stage_attribute():
    from speechtelemetry.backends.vad.silero import SileroVADBackend

    assert SileroVADBackend.STAGE == "vad"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.vad.silero as mod
    from speechtelemetry.backends.vad.silero import SileroVADBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        SileroVADBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.vad.silero as mod
    from speechtelemetry.backends.vad.silero import SileroVADBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        SileroVADBackend._check_available()


def test_get_speech_intervals_signature_matches_abc():
    from speechtelemetry.backends.vad.silero import SileroVADBackend

    sig = inspect.signature(SileroVADBackend.get_speech_intervals)
    params = list(sig.parameters)
    assert "self" in params
    assert "wav_path" in params


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.vad.silero as mod
    from speechtelemetry.backends.vad.silero import SileroVADBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        SileroVADBackend()

"""Unit tests for speechtelemetry.exceptions — hierarchy and messages."""
import pytest
from speechtelemetry.exceptions import (
    BackendError,
    BackendNotAvailableError,
    BackendNotFoundError,
    EnvironmentCheckError,
    SpeechTelemetryError,
)


def test_base_exception():
    exc = SpeechTelemetryError("base error")
    assert str(exc) == "base error"
    assert isinstance(exc, Exception)


def test_backend_not_available_is_subclass():
    exc = BackendNotAvailableError("not installed")
    assert isinstance(exc, SpeechTelemetryError)


def test_backend_not_found_is_subclass():
    exc = BackendNotFoundError("not registered")
    assert isinstance(exc, SpeechTelemetryError)


def test_environment_check_error_is_subclass():
    exc = EnvironmentCheckError("ffmpeg missing")
    assert isinstance(exc, SpeechTelemetryError)


def test_backend_error_is_subclass():
    exc = BackendError("init failed")
    assert isinstance(exc, SpeechTelemetryError)


def test_exceptions_can_be_raised_and_caught():
    with pytest.raises(SpeechTelemetryError):
        raise BackendNotAvailableError("test")

    with pytest.raises(BackendNotFoundError):
        raise BackendNotFoundError("not found")

    with pytest.raises(EnvironmentCheckError):
        raise EnvironmentCheckError("env bad")

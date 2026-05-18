"""Contract tests for PyannoteBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
import os
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend

    assert PyannoteBackend is not None


def test_inherits_from_diarization_abc():
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend
    from speechtelemetry.interfaces import DiarizationBackend

    assert issubclass(PyannoteBackend, DiarizationBackend)


def test_stage_attribute():
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend

    assert PyannoteBackend.STAGE == "diarization"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.diarization.pyannote as mod
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        PyannoteBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.diarization.pyannote as mod
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        PyannoteBackend._check_available()


def test_diarize_signature_matches_abc():
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend

    sig = inspect.signature(PyannoteBackend.diarize)
    params = list(sig.parameters)
    assert "self" in params
    assert "wav_path" in params
    assert "min_speakers" in params
    assert "max_speakers" in params


def test_raises_os_error_when_no_hf_token():
    """When dep is available but HF_TOKEN is missing, OSError is raised."""
    import speechtelemetry.backends.diarization.pyannote as mod
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend

    env_clean = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGINGFACE_TOKEN")}
    with (
        patch.object(mod, "_AVAILABLE", True),
        patch.dict(os.environ, env_clean, clear=True),
        pytest.raises(OSError),
    ):
        PyannoteBackend()


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.diarization.pyannote as mod
    from speechtelemetry.backends.diarization.pyannote import PyannoteBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        PyannoteBackend()

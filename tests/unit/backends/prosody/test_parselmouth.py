"""Contract tests for ParselmouthBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend

    assert ParselmouthBackend is not None


def test_inherits_from_prosody_abc():
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend
    from speechtelemetry.interfaces import ProsodyBackend

    assert issubclass(ParselmouthBackend, ProsodyBackend)


def test_stage_attribute():
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend

    assert ParselmouthBackend.STAGE == "prosody"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.prosody.parselmouth as mod
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        ParselmouthBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.prosody.parselmouth as mod
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        ParselmouthBackend._check_available()


def test_extract_segment_signature_matches_abc():
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend

    sig = inspect.signature(ParselmouthBackend.extract_segment)
    params = list(sig.parameters)
    assert "self" in params
    assert "wav_path" in params
    assert "start_s" in params
    assert "end_s" in params


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.prosody.parselmouth as mod
    from speechtelemetry.backends.prosody.parselmouth import ParselmouthBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        ParselmouthBackend()

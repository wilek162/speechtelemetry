"""Contract tests for WhisperXAlignmentBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend

    assert WhisperXAlignmentBackend is not None


def test_inherits_from_alignment_abc():
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend
    from speechtelemetry.interfaces import AlignmentBackend

    assert issubclass(WhisperXAlignmentBackend, AlignmentBackend)


def test_stage_attribute():
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend

    assert WhisperXAlignmentBackend.STAGE == "alignment"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.alignment.whisperx as mod
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        WhisperXAlignmentBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.alignment.whisperx as mod
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        WhisperXAlignmentBackend._check_available()


def test_align_signature_matches_abc():
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend

    sig = inspect.signature(WhisperXAlignmentBackend.align)
    params = list(sig.parameters)
    assert "self" in params
    assert "segments" in params
    assert "audio_path" in params
    assert "language" in params


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.alignment.whisperx as mod
    from speechtelemetry.backends.alignment.whisperx import WhisperXAlignmentBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        WhisperXAlignmentBackend()

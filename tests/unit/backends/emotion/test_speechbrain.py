"""Contract tests for SpeechBrainEmotionBackend — no ML loading, no I/O."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend

    assert SpeechBrainEmotionBackend is not None


def test_inherits_from_emotion_abc():
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend
    from speechtelemetry.interfaces import EmotionBackend

    assert issubclass(SpeechBrainEmotionBackend, EmotionBackend)


def test_stage_attribute():
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend

    assert SpeechBrainEmotionBackend.STAGE == "emotion"


def test_raises_backend_not_available_when_dep_missing():
    import speechtelemetry.backends.emotion.speechbrain as mod
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        SpeechBrainEmotionBackend()


def test_check_available_raises_when_dep_missing():
    import speechtelemetry.backends.emotion.speechbrain as mod
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with patch.object(mod, "_AVAILABLE", False), pytest.raises(BackendNotAvailableError):
        SpeechBrainEmotionBackend._check_available()


def test_predict_segment_signature_matches_abc():
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend

    sig = inspect.signature(SpeechBrainEmotionBackend.predict_segment)
    params = list(sig.parameters)
    assert "self" in params
    assert "wav_path" in params
    assert "start_s" in params
    assert "end_s" in params


def test_backend_not_available_message_contains_install_hint():
    import speechtelemetry.backends.emotion.speechbrain as mod
    from speechtelemetry.backends.emotion.speechbrain import SpeechBrainEmotionBackend
    from speechtelemetry.exceptions import BackendNotAvailableError

    with (
        patch.object(mod, "_AVAILABLE", False),
        pytest.raises(BackendNotAvailableError, match="pip install"),
    ):
        SpeechBrainEmotionBackend()

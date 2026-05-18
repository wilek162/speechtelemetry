"""Unit tests for speechtelemetry.registry — no real backend imports."""

import pytest

from speechtelemetry.exceptions import BackendNotFoundError
from speechtelemetry.registry import (
    _REGISTRY,
    get_backend,
    list_backends,
    list_stages,
    register,
    resolve_backend,
)


def test_list_stages_has_expected():
    stages = list_stages()
    for expected in ("asr", "vad", "alignment", "diarization", "prosody", "emotion", "exporter"):
        assert expected in stages


def test_list_backends_asr():
    backends = list_backends("asr")
    assert "faster-whisper" in backends
    assert "whisperx" in backends


def test_list_backends_vad():
    assert "silero" in list_backends("vad")


def test_list_backends_unknown_stage():
    assert list_backends("nonexistent") == []


def test_resolve_backend_unknown_raises():
    with pytest.raises(BackendNotFoundError) as exc_info:
        resolve_backend("asr", "nonexistent-backend")
    assert "nonexistent-backend" in str(exc_info.value)
    assert "asr" in str(exc_info.value)


def test_resolve_backend_unknown_stage_raises():
    with pytest.raises(BackendNotFoundError):
        resolve_backend("nonexistent-stage", "anything")


def test_register_custom_backend():
    class DummyBackend:
        STAGE = "asr"
        NAME = "dummy-test"

    register("asr", "dummy-test", DummyBackend)
    assert "dummy-test" in list_backends("asr")
    resolved = resolve_backend("asr", "dummy-test")
    assert resolved is DummyBackend


def test_register_new_stage():
    class MyBackend:
        STAGE = "custom-stage"
        NAME = "my-backend"

    register("custom-stage", "my-backend", MyBackend)
    assert "custom-stage" in list_stages()
    assert "my-backend" in list_backends("custom-stage")


def test_get_backend_unknown_raises():
    with pytest.raises(BackendNotFoundError):
        get_backend("asr", "does-not-exist-xyz")


def test_registry_string_paths_are_valid_format():
    for stage, backends in _REGISTRY.items():
        for name, path in backends.items():
            assert "." in path, f"{stage}/{name}: path must be a dotted module path"
            module, cls = path.rsplit(".", 1)
            assert module, f"{stage}/{name}: module part must not be empty"
            assert cls, f"{stage}/{name}: class part must not be empty"


def test_get_backend_instantiates_with_kwargs():
    """get_backend must pass kwargs to the backend constructor."""

    class ParameterizedBackend:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    register("test-instantiation-stage", "param-backend", ParameterizedBackend)
    backend = get_backend("test-instantiation-stage", "param-backend", value=42)
    assert isinstance(backend, ParameterizedBackend)
    assert backend.value == 42


def test_gpl_backend_emits_user_warning_on_resolve():
    """resolve_backend for the parselmouth backend must emit a GPL UserWarning."""
    import importlib as _importlib
    import warnings
    from unittest.mock import MagicMock, patch

    from speechtelemetry.registry import _CLASS_CACHE

    _CLASS_CACHE.pop("prosody:parselmouth", None)

    mock_cls = MagicMock()
    mock_module = MagicMock()
    mock_module.ParselmouthBackend = mock_cls

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with patch.object(_importlib, "import_module", return_value=mock_module):
                resolve_backend("prosody", "parselmouth")

        gpl_warnings = [w for w in captured if "GPL" in str(w.message)]
        assert len(gpl_warnings) > 0, (
            f"Expected a GPL UserWarning but got: {[str(w.message) for w in captured]}"
        )
    finally:
        _CLASS_CACHE.pop("prosody:parselmouth", None)

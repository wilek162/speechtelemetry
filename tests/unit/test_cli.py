"""Unit tests for CLI — G2 (device passthrough) and G3 (typer optional).

TDD: Tests written before implementation fixes.
No ML deps, no I/O. Patches at boundary to stay unit-level.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# ── G2: device="auto" passthrough ─────────────────────────────────────────────


def test_pipeline_config_accepts_device_auto():
    """PipelineConfig must accept device='auto' without error."""
    from speechtelemetry.config import PipelineConfig

    cfg = PipelineConfig(device="auto")
    assert cfg.device == "auto"


def test_pipeline_config_device_is_auto_by_default():
    from speechtelemetry.config import PipelineConfig

    assert PipelineConfig().device == "auto"


def test_cli_builds_config_with_device_auto(tmp_path):
    """When --device auto is passed, the PipelineConfig must have device='auto'."""
    from speechtelemetry.config import PipelineConfig
    from speechtelemetry.types import ProcessingReport, TranscriptDocument

    captured: list[PipelineConfig] = []

    def fake_enrich_media(path, config=None, output_path=None, output_dir=None):
        if config is not None:
            captured.append(config)
        return TranscriptDocument(
            source_path=str(path),
            language="en",
            duration_s=3.0,
            segments=[],
            silence_spans=[],
            processing_report=ProcessingReport(),
        )

    # Only run this test if typer is installed
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from speechtelemetry.cli.main import app

    assert app is not None, "app is None — typer not available"

    runner = CliRunner()
    fake_wav = tmp_path / "test.wav"
    fake_wav.write_bytes(b"")

    with patch("speechtelemetry.enrich_media", side_effect=fake_enrich_media):
        runner.invoke(app, ["transcribe", str(fake_wav), "--device", "auto"])

    assert len(captured) == 1, "enrich_media was not called"
    assert captured[0].device == "auto", f"Expected device='auto', got: {captured[0].device!r}"


# ── G3: typer-optional import guard ───────────────────────────────────────────


def _reload_cli_without_typer():
    """Reload cli.main with typer/rich blocked — returns the reloaded module."""
    import builtins

    original_import = builtins.__import__

    def import_blocker(name, *args, **kwargs):
        if name in ("typer", "rich", "rich.console", "rich.table"):
            raise ImportError(f"blocked: {name}")
        return original_import(name, *args, **kwargs)

    for key in list(sys.modules):
        if key.startswith("speechtelemetry.cli"):
            del sys.modules[key]

    builtins.__import__ = import_blocker
    try:
        import speechtelemetry.cli.main as cli_mod

        return cli_mod
    finally:
        builtins.__import__ = original_import


def _restore_cli():
    for key in list(sys.modules):
        if key.startswith("speechtelemetry.cli"):
            del sys.modules[key]
    import speechtelemetry.cli.main  # noqa: F401


def test_cli_importable_without_typer():
    """cli/main.py must be importable when typer is not installed."""
    cli_mod = _reload_cli_without_typer()
    assert hasattr(cli_mod, "main"), "main() must exist even without typer"
    _restore_cli()


def test_cli_app_is_none_without_typer():
    """app must be None when typer is not installed."""
    cli_mod = _reload_cli_without_typer()
    assert cli_mod.app is None, f"Expected app=None without typer, got: {cli_mod.app!r}"
    _restore_cli()


def test_cli_main_exits_1_without_typer():
    """main() must sys.exit(1) when typer is not installed."""
    cli_mod = _reload_cli_without_typer()
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()
    assert exc_info.value.code == 1
    _restore_cli()

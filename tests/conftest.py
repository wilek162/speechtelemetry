"""Shared pytest fixtures and configuration."""

import sys

import pytest

from speechtelemetry.types import ProcessingReport, Segment, SilenceSpan, TranscriptDocument


@pytest.fixture(scope="session", autouse=True)
def _force_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows.

    Library debug logging uses non-ASCII characters (→, …) that cannot be
    encoded with the Windows default cp1252 encoding.  Without this, pytest
    output fails with UnicodeEncodeError when any test produces captured log
    output that contains those characters.
    """
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


@pytest.fixture()
def minimal_doc() -> TranscriptDocument:
    return TranscriptDocument(
        source_path="fixture.wav",
        language="en",
        duration_s=3.0,
        segments=[
            Segment(start=0.0, end=1.5, text="test segment", confidence=0.9),
        ],
        silence_spans=[SilenceSpan(start=1.5, end=3.0, duration_ms=1500.0)],
        processing_report=ProcessingReport(),
    )

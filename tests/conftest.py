"""Shared pytest fixtures and configuration."""

import pytest

from speechtelemetry.types import ProcessingReport, Segment, SilenceSpan, TranscriptDocument


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

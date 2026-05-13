"""WebVTT exporter — lossy derived view.

Lossy: only text and timestamps are preserved.
"""
from __future__ import annotations

import logging

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


def _format_vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VTTExporter(Exporter):
    """Export to WebVTT (.vtt) subtitle format. Lossy — text + timestamps only."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        lines = ["WEBVTT", ""]
        for seg in doc.segments:
            start = _format_vtt_time(seg.start)
            end = _format_vtt_time(seg.end)
            text = seg.text.strip()
            if seg.speaker:
                text = f"<v {seg.speaker}>{text}"
            lines.append(f"{start} --> {end}\n{text}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.debug("VTT export written to %s", output_path)

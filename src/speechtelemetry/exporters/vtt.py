"""WebVTT exporter — lossy derived view.

Lossy: only text and timestamps are preserved.
Speaker labels use WebVTT voice spans: <v SPEAKER_00>text.

Timestamp overlap fix: same boundary-overlap issue as SRT. Each cue's start is
clamped to max(start, prev_end) so consecutive cues never overlap.
"""

from __future__ import annotations

import logging
from pathlib import Path

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


def _format_vtt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VTTExporter(Exporter):
    """Export to WebVTT (.vtt) subtitle format. Lossy — text + timestamps only."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        stem = Path(output_path).stem
        lines: list[str] = [
            "WEBVTT",
            "",
            f"NOTE speechtelemetry — {stem}",
            f"NOTE language: {doc.language or 'unknown'}  duration: {doc.duration_s:.3f}s",
            "",
        ]

        prev_end_ms = 0
        for seg in doc.segments:
            # Clamp start to previous end to prevent overlapping cues
            start_ms = max(round(seg.start * 1000), prev_end_ms)
            end_ms = max(round(seg.end * 1000), start_ms + 1)
            prev_end_ms = end_ms

            text = seg.text.strip()
            if seg.speaker:
                text = f"<v {seg.speaker}>{text}"
            lines.append(
                f"{_format_vtt_time(start_ms / 1000)} --> {_format_vtt_time(end_ms / 1000)}\n{text}\n"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.debug("VTT export written to %s", output_path)

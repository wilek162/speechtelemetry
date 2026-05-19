"""SRT subtitle exporter — lossy derived view.

Lossy: only text and timestamps are preserved.
Speaker labels are prefixed in brackets: [SPEAKER_00] text.

Timestamp overlap fix: ASR segment boundaries sometimes overlap by a few
milliseconds (e.g. seg[i].end = 1.021, seg[i+1].start = 1.000). SRT requires
strictly non-overlapping cues. Each cue's start is clamped to max(start, prev_end)
so subtitles never overlap or display simultaneously.
"""

from __future__ import annotations

import logging

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SRTExporter(Exporter):
    """Export to SubRip (.srt) subtitle format. Lossy — text + timestamps only."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        lines: list[str] = []
        prev_end_ms = 0

        for i, seg in enumerate(doc.segments, start=1):
            # Clamp start to previous end to prevent overlapping cues
            start_ms = max(round(seg.start * 1000), prev_end_ms)
            end_ms = max(round(seg.end * 1000), start_ms + 1)
            prev_end_ms = end_ms

            text = seg.text.strip()
            if seg.speaker:
                text = f"[{seg.speaker}] {text}"

            lines.append(
                f"{i}\n{_format_srt_time(start_ms / 1000)} --> {_format_srt_time(end_ms / 1000)}\n{text}\n"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.debug("SRT export written to %s (%d segments)", output_path, len(doc.segments))

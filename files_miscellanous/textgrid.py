"""Praat TextGrid exporter — for phonetics toolchain integration."""
from __future__ import annotations

import logging

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


class TextGridExporter(Exporter):
    """Export to Praat TextGrid format for phonetics/prosody analysis tools."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        duration = doc.duration_s
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            f"xmin = 0",
            f"xmax = {duration:.6f}",
            "tiers? <exists>",
            f"size = 2",
            "item []:",
        ]

        # Tier 1: Segments / utterances
        lines += [
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "utterances"',
            "        xmin = 0",
            f"        xmax = {duration:.6f}",
            f"        intervals: size = {len(doc.segments)}",
        ]
        for i, seg in enumerate(doc.segments, start=1):
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            lines += [
                f"        intervals [{i}]:",
                f"            xmin = {seg.start:.6f}",
                f"            xmax = {seg.end:.6f}",
                f'            text = "{speaker_prefix}{seg.text.strip()}"',
            ]

        # Tier 2: Speaker labels
        speakers = [seg for seg in doc.segments if seg.speaker]
        lines += [
            "    item [2]:",
            '        class = "IntervalTier"',
            '        name = "speakers"',
            "        xmin = 0",
            f"        xmax = {duration:.6f}",
            f"        intervals: size = {len(speakers)}",
        ]
        for i, seg in enumerate(speakers, start=1):
            lines += [
                f"        intervals [{i}]:",
                f"            xmin = {seg.start:.6f}",
                f"            xmax = {seg.end:.6f}",
                f'            text = "{seg.speaker}"',
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.debug("TextGrid export written to %s", output_path)

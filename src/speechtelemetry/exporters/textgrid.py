"""Praat TextGrid exporter — for phonetics toolchain integration.

Produces a valid Praat TextGrid with the following interval tiers:

  Tier 1 — utterances  : transcript text per segment, bracketed by speaker label
  Tier 2 — speakers    : speaker ID per segment interval (empty when no diarization)
  Tier 3 — emotion     : dominant emotion label per segment (empty when no emotion)
  Tier 4 — words       : individual aligned words (omitted when no word timestamps)

TextGrid validity rules enforced here:
  - Every tier covers exactly [0, xmax].
  - Intervals within a tier are contiguous and non-overlapping.
  - ASR segment boundary overlaps (e.g. seg[i].end=1.021, seg[i+1].start=1.000)
    are resolved by clamping each segment's start to max(seg.start, prev_end).
  - Gaps between segments receive empty-text intervals.
"""

from __future__ import annotations

import logging

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import Segment, TranscriptDocument

logger = logging.getLogger(__name__)

# ── Interval helpers ──────────────────────────────────────────────────────────

_Interval = tuple[float, float, str]  # (start, end, text)


def _normalize(
    items: list[_Interval],
    total_duration: float,
) -> list[_Interval]:
    """Build a valid, contiguous, non-overlapping interval list covering [0, total_duration].

    Resolves boundary overlaps by clamping each start to max(start, cursor).
    Fills gaps between items with empty-text intervals.
    """
    result: list[_Interval] = []
    cursor = 0.0

    for start, end, text in sorted(items, key=lambda x: x[0]):
        actual_start = max(start, cursor)
        if end <= actual_start:
            continue  # degenerate; skip
        if actual_start > cursor + 1e-9:
            result.append((cursor, actual_start, ""))  # gap
        result.append((actual_start, end, text))
        cursor = end

    if cursor < total_duration - 1e-9:
        result.append((cursor, total_duration, ""))  # trailing gap

    # Handle edge case: no items at all
    if not result:
        result.append((0.0, total_duration, ""))

    return result


def _tier_lines(name: str, intervals: list[_Interval], total_duration: float) -> list[str]:
    """Render a single IntervalTier as Praat TextGrid lines."""
    lines: list[str] = [
        "    item [__IDX__]:",
        '        class = "IntervalTier"',
        f'        name = "{name}"',
        "        xmin = 0",
        f"        xmax = {total_duration:.6f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for i, (start, end, text) in enumerate(intervals, start=1):
        escaped = text.replace('"', '\\"')
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {start:.6f}",
            f"            xmax = {end:.6f}",
            f'            text = "{escaped}"',
        ]
    return lines


# ── Exporter ──────────────────────────────────────────────────────────────────


class TextGridExporter(Exporter):
    """Export to Praat TextGrid format for phonetics/prosody analysis tools."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        duration = doc.duration_s
        segs: list[Segment] = doc.segments

        # ── Build raw interval data from segments ─────────────────────────
        utterance_items: list[_Interval] = []
        speaker_items: list[_Interval] = []
        emotion_items: list[_Interval] = []
        word_items: list[_Interval] = []

        for seg in segs:
            prefix = f"[{seg.speaker}] " if seg.speaker else ""
            utterance_items.append((seg.start, seg.end, f"{prefix}{seg.text.strip()}"))

            if seg.speaker:
                speaker_items.append((seg.start, seg.end, seg.speaker))

            if seg.emotion is not None:
                top = max(
                    seg.emotion.label_distribution,
                    key=seg.emotion.label_distribution.__getitem__,
                )
                emotion_items.append((seg.start, seg.end, top))

            for w in seg.words or []:
                word_items.append((w.start, w.end, w.text))

        # ── Normalize all tiers to valid intervals ────────────────────────
        utterances = _normalize(utterance_items, duration)
        speakers = _normalize(speaker_items, duration)
        emotions = _normalize(emotion_items, duration)
        words = _normalize(word_items, duration) if word_items else []

        # ── Determine tier count ──────────────────────────────────────────
        tiers: list[tuple[str, list[_Interval]]] = [
            ("utterances", utterances),
            ("speakers", speakers),
            ("emotion", emotions),
        ]
        if words:
            tiers.append(("words", words))

        # ── Render ───────────────────────────────────────────────────────
        lines: list[str] = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {duration:.6f}",
            "tiers? <exists>",
            f"size = {len(tiers)}",
            "item []:",
        ]

        for tier_idx, (tier_name, intervals) in enumerate(tiers, start=1):
            tier_block = _tier_lines(tier_name, intervals, duration)
            # Inject the 1-based tier index
            lines += [ln.replace("__IDX__", str(tier_idx)) for ln in tier_block]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.debug(
            "TextGrid export written to %s (%d tiers, %d utterance intervals)",
            output_path,
            len(tiers),
            len(utterances),
        )

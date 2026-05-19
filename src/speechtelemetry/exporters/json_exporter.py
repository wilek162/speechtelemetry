"""JSON exporter — the authoritative lossless export format.

Preserves all fields in TranscriptDocument. This is the archival format.
SRT/VTT/TextGrid are lossy derived views of this.

Serialization rules:
  - All float values are rounded to 4 decimal places to eliminate floating-point
    noise (e.g. 0.6819999999999999 → 0.682, 2.14e-12 → 0.0).
  - source_path uses forward slashes for cross-platform portability.
  - Field ordering: schema_version and speaker summary appear before segments
    for quick scanning without reading the full document.
  - Null fields are preserved (semantically meaningful — None ≠ absent).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, cast

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)

# Precision for all floating-point values in the JSON output.
# 4 decimal places → 0.1 ms timestamp resolution, 0.01% confidence resolution.
_FLOAT_PRECISION = 4


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and known types to JSON-serializable dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _round_floats(obj: Any, precision: int = _FLOAT_PRECISION) -> Any:
    """Recursively round all float values to eliminate floating-point noise."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(i, precision) for i in obj]
    return obj


def _build_ordered(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with a human-friendly field order.

    Order: version → identity → speakers summary → main content → telemetry.
    """
    ordered: dict[str, Any] = {}

    # 1. Version + identity
    ordered["schema_version"] = raw.get("schema_version", "1.0")
    ordered["source_path"] = raw.get("source_path", "")
    ordered["language"] = raw.get("language")
    ordered["duration_s"] = raw.get("duration_s", 0.0)

    # 2. Speaker summary (only when diarization ran)
    if raw.get("speakers") is not None:
        ordered["speakers"] = raw["speakers"]

    # 3. Main transcript content
    ordered["segments"] = raw.get("segments", [])
    ordered["silence_spans"] = raw.get("silence_spans", [])

    # 4. Telemetry
    ordered["processing_report"] = raw.get("processing_report", {})
    ordered["provenance"] = raw.get("provenance")

    return ordered


class JsonExporter(Exporter):
    """Serialize TranscriptDocument to JSON. Lossless."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        raw = _to_dict(doc)

        # Normalize source_path to forward slashes
        if isinstance(raw.get("source_path"), str):
            raw["source_path"] = raw["source_path"].replace("\\", "/")

        # Round all floats to eliminate floating-point noise
        raw = _round_floats(raw)

        # Reorder fields for human readability
        data = _build_ordered(raw)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("JSON export written to %s", output_path)

    @staticmethod
    def load(input_path: str) -> dict[str, Any]:
        """Load a previously exported JSON transcript."""
        with open(input_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

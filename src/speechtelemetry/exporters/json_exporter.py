"""JSON exporter — the authoritative lossless export format.

Preserves all fields in TranscriptDocument. This is the archival format.
SRT/VTT/TextGrid are lossy derived views of this.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, cast

from speechtelemetry.interfaces import Exporter
from speechtelemetry.types import TranscriptDocument

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and known types to JSON-serializable dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


class JsonExporter(Exporter):
    """Serialize TranscriptDocument to JSON. Lossless."""

    def export(self, doc: TranscriptDocument, output_path: str) -> None:
        data = _to_dict(doc)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("JSON export written to %s", output_path)

    @staticmethod
    def load(input_path: str) -> dict[str, Any]:
        """Load a previously exported JSON transcript."""
        with open(input_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

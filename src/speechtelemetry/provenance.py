"""Provenance tracking — records which backend produced which output field.

Written into TranscriptDocument for full transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackendProvenance:
    """Records the backend and model used at each pipeline stage."""

    stage: str
    backend_name: str
    model_id: Optional[str] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None


@dataclass
class PipelineProvenance:
    """Full provenance record for one pipeline run."""

    stages: list[BackendProvenance] = field(default_factory=list)

    def record(
        self,
        stage: str,
        backend_name: str,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.stages.append(
            BackendProvenance(
                stage=stage,
                backend_name=backend_name,
                model_id=model_id,
                device=device,
            )
        )

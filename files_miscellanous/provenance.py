"""Provenance tracking — records which backend produced which output field.

Written into TranscriptDocument for full transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackendProvenance:
    """Records the backend and model used at each pipeline stage."""

    stage: str
    backend_name: str
    model_id: str | None = None
    device: str | None = None
    compute_type: str | None = None


@dataclass
class PipelineProvenance:
    """Full provenance record for one pipeline run."""

    stages: list[BackendProvenance] = field(default_factory=list)

    def record(
        self,
        stage: str,
        backend_name: str,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.stages.append(
            BackendProvenance(
                stage=stage,
                backend_name=backend_name,
                model_id=model_id,
                device=device,
            )
        )

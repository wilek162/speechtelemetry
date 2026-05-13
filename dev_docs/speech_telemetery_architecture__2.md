**speechtelemetry**

Architecture Reference

Design principles, layer boundaries, extension model, and packaging
strategy

Version 0.1 \| May 2026

+-----------------------------------------------------------------------+
| This document defines the architectural principles, layer model,      |
| separation-of-concerns strategy, extension patterns, and packaging    |
| design of speechtelemetry. It is the authoritative reference for      |
| every structural decision made in this project. Developers            |
| contributing to the library, consumers embedding it in their own      |
| projects, and AI agents implementing new backends must all consult    |
| this document before making structural changes.                       |
|                                                                       |
| **Goal: make speechtelemetry trivially embeddable, safely extensible, |
| and architecturally honest --- with every concern living exactly      |
| where it belongs.**                                                   |
+-----------------------------------------------------------------------+

**Table of Contents**

**1. Architectural Principles**

Every structural decision in speechtelemetry flows from a small set of
first principles. These are not aspirational statements --- they are
enforced constraints. When a design choice violates one of these
principles, it is a defect, not a trade-off.

**1.1 The Library Is the Source of Truth**

The library (src/speechtelemetry/) is the only layer that matters for
correctness and portability. The CLI, any future REST API, and any GUI
are thin wrappers that translate user intent into library calls. They
add zero logic. This means:

-   A developer using the library as a Python import must be able to do
    everything the CLI can do --- with more control.

-   The CLI must never contain business logic that is absent from the
    library.

-   Outputs are always the canonical TranscriptDocument type --- never
    raw dicts, JSON strings, or format-specific objects at the public
    boundary.

**1.2 Dependency Rule --- Dependencies Flow Inward Only**

Inner layers know nothing about outer layers. The canonical direction of
knowledge is:

+-----------------------------------------------------------------------+
| Outermost Innermost                                                   |
|                                                                       |
| ┌────────────┐ ┌──────────────┐ ┌────────────────┐                    |
| ┌──────────────────┐                                                  |
|                                                                       |
| │ CLI / API │──▶│ core/ │──▶│ backends/ │──▶│ types / schema │        |
|                                                                       |
| │ Exporters │ │ pipeline.py │ │ (adapters) │ │ (pure data) │          |
|                                                                       |
| └────────────┘ └──────────────┘ └────────────────┘                    |
| └──────────────────┘                                                  |
|                                                                       |
| ▲                                                                     |
|                                                                       |
| External libs ──────┘ (never reach inner layers)                      |
+-----------------------------------------------------------------------+

Concretely: types.py must never import from backends/. core/pipeline.py
must never import from cli/. A backend must never import from another
backend. Any violation of this rule is an architecture bug.

**1.3 Stable Core, Pluggable Periphery**

The schema (types.py), the orchestration contract (core/pipeline.py),
and the configuration model (config.py) form a stable core that changes
infrequently and only with a version bump. Everything else --- every
backend, every exporter, every CLI command --- is a plug that snaps into
that stable core. Adding a new ASR engine must require zero changes to
types.py or pipeline.py.

**1.4 Fail Soft, Fail Honestly**

Partial output is better than a hard crash. Every stage is independently
failable. All failures are recorded in ProcessingReport.errors --- they
are never silently swallowed, and they never propagate as unhandled
exceptions to the caller.

**1.5 Local-First, Offline-Capable by Default**

No network call is ever required at runtime. Model weights are cached
locally on first download. If a model cache directory exists, it is used
without re-downloading. The library must be able to run on an air-gapped
machine after initial setup.

**1.6 Explicit Over Implicit**

Nothing meaningful happens silently. Device selection, model choices,
and backend selection are all visible in PipelineConfig and recorded in
TranscriptDocument.provenance. Users must always be able to inspect
exactly what ran, with what model, on what device.

**1.7 One Schema to Rule All Outputs**

The canonical TranscriptDocument type defined in types.py is the single
authoritative output format. All exporters (JSON, SRT, VTT, TextGrid)
are lossless or explicitly lossy projections of this type. The JSON
export is the archival format; all others are derived views. External
consumers work with TranscriptDocument --- never with raw dicts.

**2. Layer Model & Separation of Concerns**

The library is divided into five distinct layers. Each layer has a
single responsibility and a clear boundary. No layer reaches across more
than one boundary.

+-----------------------------------------------------------------------+
| ┌──────                                                               |
| ────────────────────────────────────────────────────────────────────┐ |
|                                                                       |
| │ Layer 5 --- Presentation (cli/, future api/) │                      |
|                                                                       |
| │ Concern: translate user intent to library calls; format terminal    |
| output │                                                              |
|                                                                       |
| ├──────                                                               |
| ────────────────────────────────────────────────────────────────────┤ |
|                                                                       |
| │ Layer 4 --- Exporters (exporters/) │                                |
|                                                                       |
| │ Concern: serialize TranscriptDocument to external formats           |
| (JSON/SRT/VTT) │                                                      |
|                                                                       |
| ├──────                                                               |
| ────────────────────────────────────────────────────────────────────┤ |
|                                                                       |
| │ Layer 3 --- Orchestration (core/) │                                 |
|                                                                       |
| │ Concern: run stages in order; manage state; record errors; time     |
| stages │                                                              |
|                                                                       |
| ├──────                                                               |
| ────────────────────────────────────────────────────────────────────┤ |
|                                                                       |
| │ Layer 2 --- Adapters (backends/, io/) │                             |
|                                                                       |
| │ Concern: wrap external libraries behind stable abstract interfaces  |
| │                                                                     |
|                                                                       |
| ├──────                                                               |
| ────────────────────────────────────────────────────────────────────┤ |
|                                                                       |
| │ Layer 1 --- Domain Core (types.py, config.py, interfaces.py) │      |
|                                                                       |
| │ Concern: define the schema and contracts; zero external imports │   |
|                                                                       |
| └──────                                                               |
| ────────────────────────────────────────────────────────────────────┘ |
+-----------------------------------------------------------------------+

**2.1 Layer 1 --- Domain Core**

**What lives here**

-   types.py --- All canonical dataclasses (TranscriptDocument, Segment,
    Word, SilenceSpan, ProsodyWindow, EmotionScore, ProcessingReport).
    Zero external imports. Pure Python standard library only.

-   interfaces.py --- Abstract Base Classes for every backend stage
    (ASRBackend, VADBackend, AlignmentBackend, DiarizationBackend,
    ProsodyBackend, EmotionBackend, Exporter). Zero external imports.

-   config.py --- PipelineConfig Pydantic model. Only imports pydantic.

**Rules**

-   No backend code. No ML library imports. No IO. No subprocess calls.

-   types.py may not import from any other module in this project.

-   This layer changes only when the schema changes --- a rare,
    versioned event.

+-----------------------------------------------------------------------+
| \# types.py --- illustrative structure                                |
|                                                                       |
| from \_\_future\_\_ import annotations                                |
|                                                                       |
| from dataclasses import dataclass, field                              |
|                                                                       |
| from typing import Optional                                           |
|                                                                       |
| \@dataclass                                                           |
|                                                                       |
| class Word:                                                           |
|                                                                       |
| text: str                                                             |
|                                                                       |
| start: float                                                          |
|                                                                       |
| end: float                                                            |
|                                                                       |
| confidence: float                                                     |
|                                                                       |
| alignment_backend: Optional\[str\] = None                             |
|                                                                       |
| \@dataclass                                                           |
|                                                                       |
| class EmotionScore:                                                   |
|                                                                       |
| label_distribution: dict\[str, float\] \# NEVER collapse to single    |
| label                                                                 |
|                                                                       |
| confidence: float                                                     |
|                                                                       |
| backend_name: str                                                     |
|                                                                       |
| valence: Optional\[float\] = None                                     |
|                                                                       |
| arousal: Optional\[float\] = None                                     |
|                                                                       |
| \# interfaces.py --- every backend signs this contract                |
|                                                                       |
| from abc import ABC, abstractmethod                                   |
|                                                                       |
| from speechtelemetry.types import TranscriptDocument                  |
|                                                                       |
| class ASRBackend(ABC):                                                |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def transcribe(self, wav_path: str, config) -\> list\[dict\]: \...    |
|                                                                       |
| class EmotionBackend(ABC):                                            |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def predict_segment(self, wav_path: str,                              |
|                                                                       |
| start_s: float, end_s: float) -\> dict: \...                          |
+-----------------------------------------------------------------------+

**2.2 Layer 2 --- Adapters (backends/ and io/)**

**What lives here**

-   backends/asr/, backends/vad/, backends/alignment/,
    backends/diarization/, backends/prosody/, backends/emotion/ --- One
    module per backend. Each module contains exactly one class that
    inherits from the corresponding ABC in interfaces.py.

-   io/ffmpeg.py --- FFmpeg subprocess wrapper. normalize_to_wav() only.
    No knowledge of the pipeline.

-   io/audio_normalize.py --- Post-FFmpeg WAV validation. No ML.

**Rules**

-   Each adapter class inherits from exactly one ABC. Instantiation
    raises TypeError immediately if the contract is not satisfied.

-   No adapter imports from another adapter. They are siblings, not a
    hierarchy.

-   All external imports (torch, faster_whisper, parselmouth, etc.) are
    inside adapter modules only --- never in types.py, core/, or cli/.

-   Adapter constructors must catch ImportError and raise a clear
    BackendNotAvailableError with install instructions. The library core
    must never see a raw ImportError from a missing optional backend.

+-----------------------------------------------------------------------+
| \# Pattern: guarded import in every adapter module                    |
|                                                                       |
| try:                                                                  |
|                                                                       |
| from faster_whisper import WhisperModel                               |
|                                                                       |
| \_AVAILABLE = True                                                    |
|                                                                       |
| except ImportError:                                                   |
|                                                                       |
| \_AVAILABLE = False                                                   |
|                                                                       |
| class FasterWhisperBackend(ASRBackend):                               |
|                                                                       |
| def \_\_init\_\_(self, \...):                                         |
|                                                                       |
| if not \_AVAILABLE:                                                   |
|                                                                       |
| raise BackendNotAvailableError(                                       |
|                                                                       |
| \'faster-whisper is not installed. \'                                 |
|                                                                       |
| \'Run: pip install faster-whisper\'                                   |
|                                                                       |
| )                                                                     |
|                                                                       |
| \...                                                                  |
+-----------------------------------------------------------------------+

**2.3 Layer 3 --- Orchestration (core/)**

**What lives here**

-   core/pipeline.py --- The enrich_media() implementation. Calls
    backends in order. Wraps each stage in try/except. Collects timings
    and errors into ProcessingReport. Returns TranscriptDocument.

-   core/job.py --- Manages the lifecycle of a single job: temp file
    creation, cleanup in finally blocks, GPU memory release.

-   core/provenance.py --- Records which backend, model, and device ran
    at each stage. Written into TranscriptDocument.provenance.

**Rules**

-   core/ imports from Layer 1 (types, interfaces, config) and Layer 2
    (backends via registry). Never from Layer 4 or 5.

-   core/pipeline.py must never reference a concrete backend class
    directly. It resolves backend names through registry.py only.

-   The pipeline is a pure function at heart: given a config and a file
    path, it returns a TranscriptDocument. No global state is mutated.

+-----------------------------------------------------------------------+
| \# core/pipeline.py --- skeleton                                      |
|                                                                       |
| from speechtelemetry.types import TranscriptDocument,                 |
| ProcessingReport                                                      |
|                                                                       |
| from speechtelemetry.registry import resolve_backend                  |
|                                                                       |
| import time                                                           |
|                                                                       |
| def enrich_media(input_path: str, config: PipelineConfig) -\>         |
| TranscriptDocument:                                                   |
|                                                                       |
| report = ProcessingReport()                                           |
|                                                                       |
| with Job(input_path) as job: \# manages temp files                    |
|                                                                       |
| \# Stage 1: Decode                                                    |
|                                                                       |
| t0 = time.perf_counter()                                              |
|                                                                       |
| job.wav_path = job.normalize(input_path)                              |
|                                                                       |
| report.stage_timings\[\'decode\'\] = time.perf_counter() - t0         |
|                                                                       |
| \# Stage 2: VAD                                                       |
|                                                                       |
| vad = resolve_backend(\'vad\', config.vad_backend)(config)            |
|                                                                       |
| try:                                                                  |
|                                                                       |
| silence_spans = vad.get_silence_spans(job.wav_path, \...)             |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| report.add_error(\'vad\', e)                                          |
|                                                                       |
| silence_spans = \[\]                                                  |
|                                                                       |
| \# \...repeat pattern for ASR, align, diarize, prosody, emotion\...   |
|                                                                       |
| return TranscriptDocument(\..., processing_report=report)             |
+-----------------------------------------------------------------------+

**2.4 Layer 4 --- Exporters (exporters/)**

**What lives here**

-   exporters/json_exporter.py --- Serializes TranscriptDocument to
    JSON. This is the authoritative export. Lossless.

-   exporters/srt.py, exporters/vtt.py --- Subtitle exports. Explicitly
    lossy (text + timestamps only; no prosody, no emotion).

-   exporters/textgrid.py --- Praat TextGrid format for phonetics
    toolchain integration.

**Rules**

-   Exporters receive a TranscriptDocument and an output path. They
    produce a file. Nothing else.

-   Exporters must never re-run inference. They are pure serialization
    functions.

-   Lossy exporters must document exactly what fields are dropped.

-   Exporters may not import from core/ or backends/.

**2.5 Layer 5 --- Presentation (cli/, future api/)**

**What lives here**

-   cli/main.py --- Click/Typer entry point. Parses CLI arguments,
    builds PipelineConfig, calls enrich_media(), calls the requested
    exporter.

-   cli/commands.py --- Subcommands (transcribe, benchmark, info).

**Rules**

-   Zero business logic in the CLI. If you find yourself writing an
    if/else in cli/ that is not purely about argument parsing or output
    formatting, that logic belongs in core/ or a backend.

-   The CLI is a consumer of the library, exactly like an external
    user\'s script. It must use only the public API.

-   Error messages from the CLI must be human-readable translations of
    the library\'s structured exceptions --- not raw tracebacks.

**3. Backend Extension Model --- The Strategy + Registry Pattern**

The ability to swap backends without touching core code is the most
important extensibility feature of speechtelemetry. It is implemented
via the Strategy pattern with Abstract Base Classes plus a central
Registry.

**3.1 The Strategy Pattern with ABCs**

Every pipeline stage is defined as a strategy. A strategy is an ABC in
interfaces.py that declares the contract --- method names, signatures,
and return types --- without any implementation. Concrete backends are
implementations of this strategy.

+-----------------------------------------------------------------------+
| \# interfaces.py --- the full set of ABCs                             |
|                                                                       |
| from abc import ABC, abstractmethod                                   |
|                                                                       |
| from speechtelemetry.types import \*                                  |
|                                                                       |
| class VADBackend(ABC):                                                |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def get_speech_intervals(self, wav_path: str) -\> list\[dict\]:       |
|                                                                       |
| \"\"\"Returns \[{\"start\": float, \"end\": float}, \...\]\"\"\"      |
|                                                                       |
| class ASRBackend(ABC):                                                |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def transcribe(self, wav_path: str, config) -\> tuple\[list,          |
| object\]:                                                             |
|                                                                       |
| \"\"\"Returns (segments, info).\"\"\"                                 |
|                                                                       |
| class AlignmentBackend(ABC):                                          |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def align(self, segments: list, audio_path: str, language: str) -\>   |
| list:                                                                 |
|                                                                       |
| \"\"\"Returns segments with word-level timestamps added.\"\"\"        |
|                                                                       |
| class DiarizationBackend(ABC):                                        |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def diarize(self, wav_path: str, \*\*kwargs) -\> list\[dict\]:        |
|                                                                       |
| \"\"\"Returns \[{\"start\": float, \"end\": float, \"speaker\": str}, |
| \...\]\"\"\"                                                          |
|                                                                       |
| class ProsodyBackend(ABC):                                            |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def extract_segment(self, wav_path: str,                              |
|                                                                       |
| start_s: float, end_s: float) -\> dict:                               |
|                                                                       |
| \"\"\"Returns ProsodyWindow-compatible dict.\"\"\"                    |
|                                                                       |
| class EmotionBackend(ABC):                                            |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def predict_segment(self, wav_path: str,                              |
|                                                                       |
| start_s: float, end_s: float) -\> dict:                               |
|                                                                       |
| \"\"\"Returns EmotionScore-compatible dict.\"\"\"                     |
|                                                                       |
| class Exporter(ABC):                                                  |
|                                                                       |
| \@abstractmethod                                                      |
|                                                                       |
| def export(self, doc: TranscriptDocument, output_path: str) -\> None: |
| \...                                                                  |
+-----------------------------------------------------------------------+

If a backend class does not implement all abstract methods, Python
raises TypeError at instantiation --- not at call time. This gives
developers immediate feedback at import time.

**3.2 The Registry**

The Registry (registry.py) is a runtime mapping from backend name
strings to classes. The pipeline never instantiates a backend class
directly --- it always asks the registry. This decouples the pipeline
from concrete implementations and makes the system testable with mock
backends.

+-----------------------------------------------------------------------+
| \# registry.py                                                        |
|                                                                       |
| from \_\_future\_\_ import annotations                                |
|                                                                       |
| from typing import Type                                               |
|                                                                       |
| from speechtelemetry.interfaces import (                              |
|                                                                       |
| ASRBackend, VADBackend, AlignmentBackend,                             |
|                                                                       |
| DiarizationBackend, ProsodyBackend, EmotionBackend                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| \_REGISTRY: dict\[str, dict\[str, Type\]\] = {                        |
|                                                                       |
| \'asr\': {},                                                          |
|                                                                       |
| \'vad\': {},                                                          |
|                                                                       |
| \'alignment\': {},                                                    |
|                                                                       |
| \'diarization\': {},                                                  |
|                                                                       |
| \'prosody\': {},                                                      |
|                                                                       |
| \'emotion\': {},                                                      |
|                                                                       |
| }                                                                     |
|                                                                       |
| def register(stage: str, name: str, cls: Type) -\> None:              |
|                                                                       |
| \"\"\"Register a backend class for a given stage and name.\"\"\"      |
|                                                                       |
| \_REGISTRY\[stage\]\[name\] = cls                                     |
|                                                                       |
| def resolve_backend(stage: str, name: str) -\> Type:                  |
|                                                                       |
| \"\"\"Resolve a backend name to its class. Raises KeyError if         |
| unknown.\"\"\"                                                        |
|                                                                       |
| if name not in \_REGISTRY.get(stage, {}):                             |
|                                                                       |
| available = list(\_REGISTRY.get(stage, {}).keys())                    |
|                                                                       |
| raise BackendNotFoundError(                                           |
|                                                                       |
| f\"Unknown {stage} backend: \'{name}\'. \"                            |
|                                                                       |
| f\"Available: {available}\"                                           |
|                                                                       |
| )                                                                     |
|                                                                       |
| return \_REGISTRY\[stage\]\[name\]                                    |
|                                                                       |
| \# Default registrations (in backends/\_\_init\_\_.py)                |
|                                                                       |
| from speechtelemetry.backends.vad.silero import SileroVADBackend      |
|                                                                       |
| from speechtelemetry.backends.asr.faster_whisper import               |
| FasterWhisperBackend                                                  |
|                                                                       |
| \# \...                                                               |
|                                                                       |
| register(\'vad\', \'silero\', SileroVADBackend)                       |
|                                                                       |
| register(\'asr\', \'faster-whisper\', FasterWhisperBackend)           |
|                                                                       |
| \# \...                                                               |
+-----------------------------------------------------------------------+

**3.3 How External Users Register Custom Backends**

Any user or third-party package can register a custom backend before
calling enrich_media(). The library provides a one-call registration
API. This is the primary extension mechanism for embedding
speechtelemetry into other projects.

+-----------------------------------------------------------------------+
| \# User\'s project: my_custom_asr.py                                  |
|                                                                       |
| from speechtelemetry.interfaces import ASRBackend                     |
|                                                                       |
| from speechtelemetry import registry                                  |
|                                                                       |
| class MyWhisperVariantBackend(ASRBackend):                            |
|                                                                       |
| def transcribe(self, wav_path, config):                               |
|                                                                       |
| \# \... custom implementation \...                                    |
|                                                                       |
| return segments, info                                                 |
|                                                                       |
| \# Register before first use                                          |
|                                                                       |
| registry.register(\'asr\', \'my-whisper-variant\',                    |
| MyWhisperVariantBackend)                                              |
|                                                                       |
| \# Now use it via config                                              |
|                                                                       |
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| result = enrich_media(\'audio.mp4\',                                  |
| PipelineConfig(asr_backend=\'my-whisper-variant\'))                   |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Extension Design Goal**                                             |
|                                                                       |
| A third-party developer should be able to integrate a new backend in  |
| under 30 minutes:                                                     |
|                                                                       |
| 1\. Create a class inheriting from the appropriate ABC.               |
|                                                                       |
| 2\. Call registry.register().                                         |
|                                                                       |
| 3\. Pass the backend name string in PipelineConfig.                   |
|                                                                       |
| No forks, no monkey-patching, no changes to the library source code.  |
+-----------------------------------------------------------------------+

**3.4 Third-Party Backend Packages**

Third-party backend packages can auto-register via Python entry points.
This allows pip install speechtelemetry-my-asr to make the backend
immediately available without any user code.

+-----------------------------------------------------------------------+
| \# Third-party package: pyproject.toml                                |
|                                                                       |
| \[project.entry-points.\'speechtelemetry.backends\'\]                 |
|                                                                       |
| my-whisper-variant =                                                  |
| \'speechtelemetry_my_asr:MyWhisperVariantBackend\'                    |
|                                                                       |
| \# speechtelemetry/registry.py --- auto-discovery at import time      |
|                                                                       |
| import importlib.metadata                                             |
|                                                                       |
| def \_autodiscover():                                                 |
|                                                                       |
| eps =                                                                 |
| importlib.metadata.entry_points(group=\'speechtelemetry.backends\')   |
|                                                                       |
| for ep in eps:                                                        |
|                                                                       |
| try:                                                                  |
|                                                                       |
| cls = ep.load()                                                       |
|                                                                       |
| stage = cls.STAGE \# class-level constant, e.g. \'asr\'               |
|                                                                       |
| name = cls.NAME \# class-level constant, e.g. \'my-whisper\'          |
|                                                                       |
| register(stage, name, cls)                                            |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| import warnings                                                       |
|                                                                       |
| warnings.warn(f\'Failed to load backend {ep.name}: {e}\')             |
|                                                                       |
| \_autodiscover() \# called once at module import                      |
+-----------------------------------------------------------------------+

**4. Dependency Management & Packaging Strategy**

speechtelemetry is a library intended for use in other projects.
Dependency management must therefore be conservative, explicit, and
respectful of users\' existing environments. Forcing users to install
heavy ML stacks they do not need is an adoption blocker.

**4.1 Dependency Tiers**

  ------------------------------------------------------------------------------
  **Tier**   **Description**    **Installed   **Packages**
                                By Default?**
  ---------- ------------------ ------------- ----------------------------------
  Core       Absolute minimum   Yes           pydantic\>=2, soundfile,
             to import the                    typing_extensions
             library and use
             the public API.

  ASR        faster-whisper and No (extra)    faster-whisper
             its CTranslate2
             dependency.

  Align      WhisperX forced    No (extra)    whisperx
             alignment.

  VAD        Silero VAD.        No (extra)    silero-vad, torchaudio

  Diarize    pyannote.audio     No (extra)    pyannote.audio
             diarization.

  Prosody    Parselmouth        No (extra)    praat-parselmouth, copasul
             (GPL-gated) +
             CoPaSul.

  Emotion    SpeechBrain SER.   No (extra)    speechbrain, transformers

  CLI        Click/Typer for    No (extra)    typer\[all\], rich
             the command-line
             interface.

  All        Everything above.  No (extra)    speechtelemetry\[all\]

  Dev        Testing, linting,  No (extra)    pytest, mypy, ruff, black
             type-checking
             tools.
  ------------------------------------------------------------------------------

**4.2 pyproject.toml Design**

+-----------------------------------------------------------------------+
| \# pyproject.toml                                                     |
|                                                                       |
| \[build-system\]                                                      |
|                                                                       |
| requires = \[\"hatchling\"\]                                          |
|                                                                       |
| build-backend = \"hatchling.build\"                                   |
|                                                                       |
| \[project\]                                                           |
|                                                                       |
| name = \"speechtelemetry\"                                            |
|                                                                       |
| version = \"0.1.0\"                                                   |
|                                                                       |
| description = \"Local-first speech intelligence: transcription,       |
| prosody, and emotion\"                                                |
|                                                                       |
| readme = \"README.md\"                                                |
|                                                                       |
| license = { text = \"MIT\" }                                          |
|                                                                       |
| requires-python = \"\>=3.10\"                                         |
|                                                                       |
| \# CRITICAL: Core deps ONLY. No ML libraries here.                    |
|                                                                       |
| dependencies = \[                                                     |
|                                                                       |
| \"pydantic\>=2.0\",                                                   |
|                                                                       |
| \"soundfile\>=0.12\",                                                 |
|                                                                       |
| \"typing_extensions\>=4.0; python_version\<\\\"3.11\\\"\",            |
|                                                                       |
| \]                                                                    |
|                                                                       |
| \[project.optional-dependencies\]                                     |
|                                                                       |
| asr = \[\"faster-whisper\>=1.0\"\]                                    |
|                                                                       |
| align = \[\"whisperx\>=3\"\]                                          |
|                                                                       |
| vad = \[\"silero-vad\>=5\", \"torchaudio\>=2.0\"\]                    |
|                                                                       |
| diarize = \[\"pyannote.audio\>=3.1\"\]                                |
|                                                                       |
| prosody = \[\"praat-parselmouth\>=0.4\", \"copasul\>=1.0\"\]          |
|                                                                       |
| emotion = \[\"speechbrain\>=1.0\", \"transformers\>=4.30\"\]          |
|                                                                       |
| cli = \[\"typer\[all\]\>=0.9\", \"rich\>=13\"\]                       |
|                                                                       |
| \# Convenience bundles                                                |
|                                                                       |
| default = \[                                                          |
|                                                                       |
| \"speechtelemetry\[asr,align,vad,emotion,cli\]\"                      |
|                                                                       |
| \]                                                                    |
|                                                                       |
| all = \[                                                              |
|                                                                       |
| \"speechtelemetry\[asr,align,vad,diarize,prosody,emotion,cli\]\"      |
|                                                                       |
| \]                                                                    |
|                                                                       |
| dev = \[                                                              |
|                                                                       |
| \"speechtelemetry\[all\]\",                                           |
|                                                                       |
| \"pytest\>=8\", \"pytest-cov\", \"mypy\>=1.8\", \"ruff\", \"black\"   |
|                                                                       |
| \]                                                                    |
|                                                                       |
| \[project.scripts\]                                                   |
|                                                                       |
| speechtelemetry = \"speechtelemetry.cli.main:app\"                    |
|                                                                       |
| \# Entry point group for third-party backends                         |
|                                                                       |
| \[project.entry-points.\"speechtelemetry.backends\"\]                 |
|                                                                       |
| \# (populated by third-party packages, not by us)                     |
+-----------------------------------------------------------------------+

**4.3 Installation Paths**

Users choose what they need. This prevents dependency hell when
speechtelemetry is embedded in a larger project.

+-----------------------------------------------------------------------+
| \# Minimal: import library and types only, no ML                      |
|                                                                       |
| pip install speechtelemetry                                           |
|                                                                       |
| \# Transcription + word alignment + VAD (most common use case)        |
|                                                                       |
| pip install speechtelemetry\[default\]                                |
|                                                                       |
| \# Full stack including diarization and prosody                       |
|                                                                       |
| pip install speechtelemetry\[all\]                                    |
|                                                                       |
| \# Only ASR (user provides their own VAD pipeline)                    |
|                                                                       |
| pip install speechtelemetry\[asr\]                                    |
|                                                                       |
| \# Development environment                                            |
|                                                                       |
| pip install -e .\[dev\]                                               |
+-----------------------------------------------------------------------+

**4.4 Runtime Availability Checks**

The library must not fail at import time if optional dependencies are
missing. It must fail at job start time with a clear, actionable error
message. This is enforced by the guarded import pattern in every adapter
and by a pre-flight check in core/pipeline.py.

+-----------------------------------------------------------------------+
| \# core/pipeline.py --- pre-flight check before the job starts        |
|                                                                       |
| def \_check_environment(config: PipelineConfig) -\> None:             |
|                                                                       |
| errors = \[\]                                                         |
|                                                                       |
| \# 1. FFmpeg                                                          |
|                                                                       |
| if not shutil.which(\'ffmpeg\'):                                      |
|                                                                       |
| errors.append(                                                        |
|                                                                       |
| \'FFmpeg not found. Install from https://ffmpeg.org/download.html\'   |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# 2. Backend availability                                            |
|                                                                       |
| for stage, name in \[                                                 |
|                                                                       |
| (\'asr\', config.asr_backend),                                        |
|                                                                       |
| (\'vad\', config.vad_backend),                                        |
|                                                                       |
| (\'align\', config.alignment_backend),                                |
|                                                                       |
| \]:                                                                   |
|                                                                       |
| cls = resolve_backend(stage, name)                                    |
|                                                                       |
| try:                                                                  |
|                                                                       |
| cls.\_check_available() \# class method, raises if not installed      |
|                                                                       |
| except BackendNotAvailableError as e:                                 |
|                                                                       |
| errors.append(str(e))                                                 |
|                                                                       |
| \# 3. HF token for gated backends                                     |
|                                                                       |
| if config.diarization_backend and not os.environ.get(\'HF_TOKEN\'):   |
|                                                                       |
| errors.append(                                                        |
|                                                                       |
| \'HF_TOKEN not set. Required for pyannote diarization.\'              |
|                                                                       |
| )                                                                     |
|                                                                       |
| if errors:                                                            |
|                                                                       |
| raise EnvironmentError(                                               |
|                                                                       |
| \'Pre-flight checks failed:\\n\' + \'\\n\'.join(f\' - {e}\' for e in  |
| errors)                                                               |
|                                                                       |
| )                                                                     |
+-----------------------------------------------------------------------+

**5. Public API Design**

The public API must be discoverable, minimal, and stable. It is the
primary touchpoint for developers embedding speechtelemetry in their own
projects. Every public symbol is a promise --- adding is cheap, removing
is breaking.

**5.1 Public Surface in src/speechtelemetry/\_\_init\_\_.py**

Only the following symbols are public. Everything else is private
(prefixed \_ or not re-exported from \_\_init\_\_.py).

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/\_\_init\_\_.py                                |
|                                                                       |
| \# This file IS the public API.                                       |
|                                                                       |
| from speechtelemetry.api import enrich_media, enrich_audio            |
|                                                                       |
| from speechtelemetry.config import PipelineConfig                     |
|                                                                       |
| from speechtelemetry.types import (                                   |
|                                                                       |
| TranscriptDocument,                                                   |
|                                                                       |
| Segment,                                                              |
|                                                                       |
| Word,                                                                 |
|                                                                       |
| SilenceSpan,                                                          |
|                                                                       |
| ProsodyWindow,                                                        |
|                                                                       |
| EmotionScore,                                                         |
|                                                                       |
| ProcessingReport,                                                     |
|                                                                       |
| StageError,                                                           |
|                                                                       |
| )                                                                     |
|                                                                       |
| from speechtelemetry.interfaces import (                              |
|                                                                       |
| ASRBackend,                                                           |
|                                                                       |
| VADBackend,                                                           |
|                                                                       |
| AlignmentBackend,                                                     |
|                                                                       |
| DiarizationBackend,                                                   |
|                                                                       |
| ProsodyBackend,                                                       |
|                                                                       |
| EmotionBackend,                                                       |
|                                                                       |
| Exporter,                                                             |
|                                                                       |
| )                                                                     |
|                                                                       |
| from speechtelemetry import registry                                  |
|                                                                       |
| \# Version                                                            |
|                                                                       |
| \_\_version\_\_ = \"0.1.0\"                                           |
|                                                                       |
| \# Everything else is private.                                        |
|                                                                       |
| \# Users must not import from speechtelemetry.backends.\* directly.   |
|                                                                       |
| \# Users must not import from speechtelemetry.core.\* directly.       |
+-----------------------------------------------------------------------+

**5.2 The Two Public Entry Points**

There are exactly two high-level functions. All other functions are
internal pipeline stages or helpers.

+-----------------------------------------------------------------------+
| \# api.py                                                             |
|                                                                       |
| def enrich_media(                                                     |
|                                                                       |
| input_path: str,                                                      |
|                                                                       |
| config: PipelineConfig \| None = None,                                |
|                                                                       |
| output_path: str \| None = None,                                      |
|                                                                       |
| ) -\> TranscriptDocument:                                             |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Process any audio or video file through the full pipeline.            |
|                                                                       |
| input_path : path to audio/video file                                 |
|                                                                       |
| config : PipelineConfig (defaults to PipelineConfig() if None)        |
|                                                                       |
| output_path: if given, write JSON export to this path as a side       |
| effect                                                                |
|                                                                       |
| Returns : canonical TranscriptDocument                                |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| def enrich_audio(                                                     |
|                                                                       |
| wav_path: str,                                                        |
|                                                                       |
| config: PipelineConfig \| None = None,                                |
|                                                                       |
| ) -\> TranscriptDocument:                                             |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Process a pre-normalized mono 16 kHz WAV file.                        |
|                                                                       |
| Skips the FFmpeg decode stage.                                        |
|                                                                       |
| For users who control their own audio pipeline.                       |
|                                                                       |
| \"\"\"                                                                |
+-----------------------------------------------------------------------+

**5.3 Stage-Level APIs (Advanced Usage)**

For users who want fine-grained control, each pipeline stage is also
accessible as a standalone function. These are exported from api.py and
documented as advanced/unstable.

+-----------------------------------------------------------------------+
| \# api.py --- stage-level access                                      |
|                                                                       |
| from speechtelemetry.api import (                                     |
|                                                                       |
| run_vad, \# (wav_path, config) -\> list\[SilenceSpan\]                |
|                                                                       |
| run_asr, \# (wav_path, config) -\> list\[Segment\]                    |
|                                                                       |
| run_alignment, \# (segments, wav_path, language, config) -\>          |
| list\[Segment\]                                                       |
|                                                                       |
| run_diarization, \# (wav_path, config) -\> list\[dict\]               |
|                                                                       |
| run_prosody, \# (segments, wav_path, config) -\> list\[Segment\]      |
|                                                                       |
| run_emotion, \# (segments, wav_path, config) -\> list\[Segment\]      |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Example: custom pipeline --- skip diarization, run everything else |
|                                                                       |
| segments = run_asr(wav_path, config)                                  |
|                                                                       |
| segments = run_alignment(segments, wav_path, \'en\', config)          |
|                                                                       |
| segments = run_emotion(segments, wav_path, config)                    |
|                                                                       |
| doc = TranscriptDocument(segments=segments, \...)                     |
+-----------------------------------------------------------------------+

**5.4 Version & Stability Policy**

speechtelemetry follows Semantic Versioning (semver.org).

  ------------------------------------------------------------------------
  **Version bump** **Meaning**              **Examples**
  ---------------- ------------------------ ------------------------------
  MAJOR (1.0.0)    Breaking change to       Remove a field from
                   public API               TranscriptDocument; rename
                                            enrich_media()

  MINOR (0.2.0)    Backward-compatible new  New backend stage; new
                   capability               optional field on Segment

  PATCH (0.1.1)    Bug fixes only           Fix a crash in the prosody
                                            backend; fix a typo in error
                                            messages
  ------------------------------------------------------------------------

-   Types in types.py are versioned separately. Adding an Optional field
    is a MINOR change. Removing any field is a MAJOR change.

-   Backend interfaces (interfaces.py) are versioned. Adding a new
    abstract method is a MAJOR change for all existing third-party
    backends.

-   Everything inside core/, backends/, io/ that is not re-exported from
    \_\_init\_\_.py is private and may change at any time.

**6. Testing Architecture**

The layered architecture makes each component independently testable.
Tests are divided by layer, with the innermost layers tested first and
in isolation.

**6.1 Test Pyramid**

+-----------------------------------------------------------------------+
| /\\                                                                   |
|                                                                       |
| / \\                                                                  |
|                                                                       |
| / E2E \\ Full pipeline on real audio (few, slow)                      |
|                                                                       |
| /\-\-\-\-\-\-\--\\                                                    |
|                                                                       |
| / \\                                                                  |
|                                                                       |
| / Integration \\ Pipeline stages wired together (moderate)            |
|                                                                       |
| /\-\-\-\-\-\-\-\-\-\-\-\-\--\\                                        |
|                                                                       |
| / \\                                                                  |
|                                                                       |
| / Unit Tests \\ Individual backends + types (many, fast)              |
|                                                                       |
| /\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\\                            |
+-----------------------------------------------------------------------+

**6.2 Unit Tests --- Layer 1 & 2**

Unit tests test individual backends using a 5-second canonical WAV
fixture. They use no mocks for the backend under test but may mock the
external ML library if too slow.

+-----------------------------------------------------------------------+
| \# tests/unit/backends/test_silero_vad.py                             |
|                                                                       |
| import pytest                                                         |
|                                                                       |
| from speechtelemetry.backends.vad.silero import SileroVADBackend      |
|                                                                       |
| FIXTURE = \'tests/fixtures/5s_mono_16k.wav\'                          |
|                                                                       |
| def test_returns_list_of_dicts():                                     |
|                                                                       |
| backend = SileroVADBackend()                                          |
|                                                                       |
| result = backend.get_speech_intervals(FIXTURE)                        |
|                                                                       |
| assert isinstance(result, list)                                       |
|                                                                       |
| assert all(\'start\' in r and \'end\' in r for r in result)           |
|                                                                       |
| def test_all_timestamps_in_seconds():                                 |
|                                                                       |
| backend = SileroVADBackend()                                          |
|                                                                       |
| result = backend.get_speech_intervals(FIXTURE)                        |
|                                                                       |
| for r in result:                                                      |
|                                                                       |
| assert r\[\'start\'\] \< 100 \# sanity: never \> 100s for a 5s file   |
|                                                                       |
| \# Verify ABC contract is enforced                                    |
|                                                                       |
| def test_abstract_contract():                                         |
|                                                                       |
| from speechtelemetry.interfaces import VADBackend                     |
|                                                                       |
| class Broken(VADBackend): pass                                        |
|                                                                       |
| with pytest.raises(TypeError):                                        |
|                                                                       |
| Broken() \# should fail: missing get_speech_intervals                 |
+-----------------------------------------------------------------------+

**6.3 Integration Tests --- Layer 3**

Integration tests run multiple stages together using the real pipeline
but allow backend substitution via the registry.

+-----------------------------------------------------------------------+
| \# tests/integration/test_pipeline.py                                 |
|                                                                       |
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| FIXTURE = \'tests/fixtures/30s_english_clean.wav\'                    |
|                                                                       |
| def test_full_pipeline_cpu():                                         |
|                                                                       |
| doc = enrich_media(FIXTURE, PipelineConfig(device=\'cpu\'))           |
|                                                                       |
| assert doc is not None                                                |
|                                                                       |
| assert len(doc.segments) \> 0                                         |
|                                                                       |
| assert doc.processing_report is not None                              |
|                                                                       |
| def test_pipeline_continues_on_emotion_failure(monkeypatch):          |
|                                                                       |
| \# Simulate emotion backend crash                                     |
|                                                                       |
| from speechtelemetry.backends.emotion import speechbrain as sb_module |
|                                                                       |
| monkeypatch.setattr(sb_module.SpeechBrainEmotionBackend,              |
|                                                                       |
| \'predict_segment\', lambda \*a, \*\*k: (\_ for \_ in                 |
| ()).throw(RuntimeError(\'GPU OOM\')))                                 |
|                                                                       |
| doc = enrich_media(FIXTURE, PipelineConfig())                         |
|                                                                       |
| \# pipeline must not crash                                            |
|                                                                       |
| assert doc is not None                                                |
|                                                                       |
| \# emotion stage should be recorded as failed                         |
|                                                                       |
| stages = \[e.stage for e in doc.processing_report.errors\]            |
|                                                                       |
| assert \'emotion\' in stages                                          |
+-----------------------------------------------------------------------+

**6.4 Mock Backends for Fast Testing**

A built-in MockASRBackend and MockVADBackend ship with the library in
tests/fixtures/mock_backends.py. These return deterministic, instant
outputs. They allow testing the pipeline orchestration without any ML
inference.

+-----------------------------------------------------------------------+
| \# tests/fixtures/mock_backends.py                                    |
|                                                                       |
| from speechtelemetry.interfaces import ASRBackend                     |
|                                                                       |
| from speechtelemetry import registry                                  |
|                                                                       |
| class MockASRBackend(ASRBackend):                                     |
|                                                                       |
| STAGE = \'asr\'                                                       |
|                                                                       |
| NAME = \'mock\'                                                       |
|                                                                       |
| def transcribe(self, wav_path, config):                               |
|                                                                       |
| return (\[{\'start\':0.0,\'end\':2.0,\'text\':\'hello world\'}\],     |
| object())                                                             |
|                                                                       |
| \# Register in test suite conftest.py                                 |
|                                                                       |
| registry.register(\'asr\', \'mock\', MockASRBackend)                  |
|                                                                       |
| \# In test:                                                           |
|                                                                       |
| doc = enrich_media(\'any.wav\', PipelineConfig(asr_backend=\'mock\')) |
+-----------------------------------------------------------------------+

**7. Embedding speechtelemetry in Your Own Project**

This section is specifically for developers who want to use
speechtelemetry as a library inside their own Python project. The design
intentionally makes this the primary use case.

**7.1 Install Only What You Need**

+-----------------------------------------------------------------------+
| \# pyproject.toml for a project using speechtelemetry                 |
|                                                                       |
| \[project\]                                                           |
|                                                                       |
| dependencies = \[                                                     |
|                                                                       |
| \"speechtelemetry\[asr,vad\]\>=0.1\", \# only pull in what you use    |
|                                                                       |
| \]                                                                    |
|                                                                       |
| \# Or for quick scripting:                                            |
|                                                                       |
| pip install speechtelemetry\[default\]                                |
+-----------------------------------------------------------------------+

**7.2 Use the One-Shot High-Level API**

+-----------------------------------------------------------------------+
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| \# Minimal: use all defaults                                          |
|                                                                       |
| doc = enrich_media(\"interview.mp4\")                                 |
|                                                                       |
| \# With explicit config                                               |
|                                                                       |
| doc = enrich_media(                                                   |
|                                                                       |
| \"interview.mp4\",                                                    |
|                                                                       |
| config=PipelineConfig(                                                |
|                                                                       |
| asr_backend=\"faster-whisper\",                                       |
|                                                                       |
| asr_model_size=\"medium\",                                            |
|                                                                       |
| device=\"auto\",                                                      |
|                                                                       |
| diarization_backend=None, \# skip diarization                         |
|                                                                       |
| emotion_backend=None, \# skip emotion                                 |
|                                                                       |
| )                                                                     |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Work with the result                                               |
|                                                                       |
| for seg in doc.segments:                                              |
|                                                                       |
| print(f\'{seg.start:.2f}s {seg.text}\')                               |
|                                                                       |
| for word in (seg.words or \[\]):                                      |
|                                                                       |
| print(f\' {word.start:.3f}s {word.text}\')                            |
+-----------------------------------------------------------------------+

**7.3 Plug In a Custom Backend**

+-----------------------------------------------------------------------+
| from speechtelemetry import registry                                  |
|                                                                       |
| from speechtelemetry.interfaces import EmotionBackend                 |
|                                                                       |
| \# Your custom emotion backend (e.g., wrapping a fine-tuned model)    |
|                                                                       |
| class MyFinetuned_Emotion(EmotionBackend):                            |
|                                                                       |
| def predict_segment(self, wav_path, start_s, end_s):                  |
|                                                                       |
| \# \... your inference \...                                           |
|                                                                       |
| return {                                                              |
|                                                                       |
| \'label_distribution\': {\'happy\': 0.7, \'neutral\': 0.3},           |
|                                                                       |
| \'confidence\': 0.7,                                                  |
|                                                                       |
| \'backend_name\': \'my-finetuned-v1\',                                |
|                                                                       |
| }                                                                     |
|                                                                       |
| registry.register(\'emotion\', \'my-finetuned\', MyFinetuned_Emotion) |
|                                                                       |
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| doc = enrich_media(\"audio.wav\",                                     |
| PipelineConfig(emotion_backend=\"my-finetuned\"))                     |
+-----------------------------------------------------------------------+

**7.4 Use Stage APIs for Custom Pipelines**

+-----------------------------------------------------------------------+
| from speechtelemetry.api import run_vad, run_asr, run_alignment       |
|                                                                       |
| from speechtelemetry.config import PipelineConfig                     |
|                                                                       |
| from speechtelemetry.types import TranscriptDocument                  |
|                                                                       |
| from speechtelemetry.exporters.srt import SRTExporter                 |
|                                                                       |
| \# Build a custom pipeline: transcription only, no prosody or emotion |
|                                                                       |
| config = PipelineConfig(device=\'cpu\')                               |
|                                                                       |
| silence_spans = run_vad(\'audio.wav\', config)                        |
|                                                                       |
| segments = run_asr(\'audio.wav\', config)                             |
|                                                                       |
| segments = run_alignment(segments, \'audio.wav\', \'en\', config)     |
|                                                                       |
| doc = TranscriptDocument(                                             |
|                                                                       |
| source_path=\'audio.wav\',                                            |
|                                                                       |
| language=\'en\',                                                      |
|                                                                       |
| segments=segments,                                                    |
|                                                                       |
| silence_spans=silence_spans,                                          |
|                                                                       |
| )                                                                     |
|                                                                       |
| SRTExporter().export(doc, \'output.srt\')                             |
+-----------------------------------------------------------------------+

**7.5 Work with TranscriptDocument Programmatically**

+-----------------------------------------------------------------------+
| from speechtelemetry.types import TranscriptDocument                  |
|                                                                       |
| \# Serialisation                                                      |
|                                                                       |
| import json, dataclasses                                              |
|                                                                       |
| def to_dict(doc: TranscriptDocument) -\> dict:                        |
|                                                                       |
| return dataclasses.asdict(doc) \# works if all fields are dataclasses |
|                                                                       |
| \# Or use the built-in JSON exporter                                  |
|                                                                       |
| from speechtelemetry.exporters.json_exporter import JsonExporter      |
|                                                                       |
| JsonExporter().export(doc, \'transcript.json\')                       |
|                                                                       |
| \# Loading from JSON                                                  |
|                                                                       |
| from speechtelemetry.exporters.json_exporter import JsonExporter      |
|                                                                       |
| doc = JsonExporter.load(\'transcript.json\') \# returns               |
| TranscriptDocument                                                    |
|                                                                       |
| \# Filter segments by confidence                                      |
|                                                                       |
| high_conf = \[s for s in doc.segments if s.confidence \> 0.8\]        |
|                                                                       |
| \# Get all unique speakers                                            |
|                                                                       |
| speakers = {s.speaker for s in doc.segments if s.speaker}             |
|                                                                       |
| \# Get all emotion scores                                             |
|                                                                       |
| emotions = \[s.emotion for s in doc.segments if s.emotion is not      |
| None\]                                                                |
+-----------------------------------------------------------------------+

**8. Cross-Cutting Concerns**

**8.1 Logging**

The library uses Python\'s standard logging module exclusively. No
print() statements in library code. No third-party logging frameworks.

+-----------------------------------------------------------------------+
| \# Every module in the library                                        |
|                                                                       |
| import logging                                                        |
|                                                                       |
| logger = logging.getLogger(\_\_name\_\_)                              |
|                                                                       |
| \# e.g. logger name: \'speechtelemetry.backends.asr.faster_whisper\'  |
|                                                                       |
| \# Usage                                                              |
|                                                                       |
| logger.debug(\'Loading model: %s\', model_size)                       |
|                                                                       |
| logger.info(\'ASR stage complete: %d segments in %.2fs\', n, elapsed) |
|                                                                       |
| logger.warning(\'Short segment skipped for emotion: %.3fs\',          |
| duration)                                                             |
|                                                                       |
| logger.error(\'VAD failed: %s\', exc)                                 |
|                                                                       |
| \# Users configure logging --- the library NEVER calls basicConfig()  |
|                                                                       |
| \# Good: users can silence or redirect all speechtelemetry logs via:  |
|                                                                       |
| \# logging.getLogger(\'speechtelemetry\').setLevel(logging.WARNING)   |
+-----------------------------------------------------------------------+

**8.2 Caching & Model Storage**

All model weights are cached in a single directory, configurable via
SPEECHTELEMETRY_CACHE_DIR. The default is \~/.cache/speechtelemetry/.
Backends must use this path rather than their library\'s default cache,
giving users one place to find and clear all model weights.

+-----------------------------------------------------------------------+
| \# core/cache.py                                                      |
|                                                                       |
| import os                                                             |
|                                                                       |
| from pathlib import Path                                              |
|                                                                       |
| def get_cache_dir(subdir: str = \'\') -\> Path:                       |
|                                                                       |
| root = Path(os.environ.get(\'SPEECHTELEMETRY_CACHE_DIR\',             |
|                                                                       |
| Path.home() / \'.cache\' / \'speechtelemetry\'))                      |
|                                                                       |
| path = root / subdir if subdir else root                              |
|                                                                       |
| path.mkdir(parents=True, exist_ok=True)                               |
|                                                                       |
| return path                                                           |
|                                                                       |
| \# Usage in backends                                                  |
|                                                                       |
| savedir = str(get_cache_dir(\'speechbrain\'))                         |
|                                                                       |
| savedir_fw = str(get_cache_dir(\'faster-whisper\'))                   |
+-----------------------------------------------------------------------+

**8.3 Thread Safety & Parallelism**

The library makes no guarantees about thread safety of backend
instances. Backend objects must not be shared across threads. For
parallelism:

-   Use multiprocessing (process-per-file) not threading. This avoids
    GIL issues and CUDA context sharing problems.

-   Parselmouth is explicitly not thread-safe --- never share a Sound
    object or call parselmouth functions concurrently from threads.

-   GPU backends (faster-whisper on CUDA, pyannote on CUDA) are not safe
    to call concurrently from multiple threads sharing a CUDA context.

-   The pipeline function enrich_media() is stateless and safe to call
    concurrently from separate processes.

**8.4 Type Annotations**

Every public function and method must have complete type annotations.
The codebase is checked with mypy in strict mode as part of CI.

+-----------------------------------------------------------------------+
| \# mypy configuration in pyproject.toml                               |
|                                                                       |
| \[tool.mypy\]                                                         |
|                                                                       |
| python_version = \"3.10\"                                             |
|                                                                       |
| strict = true                                                         |
|                                                                       |
| ignore_missing_imports = true \# tolerate untyped third-party stubs   |
|                                                                       |
| \[\[tool.mypy.overrides\]\]                                           |
|                                                                       |
| module = \[\"faster_whisper.\*\", \"whisperx.\*\",                    |
| \"parselmouth.\*\",                                                   |
|                                                                       |
| \"pyannote.\*\", \"speechbrain.\*\", \"silero_vad.\*\"\]              |
|                                                                       |
| ignore_missing_imports = true                                         |
+-----------------------------------------------------------------------+

**8.5 Configuration Immutability**

PipelineConfig is immutable once instantiated. Backends receive the
config object but must never mutate it. Use Pydantic\'s model_config =
ConfigDict(frozen=True) to enforce this at runtime.

+-----------------------------------------------------------------------+
| \# config.py                                                          |
|                                                                       |
| from pydantic import BaseModel, ConfigDict                            |
|                                                                       |
| class PipelineConfig(BaseModel):                                      |
|                                                                       |
| model_config = ConfigDict(frozen=True) \# immutable after             |
| instantiation                                                         |
|                                                                       |
| asr_backend: str = \'faster-whisper\'                                 |
|                                                                       |
| \# \...                                                               |
+-----------------------------------------------------------------------+

**9. Repository Layout with Rationale**

+-----------------------------------------------------------------------+
| speechtelemetry/ \# repo root                                         |
|                                                                       |
| ├── src/                                                              |
|                                                                       |
| │ └── speechtelemetry/ \# importable package (src layout prevents     |
|                                                                       |
| │ ├── \_\_init\_\_.py \# accidental non-installed imports)            |
|                                                                       |
| │ ├── api.py \# public entry points                                   |
|                                                                       |
| │ ├── config.py \# PipelineConfig                                     |
|                                                                       |
| │ ├── types.py \# canonical dataclasses                               |
|                                                                       |
| │ ├── interfaces.py \# ABCs for all backend stages                    |
|                                                                       |
| │ ├── registry.py \# backend name → class resolver                    |
|                                                                       |
| │ ├── exceptions.py \# BackendNotAvailableError, etc.                 |
|                                                                       |
| │ ├── core/                                                           |
|                                                                       |
| │ │ ├── pipeline.py \# orchestration                                  |
|                                                                       |
| │ │ ├── job.py \# temp file lifecycle                                 |
|                                                                       |
| │ │ └── provenance.py \# backend tracking                             |
|                                                                       |
| │ ├── io/                                                             |
|                                                                       |
| │ │ ├── ffmpeg.py \# decode → canonical WAV                           |
|                                                                       |
| │ │ └── audio_normalize.py \# post-FFmpeg validation                  |
|                                                                       |
| │ ├── backends/                                                       |
|                                                                       |
| │ │ ├── \_\_init\_\_.py \# registers all default backends             |
|                                                                       |
| │ │ ├── asr/                                                          |
|                                                                       |
| │ │ │ ├── faster_whisper.py                                           |
|                                                                       |
| │ │ │ └── whisperx.py                                                 |
|                                                                       |
| │ │ ├── vad/                                                          |
|                                                                       |
| │ │ │ └── silero.py                                                   |
|                                                                       |
| │ │ ├── alignment/                                                    |
|                                                                       |
| │ │ │ └── whisperx.py                                                 |
|                                                                       |
| │ │ ├── diarization/                                                  |
|                                                                       |
| │ │ │ └── pyannote.py                                                 |
|                                                                       |
| │ │ ├── prosody/                                                      |
|                                                                       |
| │ │ │ ├── parselmouth.py \# GPL-gated optional adapter                |
|                                                                       |
| │ │ │ └── copasul.py                                                  |
|                                                                       |
| │ │ └── emotion/                                                      |
|                                                                       |
| │ │ └── speechbrain.py                                                |
|                                                                       |
| │ ├── exporters/                                                      |
|                                                                       |
| │ │ ├── json_exporter.py \# authoritative; lossless                   |
|                                                                       |
| │ │ ├── srt.py                                                        |
|                                                                       |
| │ │ ├── vtt.py                                                        |
|                                                                       |
| │ │ └── textgrid.py                                                   |
|                                                                       |
| │ └── cli/                                                            |
|                                                                       |
| │ ├── main.py                                                         |
|                                                                       |
| │ └── commands.py                                                     |
|                                                                       |
| ├── tests/                                                            |
|                                                                       |
| │ ├── fixtures/ \# WAV files, golden JSON outputs                     |
|                                                                       |
| │ │ └── mock_backends.py \# instant mock backends for unit tests      |
|                                                                       |
| │ ├── unit/ \# per-module, no IO beyond fixtures                      |
|                                                                       |
| │ ├── integration/ \# multi-stage pipeline tests                      |
|                                                                       |
| │ └── e2e/ \# full file-in → JSON-out tests                           |
|                                                                       |
| ├── benchmarking/                                                     |
|                                                                       |
| │ ├── fixtures.py                                                     |
|                                                                       |
| │ ├── metrics.py                                                      |
|                                                                       |
| │ └── compare.py                                                      |
|                                                                       |
| ├── docs/                                                             |
|                                                                       |
| │ ├── architecture.md \# mirrors this document                        |
|                                                                       |
| │ ├── backends.md \# backend catalogue                                |
|                                                                       |
| │ └── embedding.md \# guide for library consumers                     |
|                                                                       |
| ├── examples/                                                         |
|                                                                       |
| │ ├── quickstart.py                                                   |
|                                                                       |
| │ ├── custom_backend.py                                               |
|                                                                       |
| │ └── stage_api_example.py                                            |
|                                                                       |
| ├── pyproject.toml                                                    |
|                                                                       |
| ├── README.md                                                         |
|                                                                       |
| └── CHANGELOG.md                                                      |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Why src/ layout?**                                                  |
|                                                                       |
| Placing the package under src/ ensures that when running tests with   |
| pytest, the                                                           |
|                                                                       |
| installed version of the package is used (not the raw source          |
| directory). This                                                      |
|                                                                       |
| prevents subtle test-passes-locally-fails-on-install bugs. All        |
| serious Python                                                        |
|                                                                       |
| libraries (attrs, httpx, structlog) use this layout.                  |
+-----------------------------------------------------------------------+

**10. Architecture Decision Log**

This section records why key architecture decisions were made. Future
contributors must read this before proposing structural changes.

  -------------------------------------------------------------------------
  **Decision**         **Rationale**                **Rejected
                                                    Alternatives**
  -------------------- ---------------------------- -----------------------
  Strategy + ABC for   Enforces the contract at     Protocols
  backends             instantiation. TypeError at  (typing.Protocol):
                       class creation is better     structurally typed,
                       than AttributeError at call  harder to enforce at
                       time. Enables third-party    runtime. Duck-typing:
                       backends without forking.    too fragile for a
                                                    multi-backend system.

  Registry pattern     Decouples pipeline from      Direct imports in
  (string → class)     concrete backends. Enables   pipeline.py: couples
                       runtime substitution.        core to every backend,
                       Enables mock backends in     makes testing hard.
                       tests. Enables third-party   Import-time factory
                       entry-point registration.    functions: same
                                                    problem.

  Optional             Users of the library must    Ship everything as
  dependencies via     not be forced to install     required deps: causes
  pyproject.toml       torch, torchaudio,           dependency hell, makes
  extras               speechbrain, etc. if they    the package unusable as
                       only want the VAD stage.     a library component.
                       Extras give users a menu.    Auto-detect-and-skip:
                                                    fragile, hard to debug.

  types.py has zero    TranscriptDocument must be   Using Pydantic for all
  external imports     importable by any tool       types: forces pydantic
                       without pulling in ML        as a dependency even on
                       stacks. Users working only   tools that only read
                       with saved JSON outputs      JSON. Using
                       should not need torch.       dataclasses + pydantic
                                                    validators: added
                                                    complexity.

  src/ layout          Prevents test suite from     Flat layout: fails for
                       accidentally importing the   editable installs in
                       non-installed source tree.   some environments.
                       Matches Python packaging
                       best practices for
                       libraries.

  Immutable            Backends receiving the       Mutable config: allows
  PipelineConfig       config object must not be    backends to adjust
                       able to modify it            config; rejected
                       mid-pipeline, causing        because it makes
                       hard-to-debug race           pipeline state
                       conditions or unexpected     unpredictable.
                       state.

  ProcessingReport on  Users must always be able to Separate report object
  every                see what ran, how long it    returned as a tuple:
  TranscriptDocument   took, and what failed ---    burdens callers.
                       without having to catch      Logging only: invisible
                       exceptions and build their   to downstream consumers
                       own report.                  of the
                                                    TranscriptDocument.
  -------------------------------------------------------------------------

*Document version 0.1. Architecture Reference for speechtelemetry v0.1.
May 2026.*

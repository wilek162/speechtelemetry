# speechtelemetry — Architecture Reference

Design principles, layer boundaries, extension model, and packaging strategy for v0.1.

This is document #2 in the source-of-truth hierarchy. It answers *why the system is shaped this way* and *how the pieces fit together*. For *what each backend does and how to call it*, see `docs/developer_reference.md`.

---

## 1. Architectural principles

Every structural decision flows from these enforced constraints. A violation is a defect, not a trade-off.

### 1.1 The library is the source of truth

`src/speechtelemetry/` is the only layer that matters for correctness and portability. The CLI, any future REST API, and any GUI are thin wrappers that translate user intent into library calls. They add zero logic.

- A developer using the library as a Python import must be able to do everything the CLI can do — with more control.
- The CLI must never contain business logic absent from the library.
- Outputs are always `TranscriptDocument` — never raw dicts or format-specific objects at the public boundary.

### 1.2 Dependencies flow inward only

Inner layers know nothing about outer layers:

```
types / interfaces / exceptions / config
        ↓
     backends / io
        ↓
    core/pipeline
        ↓
    exporters / api
        ↓
      cli / tests
```

`types.py` must never import from `backends/`. `core/pipeline.py` must never import from `cli/`. A backend must never import from another backend. Any violation is an architecture bug.

### 1.3 Stable core, pluggable periphery

`types.py`, `config.py`, and `core/pipeline.py` form a stable core that changes infrequently and only with a version bump. Every backend, exporter, and CLI command is a plug that snaps into that core. Adding a new ASR engine requires zero changes to `types.py` or `pipeline.py`.

### 1.4 Fail soft, fail honestly

Partial output is always better than a hard crash. Every optional stage is independently failable. All failures are recorded in `ProcessingReport.errors` — they are never silently swallowed and never propagate as unhandled exceptions.

The only hard failures are environment prerequisites (missing FFmpeg, missing `HF_TOKEN`, CUDA unavailable) because those make the entire job impossible. These are detected together in the pre-flight check at job start.

### 1.5 Local-first, offline-capable by default

No network call is ever required at runtime. Model weights are cached locally on first download. The library must be operable on an air-gapped machine after initial setup.

### 1.6 Explicit over implicit

Nothing meaningful happens silently. Device selection (`"cpu"` or `"cuda"`), model choices, and backend selection are all visible in `PipelineConfig`. GPU is never assumed — `device` defaults to `"cpu"` and must be explicitly set to `"cuda"` with a compatible torch installation.

Every output field that comes from an ML backend stores `backend_name` provenance. `ProcessingReport` records every stage timing, peak RAM/VRAM, and every error.

### 1.7 One schema for all outputs

`TranscriptDocument` (defined in `types.py`) is the single authoritative output format. All exporters are lossless or explicitly lossy projections of this type. The JSON export is the archival format; SRT, VTT, and TextGrid are derived views.

---

## 2. Layer model

The library is divided into five layers. Each has a single responsibility and a clear boundary. No layer reaches across more than one boundary.

### Layer 1 — Domain core

| Module | Responsibility |
|--------|---------------|
| `types.py` | All canonical dataclasses (`TranscriptDocument`, `Segment`, `Word`, `SilenceSpan`, `ProsodyWindow`, `EmotionScore`, `ProcessingReport`, `StageError`). Zero external imports. Pure Python stdlib only. |
| `interfaces.py` | Abstract Base Classes for every backend stage (`ASRBackend`, `VADBackend`, `AlignmentBackend`, `DiarizationBackend`, `ProsodyBackend`, `EmotionBackend`, `Exporter`). Zero external imports. |
| `config.py` | `PipelineConfig` Pydantic v2 model. Imports only `pydantic`. No speechtelemetry imports. |
| `exceptions.py` | All custom exceptions. No imports. |

**Rules:**
- No backend code, no ML library imports, no I/O, no subprocess calls in this layer.
- `types.py` must not import from any other module in this project.
- This layer changes only when the schema changes — a rare, versioned event.

### Layer 2 — Adapters (`backends/` and `io/`)

| Module | Responsibility |
|--------|---------------|
| `backends/<stage>/<name>.py` | One class per backend, inheriting from the corresponding ABC in `interfaces.py`. |
| `io/ffmpeg.py` | FFmpeg subprocess wrapper. `normalize_to_wav()` only. No pipeline knowledge. |
| `io/audio_normalize.py` | Post-FFmpeg WAV validation. No ML. |

**Rules:**
- Each adapter class inherits from exactly one ABC. `TypeError` is raised at instantiation if the contract is not satisfied.
- No adapter imports from another adapter. They are siblings, not a hierarchy.
- All external imports (`torch`, `faster_whisper`, `parselmouth`, etc.) live inside adapter modules only — never in `types.py`, `core/`, or `cli/`.
- Adapter constructors must catch `ImportError` and raise `BackendNotAvailableError` with install instructions. The library core must never see a raw `ImportError` from a missing optional backend.

### Layer 3 — Orchestration (`core/`)

| Module | Responsibility |
|--------|---------------|
| `core/pipeline.py` | Implements `run_pipeline()`. Calls backends in order, wraps each stage in `try/except`, collects timings and errors into `ProcessingReport`, returns `TranscriptDocument`. |
| `core/job.py` | Single-job lifecycle: temp file creation, cleanup in `finally` blocks, GPU memory release. |

**Rules:**
- `core/` imports from Layer 1 (types, interfaces, config, exceptions) and from `registry`. Never from Layer 4 or 5.
- `core/pipeline.py` must never reference a concrete backend class directly. It always resolves through `registry.py`.
- `enrich_media()` and `run_pipeline()` are pure functions at heart: given a config and a file path, they return a `TranscriptDocument`. No global state is mutated.

### Layer 4 — Exporters (`exporters/`)

| Module | Responsibility |
|--------|---------------|
| `exporters/json_exporter.py` | Serializes `TranscriptDocument` to JSON. Lossless. |
| `exporters/srt.py`, `exporters/vtt.py` | Subtitle exports. Explicitly lossy (text + timestamps only). |
| `exporters/textgrid.py` | Praat TextGrid format for phonetics toolchain integration. |

**Rules:**
- Exporters receive a `TranscriptDocument` and an output path, produce a file, and do nothing else.
- Exporters must never re-run inference.
- Lossy exporters must document exactly which fields are dropped.
- Exporters may not import from `core/` or `backends/`.

### Layer 5 — Presentation (`cli/`, future `api/`)

| Module | Responsibility |
|--------|---------------|
| `cli/main.py` | Typer entry point. Parses CLI arguments, builds `PipelineConfig`, calls `enrich_media()`, calls the requested exporter. |

**Rules:**
- Zero business logic in the CLI. If there is an `if/else` in `cli/` that is not purely about argument parsing or output formatting, that logic belongs in `core/` or a backend.
- The CLI is a consumer of the library, exactly like an external user's script. It uses only the public API.
- Error messages from the CLI must be human-readable translations of structured exceptions — not raw tracebacks.

---

## 3. Backend extension model

The ability to swap backends without touching core code is the most important extensibility feature. It is implemented via the Strategy pattern with ABCs plus a central Registry.

### 3.1 Strategy pattern with ABCs

Every pipeline stage is defined as a strategy: an ABC in `interfaces.py` that declares the contract (method names, signatures, and return types) without any implementation. If a backend class does not implement all abstract methods, Python raises `TypeError` at instantiation — immediate feedback, not a runtime failure deep in the stack.

### 3.2 The registry

`registry.py` is a runtime mapping from backend name strings to classes. The pipeline never instantiates a backend class directly — it always asks the registry. This decouples the pipeline from concrete implementations and makes the system testable with mock backends.

```python
# Adding a new ASR backend:
# 1. Create backends/asr/my_backend.py, inherit from ASRBackend
# 2. Add one line to registry.py:
_REGISTRY["asr"]["my-backend"] = "speechtelemetry.backends.asr.my_backend.MyBackend"
# That is the entire change to existing code.
```

### 3.3 Custom backend registration (external users)

Any user or third-party package can register a custom backend before calling `enrich_media()`:

```python
from speechtelemetry import registry
from speechtelemetry.interfaces import EmotionBackend

class MyEmotionModel(EmotionBackend):
    STAGE = "emotion"
    NAME  = "my-model"

    def predict_segment(self, wav_path, start_s, end_s):
        return {
            "label_distribution": {"happy": 0.8, "neutral": 0.2},
            "confidence": 0.8,
            "backend_name": "my-model-v1",
        }

registry.register("emotion", "my-model", MyEmotionModel)
```

### 3.4 Third-party backend packages (entry points)

Third-party packages can auto-register backends by declaring an entry point in their `pyproject.toml`:

```toml
[project.entry-points."speechtelemetry.backends"]
my-asr = "my_package:MyASRBackend"
```

The backend class must define class-level `STAGE` and `NAME` attributes. Auto-discovery runs at `import speechtelemetry`.

---

## 4. Dependency management and packaging

### 4.1 Dependency tiers

| Tier | Location | Rule |
|------|----------|------|
| Core | `dependencies` | MIT/BSD/Apache 2.0 only. No ML libraries. Minimal. |
| Optional adapters | `[project.optional-dependencies]` | One extra per feature area. ML libraries here. |
| GPU extras | `[cuda]` extra | CUDA-enabled torch must be pre-installed by the user. |
| Dev tools | `[dev]` extra | Testing, linting, type-checking. Never shipped to users. |

### 4.2 Installation paths

```bash
# Minimal — library and types only (no ML, for embedding)
pip install speechtelemetry

# Common — transcription + alignment + VAD + emotion + CLI
pip install speechtelemetry[default]

# Full — adds diarization + prosody
pip install speechtelemetry[all]

# GPU — after installing CUDA-enabled torch separately
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install speechtelemetry[cuda]
```

The `[cuda]` extra installs the same ML backends as `[default]`; it does not bundle CUDA or torch. Users must install the CUDA-enabled torch build themselves before installing `[cuda]`.

### 4.3 Runtime availability checks

The library must not fail at import time if optional dependencies are missing. It fails at job start with a clear, actionable error message. This is enforced by:
1. The guarded import pattern in every adapter: `try: import dep except ImportError: raise BackendNotAvailableError(...)`
2. The pre-flight check in `core/pipeline.py` that surfaces all problems together before any compute starts.

---

## 5. Public API design

### 5.1 Public surface (`__init__.py`)

Only the following symbols are public. Everything else is internal and may change without notice.

```python
# Entry points
enrich_media, enrich_audio

# Configuration
PipelineConfig

# Data model
TranscriptDocument, Segment, Word, SilenceSpan,
ProsodyWindow, EmotionScore, ProcessingReport, StageError

# Backend interfaces (for custom backend development)
ASRBackend, VADBackend, AlignmentBackend,
DiarizationBackend, ProsodyBackend, EmotionBackend, Exporter

# Registry (for custom backend registration)
registry

# Exceptions
SpeechTelemetryError, BackendNotAvailableError,
BackendNotFoundError, EnvironmentCheckError, BackendError
```

### 5.2 Stability policy

- `api.py`, `config.py`, and `types.py` form the stable public surface. They follow semantic versioning.
- Adding an `Optional` field to any type is a MINOR change. Removing or renaming a field is a MAJOR change.
- Adding a new abstract method to any ABC in `interfaces.py` is a MAJOR change for all existing third-party backends.
- Everything inside `core/`, `backends/`, `io/`, `exporters/` that is not re-exported from `__init__.py` is internal and may change in any version.

---

## 6. Testing architecture

### 6.1 Test layout

| Type | Location | Purpose |
|------|----------|---------|
| Unit | `tests/unit/` | Pure logic and data models; no ML deps, no I/O. Must complete in under 1 second total. |
| Integration | `tests/integration/` | Real pipeline flows with actual backends; gated with `@pytest.mark.slow`. |
| Fixture/golden | `tests/unit/` + `tests/fixtures/` | Canonical input → committed golden output; assert exact output. |
| Setup/preflight | `tests/unit/test_preflight.py` | Package importable, public API intact, environment sane. |

### 6.2 Testing principles

- Unit tests test the orchestration logic and data model contracts. They do not load ML models.
- Integration tests run real backends and are skipped in CI if GPU or `HF_TOKEN` is unavailable.
- Golden output tests: a committed 5-second WAV fixture produces a committed `golden_output.json`. If output changes, the test fails and the developer must explicitly update the golden file.
- No VRAM left allocated: every integration test checks that VRAM is freed after teardown.
- Mock backends (`MockASRBackend`, `MockVADBackend`) in `tests/fixtures/mock_backends.py` return deterministic, instant outputs for testing pipeline orchestration without ML inference.

---

## 7. Cross-cutting concerns

### 7.1 Logging

The library uses Python's standard `logging` module exclusively. No `print()` statements in library code. Logger names follow the module name: `logging.getLogger(__name__)`.

### 7.2 Model caching

All model weights are cached in a single directory configurable via `SPEECHTELEMETRY_CACHE_DIR`. The default is `~/.cache/speechtelemetry/`. Backends must use this path rather than their upstream library's default, giving users one place to find and clear all model weights.

### 7.3 Thread safety and parallelism

The library makes no thread-safety guarantees for backend instances. Backend objects must not be shared across threads.

- Use multiprocessing (one process per file), not threading. This avoids GIL issues and CUDA context sharing problems.
- Parselmouth is explicitly not thread-safe — never share a `Sound` object or call parselmouth functions concurrently from threads.
- GPU backends (`faster-whisper` on CUDA, `pyannote` on CUDA) are not safe to call concurrently from multiple threads sharing a CUDA context.
- `enrich_media()` is stateless and safe to call concurrently from separate processes.

### 7.4 Type annotations

Every public function and method has complete type annotations. The codebase is checked with `mypy --strict` as part of CI.

### 7.5 Configuration immutability

`PipelineConfig` is frozen after instantiation (`ConfigDict(frozen=True)`). Backends receive the config but must never mutate it.

### 7.6 GPU memory management

When switching between ML stages on GPU:
1. `del model` the large model object explicitly.
2. Call `gc.collect()`.
3. Call `torch.cuda.empty_cache()`.
4. Process audio in chunks (`max_chunk_duration_s`, default 30 s). Never load a full multi-hour WAV as a numpy array.

---

## 8. Repository layout

```
speechtelemetry/
├── src/
│   └── speechtelemetry/
│       ├── __init__.py          — public API surface
│       ├── api.py               — enrich_media(), enrich_audio(), stage APIs
│       ├── config.py            — PipelineConfig
│       ├── types.py             — canonical data model
│       ├── interfaces.py        — backend ABCs
│       ├── registry.py          — name → class resolver
│       ├── exceptions.py        — all custom exceptions
│       ├── provenance.py        — backend/model/device recording
│       ├── core/
│       │   ├── pipeline.py      — stage orchestration + fail-soft + preflight
│       │   └── job.py           — single-job lifecycle management
│       ├── io/
│       │   ├── ffmpeg.py        — FFmpeg decode wrapper
│       │   └── audio_normalize.py
│       ├── backends/
│       │   ├── asr/             — faster_whisper, whisperx, whisper_cpp, sensevoice
│       │   ├── vad/             — silero, funasr
│       │   ├── alignment/       — whisperx, forcealign
│       │   ├── diarization/     — pyannote, funasr
│       │   ├── prosody/         — parselmouth, copasul, librosa, audioflux
│       │   └── emotion/         — speechbrain, emotion2vec, emobox
│       ├── exporters/           — json, srt, vtt, textgrid
│       └── cli/
│           └── main.py
├── tests/
│   ├── unit/                    — fast, no ML deps
│   ├── integration/             — real backends; @pytest.mark.slow
│   └── fixtures/                — canonical WAV, golden JSON, mock backends
├── docs/
│   ├── spec.md                  — THIS IS #1. Product intent and scope.
│   ├── architecture.md          — THIS FILE. Design and layer model.
│   ├── developer_reference.md   — Backend APIs, install commands, caveats.
│   ├── agent_playbook.md        — Workflow for humans and AI agents.
│   ├── license_policy.md        — Per-backend license table and compliance.
│   └── setup_windows.md         — Complete Windows 11 PowerShell setup guide.
├── examples/                    — runnable integration scripts
├── pyproject.toml
└── CONTRIBUTING.md
```

---

## 9. Architecture decision log

Key decisions recorded for future contributors. Read these before proposing structural changes in the same areas.

**ADL-001: Dataclasses for types, not Pydantic models**
`types.py` uses standard dataclasses rather than Pydantic v2 models to keep the domain core dependency-free. `config.py` uses Pydantic because validation and immutability are required there. This keeps `types.py` a pure leaf with zero external imports.

**ADL-002: Lazy imports in the registry**
Backend modules are not imported at `import speechtelemetry` time. They are imported only when `resolve_backend()` is called for the first time. This keeps import time fast even if heavy ML libraries (torch, faster-whisper) are installed.

**ADL-003: No `device="auto"` in config**
The `device` field accepts only `"cpu"` or `"cuda"`. There is no `"auto"` option. GPU is never assumed. This enforces the principle that GPU support is an explicit install path (CUDA-enabled torch + `speechtelemetry[cuda]`) and not a default assumption.

**ADL-004: `prosody_backend` defaults to `[]`**
Prosody is opt-in because the most accessible backend (parselmouth) is GPL-3.0. An empty default ensures users with permissive-license requirements never accidentally pull in a GPL dependency. Users who want prosody must choose a backend explicitly.

**ADL-005: Pre-flight check reports all errors together**
Rather than failing on the first missing dependency, the pre-flight check collects all environment problems and raises a single `EnvironmentCheckError` listing everything that needs to be fixed. This saves users from the "install one thing, hit next error" loop.

**ADL-006: Fail-soft is the default; hard fails are limited to environment prerequisites**
Optional stages (alignment, diarization, prosody, emotion) fail silently into `ProcessingReport.errors` because partial output (a transcript without emotion scores) is always better than no output. Hard failures are reserved for conditions that make the entire job impossible: missing FFmpeg (cannot decode), missing `HF_TOKEN` when diarization is configured, or CUDA unavailable when `device="cuda"` is requested.

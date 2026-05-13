**speechtelemetry**

Architecture & Design Reference

*Open-source transcript, subtitle, prosody, and emotion enrichment
library*

Version 0.1 \| May 2026

Table of Contents

1\. Purpose & Scope

This document is the authoritative architecture reference for the
speechtelemetry library. It describes the structural design decisions,
separation-of-concerns strategy, design principles, and patterns that
govern how the codebase is organised, how its components interact, and
how third-party developers can integrate, extend, or swap any part of
the system.

It is a companion to the Developer & AI Agent Reference (api details,
exact method signatures, backend-specific parameters) and the Library
Spec v0.1 (product definition, defaults, performance targets). Those
documents answer \"what does each backend do and how do I call it?\" ---
this document answers \"why is the system shaped this way and how do the
pieces fit together?\"

2\. Architectural Goals

Every structural decision in speechtelemetry is driven by six goals,
derived directly from the product principles in the spec:

  --------------------------------------------------------------------------
  **Goal**          **One-line statement**    **Architectural implication**
  ----------------- ------------------------- ------------------------------
  Library-first     The library is the source All logic lives in
                    of truth. CLI, UI, and    src/speechtelemetry/. Nothing
                    API wrappers are thin     meaningful lives only in CLI
                    consumers.                or HTTP layer.

  Pluggable         Any stage can be swapped  Backend registry pattern;
  backends          without touching the      strict interface contracts; no
                    public API or the data    hardwired imports in core/.
                    model.

  Stable public     Consumers pin to a        Thin, versioned public API;
  surface           version and trust that    semantic versioning; all
                    enrich_media() +          internals are private.
                    PipelineConfig +
                    TranscriptDocument do not
                    change incompatibly.

  Local-first /     No network calls required Environment checks at startup;
  offline-capable   at runtime after models   all I/O goes through io/
                    are downloaded.           layer; no hidden HTTP calls.

  Fail-soft         Partial output is always  Every stage is wrapped in a
                    better than a hard crash. try/except in
                                              core/pipeline.py; errors go to
                                              ProcessingReport.

  License hygiene   GPL or non-commercial     Optional adapters; license
                    components must never     gate in registry.py;
                    become required defaults. documented in README.
  --------------------------------------------------------------------------

3\. Design Principles

The architecture applies the following well-established software design
principles. Each entry explains what the principle means in the context
of speechtelemetry specifically.

3.1 Separation of Concerns (SoC)

The single most important structural principle. Each directory and
module has exactly one responsibility:

  -------------------------------------------------------------------------
  **Layer / Module**      **One and only responsibility**
  ----------------------- -------------------------------------------------
  api.py                  Public entry point. Validates inputs, builds
                          config, delegates to pipeline. Nothing else.

  config.py               Declare and validate all user-tunable settings.
  (PipelineConfig)        No business logic.

  types.py                Define the canonical data model
                          (TranscriptDocument, Segment, Word, ...). Zero
                          processing logic.

  registry.py             Map backend name strings to Python classes.
                          Resolve at runtime. No pipeline logic.

  core/pipeline.py        Orchestrate stages in order. Own fail-soft
                          try/except. No backend logic.

  core/job.py             Temp-file lifecycle, job ID, cleanup. No domain
                          logic.

  core/provenance.py      Track which backend produced which output field.
                          No pipeline logic.

  io/ffmpeg.py            Decode any media to canonical mono 16 kHz WAV.
                          One function. No ML.

  io/audio_normalize.py   Post-FFmpeg validation and chunking. No decode
                          logic, no ML.

  backends/\*/            One backend per file. Implement the interface
                          contract for their stage. No orchestration.

  exporters/\*/           Serialise TranscriptDocument to one output format
                          each. No pipeline logic.

  cli/                    Parse argv, call api.py, print/write output. No
                          domain logic.

  benchmarking/           Fixtures, metrics, comparison harness. No
                          production logic.
  -------------------------------------------------------------------------

3.2 Interface Segregation & the Backend Contract

Every backend module for a given pipeline stage must implement exactly
the same public method signature as the default backend for that stage.
This is the Backend Contract. It means:

-   Consumers of a backend (core/pipeline.py) never import backend
    classes directly --- they receive them from registry.py.

-   Swapping from faster-whisper to whisper.cpp requires only a config
    change.

-   New backends can be added without touching any existing code outside
    registry.py.

  -----------------------------------------------------------------------
  **Stage**        **Required method signature**
  ---------------- ------------------------------------------------------
  VAD              get_speech_intervals(wav_path: str) →
                   list\[dict\[start,end\]\]

  ASR              transcribe(wav_path: str, language: str\|None,
                   beam_size: int) → tuple\[list\[Segment\], info\]

  Alignment        align(segments: list, audio_path: str, language: str)
                   → list\[dict\]

  Diarization      diarize(wav_path: str, min_speakers: int\|None,
                   max_speakers: int\|None) → list\[dict\]

  Prosody          extract_segment(wav_path: str, start_s: float, end_s:
                   float) → dict

  Emotion          predict_segment(wav_path: str, start_s: float, end_s:
                   float) → dict

  Exporter         export(doc: TranscriptDocument, output_path: str) →
                   None
  -----------------------------------------------------------------------

3.3 Dependency Inversion Principle (DIP)

High-level modules (core/pipeline.py) do not depend on low-level modules
(backends/asr/faster_whisper.py). Both depend on abstractions (the
Backend Contract / interface).

In practice: core/pipeline.py never contains a line like from
speechtelemetry.backends.asr.faster_whisper import FasterWhisperBackend.
It always goes through registry.py:

+-----------------------------------------------------------------------+
| \# core/pipeline.py --- correct pattern                               |
|                                                                       |
| from speechtelemetry.registry import get_backend                      |
|                                                                       |
| asr = get_backend(\"asr\", config.asr_backend) \# resolved at runtime |
|                                                                       |
| segments, info = asr.transcribe(wav_path, config.asr_language)        |
|                                                                       |
| \# registry.py --- the only place concrete classes are imported       |
|                                                                       |
| ASR_BACKENDS = {                                                      |
|                                                                       |
| \"faster-whisper\":                                                   |
| \"speechtelemetry.backends.asr.faster_whisper.FasterWhisperBackend\", |
|                                                                       |
| \"whisperx\":                                                         |
| \"speechtelemetry.backends.asr.whisperx.WhisperXASRBackend\",         |
|                                                                       |
| \"whisper.cpp\":                                                      |
| \"speechtelemetry.backends.asr.whisper_cpp.WhisperCppBackend\",       |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

3.4 Single Responsibility Principle (SRP)

Each module does one thing. The checklist is simple: if you can describe
a module\'s job using the word \"and\", it should be split. Examples:

-   io/ffmpeg.py: \"decode media to WAV\". No \"and\".

-   core/pipeline.py: \"orchestrate stages\". No backend logic inside.

-   types.py: \"define the data model\". No validation methods, no
    serialisation.

-   exporters/json.py: \"serialise to JSON\". No pipeline logic.

3.5 Open/Closed Principle (OCP)

The system is open for extension (new backends, new exporters, new
stages) and closed for modification (adding a new emotion backend does
not require editing core/pipeline.py or api.py).

Mechanism: registry.py is the only extension point. A contributor adds a
new backend file, adds one line to the appropriate dict in registry.py,
and the system picks it up automatically via PipelineConfig.

3.6 Explicit over Implicit

Every piece of information that affects output behaviour must be visible
in the result object. No silent fallbacks, no hidden behaviour, no model
internals exposed as certainty.

-   ProcessingReport records every stage timing, peak RAM/VRAM, and
    every error.

-   Every output field that comes from an ML backend stores backend_name
    provenance.

-   EmotionScore always stores the full probability distribution ---
    never just the top label.

-   Alignment failures mark alignment_backend=\"none\" rather than
    silently using original timestamps.

3.7 Fail-Soft by Default

Every stage call in core/pipeline.py is wrapped in a try/except.
Failures append a StageError to ProcessingReport and return None or an
empty value for that field. The only hard failures are environment
prerequisites (missing FFmpeg, missing HF_TOKEN) because those make the
whole job impossible.

+-----------------------------------------------------------------------+
| \# core/pipeline.py --- fail-soft pattern                             |
|                                                                       |
| try:                                                                  |
|                                                                       |
| prosody = prosody_backend.extract_segment(wav, seg.start, seg.end)    |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| report.errors.append(StageError(                                      |
|                                                                       |
| stage=\"prosody\", message=str(e),                                    |
|                                                                       |
| exception_type=type(e).\_\_name\_\_, segment_index=i                  |
|                                                                       |
| ))                                                                    |
|                                                                       |
| prosody = None                                                        |
+-----------------------------------------------------------------------+

4\. Component Architecture

The diagram below shows all layers, their modules, and the data flows
between them. Arrows represent function calls or data being passed; they
always flow downward (higher layer calls lower layer, never the
reverse).

+-----------------------------------------------------------------------+
| ┌─────────────────────────────────────────────────────────┐           |
|                                                                       |
| │ CONSUMER LAYER │                                                    |
|                                                                       |
| │ CLI (cli/main.py) Python import HTTP wrapper │                      |
|                                                                       |
| │ └──────────────────┴─────────────────┘ │                            |
|                                                                       |
| │ api.py │                                                            |
|                                                                       |
| │ enrich_media(input_path, config) │                                  |
|                                                                       |
| └────────────────────────┬────────────────────────────────┘           |
|                                                                       |
| │ TranscriptDocument                                                  |
|                                                                       |
| ┌────────────────────────▼────────────────────────────────┐           |
|                                                                       |
| │ ORCHESTRATION LAYER │                                               |
|                                                                       |
| │ core/pipeline.py core/job.py core/provenance.py │                   |
|                                                                       |
| │ ┌────────────────────────────────────────────────────┐ │            |
|                                                                       |
| │ │ 1.Decode → 2.Norm → 3.VAD → 4.ASR → │ │                           |
|                                                                       |
| │ │ 5.Align → 6.Diar → 7.Prosody → 8.Emotion → │ │                    |
|                                                                       |
| │ │ 9.Export │ │                                                      |
|                                                                       |
| │ └────────────────────────────────────────────────────┘ │            |
|                                                                       |
| │ registry.py (name → class resolver) │                               |
|                                                                       |
| └──────────┬────────────┬────────────┬────────────────────┘           |
|                                                                       |
| │ │ │                                                                 |
|                                                                       |
| ┌──────────▼──┐ ┌──────▼──────┐ ┌─▼──────────────────┐                |
|                                                                       |
| │ I/O LAYER │ │BACKEND LAYER│ │ EXPORT LAYER │                        |
|                                                                       |
| │ io/ffmpeg │ │ backends/ │ │ exporters/ │                            |
|                                                                       |
| │ io/norm │ │ asr/ │ │ json.py │                                      |
|                                                                       |
| └─────────────┘ │ alignment/ │ │ srt.py │                             |
|                                                                       |
| │ vad/ │ │ vtt.py │                                                   |
|                                                                       |
| ┌────────────┐ │ diarize/ │ │ textgrid.py │                           |
|                                                                       |
| │DATA MODEL │ │ prosody/ │ └────────────────────-┘                    |
|                                                                       |
| │ types.py │ │ emotion/ │                                             |
|                                                                       |
| │ config.py │ └─────────────┘                                         |
|                                                                       |
| └────────────┘                                                        |
+-----------------------------------------------------------------------+

4.1 Layer Responsibilities

  --------------------------------------------------------------------------
  **Layer**       **Modules**              **What it owns**
  --------------- ------------------------ ---------------------------------
  Consumer        cli/, any HTTP wrapper,  User-facing interface. Calls
                  Jupyter notebook         api.py. Zero domain logic.

  Public API      api.py, config.py        Entry point, input validation,
                                           config construction. Delegates
                                           immediately to pipeline.

  Orchestration   core/pipeline.py,        Stage ordering, fail-soft error
                  core/job.py,             handling, temp-file lifecycle,
                  core/provenance.py,      backend resolution, provenance
                  registry.py              tracking.

  I/O             io/ffmpeg.py,            All file system and media I/O.
                  io/audio_normalize.py    Converts any input to canonical
                                           WAV. No ML.

  Backend         backends/\*\*/\*.py      ML inference for each stage. One
                                           file per backend. Implement
                                           interface contract.

  Data Model      types.py                 Canonical typed output objects.
                                           Pure data. No processing logic
                                           anywhere in this file.

  Export          exporters/\*.py          Serialise TranscriptDocument to
                                           each output format. Lossless
                                           (JSON) and lossy (SRT/VTT).

  Benchmark       benchmarking/            Test fixtures, metrics
                                           calculation, backend comparison.
                                           Never imported by production
                                           code.
  --------------------------------------------------------------------------

5\. Canonical Data Model

All pipeline outputs are stored in a typed canonical object tree rooted
at TranscriptDocument. The model is defined entirely in types.py using
Pydantic v2. The golden rule: types.py is pure data. It has no methods
that call backends, modify audio, or perform I/O.

5.1 Object Hierarchy

+-----------------------------------------------------------------------+
| TranscriptDocument                                                    |
|                                                                       |
| ├─ source_path: str                                                   |
|                                                                       |
| ├─ language: str                                                      |
|                                                                       |
| ├─ duration_s: float                                                  |
|                                                                       |
| ├─ processing_report: ProcessingReport                                |
|                                                                       |
| │ ├─ stage_timings: dict\[str, float\]                                |
|                                                                       |
| │ ├─ peak_ram_mb: float                                               |
|                                                                       |
| │ ├─ peak_vram_mb: float                                              |
|                                                                       |
| │ ├─ real_time_factor: float                                          |
|                                                                       |
| │ └─ errors: list\[StageError\]                                       |
|                                                                       |
| ├─ silence_spans: list\[SilenceSpan\]                                 |
|                                                                       |
| │ └─ {start, end, duration_ms, reason}                                |
|                                                                       |
| └─ segments: list\[Segment\]                                          |
|                                                                       |
| ├─ start, end, text, speaker, confidence                              |
|                                                                       |
| ├─ silence_before_ms, silence_after_ms                                |
|                                                                       |
| ├─ prosody: ProsodyWindow                                             |
|                                                                       |
| │ └─ {f0_mean, f0_variance, energy_mean, energy_variance,             |
|                                                                       |
| │ speech_rate_sps, pause_density, voice_quality_hnr}                  |
|                                                                       |
| ├─ emotion: EmotionScore                                              |
|                                                                       |
| │ └─ {label_distribution: dict\[str,float\], valence,                 |
|                                                                       |
| │ arousal, confidence, backend_name}                                  |
|                                                                       |
| └─ words: list\[Word\]                                                |
|                                                                       |
| └─ {text, start, end, confidence, token_id, alignment_backend}        |
+-----------------------------------------------------------------------+

5.2 Data Model Rules

-   types.py imports nothing from any other speechtelemetry module. It
    is a leaf in the dependency graph.

-   Every field that comes from an ML backend carries backend_name
    provenance.

-   EmotionScore.label_distribution is always a full dict --- never
    collapsed to a single string.

-   Null / missing values use Optional\[T\] = None, not sentinel strings
    like \"unknown\".

-   ProcessingReport.errors is always present (may be empty list), never
    None.

-   SilenceSpan.reason uses a Literal type: \"speech_gap\" \|
    \"non_speech\" \| \"overlap\".

6\. Backend Registry Pattern

The registry is the single place where backend name strings are mapped
to Python classes. It is the only file in the entire codebase that
directly imports backend modules.

6.1 Registry Structure

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/registry.py                                    |
|                                                                       |
| import importlib                                                      |
|                                                                       |
| from typing import Any                                                |
|                                                                       |
| \# Maps stage -\> config_name -\> \"module.path.ClassName\"           |
|                                                                       |
| \# Lazy string paths --- imports only happen when get_backend() is    |
| called.                                                               |
|                                                                       |
| \# Add one line here to register a new backend. Nothing else changes. |
|                                                                       |
| REGISTRY: dict\[str, dict\[str, str\]\] = {                           |
|                                                                       |
| \"asr\": {                                                            |
|                                                                       |
| \"faster-whisper\":                                                   |
| \"speechtelemetry.backends.asr.faster_whisper.FasterWhisperBackend\", |
|                                                                       |
| \"whisperx\":                                                         |
| \"speechtelemetry.backends.asr.whisperx.WhisperXASRBackend\",         |
|                                                                       |
| \"whisper.cpp\":                                                      |
| \"speechtelemetry.backends.asr.whisper_cpp.WhisperCppBackend\",       |
|                                                                       |
| \"sensevoice\":                                                       |
| \"speechtelemetry.backends.asr.sensevoice.SenseVoiceBackend\",        |
|                                                                       |
| },                                                                    |
|                                                                       |
| \"vad\": {                                                            |
|                                                                       |
| \"silero\": \"speechtelemetry.backends.vad.silero.SileroVADBackend\", |
|                                                                       |
| \"funasr\": \"speechtelemetry.backends.vad.funasr.FunASRVADBackend\", |
|                                                                       |
| },                                                                    |
|                                                                       |
| \"alignment\": { \... },                                              |
|                                                                       |
| \"diarization\": { \... },                                            |
|                                                                       |
| \"prosody\": { \... },                                                |
|                                                                       |
| \"emotion\": { \... },                                                |
|                                                                       |
| }                                                                     |
|                                                                       |
| LICENSE_FLAGS: dict\[str, str\] = {                                   |
|                                                                       |
| \"speechtelemetry.backends.prosody.parselmouth.ParselmouthBackend\":  |
| \"GPL-3.0\",                                                          |
|                                                                       |
| }                                                                     |
|                                                                       |
| def get_backend(stage: str, name: str, \*\*kwargs) -\> Any:           |
|                                                                       |
| path = REGISTRY\[stage\]\[name\]                                      |
|                                                                       |
| module_path, class_name = path.rsplit(\".\", 1)                       |
|                                                                       |
| try:                                                                  |
|                                                                       |
| module = importlib.import_module(module_path)                         |
|                                                                       |
| except ImportError as e:                                              |
|                                                                       |
| raise ImportError(                                                    |
|                                                                       |
| f\"Backend \'{name}\' for stage \'{stage}\' requires extra            |
| dependencies.\\n\"                                                    |
|                                                                       |
| f\"Run: pip install {name}\\n\"                                       |
|                                                                       |
| f\"Original error: {e}\"                                              |
|                                                                       |
| ) from e                                                              |
|                                                                       |
| cls = getattr(module, class_name)                                     |
|                                                                       |
| return cls(\*\*kwargs)                                                |
+-----------------------------------------------------------------------+

6.2 Benefits of the Registry Pattern

  -----------------------------------------------------------------------
  **Benefit**        **Explanation**
  ------------------ ----------------------------------------------------
  Lazy imports       Backend dependencies (torch, whisperx, speechbrain)
                     are only imported when that backend is actually
                     used. A user who only needs ASR+export does not pay
                     the startup cost of importing SpeechBrain.

  Graceful           If a backend package is not installed, get_backend()
  unavailability     raises a clear ImportError with the pip install
                     command. No cryptic AttributeError deep in the
                     stack.

  License gate       LICENSE_FLAGS allows the pipeline to warn at runtime
                     when a GPL backend is configured, so developers are
                     never surprised.

  Testability        Tests can inject a mock backend by calling
                     get_backend() with a mock class registered in the
                     test fixture --- no monkeypatching of imports
                     required.

  One-line extension Adding a new backend to the project requires adding
                     one line to REGISTRY and zero changes to any other
                     production file.
  -----------------------------------------------------------------------

7\. Pipeline Orchestration

core/pipeline.py is the only module that knows the stage ordering. It
owns all fail-soft logic. It owns nothing else.

7.1 Stage Execution Flow

+-----------------------------------------------------------------------+
| \# core/pipeline.py (simplified pseudo-code)                          |
|                                                                       |
| def run(config: PipelineConfig) -\> TranscriptDocument:               |
|                                                                       |
| job = Job() \# creates temp dir, job ID                               |
|                                                                       |
| report = ProcessingReport()                                           |
|                                                                       |
| \# ── Stage 1: Decode ──────────────────────                          |
|                                                                       |
| t0 = perf_counter()                                                   |
|                                                                       |
| wav_path = ffmpeg.normalize_to_wav(job.input_path, job.temp_wav)      |
|                                                                       |
| report.stage_timings\[\"decode\"\] = perf_counter() - t0              |
|                                                                       |
| \# ── Stage 3: VAD ─────────────────────────                          |
|                                                                       |
| vad = get_backend(\"vad\", config.vad_backend)                        |
|                                                                       |
| try:                                                                  |
|                                                                       |
| t0 = perf_counter()                                                   |
|                                                                       |
| speech_intervals = vad.get_speech_intervals(wav_path)                 |
|                                                                       |
| report.stage_timings\[\"vad\"\] = perf_counter() - t0                 |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| report.errors.append(StageError(\"vad\", str(e),                      |
| type(e).\_\_name\_\_))                                                |
|                                                                       |
| speech_intervals = \[{\"start\": 0, \"end\": audio_duration}\]        |
|                                                                       |
| \# ── Stages 4-8: ASR, Align, Diar, Prosody, Emotion ──               |
|                                                                       |
| \# (same pattern: get_backend → try/except → record timing/error)     |
|                                                                       |
| \# ── Stage 9: Export ──────────────────────                          |
|                                                                       |
| for fmt in config.export_formats:                                     |
|                                                                       |
| exporter = get_backend(\"exporter\", fmt)                             |
|                                                                       |
| exporter.export(doc, job.output_path(fmt))                            |
|                                                                       |
| job.cleanup() \# delete temp files                                    |
|                                                                       |
| return doc                                                            |
+-----------------------------------------------------------------------+

7.2 Hard Fails vs. Soft Fails

  -----------------------------------------------------------------------
  **Situation**            **Policy**
  ------------------------ ----------------------------------------------
  FFmpeg not on PATH       Hard fail: EnvironmentError at job start.
                           Pipeline cannot proceed without audio decode.

  HF_TOKEN missing when    Hard fail: EnvironmentError at job start. Fail
  diarization is enabled   early, fail clearly.

  Model download fails     Hard fail: BackendError with URL and expected
                           cache path.

  VAD returns no speech    Soft fail: pass full audio to ASR, log
                           StageError, continue.

  ASR returns empty        Soft fail: return TranscriptDocument with
  transcript               empty segments list.

  Alignment fails for a    Soft fail: keep original timestamps, mark
  segment                  alignment_backend=\"none\".

  Diarization fails        Soft fail: return transcript without speaker
  entirely                 labels (all None).

  Prosody fails for a      Soft fail: set segment.prosody = None, log
  segment                  StageError.

  Emotion fails for a      Soft fail: set segment.emotion = None, log
  segment                  StageError.

  Prosody segment \< 40 ms Soft fail: skip, log warning, return None
                           prosody for that segment.
  -----------------------------------------------------------------------

8\. Public API Design

The public API surface is intentionally minimal. Only three symbols need
to be in a user\'s import statement to do everything.

8.1 The Three Public Symbols

  -----------------------------------------------------------------------
  **Symbol**                      **Purpose**
  ------------------------------- ---------------------------------------
  enrich_media(input_path,        One-shot high-level API. Runs the full
  config) → TranscriptDocument    pipeline and returns the canonical
                                  result object.

  PipelineConfig                  Pydantic model. All user-tunable
                                  settings with sensible defaults. Import
                                  and configure; nothing else required.

  TranscriptDocument (+ subtypes) The canonical result. Import to
                                  type-hint consumer code. All subtypes
                                  (Segment, Word, etc.) exported from the
                                  top-level package.
  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| \# Everything a normal user needs --- three symbols, one import       |
|                                                                       |
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| result = enrich_media(                                                |
|                                                                       |
| input_path=\"interview.mp4\",                                         |
|                                                                       |
| config=PipelineConfig(                                                |
|                                                                       |
| asr_backend=\"faster-whisper\",                                       |
|                                                                       |
| vad_backend=\"silero\",                                               |
|                                                                       |
| prosody_backend=\[\"parselmouth\"\],                                  |
|                                                                       |
| emotion_backend=\"speechbrain\",                                      |
|                                                                       |
| device=\"auto\",                                                      |
|                                                                       |
| ),                                                                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# result is a fully-typed TranscriptDocument                         |
|                                                                       |
| for seg in result.segments:                                           |
|                                                                       |
| print(seg.start, seg.text, seg.emotion.top_label)                     |
+-----------------------------------------------------------------------+

8.2 Stage-Level APIs

For users who need to run individual pipeline stages (e.g. only VAD, or
only prosody on pre-existing timestamps), each backend can be
instantiated and called directly. The stage APIs are the same method
contracts as described in Section 3.2.

+-----------------------------------------------------------------------+
| \# Advanced use: call individual stages                               |
|                                                                       |
| from speechtelemetry.registry import get_backend                      |
|                                                                       |
| vad = get_backend(\"vad\", \"silero\")                                |
|                                                                       |
| speech_intervals = vad.get_speech_intervals(\"audio.wav\")            |
|                                                                       |
| prosody = get_backend(\"prosody\", \"parselmouth\")                   |
|                                                                       |
| features = prosody.extract_segment(\"audio.wav\", start_s=1.2,        |
| end_s=4.8)                                                            |
+-----------------------------------------------------------------------+

8.3 API Stability Policy

-   api.py, config.py, and types.py form the stable public surface. They
    follow semantic versioning.

-   registry.py is semi-public: its REGISTRY dict is stable;
    get_backend() is stable; internal structure may change in minor
    versions.

-   Everything in core/, io/, backends/, exporters/ is internal.
    Breaking changes to internal modules do not require a major version
    bump.

-   New PipelineConfig fields are always additive (new optional field
    with a default). Removing or renaming a field is a major-version
    change.

9\. Making It Easy for Third Parties

This is one of the most important sections. A library that is correct
but hard to use will not be used. The following practices are the
concrete mechanisms that make speechtelemetry easy to integrate.

9.1 One-Import Entry Point

The entire library is usable with a single import line.
src/speechtelemetry/\_\_init\_\_.py exports exactly what a consumer
needs:

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/\_\_init\_\_.py                                |
|                                                                       |
| from .api import enrich_media                                         |
|                                                                       |
| from .config import PipelineConfig                                    |
|                                                                       |
| from .types import (                                                  |
|                                                                       |
| TranscriptDocument, Segment, Word,                                    |
|                                                                       |
| SilenceSpan, ProsodyWindow, EmotionScore,                             |
|                                                                       |
| ProcessingReport, StageError,                                         |
|                                                                       |
| )                                                                     |
|                                                                       |
| \_\_version\_\_ = \"0.1.0\"                                           |
|                                                                       |
| \_\_all\_\_ = \[                                                      |
|                                                                       |
| \"enrich_media\", \"PipelineConfig\",                                 |
|                                                                       |
| \"TranscriptDocument\", \"Segment\", \"Word\",                        |
|                                                                       |
| \"SilenceSpan\", \"ProsodyWindow\", \"EmotionScore\",                 |
|                                                                       |
| \"ProcessingReport\", \"StageError\",                                 |
|                                                                       |
| \]                                                                    |
+-----------------------------------------------------------------------+

9.2 Sensible Defaults for Every Setting

PipelineConfig has defaults for every field. A new user can call
enrich_media(\"file.mp4\", config=PipelineConfig()) and get a useful
result. Defaults are chosen to be correct on any machine (CPU-safe, no
token required, permissive licenses only):

  -------------------------------------------------------------------------
  **Config field**      **Default and rationale**
  --------------------- ---------------------------------------------------
  device                \"auto\" --- CUDA if available, otherwise CPU.
                        Never forces a choice.

  asr_backend           \"faster-whisper\" --- MIT license, best
                        speed/quality, CPU+GPU.

  vad_backend           \"silero\" --- MIT license, very fast, no token
                        needed.

  alignment_backend     \"whisperx\" --- best word-level timestamps.

  diarization_backend   None --- disabled by default. Requires HF_TOKEN;
                        must be opt-in.

  prosody_backend       \[\"parselmouth\"\] --- GPL adapter, but opt-in
                        default for best quality.

  emotion_backend       \"speechbrain\" --- Apache 2.0, auto-downloads, no
                        token needed.

  export_formats        \[\"json\"\] --- JSON is the authoritative lossless
                        format.

  chunk_audio           True --- always safer for memory; no reason to
                        disable.
  -------------------------------------------------------------------------

9.3 Clear, Actionable Error Messages

Every EnvironmentError and ImportError raised by speechtelemetry must
include the exact command needed to fix the problem. No generic
\"dependency missing\" messages.

+-----------------------------------------------------------------------+
| \# Example: FFmpeg missing                                            |
|                                                                       |
| raise EnvironmentError(                                               |
|                                                                       |
| \"FFmpeg not found on PATH.\\n\"                                      |
|                                                                       |
| \"Install it with: winget install \--id=Gyan.FFmpeg -e\\n\"           |
|                                                                       |
| \"Then restart your terminal and verify with: ffmpeg -version\"       |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Example: backend not installed                                     |
|                                                                       |
| raise ImportError(                                                    |
|                                                                       |
| \"Backend \'pyannote\' requires pyannote.audio.\\n\"                  |
|                                                                       |
| \"Install with: pip install pyannote.audio\\n\"                       |
|                                                                       |
| \"Also required: set HF_TOKEN env var and accept model license        |
| at\\n\"                                                               |
|                                                                       |
| \" https://hf.co/pyannote/speaker-diarization-community-1\"           |
|                                                                       |
| )                                                                     |
+-----------------------------------------------------------------------+

9.4 Pre-flight Environment Check

core/pipeline.py runs a pre-flight check at the start of every job. This
surfaces all problems immediately, before any compute happens, rather
than failing 20 minutes into processing a large file:

+-----------------------------------------------------------------------+
| \# core/pipeline.py                                                   |
|                                                                       |
| def preflight_check(config: PipelineConfig) -\> None:                 |
|                                                                       |
| \# 1. FFmpeg (always required)                                        |
|                                                                       |
| if not shutil.which(\"ffmpeg\"):                                      |
|                                                                       |
| raise EnvironmentError(\"FFmpeg not found on PATH. \...\")            |
|                                                                       |
| \# 2. HF_TOKEN (only if diarization enabled)                          |
|                                                                       |
| if config.diarization_backend and not os.environ.get(\"HF_TOKEN\"):   |
|                                                                       |
| raise EnvironmentError(\"HF_TOKEN not set. \...\")                    |
|                                                                       |
| \# 3. CUDA (if explicitly requested)                                  |
|                                                                       |
| if config.device == \"cuda\":                                         |
|                                                                       |
| import torch                                                          |
|                                                                       |
| if not torch.cuda.is_available():                                     |
|                                                                       |
| raise EnvironmentError(\"device=\'cuda\' requested but CUDA is not    |
| available.\")                                                         |
|                                                                       |
| \# 4. Check all requested backends are importable                     |
|                                                                       |
| for stage, name in config.iter_backends():                            |
|                                                                       |
| try:                                                                  |
|                                                                       |
| get_backend(stage, name) \# will raise ImportError if not installed   |
|                                                                       |
| except ImportError as e:                                              |
|                                                                       |
| raise EnvironmentError(f\"Backend \'{name}\' is not                   |
| available.\\n{e}\") from e                                            |
+-----------------------------------------------------------------------+

9.5 Typed Results

TranscriptDocument and all subtypes are Pydantic v2 models. This means:

-   Full IDE autocomplete on result objects --- consumers see every
    available field.

-   Runtime type validation --- passing a wrong type to PipelineConfig
    raises a clear ValidationError immediately.

-   Free JSON serialisation: result.model_dump_json() produces the
    canonical JSON export without any extra exporter call.

-   Easy schema introspection: TranscriptDocument.model_json_schema()
    produces a JSON Schema document that API wrappers and documentation
    generators can consume.

9.6 Minimal Installation Paths

pyproject.toml defines layered extras so users install only what they
need:

+-----------------------------------------------------------------------+
| \# pyproject.toml                                                     |
|                                                                       |
| \[project.optional-dependencies\]                                     |
|                                                                       |
| core = \[\"faster-whisper\", \"silero-vad\", \"speechbrain\",         |
| \"pydantic\>=2\", \"soundfile\"\]                                     |
|                                                                       |
| align = \[\"whisperx\"\]                                              |
|                                                                       |
| diar = \[\"pyannote.audio\"\]                                         |
|                                                                       |
| prosody = \[\"praat-parselmouth\", \"copasul\"\]                      |
|                                                                       |
| gpu = \[\"torch\", \"torchaudio\"\] \# user installs torch with CUDA  |
| themselves                                                            |
|                                                                       |
| all = \[\"speechtelemetry\[core,align,diar,prosody\]\"\]              |
|                                                                       |
| \# Usage:                                                             |
|                                                                       |
| \# pip install speechtelemetry\[core\] \# ASR + VAD + emotion only    |
|                                                                       |
| \# pip install speechtelemetry\[core,align\] \# + word timestamps     |
|                                                                       |
| \# pip install speechtelemetry\[all\] \# everything                   |
+-----------------------------------------------------------------------+

9.7 Examples Repository

The examples/ directory provides runnable scripts for the most common
integration scenarios. These are part of the public contract --- they
are tested in CI and must not be broken by library changes.

  -----------------------------------------------------------------------------
  **File**                       **What it demonstrates**
  ------------------------------ ----------------------------------------------
  examples/basic_transcribe.py   Minimal usage: enrich_media() with default
                                 config.

  examples/with_diarization.py   Adding speaker diarization; setting HF_TOKEN.

  examples/cpu_only.py           Forcing CPU mode; using int8 quantization.

  examples/custom_backend.py     Registering a custom backend at runtime
                                 without editing registry.py.

  examples/export_srt.py         Exporting to SRT subtitle format.

  examples/stage_by_stage.py     Using individual stage backends directly.

  examples/read_result.py        Navigating the TranscriptDocument object tree.
  -----------------------------------------------------------------------------

10\. Dependency & License Architecture

License hygiene is part of the architecture, not just the documentation.
The following rules are enforced by the code structure itself:

  -----------------------------------------------------------------------
  **Rule**              **Enforcement mechanism**
  --------------------- -------------------------------------------------
  GPL backends are      parselmouth and any other GPL packages are only
  never required        imported inside their backend adapter files. They
  defaults              are not listed under \[project.dependencies\] in
                        pyproject.toml --- only under
                        \[optional-dependencies\].

  Non-commercial        Any non-commercial or research-only component is
  components never ship excluded from core extras. Registry entry
  as required           includes a comment flagging the license. CLI
                        warns at job start if such a backend is enabled.

  Core library has no   api.py, config.py, types.py, registry.py import
  ML-framework          no torch, no tensorflow. ML imports only happen
  dependency at import  inside backend modules, which are lazy-loaded.
  time

  FFmpeg LGPL build     Documentation specifies winget install
  recommended           Gyan.FFmpeg which defaults to the LGPL build. A
                        separate install guide documents the GPL vs LGPL
                        distinction.

  HF token never in     Any backend requiring HuggingFace authentication
  source                reads from os.environ. Hardcoded tokens raise a
                        linting error via a pre-commit hook.
  -----------------------------------------------------------------------

11\. Testing Architecture

11.1 Test Layout

+-----------------------------------------------------------------------+
| tests/                                                                |
|                                                                       |
| unit/                                                                 |
|                                                                       |
| test_types.py \# Pydantic model validation                            |
|                                                                       |
| test_registry.py \# get_backend() resolution, ImportError messages    |
|                                                                       |
| test_pipeline_routing.py \# Stage ordering, fail-soft logic (backends |
| mocked)                                                               |
|                                                                       |
| test_ffmpeg.py \# normalize_to_wav() with a real 5-second WAV fixture |
|                                                                       |
| test_exporters.py \# JSON/SRT/VTT/TextGrid output from a fixture doc  |
|                                                                       |
| integration/                                                          |
|                                                                       |
| test_full_pipeline.py \# enrich_media() on a 30s audio fixture,       |
| CPU-only                                                              |
|                                                                       |
| test_vad_asr.py \# VAD + ASR only, no alignment                       |
|                                                                       |
| test_with_diarization.py \# Skipped if HF_TOKEN not set               |
|                                                                       |
| benchmarks/                                                           |
|                                                                       |
| bench_rtf.py \# Real-time factor measurement                          |
|                                                                       |
| bench_memory.py \# Peak RAM/VRAM measurement                          |
|                                                                       |
| fixtures/                                                             |
|                                                                       |
| canonical_5s.wav \# 5-second clean speech WAV (committed)             |
|                                                                       |
| canonical_30s.wav \# 30-second multi-speaker WAV (committed)          |
|                                                                       |
| golden_output.json \# Expected TranscriptDocument for 5s fixture      |
+-----------------------------------------------------------------------+

11.2 Testing Principles

-   Unit tests mock all backends. They test the orchestration logic, not
    the ML models.

-   Integration tests run real backends. They are skipped in CI if GPU
    or HF_TOKEN is unavailable.

-   Golden output tests: the 5s fixture produces a committed
    golden_output.json. If the output changes, the test fails and the
    developer must explicitly update the golden file, preventing silent
    regressions.

-   No GPU left allocated: every integration test checks that VRAM is
    freed after teardown.

-   Pre-commit hooks: black, ruff, and a custom hook that rejects
    hardcoded HF tokens.

12\. Full Package Layout

+-----------------------------------------------------------------------+
| speechtelemetry/ ← repo root                                          |
|                                                                       |
| ├── pyproject.toml ← build config, extras, metadata                   |
|                                                                       |
| ├── README.md ← quickstart, badge, install instructions               |
|                                                                       |
| ├── CHANGELOG.md                                                      |
|                                                                       |
| ├── LICENSE ← MIT (library code)                                      |
|                                                                       |
| ├── docs/                                                             |
|                                                                       |
| │ ├── architecture.md ← this document (Markdown mirror)               |
|                                                                       |
| │ ├── backends.md ← per-backend detail (from dev reference)           |
|                                                                       |
| │ ├── windows_setup.md ← Windows 11 PowerShell setup guide            |
|                                                                       |
| │ └── license_policy.md                                               |
|                                                                       |
| ├── examples/                                                         |
|                                                                       |
| │ ├── basic_transcribe.py                                             |
|                                                                       |
| │ ├── with_diarization.py                                             |
|                                                                       |
| │ ├── cpu_only.py                                                     |
|                                                                       |
| │ ├── custom_backend.py                                               |
|                                                                       |
| │ ├── export_srt.py                                                   |
|                                                                       |
| │ ├── stage_by_stage.py                                               |
|                                                                       |
| │ └── read_result.py                                                  |
|                                                                       |
| ├── src/                                                              |
|                                                                       |
| │ └── speechtelemetry/                                                |
|                                                                       |
| │ ├── \_\_init\_\_.py ← exports: enrich_media, PipelineConfig, types  |
|                                                                       |
| │ ├── api.py ← enrich_media() entry point                             |
|                                                                       |
| │ ├── config.py ← PipelineConfig (Pydantic model)                     |
|                                                                       |
| │ ├── types.py ← ALL canonical dataclasses / Pydantic models          |
|                                                                       |
| │ ├── registry.py ← backend name → class resolver                     |
|                                                                       |
| │ ├── core/                                                           |
|                                                                       |
| │ │ ├── pipeline.py ← stage orchestration + fail-soft logic           |
|                                                                       |
| │ │ ├── job.py ← job lifecycle + temp file management                 |
|                                                                       |
| │ │ └── provenance.py ← backend provenance tracking                   |
|                                                                       |
| │ ├── io/                                                             |
|                                                                       |
| │ │ ├── ffmpeg.py ← normalize_to_wav() via subprocess                 |
|                                                                       |
| │ │ └── audio_normalize.py ← post-FFmpeg validation + chunking        |
|                                                                       |
| │ ├── backends/                                                       |
|                                                                       |
| │ │ ├── asr/                                                          |
|                                                                       |
| │ │ │ ├── faster_whisper.py ← DEFAULT                                 |
|                                                                       |
| │ │ │ ├── whisperx.py ← fallback / combined mode                      |
|                                                                       |
| │ │ │ ├── whisper_cpp.py                                              |
|                                                                       |
| │ │ │ └── sensevoice.py                                               |
|                                                                       |
| │ │ ├── alignment/                                                    |
|                                                                       |
| │ │ │ ├── whisperx.py ← DEFAULT                                       |
|                                                                       |
| │ │ │ └── forcealign.py                                               |
|                                                                       |
| │ │ ├── vad/                                                          |
|                                                                       |
| │ │ │ ├── silero.py ← DEFAULT                                         |
|                                                                       |
| │ │ │ └── funasr.py                                                   |
|                                                                       |
| │ │ ├── diarization/                                                  |
|                                                                       |
| │ │ │ ├── pyannote.py ← DEFAULT (optional)                            |
|                                                                       |
| │ │ │ └── funasr.py                                                   |
|                                                                       |
| │ │ ├── prosody/                                                      |
|                                                                       |
| │ │ │ ├── parselmouth.py ← DEFAULT (GPL adapter)                      |
|                                                                       |
| │ │ │ ├── copasul.py                                                  |
|                                                                       |
| │ │ │ ├── librosa.py                                                  |
|                                                                       |
| │ │ │ └── audioflux.py                                                |
|                                                                       |
| │ │ └── emotion/                                                      |
|                                                                       |
| │ │ ├── speechbrain.py ← DEFAULT                                      |
|                                                                       |
| │ │ ├── emotion2vec.py                                                |
|                                                                       |
| │ │ └── emobox.py                                                     |
|                                                                       |
| │ ├── exporters/                                                      |
|                                                                       |
| │ │ ├── json.py ← authoritative lossless export                       |
|                                                                       |
| │ │ ├── srt.py                                                        |
|                                                                       |
| │ │ ├── vtt.py                                                        |
|                                                                       |
| │ │ └── textgrid.py                                                   |
|                                                                       |
| │ ├── benchmarking/                                                   |
|                                                                       |
| │ │ ├── fixtures.py                                                   |
|                                                                       |
| │ │ ├── metrics.py                                                    |
|                                                                       |
| │ │ └── compare.py                                                    |
|                                                                       |
| │ └── cli/                                                            |
|                                                                       |
| │ ├── main.py                                                         |
|                                                                       |
| │ └── commands.py                                                     |
|                                                                       |
| └── tests/                                                            |
|                                                                       |
| ├── unit/                                                             |
|                                                                       |
| ├── integration/                                                      |
|                                                                       |
| ├── benchmarks/                                                       |
|                                                                       |
| └── fixtures/                                                         |
|                                                                       |
| ├── canonical_5s.wav                                                  |
|                                                                       |
| ├── canonical_30s.wav                                                 |
|                                                                       |
| └── golden_output.json                                                |
+-----------------------------------------------------------------------+

13\. Windows 11 Setup --- Full PowerShell Instructions

This section provides every command needed to go from a clean Windows 11
machine to a fully operational speechtelemetry development environment.
Run all commands in a PowerShell terminal opened as Administrator where
noted.

13.1 Prerequisites Check

Open PowerShell (not PowerShell ISE, not CMD). Check what is already
installed:

+-----------------------------------------------------------------------+
| \# Check Python                                                       |
|                                                                       |
| python \--version                                                     |
|                                                                       |
| \# Expected: Python 3.10.x or 3.11.x                                  |
|                                                                       |
| \# Check pip                                                          |
|                                                                       |
| pip \--version                                                        |
|                                                                       |
| \# Check Git                                                          |
|                                                                       |
| git \--version                                                        |
|                                                                       |
| \# Check FFmpeg                                                       |
|                                                                       |
| ffmpeg -version                                                       |
|                                                                       |
| \# Check CUDA (optional, skip if CPU-only)                            |
|                                                                       |
| nvidia-smi                                                            |
+-----------------------------------------------------------------------+

13.2 Install Python 3.11 (if needed)

Python 3.11 is the recommended version. Use winget (built into Windows
11):

+-----------------------------------------------------------------------+
| \# Open PowerShell as Administrator                                   |
|                                                                       |
| winget install \--id Python.Python.3.11 -e \--source winget           |
|                                                                       |
| \# After install, restart PowerShell, then verify:                    |
|                                                                       |
| python \--version                                                     |
|                                                                       |
| \# Expected: Python 3.11.x                                            |
|                                                                       |
| \# Upgrade pip                                                        |
|                                                                       |
| python -m pip install \--upgrade pip                                  |
+-----------------------------------------------------------------------+

13.3 Install FFmpeg

FFmpeg is mandatory. The Gyan.dev build via winget defaults to the LGPL
variant (recommended for license hygiene):

+-----------------------------------------------------------------------+
| \# Install FFmpeg via winget (LGPL build)                             |
|                                                                       |
| winget install \--id=Gyan.FFmpeg -e                                   |
|                                                                       |
| \# winget adds FFmpeg to PATH automatically.                          |
|                                                                       |
| \# Restart PowerShell, then verify:                                   |
|                                                                       |
| ffmpeg -version                                                       |
|                                                                       |
| \# Expected: ffmpeg version 7.x.x built with gcc \...                 |
|                                                                       |
| \# If winget is unavailable, manual install:                          |
|                                                                       |
| \# 1. Download from                                                   |
| https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip      |
|                                                                       |
| \# 2. Extract to C:\\ffmpeg\\                                         |
|                                                                       |
| \# 3. Add C:\\ffmpeg\\bin to system PATH:                             |
|                                                                       |
| \[System.Environment\]::SetEnvironmentVariable(                       |
|                                                                       |
| \"PATH\",                                                             |
|                                                                       |
| \$env:PATH + \";C:\\ffmpeg\\bin\",                                    |
|                                                                       |
| \[System.EnvironmentVariableTarget\]::Machine                         |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# 4. Restart PowerShell and verify: ffmpeg -version                  |
+-----------------------------------------------------------------------+

13.4 Install Git

+-----------------------------------------------------------------------+
| winget install \--id Git.Git -e \--source winget                      |
|                                                                       |
| \# Restart PowerShell after install                                   |
|                                                                       |
| git \--version                                                        |
+-----------------------------------------------------------------------+

13.5 Clone the Repository

+-----------------------------------------------------------------------+
| \# Navigate to your projects folder                                   |
|                                                                       |
| cd C:\\Users\\\$env:USERNAME\\Projects \# or wherever you prefer      |
|                                                                       |
| \# Clone                                                              |
|                                                                       |
| git clone https://github.com/your-org/speechtelemetry.git             |
|                                                                       |
| cd speechtelemetry                                                    |
+-----------------------------------------------------------------------+

13.6 Create and Activate Virtual Environment

Always use a virtual environment. Never install speechtelemetry into the
system Python.

+-----------------------------------------------------------------------+
| \# Inside the speechtelemetry repo directory:                         |
|                                                                       |
| python -m venv .venv                                                  |
|                                                                       |
| \# Activate (PowerShell)                                              |
|                                                                       |
| .venv\\Scripts\\Activate.ps1                                          |
|                                                                       |
| \# If you get an execution policy error, run this first (once per     |
| machine):                                                             |
|                                                                       |
| Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  |
|                                                                       |
| \# Then re-run the activate command above.                            |
|                                                                       |
| \# Your prompt should now show (.venv) at the start.                  |
|                                                                       |
| \# Verify you are in the venv:                                        |
|                                                                       |
| where python                                                          |
|                                                                       |
| \# Expected: C:\\\...\\speechtelemetry\\.venv\\Scripts\\python.exe    |
|                                                                       |
| \# Upgrade pip inside the venv                                        |
|                                                                       |
| python -m pip install \--upgrade pip                                  |
+-----------------------------------------------------------------------+

13.7 Install PyTorch

PyTorch must be installed before any ML backends. Choose the correct
variant:

+-----------------------------------------------------------------------+
| \# ── Option A: CPU-only (always works, no GPU needed) ────────────   |
|                                                                       |
| pip install torch torchaudio \--index-url                             |
| https://download.pytorch.org/whl/cpu                                  |
|                                                                       |
| \# ── Option B: CUDA 12.4 (requires NVIDIA GPU + CUDA Toolkit 12.x)   |
|                                                                       |
| pip install torch torchaudio \--index-url                             |
| https://download.pytorch.org/whl/cu124                                |
|                                                                       |
| \# Verify PyTorch install:                                            |
|                                                                       |
| python -c \"import torch; print(torch.\_\_version\_\_)\"              |
|                                                                       |
| \# Verify CUDA (Option B only):                                       |
|                                                                       |
| python -c \"import torch; print(torch.cuda.is_available(),            |
| torch.cuda.get_device_name(0))\"                                      |
+-----------------------------------------------------------------------+

13.8 Install speechtelemetry Core Dependencies

+-----------------------------------------------------------------------+
| \# Install the library in editable mode with core extras              |
|                                                                       |
| pip install -e \".\[core\]\"                                          |
|                                                                       |
| \# Or install extras individually:                                    |
|                                                                       |
| \# Core: ASR + VAD + Emotion (no alignment, no diarization)           |
|                                                                       |
| pip install faster-whisper silero-vad speechbrain pydantic\>=2        |
| soundfile ffmpeg-normalize                                            |
|                                                                       |
| \# Add word-level alignment:                                          |
|                                                                       |
| pip install whisperx                                                  |
|                                                                       |
| \# Add prosody extraction (GPL --- optional, read license notes):     |
|                                                                       |
| pip install praat-parselmouth copasul                                 |
|                                                                       |
| \# Add diarization (requires HuggingFace token --- see 13.9):         |
|                                                                       |
| pip install pyannote.audio                                            |
|                                                                       |
| \# Verify core install:                                               |
|                                                                       |
| python -c \"from speechtelemetry import enrich_media, PipelineConfig; |
| print(\'OK\')\"                                                       |
+-----------------------------------------------------------------------+

13.9 HuggingFace Token Setup (for Diarization)

Required only if diarization_backend=\"pyannote\" is enabled.

+-----------------------------------------------------------------------+
| \# Step 1: Create a HuggingFace account at https://hf.co              |
|                                                                       |
| \# Step 2: Accept the pyannote model license at:                      |
|                                                                       |
| \# https://hf.co/pyannote/speaker-diarization-community-1             |
|                                                                       |
| \# Step 3: Generate a read token at:                                  |
|                                                                       |
| \# https://hf.co/settings/tokens                                      |
|                                                                       |
| \# Step 4: Set HF_TOKEN as a persistent environment variable in       |
| Windows:                                                              |
|                                                                       |
| \[System.Environment\]::SetEnvironmentVariable(                       |
|                                                                       |
| \"HF_TOKEN\",                                                         |
|                                                                       |
| \"hf_YOUR_TOKEN_HERE\",                                               |
|                                                                       |
| \[System.EnvironmentVariableTarget\]::User                            |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Step 5: Restart PowerShell, then verify:                           |
|                                                                       |
| \$env:HF_TOKEN                                                        |
|                                                                       |
| \# Expected: hf\_\...                                                 |
|                                                                       |
| \# Never put your token in source code.                               |
|                                                                       |
| \# Verify diarization backend is importable:                          |
|                                                                       |
| python -c \"from pyannote.audio import Pipeline; print(\'pyannote     |
| OK\')\"                                                               |
+-----------------------------------------------------------------------+

13.10 (Optional) NVIDIA CUDA Toolkit

Required only for GPU-accelerated inference. Skip if CPU-only.

+-----------------------------------------------------------------------+
| \# Download CUDA Toolkit 12.x from:                                   |
|                                                                       |
| \# https://developer.nvidia.com/cuda-downloads                        |
|                                                                       |
| \# Choose: Windows \> x86_64 \> 11 \> exe (local)                     |
|                                                                       |
| \# Run the installer with default settings.                           |
|                                                                       |
| \# After install, add CUDA bin to PATH if not auto-added:             |
|                                                                       |
| \$cudaPath = \"C:\\Program Files\\NVIDIA GPU Computing                |
| Toolkit\\CUDA\\v12.4\\bin\"                                           |
|                                                                       |
| \[System.Environment\]::SetEnvironmentVariable(                       |
|                                                                       |
| \"PATH\",                                                             |
|                                                                       |
| \                                                                     |
| [System.Environment\]::GetEnvironmentVariable(\"PATH\",\"Machine\") + |
| \";\$cudaPath\",                                                      |
|                                                                       |
| \[System.EnvironmentVariableTarget\]::Machine                         |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Restart PowerShell, then verify:                                   |
|                                                                       |
| nvcc \--version                                                       |
|                                                                       |
| \# Expected: Cuda compilation tools, release 12.4                     |
+-----------------------------------------------------------------------+

13.11 Verify Full Installation

+-----------------------------------------------------------------------+
| \# Run the built-in preflight check:                                  |
|                                                                       |
| python -c \"                                                          |
|                                                                       |
| from speechtelemetry.core.pipeline import preflight_check             |
|                                                                       |
| from speechtelemetry import PipelineConfig                            |
|                                                                       |
| preflight_check(PipelineConfig())                                     |
|                                                                       |
| print(\'All checks passed.\')                                         |
|                                                                       |
| \"                                                                    |
|                                                                       |
| \# Run unit tests (no GPU needed):                                    |
|                                                                       |
| pip install pytest                                                    |
|                                                                       |
| pytest tests/unit/ -v                                                 |
|                                                                       |
| \# Run a quick end-to-end transcription on the built-in fixture:      |
|                                                                       |
| python examples/basic_transcribe.py                                   |
|                                                                       |
| \# Expected output: a TranscriptDocument printed as JSON.             |
+-----------------------------------------------------------------------+

13.12 Disable pyannote Telemetry (Optional)

+-----------------------------------------------------------------------+
| \# pyannote.audio sends anonymous usage metrics by default.           |
|                                                                       |
| \# To disable, set this environment variable permanently:             |
|                                                                       |
| \[System.Environment\]::SetEnvironmentVariable(                       |
|                                                                       |
| \"PYANNOTE_METRICS_ENABLED\",                                         |
|                                                                       |
| \"0\",                                                                |
|                                                                       |
| \[System.EnvironmentVariableTarget\]::User                            |
|                                                                       |
| )                                                                     |
+-----------------------------------------------------------------------+

13.13 Summary --- Complete Install Script

The complete sequence for a CPU-only developer setup on a clean Windows
11 machine, copy-paste ready:

+-----------------------------------------------------------------------+
| \# === speechtelemetry --- Windows 11 Developer Setup (CPU-only) ===  |
|                                                                       |
| \# Run each block in an Administrator PowerShell unless noted.        |
|                                                                       |
| \# 1. Install prerequisites                                           |
|                                                                       |
| winget install \--id Python.Python.3.11 -e \--source winget           |
|                                                                       |
| winget install \--id=Gyan.FFmpeg -e                                   |
|                                                                       |
| winget install \--id Git.Git -e \--source winget                      |
|                                                                       |
| \# Restart PowerShell after this block.                               |
|                                                                       |
| \# 2. Allow scripts (run once per machine)                            |
|                                                                       |
| Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  |
|                                                                       |
| \# 3. Clone and set up venv                                           |
|                                                                       |
| cd C:\\Users\\\$env:USERNAME\\Projects                                |
|                                                                       |
| git clone https://github.com/your-org/speechtelemetry.git             |
|                                                                       |
| cd speechtelemetry                                                    |
|                                                                       |
| python -m venv .venv                                                  |
|                                                                       |
| .venv\\Scripts\\Activate.ps1                                          |
|                                                                       |
| python -m pip install \--upgrade pip                                  |
|                                                                       |
| \# 4. Install PyTorch (CPU-only)                                      |
|                                                                       |
| pip install torch torchaudio \--index-url                             |
| https://download.pytorch.org/whl/cpu                                  |
|                                                                       |
| \# 5. Install speechtelemetry core                                    |
|                                                                       |
| pip install -e \".\[core\]\"                                          |
|                                                                       |
| \# 6. (Optional) Add alignment and prosody                            |
|                                                                       |
| pip install whisperx praat-parselmouth                                |
|                                                                       |
| \# 7. Verify                                                          |
|                                                                       |
| python -c \"from speechtelemetry import enrich_media, PipelineConfig; |
| print(\'OK\')\"                                                       |
|                                                                       |
| ffmpeg -version                                                       |
|                                                                       |
| python -c \"import torch; print(torch.\_\_version\_\_)\"              |
|                                                                       |
| \# === Done. Run examples/basic_transcribe.py to test. ===            |
+-----------------------------------------------------------------------+

14\. Environment Variables Reference

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Variable**                **Purpose**                              **How to set (PowerShell, persistent)**
  --------------------------- ---------------------------------------- ------------------------------------------------------------------------------------------------
  HF_TOKEN                    HuggingFace read token. Required for     \[Environment\]::SetEnvironmentVariable(\"HF_TOKEN\",\"hf_xxx\",\"User\")
                              pyannote diarization and WhisperX
                              diarization.

  PYANNOTE_METRICS_ENABLED    Set to \"0\" to disable pyannote         \[Environment\]::SetEnvironmentVariable(\"PYANNOTE_METRICS_ENABLED\",\"0\",\"User\")
                              anonymous telemetry.

  SPEECHTELEMETRY_CACHE_DIR   Override default model cache directory.  \[Environment\]::SetEnvironmentVariable(\"SPEECHTELEMETRY_CACHE_DIR\",\"D:\\models\",\"User\")
                              Default:
                              %USERPROFILE%\\.cache\\speechtelemetry

  CUDA_VISIBLE_DEVICES        Select which GPU(s) to use on a          \[Environment\]::SetEnvironmentVariable(\"CUDA_VISIBLE_DEVICES\",\"0\",\"User\")
                              multi-GPU machine.

  OMP_NUM_THREADS             Limit OpenMP threads for CPU-only        \[Environment\]::SetEnvironmentVariable(\"OMP_NUM_THREADS\",\"4\",\"User\")
                              inference on shared machines.
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

15\. Architectural Decision Log

This section records key decisions that were made and why. Future
contributors should read these before proposing changes that touch the
same areas.

  -----------------------------------------------------------------------
  **Decision**             **Rationale**
  ------------------------ ----------------------------------------------
  Pydantic v2 for data     Free runtime validation, JSON schema
  model, not plain         generation, IDE autocomplete, and
  dataclasses              model_dump_json() without extra serialisation
                           code. The type-safety benefit to consumers
                           outweighs the dependency cost.

  Backend Contract via     Python\'s duck typing makes formal ABCs
  method signatures, not   unnecessary. The method signature documented
  abstract base classes    in the registry and enforced by tests is
                           sufficient. Avoids import-time inheritance
                           coupling between backends.

  registry.py with lazy    Prevents all ML frameworks from loading at
  string imports, not      import time. A user calling enrich_media()
  direct imports           with only ASR should not wait for SpeechBrain
                           to initialise.

  Separate io/ layer from  FFmpeg and audio normalisation are pure I/O
  backends/                with no ML. Keeping them separate makes them
                           testable without any ML dependencies and
                           reusable by non-ML tools.

  Diarization disabled by  Requires HuggingFace account, model license
  default                  acceptance, and HF_TOKEN. Making it opt-in
                           prevents first-run failures for new users.

  parselmouth in optional  GPL-3.0 license. Making it a required import
  adapter only (not        would contaminate the library\'s license.
  default prosody)         Listed as optional with a clear warning in
                           registry.py.

  Export JSON as           SRT and VTT cannot represent speaker labels,
  authoritative, SRT/VTT   prosody, emotion, or confidence. JSON
  as lossy derived views   preserves everything. Consumers that only need
                           subtitles use the lossy exporters; consumers
                           that need full fidelity use JSON.

  Pre-flight check at job  Surfaces all problems (missing FFmpeg, missing
  start, not on first      token, unavailable backend) before any compute
  backend use              happens. Avoids a 30-minute transcription job
                           failing with a dependency error at the last
                           step.

  Windows first, then      Dictated by target use-case. Consequences:
  Linux                    test CI runs on Windows; installation docs
                           lead with PowerShell; FFmpeg docs use winget;
                           CUDA docs use the NVIDIA Windows installer.
  -----------------------------------------------------------------------

16\. Quick Reference

16.1 Where to Put New Code

  ---------------------------------------------------------------------------
  **I want to...**          **Where it goes**
  ------------------------- -------------------------------------------------
  Add a new ASR backend     src/speechtelemetry/backends/asr/mybackend.py +
                            one line in registry.py

  Add a new export format   src/speechtelemetry/exporters/myformat.py +
                            register in registry.py

  Change the public API     api.py --- but avoid breaking changes; bump MAJOR
                            version if unavoidable

  Add a new config option   config.py (PipelineConfig) --- always with a
                            default value

  Add a new output field    types.py --- Optional\[T\] = None to remain
                            backward-compatible

  Add a CLI command         cli/commands.py --- CLI is a thin consumer; no
                            domain logic

  Add a metric / benchmark  benchmarking/metrics.py or
                            benchmarking/compare.py

  Add a test fixture        tests/fixtures/ --- commit the fixture file,
                            update golden_output.json
  ---------------------------------------------------------------------------

16.2 Design Principles Cheat Sheet

  -----------------------------------------------------------------------
  **Principle**         **The one-sentence test**
  --------------------- -------------------------------------------------
  Separation of         Can you describe this module\'s job without using
  Concerns              \"and\"?

  Single Responsibility If this file changes, does only one kind of
                        reason cause the change?

  Open/Closed           Can you add a new backend without modifying
                        core/pipeline.py or api.py?

  Dependency Inversion  Does core/pipeline.py import any concrete backend
                        class directly?

  Interface Segregation Does every backend for a stage implement exactly
                        the same method signature?

  Fail-Soft             If this stage fails, does the pipeline return
                        partial output rather than crashing?

  Explicit over         Is every decision that affects the output visible
  Implicit              in the result object?
  -----------------------------------------------------------------------

*speechtelemetry Architecture Reference \| v0.1 \| May 2026*

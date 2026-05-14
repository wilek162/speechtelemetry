# speechtelemetry — Library Spec v0.1

Product intent, scope, non-goals, and baseline behavior for v0.1.

This is document #1 in the source-of-truth hierarchy. When this document conflicts with any other, this document wins.

---

## 1. Product definition

speechtelemetry is a **library-first, local-first** speech intelligence engine.

- The library (`src/speechtelemetry/`) is the source of truth. CLI, UI, and API wrappers are secondary and must add zero business logic.
- Target platforms: **Windows first, then Linux**. CPU-only operation must always work. CUDA GPU acceleration is optional but first-class when explicitly configured.
- Core output: a canonical, fully-typed `TranscriptDocument` containing transcript, word timing, silence spans, optional speaker labels, optional prosody features, emotion estimates, and backend provenance for every stage.

---

## 2. Product principles

- **Local-first and offline-capable by default.** No cloud API keys required at runtime. Model weights are cached locally on first download and reused from cache on every subsequent run.
- **Deterministic, typed, inspectable outputs.** Every output is a Pydantic v2 model with complete type annotations. No raw dicts at the public boundary. Consumers see exactly what ran.
- **Probabilistic emotion, not fake certainty.** Emotion results store the full label probability distribution — never a single collapsed label. Callers decide how to surface it.
- **Pluggable backends behind one stable schema.** Swapping from `faster-whisper` to `whisper.cpp` requires only a config change. `types.py`, `core/pipeline.py`, and `api.py` do not change.
- **Fail soft: partial output is better than a hard crash.** Every optional stage is independently failable. Failures are recorded in `ProcessingReport.errors` and pipeline execution continues with degraded output.
- **Prefer proven tooling over novelty for the default path.** Default backends are chosen for reliability, permissive licensing, and CPU compatibility, not for benchmarks on the latest paper.
- **Explicit over implicit.** Device selection, model choices, and backend selection are all visible in `PipelineConfig` and recorded in `TranscriptDocument`. Nothing meaningful happens silently. GPU is never assumed.

---

## 3. Non-goals for v0.1

- Streaming or real-time transcription
- Conversational agent features
- Automatic summarization, chaptering, translation, or entity extraction
- Human-level or production emotion claims
- Mandatory speaker diarization (it is always opt-in)
- Web SaaS dependency or cloud services at runtime
- Training pipelines or model fine-tuning
- A monolithic app with hardcoded model choices

---

## 4. Pipeline

```
Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
```

| Stage | Default backend | License | Required |
|-------|----------------|---------|----------|
| Decode | FFmpeg (system) | LGPL 2.1 | Yes |
| Normalize | FFmpeg (system) | LGPL 2.1 | Yes |
| VAD | silero-vad | MIT | Yes |
| ASR | faster-whisper | MIT | Yes |
| Alignment | whisperx | BSD-4-Clause | Yes |
| Diarization | pyannote.audio | MIT + CC-BY-4.0 weights | No (opt-in, HF_TOKEN required) |
| Prosody | — | — | No (opt-in; no default backend) |
| Emotion | speechbrain | Apache 2.0 | Yes |
| Export | built-in | MIT | Yes (JSON always) |

FFmpeg is a system prerequisite, not a Python package. It must be on `PATH`.

---

## 5. Public API shape

```python
from speechtelemetry import enrich_media, enrich_audio, PipelineConfig

# One-shot: any media file (MP4, MP3, WAV, MKV, …)
doc = enrich_media("interview.mp4", config=PipelineConfig())

# One-shot: already a mono 16 kHz WAV — skip FFmpeg decode
doc = enrich_audio("interview.wav", config=PipelineConfig())
```

Stage-level functions (`run_vad`, `run_asr`, `run_alignment`, `run_diarization`, `run_prosody`, `run_emotion`) are also available from `speechtelemetry.api` for custom pipeline assembly. These are advanced/unstable.

The library is fully usable with a single import:

```python
from speechtelemetry import enrich_media, PipelineConfig
```

---

## 6. Canonical data model

```
TranscriptDocument
  ├── source_path: str
  ├── language: Optional[str]           — ISO 639-1; None if not detected
  ├── duration_s: float
  ├── segments: list[Segment]
  │     ├── start, end: float           — seconds
  │     ├── text: str
  │     ├── confidence: float           — [0.0, 1.0]
  │     ├── speaker: Optional[str]      — "SPEAKER_00"; None if diarization off
  │     ├── silence_before_ms, silence_after_ms: Optional[float]
  │     ├── words: Optional[list[Word]]
  │     │     └── text, start, end, confidence, token_id, alignment_backend
  │     ├── prosody: Optional[ProsodyWindow]
  │     │     └── f0_mean, f0_variance, energy_mean, energy_variance,
  │     │         speech_rate_sps, pause_density, voice_quality_hnr,
  │     │         jitter, shimmer, backend_name
  │     └── emotion: Optional[EmotionScore]
  │           └── label_distribution: dict[str, float]  — ALWAYS a full dict
  │               confidence, backend_name, valence, arousal
  ├── silence_spans: list[SilenceSpan]
  │     └── start, end, duration_ms, reason: "speech_gap"|"non_speech"|"overlap"
  └── processing_report: ProcessingReport
        └── real_time_factor, peak_ram_mb, peak_vram_mb,
            stage_timings: dict[str, float],
            errors: list[StageError]    — ALWAYS a list, never None
```

Rules enforced in `types.py`:
- `types.py` imports nothing from any other speechtelemetry module (leaf node in the dependency graph).
- Every ML-derived field carries `backend_name` provenance.
- `EmotionScore.label_distribution` is always a full probability dict. Consumers decide how to render it.
- Missing values use `Optional[T] = None`, never sentinel strings like `"unknown"`.
- `ProcessingReport.errors` is always a list (may be empty), never `None`.

---

## 7. Defaults and behavior

| Setting | Default | Rationale |
|---------|---------|-----------|
| `device` | `"auto"` | `"auto"` selects CUDA if available, else CPU. Resolution is delegated to the backend layer. Use `"cpu"` to force CPU; `"cuda"` to require GPU (fails if unavailable). |
| `asr_backend` | `"faster-whisper"` | MIT licensed; best CPU/GPU throughput ratio |
| `asr_model_size` | `"large-v3"` | Best accuracy; swap to `medium` or `small` for speed |
| `asr_compute_type` | `None` | Auto: `float16` on GPU, `int8` on CPU |
| `asr_language` | `None` | Auto-detect; set explicitly to skip detection and save latency |
| `asr_beam_size` | `5` | Standard beam search width |
| `vad_backend` | `"silero"` | MIT, ~2 MB model, runs well on CPU |
| `vad_threshold` | `0.5` | Speech/silence decision boundary |
| `alignment_backend` | `"whisperx"` | Best word-timestamp accuracy available |
| `diarization_backend` | `None` | Disabled — requires HF_TOKEN and model license acceptance |
| `prosody_backend` | `[]` | Opt-in — no default because parselmouth is GPL-3.0 |
| `emotion_backend` | `"speechbrain"` | Apache 2.0, IEMOCAP-trained wav2vec2 |
| `export_formats` | `["json"]` | JSON is the authoritative lossless format |
| `chunk_audio` | `True` | Prevents memory spikes on long files |
| `max_chunk_duration_s` | `30.0` | Chunk size for chunked inference |

---

## 8. License constraints

- Core `dependencies` in `pyproject.toml` must be **MIT, BSD, or Apache 2.0 only**. No GPL in core.
- GPL-licensed backends (parselmouth) are listed only in optional extras. A warning is emitted via `warnings.warn()` at backend resolution time.
- Backends requiring model license acceptance (pyannote) are guarded by a pre-flight `HF_TOKEN` check before any compute starts.
- Model weights are **never bundled**. They are downloaded to `SPEECHTELEMETRY_CACHE_DIR` (default: `~/.cache/speechtelemetry/`) on first use.
- GPU extras (`speechtelemetry[cuda]`) require CUDA-enabled PyTorch to be pre-installed by the user. The package does not bundle CUDA.

See `docs/license_policy.md` for the full per-backend license table and compliance guidance.

---

## 9. Performance targets (v0.1)

| Metric | CPU (large-v3 int8) | GPU (large-v3 float16) |
|--------|--------------------|-----------------------|
| RTF | < 2.0 | < 0.2 |
| Peak RAM | < 8 GB | < 8 GB system RAM |
| Peak VRAM | — | < 8 GB |

RTF = `wall_time_s / audio_duration_s`. Measured end-to-end including decode and export.

Every job records RTF, peak RAM, peak VRAM, and per-stage timings in `ProcessingReport`.

---

## 10. MVP scope

**In scope:**
- Local file ingest (any FFmpeg-supported format)
- Canonical audio normalization to mono 16 kHz WAV
- Transcription with word-level timestamps
- Dead-air and silence detection
- Optional speaker diarization
- Optional prosody feature extraction
- Emotion probability distribution per segment
- Structured JSON export plus SRT/VTT/TextGrid subtitle export
- CLI (`speechtelemetry transcribe`, `speechtelemetry info`) and importable Python library

**Not in scope for v0.1:**
- Real-time or streaming transcription
- Summarization, chaptering, or translation
- Web UI or cloud services
- Training pipelines

---

## 11. Developer experience requirements

- One install path for normal users. No manual model graph surgery.
- `PipelineConfig()` with zero arguments produces a working CPU pipeline.
- All environment problems (missing FFmpeg, missing `HF_TOKEN`, CUDA unavailable, backend not installed) are reported together at job start — before any compute runs.
- Every error message includes the exact command needed to fix the problem.
- Reproducible fixtures and golden outputs in the repository.
- No hidden behavior based on model internals. Everything meaningful is explicit in the result object.

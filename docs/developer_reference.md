# speechtelemetry — Developer & Agent Reference

Backend APIs, install commands, configuration reference, and caveats for v0.1.

This is document #3 in the source-of-truth hierarchy. For design decisions, see `docs/architecture.md`. For product intent, see `docs/spec.md`.

---

## 1. Project overview

speechtelemetry is a library-first, local-first speech intelligence engine. The pipeline processes any audio or video file and returns a fully-typed `TranscriptDocument`.

### Pipeline stages

```
Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
```

Every stage is fail-soft (except hard prerequisites — see Section 7). Failures are recorded in `ProcessingReport.errors` and processing continues.

### Primary public API

```python
from speechtelemetry import enrich_media, enrich_audio, PipelineConfig

# Any media file (MP4, MP3, WAV, MKV, …) — FFmpeg decode included
doc = enrich_media("interview.mp4", config=PipelineConfig())

# Pre-normalized mono 16 kHz WAV — skips FFmpeg decode
doc = enrich_audio("interview.wav", config=PipelineConfig())
```

Both return a `TranscriptDocument`. See `src/speechtelemetry/types.py` for the full schema.

---

## 2. Installation

### 2.1 Python and virtual environment

**Python 3.10 or 3.11 is recommended.** Python 3.12 is supported but some audio backend wheels may be missing on Windows. Always use a virtual environment.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Verify venv is active (prefix should differ from base)
python -c "import sys; print(sys.prefix)"
```

### 2.2 FFmpeg — mandatory system dependency

FFmpeg must be installed system-wide and available on `PATH`. It is the universal decode layer for all media input.

```powershell
# Windows — LGPL build (recommended for license hygiene)
winget install --id=Gyan.FFmpeg -e

# Verify
ffmpeg -version
```

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### 2.3 PyTorch — install before any ML backend

PyTorch must be installed **before** any ML backends (`speechtelemetry[asr]`, etc.). Choose one variant:

**CPU-only (default, works everywhere):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1 (NVIDIA GPU, most common):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (older NVIDIA GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4 (latest NVIDIA GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To find your installed CUDA version: `nvcc --version` or `nvidia-smi`.

### 2.4 speechtelemetry install paths

After installing PyTorch, install speechtelemetry:

| Goal | Command |
|------|---------|
| Library only (no ML, for type-checking or embedding) | `pip install speechtelemetry` |
| Transcription + alignment + VAD + emotion + CLI | `pip install speechtelemetry[default]` |
| Full stack + diarization + prosody | `pip install speechtelemetry[all]` |
| GPU-accelerated (after CUDA torch install) | `pip install speechtelemetry[cuda]` |
| Development environment | `pip install -e ".[dev]"` |

### 2.5 GPU (CUDA) setup — full path

GPU acceleration requires three things: NVIDIA GPU, CUDA Toolkit 12.x, and a CUDA-enabled torch build.

```powershell
# Step 1: Verify GPU and CUDA
nvidia-smi                    # confirms GPU driver and CUDA version
nvcc --version                # confirms CUDA Toolkit is installed

# Step 2: Install CUDA Toolkit if needed
# Download from: https://developer.nvidia.com/cuda-downloads
# Choose: Windows > x86_64 > 11 > exe (local)

# Step 3: Create venv and install CUDA-enabled torch FIRST
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Verify torch sees the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Step 5: Install speechtelemetry with cuda extras
pip install speechtelemetry[cuda]

# Step 6: Use device="cuda" in config
```

```python
from speechtelemetry import enrich_media, PipelineConfig

doc = enrich_media(
    "interview.mp4",
    config=PipelineConfig(device="cuda"),
)
```

`device="cpu"` is always the safe default. `device="cuda"` is an explicit opt-in that requires the above setup.

### 2.6 HuggingFace token — required for diarization

Speaker diarization with pyannote.audio requires a HuggingFace token and model license acceptance.

```powershell
# PowerShell — set for current session
$env:HF_TOKEN = "hf_YOUR_TOKEN_HERE"

# Persist across sessions (add to your PowerShell profile)
[System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_YOUR_TOKEN_HERE", "User")
```

Steps to get a token:
1. Create account at [hf.co](https://huggingface.co)
2. Accept model license at [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
3. Generate a read token at [hf.co/settings/tokens](https://hf.co/settings/tokens)
4. Set `HF_TOKEN` as shown above

---

## 3. Backend reference

### 3.1 FFmpeg — media decode and normalization

**Role:** Every input file (any format FFmpeg supports) is decoded to a canonical mono 16 kHz 16-bit PCM WAV before any Python backend touches it.

**Install:** System package. Not a Python dependency.

**Canonical normalization command (used internally by `io/ffmpeg.py`):**
```
ffmpeg -i input.mp4 -ac 1 -ar 16000 -acodec pcm_s16le output.wav
```

**Key parameters:**
- `-ac 1` — mono
- `-ar 16000` — 16 kHz sample rate
- `-acodec pcm_s16le` — 16-bit signed little-endian PCM

---

### 3.2 Silero VAD — Voice Activity Detection

**Role:** Runs after normalization. Produces speech timestamp intervals that (a) filter audio before ASR to reduce wasted compute and hallucination, and (b) populate `SilenceSpan` objects in the output.

**Install:**
```bash
pip install silero-vad
```

**License:** MIT — no restrictions, no keys, no telemetry.

**Model:** ~2 MB JIT model. Downloads from HuggingFace Hub on first use. Cached at `SPEECHTELEMETRY_CACHE_DIR`.

**Supported sample rates:** 8 kHz and 16 kHz only. The normalization stage ensures 16 kHz input.

**Key tuning parameters (via `PipelineConfig`):**
- `vad_threshold` (default `0.5`) — higher values = stricter, fewer false positives
- `vad_min_speech_ms` (default `250`) — minimum speech segment duration
- `vad_min_silence_ms` (default `300`) — minimum silence gap before splitting

---

### 3.3 faster-whisper — ASR (default backend)

**Role:** Transcribes VAD-filtered audio segments into text with segment-level timestamps. Does not produce word-level timestamps — that is the alignment stage.

**Install:**
```bash
pip install faster-whisper
```

For GPU, requires `cuBLAS` and `cuDNN` matching your CUDA version (auto-installed with the CUDA torch build in most cases).

**License:** MIT. Models on HuggingFace Hub under OpenAI license (permissive for commercial use).

**How it works:** Reimplements OpenAI Whisper using CTranslate2, a high-performance inference engine for Transformer models. Up to 4x faster than the original implementation with identical accuracy. Supports int8 quantization on both CPU and GPU.

**Key tuning parameters (via `PipelineConfig`):**
- `asr_model_size` (default `"large-v3"`) — `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Use `small` or `medium` for speed on CPU.
- `asr_compute_type` (default `None`) — Auto selects `float16` on GPU and `int8` on CPU. Can override with `"float32"`, `"int8"`, `"int8_float16"`.
- `asr_language` (default `None`) — Auto-detect. Set an ISO 639-1 code (e.g., `"en"`, `"de"`) to skip detection and reduce latency.
- `asr_beam_size` (default `5`) — Beam search width. Reduce to `1` (greedy) for maximum speed.

**Performance guidance:**
- CPU (`int8`, `large-v3`): RTF ~1.5–2.0 on a modern multi-core desktop
- GPU (`float16`, `large-v3`): RTF ~0.1–0.2 on an RTX 3080/4080

---

### 3.4 WhisperX — word alignment (and optional combined ASR+diarization)

**Role (primary):** Word-level forced alignment. Refines faster-whisper segment timestamps to word-level precision using wav2vec2 forced alignment.

**Role (optional):** Combined ASR + alignment + diarization backend when tight integration is preferred. Use faster-whisper for standalone ASR and WhisperX only for alignment unless the combined mode is explicitly configured.

**Install:**
```bash
pip install whisperx>=3
```

HuggingFace token required for the diarization sub-pipeline (pyannote.audio within WhisperX).

**License:** BSD-4-Clause. Compatible with commercial use; consult legal if redistributing.

**Usage note:** WhisperX downloads language-specific wav2vec2 alignment models on first use per language. Set `SPEECHTELEMETRY_CACHE_DIR` to control where they land.

---

### 3.5 pyannote.audio — speaker diarization

**Role:** Assigns speaker labels to time intervals in the transcript. Runs after alignment. Diarization is always optional — if disabled or failed, the pipeline returns a transcript without speaker labels.

**Install:**
```bash
pip install pyannote.audio>=3.1
```

**License:** pyannote.audio library — MIT. Model weights — CC-BY-4.0. **Always check the model card license before using in a commercial product.**

**Prerequisites:**
- `HF_TOKEN` environment variable set
- Model license accepted at [hf.co/pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)

**Recommended model:** `pyannote/speaker-diarization-community-1` — better speaker counting accuracy.

**Fallback model:** `pyannote/speaker-diarization-3.1` — more stable, widely tested.

**Key tuning parameters:**
- `diarization_min_speakers`, `diarization_max_speakers` — constrain the expected speaker count. Set both to the same value if the number is known.

**Caveats:**
- Diarization quality degrades on overlapping speech, far-field microphones, noisy recordings, and very short utterances.
- Uses GPU when `device="cuda"` is set.

---

### 3.6 praat-parselmouth — prosody extraction

**Role:** Extracts core prosody features per segment: fundamental frequency (F0/pitch), intensity (energy), jitter, shimmer, and harmonics-to-noise ratio (HNR).

**Install:**
```bash
pip install praat-parselmouth>=0.4
```

**License: GPL-3.0.** Any code that imports and redistributes parselmouth may need to be GPL-3.0 or compatible. Use the `librosa` backend instead for permissive-license projects.

**Current version:** 0.4.7+. Supports Python 3.7+.

**Enable prosody:**
```python
config = PipelineConfig(prosody_backend=["parselmouth"])
```

Prosody is disabled by default (`prosody_backend=[]`). Users must opt in.

**Minimum segment length:** 40 ms. Segments shorter than this are skipped silently.

**Key features extracted:**
- `f0_mean`, `f0_variance` — fundamental frequency (Hz)
- `energy_mean`, `energy_variance` — intensity (dB)
- `jitter` — local pitch period irregularity
- `shimmer` — local amplitude irregularity
- `voice_quality_hnr` — harmonics-to-noise ratio (dB)

---

### 3.7 CoPaSul — advanced prosody stylization

**Role:** Deeper prosody analysis beyond basic F0/energy statistics. Models intonation as a superposition of global and local polynomial contours. Extracts register features, prosodic boundary indicators, rhythm measures, and bottom-up contour classes.

**Install:**
```bash
pip install copasul>=1.0
```

**License:** MIT.

**Input requirements:** WAV files plus TextGrid or annotation files marking segment boundaries.

**Enable:**
```python
config = PipelineConfig(prosody_backend=["copasul"])
```

---

### 3.8 SpeechBrain — speech emotion recognition (default)

**Role:** Runs on each segment (utterance level) and returns a full probability distribution over emotion classes, not a single predicted label. The library stores full distributions and confidence scores — never collapses to a single label.

**Install:**
```bash
pip install speechbrain>=1.0 transformers>=4.30
```

**License:** Apache 2.0 — fully permissive for commercial use.

**Default model:** `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` — auto-downloads from HuggingFace on first use.

**Emotion classes (IEMOCAP):** `neutral`, `happy`, `sad`, `anger`. Accuracy: ~75% on the IEMOCAP test set. Cross-corpus accuracy varies significantly — treat emotion scores as estimates, not ground truth.

**GPU:** uses `device="cuda"` from `PipelineConfig` automatically.

**Minimum segment length:** 0.5 s. Shorter segments are silently skipped.

**Caveats:**
- Cross-corpus SER (speech emotion recognition) is a hard research problem. Benchmark scores do not transfer cleanly to real-world audio.
- Emotion recognition is framed as estimation, not truth. Never present a single label as a factual claim.

---

### 3.9 Planned backends (not yet implemented)

The registry lists the following backends as future targets. Specifying one of these names in `PipelineConfig` raises `BackendNotFoundError` or `ImportError` (module not found) today — they are placeholders signalling contributor intent, not available features. Each entry includes the target install command once implementation ships.

#### ASR

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `whisperx` (combined ASR mode) | `pip install whisperx>=3` | BSD-4-Clause | Combined ASR + alignment. Currently implemented as alignment-only backend. |
| `whisper.cpp` | `pip install whispercpp` | MIT | CTranslate2 alternative; better Windows compat |
| `sensevoice` | `pip install funasr` | Apache 2.0 | FunASR SenseVoice multilingual model |

#### VAD

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `funasr` | `pip install funasr` | Apache 2.0 | FunASR VAD model |

#### Alignment

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `forcealign` | `pip install forcealign` | MIT | Alternative to WhisperX for forced alignment |

#### Diarization

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `funasr` | `pip install funasr` | Apache 2.0 | FunASR speaker diarization; no HF_TOKEN required |

#### Prosody

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `librosa` | `pip install librosa` | MIT | Permissive prosody alternative to parselmouth |
| `copasul` | `pip install copasul` | MIT | Advanced intonation stylization |
| `audioflux` | `pip install audioflux` | Apache 2.0 | Low-level audio feature extraction |

#### Emotion

| Name | Install (future) | License | Notes |
|------|-----------------|---------|-------|
| `emotion2vec` | `pip install funasr` | Apache 2.0 | emotion2vec+ model via FunASR |
| `emobox` | `pip install emobox` | MIT | Multi-corpus emotion models |

To implement any of the above, follow the **Adding a new backend** steps in `docs/agent_playbook.md`.

---

## 4. Configuration reference

### 4.1 PipelineConfig fields

```python
from speechtelemetry import PipelineConfig

config = PipelineConfig(
    # ── Device ────────────────────────────────────────────────────────────
    device="cpu",                      # "cpu" | "cuda"
                                       # "cpu" is always the safe default.
                                       # "cuda" requires CUDA-enabled torch.

    # ── ASR ───────────────────────────────────────────────────────────────
    asr_backend="faster-whisper",      # "faster-whisper" | "whisperx" | "whisper.cpp" | "sensevoice"
    asr_model_size="large-v3",         # "tiny" | "base" | "small" | "medium" | "large-v2" | "large-v3"
    asr_compute_type=None,             # None = auto (float16 GPU, int8 CPU)
    asr_language=None,                 # None = auto-detect; e.g. "en", "de", "fr"
    asr_beam_size=5,                   # 1..10; 1 = greedy (fastest)

    # ── VAD ───────────────────────────────────────────────────────────────
    vad_backend="silero",              # "silero" | "funasr"
    vad_threshold=0.5,                 # 0.0..1.0; higher = stricter
    vad_min_silence_ms=300,            # minimum silence gap (ms)
    vad_min_speech_ms=250,             # minimum speech segment (ms)

    # ── Alignment ─────────────────────────────────────────────────────────
    alignment_backend="whisperx",      # "whisperx" | "forcealign"

    # ── Diarization ───────────────────────────────────────────────────────
    diarization_backend=None,          # None | "pyannote" | "funasr"
                                       # None disables diarization (default)
    diarization_min_speakers=None,     # int | None
    diarization_max_speakers=None,     # int | None

    # ── Prosody ───────────────────────────────────────────────────────────
    prosody_backend=[],                # list; empty = disabled (default)
                                       # options: "parselmouth" (GPL-3.0),
                                       #          "copasul" (MIT),
                                       #          "librosa" (MIT),
                                       #          "audioflux" (Apache 2.0)

    # ── Emotion ───────────────────────────────────────────────────────────
    emotion_backend="speechbrain",     # "speechbrain" | "emotion2vec" | "emobox" | None

    # ── Export ────────────────────────────────────────────────────────────
    export_formats=["json"],           # any of: "json", "srt", "vtt", "textgrid"

    # ── Performance ───────────────────────────────────────────────────────
    chunk_audio=True,                  # process in chunks (recommended)
    max_chunk_duration_s=30.0,         # chunk size in seconds
)
```

`PipelineConfig` is immutable after instantiation. All fields have defaults. `PipelineConfig()` with zero arguments produces a working CPU pipeline.

### 4.2 Environment variables

| Variable | Purpose | Required |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace read token for pyannote diarization | Only when `diarization_backend="pyannote"` |
| `HUGGINGFACE_TOKEN` | Alias for `HF_TOKEN` | Same |
| `SPEECHTELEMETRY_CACHE_DIR` | Override model weight cache directory | No (default: `~/.cache/speechtelemetry/`) |
| `SPEECHTELEMETRY_LOG_LEVEL` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | No (default: `WARNING`) |
| `PYANNOTE_NO_ANALYTICS` | Disable pyannote telemetry (`1` = off) | No |

---

## 5. Error handling and fail-soft policy

### Pre-flight errors (hard failures)

The following conditions raise `EnvironmentCheckError` before any compute starts. All problems are reported together in a single exception.

- FFmpeg not found on `PATH`
- `HF_TOKEN` not set when `diarization_backend` is not `None`
- `device="cuda"` but `torch.cuda.is_available()` returns `False`
- A requested backend is not installed (import would fail)

### Stage errors (soft failures)

Every other failure is a soft failure. The stage records a `StageError` in `ProcessingReport.errors` and returns `None` or empty values for that output field.

```python
doc = enrich_media("interview.mp4", config=PipelineConfig())

if doc.processing_report.errors:
    for err in doc.processing_report.errors:
        print(f"[{err.stage}] {err.exception_type}: {err.message}")
```

### Exception hierarchy

```
SpeechTelemetryError
├── EnvironmentCheckError   — pre-flight failures; raised before any compute
├── BackendNotFoundError    — requested backend name not in registry
├── BackendNotAvailableError— backend found but its dependency is not installed
└── BackendError            — backend raised an unexpected error (also soft-caught in pipeline)
```

---

## 6. Performance targets and telemetry

Every `TranscriptDocument` includes a `ProcessingReport` with:

| Metric | How it is measured |
|--------|--------------------|
| `real_time_factor` | `wall_time_s / audio_duration_s` |
| `peak_ram_mb` | `tracemalloc` peak allocated |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated()` (0 if no GPU) |
| `stage_timings` | `time.perf_counter()` around each stage |

**Target RTF values (v0.1):**

| Config | Expected RTF |
|--------|-------------|
| CPU, `large-v3`, int8 | < 2.0 |
| GPU, `large-v3`, float16 | < 0.2 |
| CPU, `small`, int8 | < 0.5 |

RTF < 1.0 means the pipeline is faster than real-time.

**Memory management on GPU:**
- Always delete model objects (`del model`) and call `gc.collect()` between ML stages.
- Call `torch.cuda.empty_cache()` after GPU stage teardown.
- Never load a full multi-hour WAV as a single numpy array. Use `chunk_audio=True`.
- For parselmouth: load the full WAV once as `parselmouth.Sound`, then call `extract_part()` per segment. Do not re-open the file per segment.

---

## 7. Developer and agent checklist

Use this checklist before merging any new backend or pipeline stage.

### Environment
- [ ] FFmpeg is installed and on `PATH`. Verified with `ffmpeg -version`.
- [ ] Python 3.10 or 3.11 virtual environment created and activated.
- [ ] PyTorch installed (CPU or CUDA variant) before any ML backend.
- [ ] `HF_TOKEN` set if diarization is being tested.

### Backend implementation
- [ ] New backend placed in correct `backends/<stage>/` subdirectory.
- [ ] Method signature matches the ABC in `interfaces.py` exactly.
- [ ] `try/except ImportError → BackendNotAvailableError` guard at the top of the module.
- [ ] Backend name registered in `registry.py` with one line.
- [ ] License checked and documented in `registry.py` comment and `docs/license_policy.md`.
- [ ] GPL-licensed backends are in optional adapters only and emit a warning via `_LICENSE_FLAGS`.

### Data model
- [ ] All outputs mapped to `types.py` canonical types.
- [ ] `EmotionScore` stores full `label_distribution` — never a single label.
- [ ] `ProcessingReport` populated with stage timings and errors.
- [ ] `backend_name` set in every ML-derived output field.

### Error handling
- [ ] All backends wrapped in `try/except` inside `core/pipeline.py`.
- [ ] Failures append to `ProcessingReport.errors` and return `None`/empty for that field.
- [ ] Short segments (< 40 ms for prosody, < 500 ms for emotion) handled gracefully.
- [ ] Missing model weights, missing FFmpeg, missing `HF_TOKEN` all raise clear `EnvironmentError` at job start.

### Testing
- [ ] Unit test for each backend using a short canonical WAV fixture.
- [ ] Integration test running `enrich_media()` on a 30 s sample.
- [ ] All four test types covered as appropriate.
- [ ] No VRAM left allocated after test teardown.

### Quality gates
- [ ] `python -m pytest tests/unit/ -q` exits 0.
- [ ] `ruff check src/ tests/` exits 0.
- [ ] `ruff format --check src/ tests/` exits 0.
- [ ] `mypy src/` exits 0.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.

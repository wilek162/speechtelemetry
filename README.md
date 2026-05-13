# speechtelemetry

**Local-first speech intelligence.** Transcription, word timing, silence detection, speaker diarization, prosody, and emotion — in one library.

[![PyPI version](https://badge.fury.io/py/speechtelemetry.svg)](https://badge.fury.io/py/speechtelemetry)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/speechtelemetry/speechtelemetry/actions/workflows/ci.yml/badge.svg)](https://github.com/speechtelemetry/speechtelemetry/actions/workflows/ci.yml)

---

## What it does

speechtelemetry processes any audio or video file through a local pipeline and returns a fully-typed, inspectable `TranscriptDocument` containing:

- **Transcript** — segments and words with timestamps and confidence scores
- **Silence spans** — dead-air detection with reason classification
- **Speaker labels** — optional diarization (who spoke when)
- **Prosody** — F0/pitch, energy, speech rate, voice quality per segment (opt-in)
- **Emotion** — probability distribution over emotion classes per utterance
- **Processing report** — timing, RAM/VRAM, and structured errors for every stage

All processing is **local and offline-capable**. No cloud API keys required.

---

## Quickstart

```bash
pip install speechtelemetry[default]
```

```python
from speechtelemetry import enrich_media, PipelineConfig

result = enrich_media(
    input_path="interview.mp4",
    config=PipelineConfig(),          # device="auto" by default
)

for seg in result.segments:
    print(f"{seg.start:.2f}s  [{seg.speaker}]  {seg.text}")
    if seg.emotion:
        print(f"  emotion: {seg.emotion.label_distribution}")
```

---

## Installation

### Prerequisites

- **Python 3.10 or 3.11** (3.12 supported; some audio backend wheels may be missing on Windows)
- **FFmpeg** — mandatory for media decode

  ```powershell
  # Windows
  winget install --id=Gyan.FFmpeg -e

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

- **Virtual environment** — always use one

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1    # Windows
  # source .venv/bin/activate     # Linux/macOS
  ```

### CPU-only install (default, works everywhere)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechtelemetry[default]
```

### GPU (CUDA) install

GPU support is an explicit install path, not a default assumption. Install a CUDA-enabled torch **before** installing speechtelemetry, then use `device="cuda"` in the config.

```bash
# Step 1: install CUDA-enabled torch (replace cu121 with your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: verify torch sees your GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Step 3: install speechtelemetry with cuda extras
pip install speechtelemetry[cuda]
```

```python
from speechtelemetry import enrich_media, PipelineConfig

result = enrich_media(
    "interview.mp4",
    config=PipelineConfig(device="cuda"),   # explicit opt-in
)
print(f"RTF: {result.processing_report.real_time_factor:.2f}")
```

**CUDA index URLs by version:**

| CUDA version | Index URL |
|-------------|-----------|
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |

Find your CUDA version with `nvcc --version` or `nvidia-smi`.

### Install options

| Goal | Command |
|------|---------|
| Library + types only (no ML) | `pip install speechtelemetry` |
| Transcription + VAD + emotion + CLI | `pip install speechtelemetry[default]` |
| Full stack (+ diarization + prosody) | `pip install speechtelemetry[all]` |
| GPU-accelerated (after CUDA torch install) | `pip install speechtelemetry[cuda]` |
| Development environment | `pip install -e ".[dev]"` |

For the complete Windows 11 PowerShell setup guide, see [docs/setup_windows.md](docs/setup_windows.md).

---

## Pipeline

```
Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
```

| Stage | Default backend | License | Required |
|-------|----------------|---------|----------|
| Decode | FFmpeg (system) | LGPL 2.1 | Yes |
| VAD | silero-vad | MIT | Yes |
| ASR | faster-whisper | MIT | Yes |
| Alignment | whisperx | BSD-4-Clause | Yes |
| Diarization | pyannote.audio *(opt-in)* | MIT + CC-BY-4.0 | No |
| Prosody | *(opt-in, no default)* | varies | No |
| Emotion | speechbrain | Apache 2.0 | Yes |
| Export | built-in | MIT | Yes |

Prosody has no default backend because the most accessible option (parselmouth) is GPL-3.0. Choose `"librosa"` (MIT) for permissive-license projects.

---

## Configuration

```python
from speechtelemetry import PipelineConfig

config = PipelineConfig(
    device="auto",                     # "auto" (default), "cpu", or "cuda"
    asr_backend="faster-whisper",
    asr_model_size="large-v3",         # "tiny", "small", "medium", "large-v3"
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,          # None disables diarization (no HF_TOKEN needed)
    prosody_backend=[],                # [] disables prosody (opt-in; parselmouth=GPL-3.0)
    emotion_backend="speechbrain",
    export_formats=["json", "srt"],
)
```

All fields have defaults. `PipelineConfig()` with zero arguments produces a working CPU pipeline.

---

## Architecture

speechtelemetry is built on strict separation of concerns:

- **`types.py`** — canonical data model (pure data, zero imports)
- **`interfaces.py`** — ABCs for every backend stage
- **`registry.py`** — runtime name → class resolver; the only place backends are imported
- **`core/pipeline.py`** — stage orchestration with fail-soft error handling
- **`backends/`** — one file per backend, all behind the interface contract
- **`exporters/`** — serialize `TranscriptDocument` to JSON, SRT, VTT, TextGrid

Adding a new backend requires: one new file in `backends/` + one line in `registry.py`. Nothing else changes.

See [docs/architecture.md](docs/architecture.md) for the full design reference.

---

## Custom backends

```python
from speechtelemetry import registry
from speechtelemetry.interfaces import EmotionBackend

class MyEmotionBackend(EmotionBackend):
    STAGE = "emotion"
    NAME  = "my-model"

    def predict_segment(self, wav_path, start_s, end_s):
        return {
            "label_distribution": {"happy": 0.8, "neutral": 0.2},
            "confidence": 0.8,
            "backend_name": "my-model-v1",
        }

registry.register("emotion", "my-model", MyEmotionBackend)
```

---

## License

The library is **MIT licensed**. Some optional backends have different licenses:

- `praat-parselmouth` — GPL-3.0 (optional adapter only; use `librosa` for permissive projects)
- `pyannote.audio` models — CC-BY-4.0 (requires HuggingFace account + license acceptance)
- FFmpeg — LGPL 2.1 (use the LGPL build; see [docs/license_policy.md](docs/license_policy.md))

See [docs/license_policy.md](docs/license_policy.md) for the full per-backend license table and compliance guidance.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The [agent playbook](docs/agent_playbook.md) documents how both human and AI contributors should work inside the project.

**Documentation:**
- [docs/spec.md](docs/spec.md) — product intent, scope, non-goals
- [docs/architecture.md](docs/architecture.md) — layer boundaries, design decisions
- [docs/developer_reference.md](docs/developer_reference.md) — backend APIs, install commands, config reference
- [docs/setup_windows.md](docs/setup_windows.md) — complete Windows 11 PowerShell setup guide

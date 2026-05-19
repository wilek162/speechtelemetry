# speechtelemetry

**Local-first speech intelligence.** Transcription, word timing, silence detection, speaker diarization, prosody, and emotion — in one library.

[![CI](https://github.com/wilek162/speechtelemetry/actions/workflows/ci.yml/badge.svg)](https://github.com/wilek162/speechtelemetry/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/speechtelemetry.svg)](https://badge.fury.io/py/speechtelemetry)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What it does

speechtelemetry processes any audio or video file through a fully local pipeline and returns a typed, inspectable `TranscriptDocument`:

- **Transcript** — segments and word-level timestamps with confidence scores
- **Speaker labels** — who spoke when (optional diarization)
- **Emotion** — full probability distribution over emotion classes per utterance
- **Silence spans** — dead-air detection with reason classification
- **Prosody** — F0/pitch, energy, jitter, shimmer per segment (opt-in)
- **Speaker profiles** — per-speaker stats: speaking time, turn count, dominant emotion
- **Processing report** — stage timings, peak RAM/VRAM, structured error log

All processing is **local and offline-capable**. No cloud API keys required at runtime.

---

## Install

### Step 1 — Prerequisites

**Python 3.10 or 3.11** and **FFmpeg** (mandatory — used for media decode):

```bash
# Windows
winget install --id=Gyan.FFmpeg -e

# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

Verify: `ffmpeg -version`

### Step 2 — PyTorch (must be installed before speechtelemetry)

PyTorch is not bundled — choose your variant:

```bash
# CPU-only (works everywhere, no GPU required)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (latest NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (older NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Find your CUDA version: `nvcc --version` or `nvidia-smi`.

### Step 3 — Install speechtelemetry

```bash
# DEFAULT — transcription + word alignment + VAD + emotion + CLI
pip install speechtelemetry[default]
```

That's it for most users. See the [install variants](#install-variants) table below for everything else.

---

## Quickstart

```python
from speechtelemetry import enrich_media, PipelineConfig

doc = enrich_media(
    "interview.mp4",
    config=PipelineConfig(),   # zero-config CPU pipeline
)

for seg in doc.segments:
    emotion = seg.emotion.top_label if seg.emotion else "—"
    print(f"[{seg.start:.2f}s] [{seg.speaker or '?'}] {seg.text}  ({emotion})")
```

Example output:
```
[0.00s] [SPEAKER_00] You're pointy.  (neu)
[1.00s] [SPEAKER_00] Elon Musk has deserved every single penny.  (ang)
[7.60s] [SPEAKER_01] That is the most ludicrous image you have ever made.  (ang)
```

**Save to JSON, SRT, VTT, and TextGrid in one call:**

```python
doc = enrich_media(
    "interview.mp4",
    config=PipelineConfig(export_formats=["json", "srt", "vtt", "textgrid"]),
    output_dir="transcripts/",
)
# transcripts/interview.json   — full lossless transcript (schema v1.0)
# transcripts/interview.srt    — subtitles with speaker labels
# transcripts/interview.vtt    — WebVTT with voice tags
# transcripts/interview.TextGrid — Praat-compatible 4-tier annotation
```

**Speaker diarization** (requires HuggingFace token — see [setup](#speaker-diarization)):

```python
doc = enrich_media(
    "panel.mp4",
    config=PipelineConfig(diarization_backend="pyannote"),
)
for sp in doc.speakers:
    print(f"{sp.speaker_id}: {sp.speaking_time_s:.1f}s  dominant={sp.dominant_emotion}")
# SPEAKER_00: 33.2s  dominant=ang
# SPEAKER_01: 25.4s  dominant=hap
```

---

## Install variants

All variants require PyTorch to be installed first (Step 2 above).

| Install command | What you get | Notes |
|----------------|-------------|-------|
| `pip install speechtelemetry` | Library + types only — no ML, no CLI | For embedding or type-checking only |
| `pip install speechtelemetry[default]` | **Transcription + VAD + alignment + emotion + CLI** | ✅ Start here |
| `pip install speechtelemetry[all]` | Everything in `[default]` + diarization + prosody | Needs HF_TOKEN for diarization; parselmouth is GPL-3.0 |
| `pip install speechtelemetry[cuda]` | Same as `[default]`, optimised for CUDA | Install CUDA-enabled torch first |
| `pip install -e ".[dev]"` | Full dev environment (all extras + testing tools) | For contributors |

### What `[default]` includes

| Component | Package | License | Purpose |
|-----------|---------|---------|---------|
| ASR | `faster-whisper` | MIT | Speech-to-text with segment timestamps |
| VAD | `silero-vad` | MIT | Voice activity detection, silence spans |
| Alignment | `whisperx` | BSD-4-Clause | Word-level forced alignment |
| Emotion | `speechbrain` + `transformers` | Apache 2.0 | Emotion probability distribution |
| CLI | `typer` + `rich` | MIT | `speechtelemetry transcribe` command |

### Add-on extras (install on top of `[default]`)

```bash
# Speaker diarization (who spoke when)
pip install pyannote.audio>=3.1
# Then set HF_TOKEN — see Speaker Diarization section below

# Prosody (F0, energy, jitter, shimmer)
pip install praat-parselmouth>=0.4   # GPL-3.0 — check your license requirements
# or
pip install librosa                  # MIT (planned backend — coming soon)
```

---

## CLI

```bash
# Transcribe a file (default: JSON output in same directory)
speechtelemetry transcribe interview.mp4

# Multiple formats
speechtelemetry transcribe interview.mp4 --formats json,srt,vtt,textgrid

# With diarization (requires HF_TOKEN)
speechtelemetry transcribe panel.mp4 --diarize

# Explicit output path
speechtelemetry transcribe interview.mp4 --output out/transcript.json

# Use a smaller model for faster CPU transcription
speechtelemetry transcribe interview.mp4 --model small

# List available backends and environment status
speechtelemetry info
```

---

## Configuration

```python
from speechtelemetry import PipelineConfig

config = PipelineConfig(
    # Device
    device="auto",                    # "auto" → CUDA if available, else CPU
                                      # "cpu"  → force CPU
                                      # "cuda" → require GPU (fails if unavailable)

    # ASR
    asr_backend="faster-whisper",
    asr_model_size="large-v3",        # "tiny" | "small" | "medium" | "large-v3"
    asr_language=None,                # None = auto-detect; "en", "de", "fr", …

    # Diarization (disabled by default — requires HF_TOKEN)
    diarization_backend=None,         # None | "pyannote"

    # Prosody (disabled by default — opt-in because parselmouth is GPL-3.0)
    prosody_backend=[],               # [] | ["parselmouth"] | ["librosa"]

    # Emotion
    emotion_backend="speechbrain",    # "speechbrain" | None

    # Export
    export_formats=["json"],          # any of: "json", "srt", "vtt", "textgrid"
)
```

All fields have defaults. `PipelineConfig()` with zero arguments produces a working CPU pipeline.

---

## Speaker diarization

Speaker diarization is **opt-in** and requires:

1. A free HuggingFace account at [hf.co](https://huggingface.co)
2. License acceptance at [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
3. A read token from [hf.co/settings/tokens](https://hf.co/settings/tokens)
4. `pyannote.audio` installed: `pip install pyannote.audio>=3.1`

```bash
# Set your token
export HF_TOKEN="hf_your_token_here"          # Linux/macOS
$env:HF_TOKEN = "hf_your_token_here"          # PowerShell
```

```python
doc = enrich_media(
    "panel.mp4",
    config=PipelineConfig(diarization_backend="pyannote"),
)
```

The pyannote model library is MIT-licensed; its model weights are CC-BY-4.0. Check the model card before commercial use.

---

## Output schema

Every pipeline run returns a `TranscriptDocument` (schema `v1.0`):

```json
{
  "schema_version": "1.0",
  "source_path": "interview.mp4",
  "language": "en",
  "duration_s": 57.877,
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "speaking_time_s": 33.215,
      "turn_count": 20,
      "word_count": 118,
      "mean_segment_confidence": 0.511,
      "dominant_emotion": "ang",
      "emotion_distribution": {"neu": 0.15, "ang": 0.80, "hap": 0.05, "sad": 0.0}
    }
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 1.021,
      "text": "You're pointy.",
      "confidence": 0.682,
      "speaker": "SPEAKER_00",
      "words": [{"text": "You're", "start": 0.0, "end": 0.354, "confidence": 0.54, "alignment_backend": "whisperx"}],
      "emotion": {
        "label_distribution": {"neu": 1.0, "ang": 0.0, "hap": 0.0, "sad": 0.0},
        "confidence": 1.0,
        "backend_name": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
      }
    }
  ],
  "silence_spans": [...],
  "processing_report": {"real_time_factor": 4.743, "peak_ram_mb": 60.6, "errors": []}
}
```

Every float is rounded to 4 decimal places. Source paths use forward slashes. `speakers` is omitted when diarization is disabled.

---

## Pipeline

```
Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
```

Every stage is **fail-soft**: failures are recorded in `processing_report.errors` and the pipeline continues. Hard failures (missing FFmpeg, missing `HF_TOKEN`, CUDA unavailable when `device="cuda"`) are reported together before any compute starts.

---

## Custom backends

Register a custom backend without forking the library:

```python
from speechtelemetry import registry
from speechtelemetry.interfaces import EmotionBackend

class MyEmotionBackend(EmotionBackend):
    STAGE = "emotion"
    NAME  = "my-model"

    def predict_segment(self, wav_path, start_s, end_s):
        return {
            "label_distribution": {"positive": 0.8, "negative": 0.2},
            "confidence": 0.8,
            "backend_name": "my-model-v1",
        }

registry.register("emotion", "my-model", MyEmotionBackend)

from speechtelemetry import enrich_media, PipelineConfig
doc = enrich_media("audio.mp3", config=PipelineConfig(emotion_backend="my-model"))
```

Third-party packages can auto-register backends via entry points in their `pyproject.toml`:
```toml
[project.entry-points."speechtelemetry.backends"]
my-asr = "my_package:MyASRBackend"
```

---

## Architecture

```
types / interfaces / exceptions / config   ← domain core (zero ML imports)
              ↓
        backends / io                       ← one file per backend
              ↓
       core/pipeline                        ← stage orchestration, fail-soft
              ↓
       exporters / api                      ← JSON, SRT, VTT, TextGrid
              ↓
          cli / tests
```

Adding a backend = one new file in `backends/` + one line in `registry.py`. Nothing else changes. See [docs/architecture.md](docs/architecture.md).

---

## License

The core library is **MIT**. Optional backends have different licenses:

| Component | License | Notes |
|-----------|---------|-------|
| Core library | MIT | |
| faster-whisper | MIT | |
| silero-vad | MIT | |
| whisperx | BSD-4-Clause | Commercial use: check redistribution terms |
| speechbrain | Apache 2.0 | |
| pyannote.audio | MIT (library) + CC-BY-4.0 (weights) | Accept model license on HuggingFace |
| praat-parselmouth | **GPL-3.0** | Optional only — do not use in proprietary products without legal review |
| FFmpeg | LGPL 2.1 | Use the LGPL build (not GPL) |

See [docs/license_policy.md](docs/license_policy.md) for the full per-backend table and compliance guidance.

---

## Contributing

```bash
git clone https://github.com/wilek162/speechtelemetry.git
cd speechtelemetry
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # Linux/macOS
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
python -m pytest tests/unit/ -q       # must be green before any change
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/agent_playbook.md](docs/agent_playbook.md).

**Documentation index:**
- [docs/spec.md](docs/spec.md) — product intent and scope
- [docs/architecture.md](docs/architecture.md) — layer model and design decisions
- [docs/developer_reference.md](docs/developer_reference.md) — backend APIs, config reference, install caveats
- [docs/setup_windows.md](docs/setup_windows.md) — complete Windows 11 setup guide

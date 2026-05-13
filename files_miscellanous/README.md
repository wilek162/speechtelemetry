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
- **Prosody** — F0/pitch, energy, speech rate, voice quality per segment
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
    config=PipelineConfig(device="auto"),
)

for seg in result.segments:
    print(f"{seg.start:.2f}s  [{seg.speaker}]  {seg.text}")
    if seg.emotion:
        print(f"  emotion: {seg.emotion.label_distribution}")
```

---

## Installation

**Minimal (library + types only, no ML):**
```bash
pip install speechtelemetry
```

**Common use case (transcription + word alignment + VAD + CLI):**
```bash
pip install speechtelemetry[default]
```

**Full stack (includes diarization + prosody):**
```bash
pip install speechtelemetry[all]
```

**Development environment:**
```bash
pip install -e ".[dev]"
```

### Prerequisites

- **Python 3.10 or 3.11**
- **FFmpeg** — mandatory for media decode

  ```bash
  # Windows
  winget install --id=Gyan.FFmpeg -e

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

- **PyTorch** — install before ML backends

  ```bash
  # CPU-only
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

  # CUDA 12.4
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

See [docs/setup_windows.md](docs/setup_windows.md) for the complete Windows 11 PowerShell guide.

---

## Pipeline

```
Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
```

| Stage | Default backend | License |
|-------|----------------|---------|
| Decode | FFmpeg | LGPL 2.1 |
| VAD | silero-vad | MIT |
| ASR | faster-whisper | MIT |
| Alignment | whisperx | BSD-4-Clause |
| Diarization | pyannote.audio *(optional)* | MIT + CC-BY-4.0 |
| Prosody | praat-parselmouth *(optional)* | GPL-3.0 |
| Emotion | speechbrain | Apache 2.0 |
| Export | built-in | MIT |

---

## Configuration

```python
from speechtelemetry import PipelineConfig

config = PipelineConfig(
    device="auto",                    # auto | cpu | cuda
    asr_backend="faster-whisper",
    asr_model_size="large-v3",
    vad_backend="silero",
    alignment_backend="whisperx",
    diarization_backend=None,         # None disables diarization
    prosody_backend=["parselmouth"],
    emotion_backend="speechbrain",
    export_formats=["json", "srt"],
)
```

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

- `praat-parselmouth` — GPL-3.0 (optional adapter only; not a required default)
- `pyannote.audio` models — CC-BY-4.0 (requires HuggingFace account + license acceptance)
- FFmpeg — LGPL 2.1 (use the LGPL build; see [docs/license_policy.md](docs/license_policy.md))

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The [AI agent playbook](docs/agent_playbook.md) documents how both human and AI contributors should work inside the project.

# Changelog

All notable changes to speechtelemetry are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project scaffolding and architecture
- `TranscriptDocument` canonical data model (`types.py`)
- `PipelineConfig` Pydantic v2 configuration model (`config.py`)
- Backend interface ABCs (`interfaces.py`)
- Backend registry with lazy imports and entry-point auto-discovery (`registry.py`)
- FFmpeg media decode stage (`io/ffmpeg.py`)
- Silero VAD backend (`backends/vad/silero.py`)
- faster-whisper ASR backend (`backends/asr/faster_whisper.py`)
- WhisperX alignment backend (`backends/alignment/whisperx.py`)
- pyannote.audio diarization backend (`backends/diarization/pyannote.py`)
- Parselmouth prosody backend (`backends/prosody/parselmouth.py`)
- SpeechBrain emotion backend (`backends/emotion/speechbrain.py`)
- JSON, SRT, VTT, TextGrid exporters
- CLI via Typer (`speechtelemetry transcribe`, `speechtelemetry info`)
- Unit and integration test scaffolding
- GitHub Actions CI workflow
- `docs/spec.md` — product intent, scope, and baseline behavior
- `docs/architecture.md` — layer model, extension model, design decisions
- `docs/developer_reference.md` — backend APIs, install commands, configuration reference
- `docs/setup_windows.md` — complete Windows 11 PowerShell setup guide

### Changed
- `PipelineConfig.device` now accepts only `"cpu"` or `"cuda"`. The `"auto"` value has been removed. GPU is an explicit opt-in; `"cpu"` is the default.
- `PipelineConfig.prosody_backend` defaults to `[]` (empty). Prosody is now opt-in. Previously defaulted to `["parselmouth"]`, which is GPL-3.0.
- `pyproject.toml`: added `[cuda]` optional extra documenting the explicit GPU install path.
- Pre-flight check error message for `device="cuda"` updated to include the correct `pip install` command.

---

## [0.1.0] — TBD

_First public release._

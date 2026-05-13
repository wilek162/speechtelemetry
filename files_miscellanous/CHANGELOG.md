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
- Full documentation: architecture, backend reference, Windows setup, license policy

---

## [0.1.0] — TBD

_First public release._

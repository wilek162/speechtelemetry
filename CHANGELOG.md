# Changelog

All notable changes to speechtelemetry are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- **GPL violation**: CLI was enabling parselmouth (GPL-3.0) by default via `--no-prosody` inversion logic; `prosody_backend` now correctly defaults to `[]` in all paths (spec §7, ADL-004)
- **Incomplete export report**: JSON exports now include final `real_time_factor`, `peak_ram_mb`, and `peak_vram_mb`; report finalization moved before export stage
- **faster-whisper debug log**: Removed `... and 0` expression that always logged `RTF=0.00`
- **Parselmouth `_sound_cache`**: Moved from class-level (shared across instances) to instance-level, preventing cross-instance cache pollution
- **Unused `type: ignore` comments**: Removed stale suppression comments in `pipeline.py` and `cli/main.py`

### Changed
- `registry.py`: Replaced deprecated `typing.Type` with builtin `type` (UP006/UP035)
- Type annotations throughout backends, `interfaces.py`, `api.py`, `exporters/json_exporter.py`: added missing generic type arguments (`dict[str, Any]`, `list[dict[str, Any]]`) to satisfy `mypy --strict`
- `cli/main.py`: B008 `# noqa` comments moved to opening lines of multi-line `typer.Option`/`typer.Argument` calls
- `cli/main.py`: Added `# type: ignore[arg-type]` for CLI string → `Literal[...]` field assignments that Pydantic validates at runtime

### Tests
- `test_preflight.py`: Replaced 6 empty `pass`-body tests with real import assertions covering types, interfaces, exceptions, exporters, I/O, and core modules

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
- `_attach_silence_gaps()` in `core/pipeline.py` — populates `Segment.silence_before_ms` and `silence_after_ms` from computed silence spans (G4)
- `provenance` field on `TranscriptDocument` — wired `PipelineProvenance` records VAD, ASR, and alignment backend names and model IDs per run (G8)
- `tests/fixtures/mock_backends.py` — deterministic ML-free mock backends for unit testing
- `tests/fixtures/sample_16k_mono.wav` — deterministic 3 s 16 kHz mono sine wave fixture
- `ASRBackend.transcribe()` now accepts optional `chunk_size_s` parameter; pipeline passes it when `chunk_audio=True` (G10)
- Unit tests for `audio_normalize`, `Job`, pipeline orchestration, and CLI

### Fixed
- `Job.decode()` now calls `validate_wav()` immediately after `normalize_to_wav()`, ensuring invalid WAV files are rejected before any ML backend runs (G1)
- `PipelineConfig.device` restored to accept `"auto"`, `"cpu"`, and `"cuda"`. Default is `"auto"` (CUDA if available, else CPU). Resolution is delegated to the backend layer, not the CLI (G2).

### Changed
- `PipelineConfig.prosody_backend` defaults to `[]` (empty). Prosody is now opt-in. Previously defaulted to `["parselmouth"]`, which is GPL-3.0.
- `pyproject.toml`: added `[cuda]` optional extra documenting the explicit GPU install path.

---

## [0.1.0] — TBD

_First public release._

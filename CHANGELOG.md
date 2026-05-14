# Changelog

All notable changes to speechtelemetry are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Debug logging throughout the pipeline**: Each stage now emits `DEBUG`-level log lines at entry and exit with key metrics (elapsed time, item counts, config params used). Stages covered: decode, VAD, ASR, alignment, diarization, prosody (per-segment), emotion (per-segment + scored/total summary). Enable with `SPEECHTELEMETRY_LOG_LEVEL=DEBUG` or `--log-cli-level=DEBUG` in pytest.
- **`pyproject.toml`**: Added `log_level = "DEBUG"` (caplog capture level) and `log_cli_level = "WARNING"` (terminal log threshold) to `[tool.pytest.ini_options]` — debug logs are captured in `caplog` fixtures and visible when running `pytest --log-cli-level=DEBUG`.

### Fixed
- **`pyproject.toml` ruff exclude**: Added `files_miscellanous` to `[tool.ruff] exclude` — reference copy directory was failing pre-commit with stale B027/B008/SIM102 violations.
- **`docs/agent_playbook.md` step 9**: Added mandatory `pre-commit run --all-files` step before every commit; added `pre-commit run --all-files` exits 0 to PR checklist.
- **`pyproject.toml` entry point**: `speechtelemetry.cli.main:app` → `speechtelemetry.cli.main:main` — `app` is `None` when typer is not installed, causing `TypeError` on invocation; `main()` handles the missing-typer case correctly.
- **`pyproject.toml` dev deps**: Removed `black>=24`; ruff-format is the sole formatter. Removed `[tool.black]` config section.
- **`.github/workflows/ci.yml`**: Removed `black` from lint job pip install — it was installed but never invoked.
- **`docs/developer_reference.md` checklist**: `black --check src/ tests/` → `ruff format --check src/ tests/` (doc §7, quality gates).
- **`docs/agent_playbook.md` PR checklist**: `black --check` → `ruff format --check` (PR checklist entry).
- **`.gitignore`**: Uncommented `mock_data/`, replaced `mock_output/*.{json,srt,vtt,TextGrid}` pattern fragments with `mock_output/` (full directory), and added `wav2vec2_checkpoints/` — all three were showing as untracked in git status.
- **Pipeline config not wired to backend constructors**: `run_pipeline()` was calling `get_backend(stage, name)` with no kwargs for every stage, silently discarding `asr_model_size`, `device`, `compute_type`, `vad_threshold`, `vad_min_silence_ms`, `vad_min_speech_ms` from `PipelineConfig`. Each stage now passes its relevant config settings to the backend constructor.
- **`SpeechBrainEmotionBackend.predict_segment()` failed on Windows**: `classify_file()` routed audio through `torchaudio.load()` which fails on Windows temp paths. Replaced with `classify_batch(tensor)` using a torch tensor directly — no temp file, no path handling, no torchaudio involvement.
- **SpeechBrain 1.x `LazyModule` masking real exceptions**: `LazyModule.__getattr__` raises `ImportError` when CPython's `inspect.getmodule()` calls `hasattr(module, '__file__')` during stack-frame inspection. This masked real backend errors (alignment, emotion) with unrelated `k2`/`flair` not-installed errors. Fixed by patching `LazyModule.__getattr__` in `core/pipeline.py` to return `None` for pure metadata attributes instead of triggering the full import.
- **`Segment.confidence` always 0.0 after alignment**: `_build_segments()` fell back to `avg_logprob` (ASR log-probability, negative) then to 0.0. After WhisperX alignment, `avg_logprob` is absent from aligned segments. Replaced the one-liner with `_segment_confidence()`: priority is (1) explicit `confidence` key, (2) `math.exp(avg_logprob)` → [0,1], (3) mean word alignment score.
- **`ffmpeg.py`**: `-loglevel error` flag moved before `-i` input; it is a global FFmpeg option and must not follow the output path.
- **`registry.py`**: `_CLASS_CACHE` dict declaration moved before `register()`, which references it — eliminates forward-reference code smell.
- **`srt.py`**: Removed `SRTExporter.DROPPED_FIELDS` class attribute — it was never read and documented nothing that the class name didn't already convey.
- **`.pre-commit-config.yaml`**: Removed redundant `black` hook; `ruff-format` is the sole formatter. Added `detect-private-key`, `check-ast`, and secret-pattern hooks for OpenAI keys and `os.environ` secret assignments. Tightened `check-added-large-files` to 500 KB.
- **`docs/agent_playbook.md`**: Step 8 lint command corrected from `black src/ tests/` to `ruff format --check src/ tests/` — the pre-commit config uses `ruff-format` as the sole formatter; `black` is not a pre-commit hook.
- **GPL violation**: CLI was enabling parselmouth (GPL-3.0) by default via `--no-prosody` inversion logic; `prosody_backend` now correctly defaults to `[]` in all paths (spec §7, ADL-004).
- **Incomplete export report**: JSON exports now include final `real_time_factor`, `peak_ram_mb`, and `peak_vram_mb`; report finalization moved before export stage.
- **faster-whisper debug log**: Removed `... and 0` expression that always logged `RTF=0.00`.
- **Parselmouth `_sound_cache`**: Moved from class-level (shared across instances) to instance-level, preventing cross-instance cache pollution.
- **Unused `type: ignore` comments**: Removed stale suppression comments in `pipeline.py` and `cli/main.py`.

### Changed
- `registry.py`: Replaced deprecated `typing.Type` with builtin `type` (UP006/UP035)
- Type annotations throughout backends, `interfaces.py`, `api.py`, `exporters/json_exporter.py`: added missing generic type arguments (`dict[str, Any]`, `list[dict[str, Any]]`) to satisfy `mypy --strict`
- `cli/main.py`: B008 `# noqa` comments moved to opening lines of multi-line `typer.Option`/`typer.Argument` calls
- `cli/main.py`: Added `# type: ignore[arg-type]` for CLI string → `Literal[...]` field assignments that Pydantic validates at runtime

### Tests
- `test_preflight.py`: Replaced 6 empty `pass`-body tests with real import assertions covering types, interfaces, exceptions, exporters, I/O, and core modules

### Added
- `.github/workflows/ci.yml` — GitHub Actions CI: lint (ruff + mypy), unit tests on Python 3.10 + 3.11 with coverage, secret scan; runs on push/PR to master
- `tests/integration/test_cli_e2e.py` — 6 E2E integration tests against `mock_data/BLOOD_liquidity_Captions_V2.mp4` using real FFmpeg decode + mock ML backends; `test_mock_output_all_formats` writes all four formats to `mock_output/`; marked `@pytest.mark.slow`, auto-skipped when media or FFmpeg absent
- `run_pipeline()` and `enrich_media()` now accept `output_dir: str | Path | None` — fixes the dead Stage 9 export path; multi-format export to a directory now works
- `enrich_audio()` gains `output_dir` parameter matching `enrich_media` for API symmetry
- `tests/integration/test_real_pipeline.py` — 4 integration tests using real ML backends (faster-whisper `small`, Silero VAD, WhisperX alignment, SpeechBrain emotion) on the actual MP4; all 5 pipeline stages produce real output; writes to `mock_output/`; RTF ~0.75 on CPU
- 4 new unit tests verifying that pipeline passes config kwargs (model size, device, VAD thresholds) to backend constructors
- 2 new unit tests verifying `_segment_confidence()` converts `avg_logprob` via `math.exp()` and falls back to mean word score after alignment
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

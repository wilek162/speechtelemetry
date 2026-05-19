# Changelog

All notable changes to speechtelemetry are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- **Emotion and prosody stages never recorded in provenance**: `_attach_emotion()` and `_attach_prosody()` ran successfully and populated `Segment.emotion` / `Segment.prosody`, but never called `provenance.record()`. Both helpers now accept an optional `provenance: PipelineProvenance | None` parameter; `run_pipeline()` passes the active `provenance` object so all six pipeline stages are correctly recorded in `TranscriptDocument.provenance.stages`.
- **MockEmotionBackend used wrong emotion labels**: `tests/fixtures/mock_backends.py` returned `{"neutral": 0.8, "happy": 0.2}` — full-word labels not matching the real SpeechBrain IEMOCAP model output (`neu`, `ang`, `hap`, `sad`). Updated to return all four abbreviated IEMOCAP labels summing to 1.0, matching the real model's output format so mock-based tests accurately reflect production output structure.
- **`speechtelemetry info` crashes on Windows cp1252 terminals**: The Unicode check mark `✓` (U+2713) and cross `✗` (U+2717) in the FFmpeg status line cannot be encoded by the Windows legacy cp1252 codepage, raising `UnicodeEncodeError` at runtime. Replaced with ASCII-safe Rich markup `[green]found[/green]` / `[red]NOT FOUND[/red]`.

### Added
- **Pipeline invariant test suite** (`tests/unit/test_pipeline_invariants.py`): 36 invariant tests that run the full pipeline with mock backends on the real `sample_16k_mono.wav` fixture and assert: TranscriptDocument type/structure, duration positive+finite, segment ordering, timestamps in bounds, confidence in [0,1], word timestamps valid, alignment_backend set, emotion None when disabled, emotion distribution sums to 1.0, emotion labels match IEMOCAP set, provenance records VAD/ASR/emotion, emotion absent from provenance when disabled, silence span invariants.
- **Output quality validation tests** (`tests/unit/test_output_quality.py`): 21 tests that read the real pipeline output JSON files from `mock_output/` (skipped when absent) and assert BLOOD + ABSOLUTELYNOT outputs meet structural and content quality standards: emotion IEMOCAP labels, distribution sums, confidence ranges, timestamp ordering, speaker label format, coverage thresholds, provenance completeness, RTF validity.
- **Golden output structural test (B7)**: `tests/unit/test_golden_output.py` — 10 unit tests that run `enrich_audio()` on `tests/fixtures/sample_16k_mono.wav` with ML-free mock backends and assert: duration within 5% of 3.0s, segments list, zero pipeline errors, stage timings populated, silence spans present, emotion scores on eligible segments, full probability distribution in `EmotionScore.label_distribution`, JSON round-trip with all canonical fields, and per-segment structural schema.
- **Planned backends documented (B6)**: `docs/developer_reference.md` now includes a "Planned backends" section (§3.9) listing all 11 registry stubs with target install commands and license notes. `registry.py` now has `# future` comments on each unimplemented stub so they are clearly distinguished from implemented backends.
- **Diarization integration test now includes emotion**: `test_diarization_speaker_detection.py` updated to run `emotion_backend="speechbrain"` so the diarization mock output (`mock_output/diarization/ABSOLUTELYNOT_Sequence02.*`) includes real emotion scores per segment.
- **Multi-speaker diarization end-to-end**: New integration test `tests/integration/test_diarization_speaker_detection.py` validates the full pipeline on `ABSOLUTELYNOT_Sequence02.mp4` with pyannote.audio — asserts ≥2 distinct speakers, ≥70% coverage, SPEAKER_NN label format, diarization provenance, and all four export formats containing speaker information.
- **Pre-commit ruff scope fix**: Added `exclude: ^files_miscellanous/` to both ruff hooks in `.pre-commit-config.yaml` so draft files in that directory are not scanned, consistent with the `exclude` already set in `pyproject.toml`.

### Fixed
- **Diarization provenance not recorded**: `run_pipeline()` called `provenance.record()` for VAD, ASR, and alignment but never for diarization. Added `provenance.record("diarization", config.diarization_backend)` after a successful diarization run.
- **Speaker assignment used segment midpoint only**: `_build_segments()` assigned speakers by checking whether the segment midpoint fell within a diarization turn; if the midpoint was outside every turn (e.g. near a turn boundary), the segment got `speaker=None`. Replaced with a `_assign_speaker()` helper that picks the turn with **maximum overlap** with the segment — segments straddling a turn boundary now receive the correct speaker label.
- **pyannote 3.x / huggingface_hub ≥ 0.23 incompatibility**: pyannote 3.x passes `use_auth_token=` to `hf_hub_download()` throughout its codebase, but huggingface_hub ≥ 0.23 renamed the parameter to `token`. Applied a compatibility patch in `backends/diarization/pyannote.py` that patches the `hf_hub_download` local reference in every relevant pyannote module at import time.
- **pyannote 3.x / PyTorch 2.6+ weights_only incompatibility**: PyTorch 2.6+ changed `torch.load` default to `weights_only=True`; pyannote checkpoint files contain non-tensor objects (`TorchVersion`, `Specifications`, etc.) that are rejected. Applied a compatibility patch that defaults `weights_only=False` for `pl_load` calls originating from pyannote/pytorch_lightning model loading.
- **SpeechBrain LazyModule masking diarization errors**: The existing `_patch_speechbrain_lazy_modules()` fix in `core/pipeline.py` did not cover code paths where the pyannote backend was loaded in isolation (e.g. integration tests that import `PyannoteBackend` directly). Applied the same LazyModule patch inside `_patch_pyannote_compat()` so it runs whenever the diarization backend is imported.
- **pyannote gated model fallback**: Both recommended pyannote models (`speaker-diarization-community-1` and `speaker-diarization-3.1`) require explicit HuggingFace license acceptance. Added automatic fallback to `tensorlake/speaker-diarization-3.1` (MIT-licensed community mirror, no gating required) so diarization works out of the box without requiring the user to visit the HF model page first.

### Tests
- **Unit `test_build_segments_assigns_speaker_by_highest_overlap`**: Verifies that when two turns overlap a segment, the one with the largest overlap wins.
- **Unit `test_build_segments_assigns_speaker_even_when_midpoint_outside_turn`**: Verifies overlap-based assignment works when the segment midpoint is strictly outside the winning turn.
- **Unit `test_build_segments_returns_none_speaker_when_segment_outside_all_turns`**: Verifies speaker is `None` when the segment has zero overlap with every diarization turn.
- **Unit `test_pipeline_records_diarization_in_provenance`**: Verifies `doc.provenance.stages` contains a diarization entry after a successful diarization run.

### Fixed
- **`SRTExporter`/`VTTExporter` 1ms truncation**: `_format_srt_time` and `_format_vtt_time` used `int((seconds % 1) * 1000)` — floating-point `10.18 * 1000 = 10179.999...` was truncated to 179ms, producing `10,179` instead of `10,180`. Fixed by computing `round(seconds * 1000)` first and using `divmod` on integer milliseconds throughout. Tests added for the `10.18` edge case.
- **Mock e2e test overwrote real pipeline output**: `test_mock_output_all_formats` in `test_cli_e2e.py` wrote to `mock_output/` — the same directory as `test_real_pipeline_writes_all_formats_to_mock_output`. Running all integration tests caused the mock data ("hello world", "test audio") to overwrite the real transcript, making it appear the pipeline hadn't processed the actual MP4. Fixed: mock e2e test now writes to `mock_output/cli_e2e/` so the two outputs are permanently separate.
- **`Word.alignment_backend` always `None`**: `_build_segments()` never set `Word.alignment_backend` even when WhisperX ran. Added `alignment_backend` parameter to `_build_segments()`; pipeline passes the backend name when alignment succeeds and `None` when it was skipped or failed. JSON output now shows `"alignment_backend": "whisperx"` on every word.
- **`PipelineProvenance.record()` couldn't capture `compute_type`**: `BackendProvenance.compute_type` existed but `record()` had no parameter for it, so it was always `None`. Added `compute_type` parameter to `record()`; pipeline now passes `config.asr_compute_type` when recording ASR provenance. Verified in JSON output: `"compute_type": "int8"` appears on the ASR provenance entry.
- **Dead `savedir` parameter in `SpeechBrainEmotionBackend.__init__`**: The `savedir` default argument was immediately overridden by the `SPEECHTELEMETRY_CACHE_DIR` env-var logic — the parameter was dead code and misleading. Removed; the cache directory is now computed purely from the env var.
- **CLI silently dropped configured `export_formats`**: `transcribe` set `export_formats` in config but never passed `output_dir` to `enrich_media()`, so all formats except an explicit `--output` JSON were silently discarded. Fixed: when `--output` is not given the CLI now passes `output_dir=str(input_file.parent)` so all configured formats are written to the input file's directory (matching the `--output` help text "default: same dir as input").

### Tests
- **Integration `test_real_pipeline_report_has_timings`**: Added `alignment` timing assertion and `real_time_factor > 0` assertion alongside the existing decode/vad/asr checks.
- **Integration `test_real_pipeline_writes_all_formats_to_mock_output`**: Added assertions for word-level timestamps (`alignment_backend == "whisperx"`, `start`/`end` validity), silence spans (`duration_ms > 0`), and provenance content (vad/asr/alignment stages, `model_id`, `backend_name`).
- **Integration `test_enrich_media_report_has_stage_timings`**: Added `alignment` timing assertion.
- **Unit `test_build_segments_sets_alignment_backend_on_words`**: Verifies `Word.alignment_backend` is set when `alignment_backend` is passed to `_build_segments`.
- **Unit `test_build_segments_alignment_backend_none_by_default`**: Verifies `Word.alignment_backend` is `None` when alignment did not run.
- **Unit `test_pipeline_continues_when_alignment_fails`**: Verifies fail-soft: pipeline produces segments and records a `StageError` when the alignment backend raises.
- **Unit `test_pipeline_continues_when_emotion_fails`**: Verifies fail-soft: segments get `emotion=None` and `ProcessingReport.errors` is populated when per-segment emotion prediction raises.
- **Unit `test_pipeline_records_compute_type_in_asr_provenance`**: Verifies `compute_type` is captured in provenance when set in config.

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

## [0.1.0] — 2026-05-19

_First public release._

### Summary

- Full pipeline: Decode → Normalize → VAD → ASR → Align → Diarize → Prosody → Emotion → Export
- Six default ML backends: Silero VAD, faster-whisper ASR, WhisperX alignment, pyannote diarization, parselmouth prosody, SpeechBrain emotion
- Four export formats: JSON, SRT, VTT, TextGrid
- CLI: `speechtelemetry transcribe`, `speechtelemetry info`
- 189 unit tests, 73.71% coverage, all quality gates green

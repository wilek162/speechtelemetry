# Required Pipeline Completion — Design Spec v0.1

**Date:** 2026-05-13
**Scope:** Required stages only (Decode → Normalize → VAD → ASR → Align → Emotion → Export).
Diarization and Prosody are excluded — they remain optional, untouched stubs.

---

## 1. Goal

Bring the required pipeline from a scaffolded-but-gapped state to a production-ready,
integration-tested, lint-clean baseline that satisfies every criterion in the playbook PR
checklist. No structural changes. Gaps within the existing 5-layer architecture are filled.

---

## 2. Gap Inventory

| ID | Location | Description | Resolution |
|----|----------|-------------|------------|
| G1 | `core/job.py` | `Job.decode()` calls `normalize_to_wav()` but never calls `validate_wav()` | Call `validate_wav()` immediately after `normalize_to_wav()`; wrap `ValueError` into `EnvironmentCheckError` |
| G2 | `cli/main.py` | Default `device="auto"` is not a valid `PipelineConfig` value (`"cpu"` \| `"cuda"` only) | Detect GPU availability in CLI and translate to the correct literal before building `PipelineConfig` |
| G3 | `pyproject.toml` | CLI entry point is `speechtelemetry.cli.main:app`; `app = None` when typer absent | Change entry point to `speechtelemetry.cli.main:main` |
| G4 | `core/pipeline.py` | `Segment.silence_before_ms` and `silence_after_ms` are defined in `types.py` but never populated | Compute from `silence_spans` after `_build_segments()` and attach to each `Segment` |
| G5 | `tests/fixtures/` | `mock_backends.py` referenced in architecture doc but absent; pipeline orchestration tests cannot run without ML | Create `MockVADBackend`, `MockASRBackend`, `MockAlignmentBackend`, `MockEmotionBackend` |
| G6 | `tests/fixtures/` | `sample_16k_mono.wav` absent; smoke test always skipped | Generate deterministic 3 s 16 kHz mono sine-wave WAV using numpy + soundfile; commit to repo |
| G7 | `tests/unit/` | `test_job.py`, `test_pipeline_orchestration.py`, `test_cli.py`, `test_audio_normalize.py` absent | Write these before touching any implementation (TDD) |
| G8 | `provenance.py` | `BackendProvenance` / `PipelineProvenance` defined but not wired into `TranscriptDocument` or `pipeline.py` | Defer to v0.2; add `# TODO v0.2: wire PipelineProvenance into TranscriptDocument` in pipeline.py |
| G9 | repo root | `CHANGELOG.md` and `README.md` absent (referenced in pyproject.toml) | Create both |
| G10 | `core/pipeline.py` | `chunk_audio` / `max_chunk_duration_s` config fields not used | Document as v0.2 with a comment; backends manage their own memory in v0.1 |

---

## 3. Architecture

No layer boundaries change. The dependency order is unchanged:

```
types / interfaces / exceptions / config   (Layer 1 — domain core)
         ↓
    backends / io                          (Layer 2 — adapters)
         ↓
    core/pipeline + core/job               (Layer 3 — orchestration)
         ↓
    exporters / api                        (Layer 4 — exporters + public API)
         ↓
    cli / tests                            (Layer 5 — presentation + verification)
```

---

## 4. Data Flow (post-fix)

```
Input file
  └─ Job.__enter__()             creates temp dir
  └─ normalize_to_wav()          Stage 1+2: FFmpeg → mono 16 kHz 16-bit PCM WAV
  └─ validate_wav() [G1]         Stage 2 guard: assert 16 kHz, mono; raises on mismatch
  └─ SileroVADBackend            Stage 3: speech_intervals + silence_spans
  └─ FasterWhisperBackend        Stage 4: raw_segments + detected_language
  └─ WhisperXAlignmentBackend    Stage 5: word-level timestamps on segments
  └─ _build_segments()           assemble Segment objects
  └─ _attach_silence_gaps() [G4] populate silence_before_ms / silence_after_ms
  └─ SpeechBrainEmotionBackend   Stage 8: EmotionScore per segment
  └─ JsonExporter (+ SRT/VTT/TextGrid if configured)   Stage 9
  └─ TranscriptDocument returned
  └─ Job.__exit__()              temp dir cleanup; GPU memory release
```

Each stage is wrapped in `try/except`. Failures write to `ProcessingReport.errors`; the
pipeline never hard-crashes on optional stages.

Hard failures (raised before compute):
- `normalize_to_wav()` fails → `RuntimeError` surfaces as preflight failure
- `validate_wav()` fails → `ValueError` wrapped into `EnvironmentCheckError`
- Backend not importable → `BackendNotAvailableError` from preflight check

---

## 5. Test Design (TDD — written before implementation)

### 5.1 New unit test files

#### `tests/unit/test_audio_normalize.py`
| Test | Expected outcome |
|------|-----------------|
| `test_validate_wav_passes_valid_file` | `validate_wav()` returns info dict for a canonical 16 kHz mono WAV |
| `test_validate_wav_rejects_wrong_rate` | `ValueError` raised when sample rate ≠ 16 000 |
| `test_validate_wav_rejects_stereo` | `ValueError` raised when channels ≠ 1 |
| `test_iter_chunks_exact_division` | 6 s WAV with chunk=2 s yields 3 chunks: (0,2), (2,4), (4,6) |
| `test_iter_chunks_remainder` | 5 s WAV with chunk=2 s yields (0,2), (2,4), (4,5) |
| `test_iter_chunks_shorter_than_chunk` | 1 s WAV with chunk=2 s yields single (0,1) |

#### `tests/unit/test_job.py`
| Test | Expected outcome |
|------|-----------------|
| `test_job_creates_temp_dir` | `_temp_dir` is set after `__enter__` |
| `test_job_cleans_up_on_exit` | temp dir absent after `__exit__` |
| `test_job_cleans_up_on_exception` | temp dir absent even when pipeline raises inside `with Job()` |
| `test_temp_wav_path_format` | `job.temp_wav` ends with `audio.wav` inside temp dir |
| `test_output_path_extensions` | correct `.json`, `.srt`, `.vtt`, `.TextGrid` for each format |
| `test_decode_calls_validate_wav` | monkeypatching `normalize_to_wav` + `validate_wav`; asserts `validate_wav` called with result of `normalize_to_wav` |
| `test_decode_wraps_validate_error` | `validate_wav` raises `ValueError` → `Job.decode()` re-raises as `EnvironmentCheckError` |

#### `tests/fixtures/mock_backends.py`
Deterministic, ML-free stub backends:

```python
# Expected interface for each mock:
MockVADBackend.get_speech_intervals(wav_path)
  → [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]

MockASRBackend.transcribe(wav_path, language, beam_size)
  → ([{"start": 0.0, "end": 1.0, "text": "hello world", "avg_logprob": -0.1}], MockInfo("en"))

MockAlignmentBackend.align(segments, audio_path, language)
  → segments each with "words": [{"word": "hello", "start": 0.0, "end": 0.5, "score": 0.99}]

MockEmotionBackend.predict_segment(wav_path, start_s, end_s)
  → {"label_distribution": {"neutral": 0.7, "happy": 0.3}, "confidence": 0.7, "backend_name": "mock"}
```

#### `tests/unit/test_pipeline_orchestration.py`
Uses mock backends registered via `registry.register()` before each test.

| Test | Expected outcome |
|------|-----------------|
| `test_full_pipeline_happy_path` | Returns `TranscriptDocument` with segments, no errors in report |
| `test_vad_failure_falls_back_to_full_audio` | VAD mock raises; speech_intervals = full audio; pipeline completes; 1 error in report |
| `test_asr_failure_returns_empty_segments` | ASR mock raises; `segments == []`; 1 error in report |
| `test_alignment_failure_retains_segments` | Alignment mock raises; segments present without `words`; 1 error in report |
| `test_emotion_failure_retains_segments` | Emotion mock raises; segments present without `emotion`; 1 error in report |
| `test_silence_fields_populated` | `Segment.silence_before_ms` and `silence_after_ms` set based on VAD output |
| `test_stage_timings_recorded` | `processing_report.stage_timings` contains keys for vad, asr, alignment, emotion |
| `test_preflight_catches_missing_ffmpeg` | monkeypatch `shutil.which("ffmpeg")` → None; `EnvironmentCheckError` raised |
| `test_skip_decode_flag` | `run_pipeline(..., skip_decode=True)` skips `Job.decode()` |

#### `tests/unit/test_cli.py`
| Test | Expected outcome |
|------|-----------------|
| `test_device_auto_maps_to_cpu_without_cuda` | monkeypatch `torch.cuda.is_available()` → False; config built with `device="cpu"` |
| `test_device_auto_maps_to_cuda_with_cuda` | monkeypatch → True; config built with `device="cuda"` |
| `test_formats_parsed_correctly` | `--formats json,srt` produces `export_formats=["json", "srt"]` |
| `test_missing_file_exits_1` | non-existent input file; typer exits with code 1 |
| `test_cli_imports_without_typer` | mock `import typer` as ImportError; `main()` prints install hint and exits 1 |

### 5.2 Integration fixture (no ML)

`tests/fixtures/generate_fixture.py`: script that writes a 3 s, 16 kHz, mono, 16-bit PCM WAV
containing a 440 Hz sine wave using `numpy` + `soundfile`. The resulting
`tests/fixtures/sample_16k_mono.wav` is committed to the repository.

Fixture is valid input for the smoke test and for `validate_wav()` tests.

### 5.3 Integration smoke test (existing, now completable)

`tests/integration/test_pipeline_smoke.py` — already written. Runs once:
- `sample_16k_mono.wav` exists
- ML backends installed

Gated by `@pytest.mark.slow`.

---

## 6. Implementation Order

Following the playbook: tests first, then fixes. Each numbered item is a discrete PR-worthy task.

```
Step 1.  Write tests/fixtures/mock_backends.py
Step 2.  Write tests/unit/test_audio_normalize.py     → run → all fail
Step 3.  Write tests/unit/test_job.py                 → run → all fail
Step 4.  Write tests/unit/test_pipeline_orchestration.py → run → all fail
Step 5.  Write tests/unit/test_cli.py                 → run → all fail
Step 6.  Fix G1: wire validate_wav() into Job.decode()  → test_job tests pass
Step 7.  Fix G4: populate silence_before/after_ms       → orchestration tests pass
Step 8.  Fix G2: CLI device="auto" handling             → test_cli tests pass
Step 9.  Fix G3: pyproject.toml CLI entry point         → verified manually
Step 10. Add G8 TODO comment in pipeline.py
Step 11. Add G10 TODO comment in pipeline.py
Step 12. Write tests/fixtures/generate_fixture.py + commit sample WAV
Step 13. Add CHANGELOG.md and README.md
Step 14. Run: pytest tests/unit/ -q                     → 0 failures
Step 15. Run: ruff check src/ tests/
Step 16. Run: black --check src/ tests/
Step 17. Run: mypy src/
Step 18. Fix any lint/type errors
Step 19. Update CHANGELOG.md [Unreleased] section
Step 20. Commit
```

---

## 7. Acceptance Criteria

| # | Criterion | Command / check |
|---|-----------|-----------------|
| AC1 | All pre-existing 69 unit tests pass | `pytest tests/unit/ -q` |
| AC2 | All new unit tests pass | `pytest tests/unit/ -q` |
| AC3 | `validate_wav()` called in `Job.decode()` | `test_job.py::test_decode_calls_validate_wav` green |
| AC4 | Bad WAV raises `EnvironmentCheckError` | `test_job.py::test_decode_wraps_validate_error` green |
| AC5 | CLI `device="auto"` resolves without PipelineConfig error | `test_cli.py::test_device_auto_maps_to_cpu_without_cuda` green |
| AC6 | `silence_before_ms` / `silence_after_ms` populated | `test_pipeline_orchestration.py::test_silence_fields_populated` green |
| AC7 | CLI entry point is `main` | `pyproject.toml` inspection |
| AC8 | `sample_16k_mono.wav` committed and valid | `validate_wav()` passes on it |
| AC9 | `ruff check src/ tests/` exits 0 | CI gate |
| AC10 | `black --check src/ tests/` exits 0 | CI gate |
| AC11 | `mypy src/` exits 0 | CI gate |
| AC12 | `CHANGELOG.md` exists with `[Unreleased]` section | File present |
| AC13 | `README.md` exists | File present |

---

## 8. Out of Scope

- Speaker diarization (Stage 6) — remains an optional stub
- Prosody extraction (Stage 7) — remains an optional stub
- `PipelineProvenance` wiring — deferred to v0.2
- Chunked audio processing — deferred to v0.2; config fields retained for forward compatibility
- Alternative backends (whisper.cpp, sensevoice, funasr, forcealign, librosa, emotion2vec, emobox) — not in required pipeline
- Streaming / real-time transcription — explicitly out of scope per spec.md

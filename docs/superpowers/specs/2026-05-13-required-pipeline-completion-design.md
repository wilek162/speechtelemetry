# Required Pipeline Completion — Design Spec v0.1 (Updated)

**Date:** 2026-05-13
**Scope:** Required stages only (Decode → Normalize → VAD → ASR → Align → Emotion → Export).
Diarization and Prosody are excluded — they remain optional, untouched stubs.

---

## 1. Goal

Bring the required pipeline from a scaffolded-but-gapped state to a production-ready,
integration-tested, lint-clean baseline that satisfies every criterion in the playbook PR
checklist.

No structural changes. Gaps within the existing 5-layer architecture are filled.

---

## 2. Gap Inventory

| ID  | Location                         | Description                                                          | Resolution                                                                                                            |
| --- | -------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| G1  | `core/job.py`                    | `Job.decode()` calls `normalize_to_wav()` but never validates output | Call `validate_wav()` immediately after `normalize_to_wav()`; wrap errors consistently using existing exception types |
| G2  | `cli/main.py`                    | CLI attempts to resolve `device="auto"` manually                     | Do NOT resolve in CLI. Pass `"auto"` through to `PipelineConfig`; backend layer handles resolution                    |
| G3  | `cli/main.py` / `pyproject.toml` | CLI breaks if `typer` is not installed                               | Keep entry point as `speechtelemetry.cli.main:app`; make CLI import-safe and fail gracefully if typer is missing      |
| G4  | `core/pipeline.py`               | `Segment.silence_before_ms` and `silence_after_ms` never populated   | Compute from `silence_spans` after `_build_segments()` and attach to each `Segment`                                   |
| G5  | `tests/fixtures/`                | `mock_backends.py` missing                                           | Create deterministic ML-free mock backends for VAD, ASR, alignment, emotion                                           |
| G6  | `tests/fixtures/`                | `sample_16k_mono.wav` missing                                        | Generate deterministic 3 s 16 kHz mono WAV using numpy + soundfile and commit                                         |
| G7  | `tests/unit/`                    | Core unit tests missing                                              | Write all unit tests before implementation (TDD)                                                                      |
| G8  | `core/pipeline.py`               | Provenance structures defined but not wired                          | Wire minimal `PipelineProvenance` + backend provenance into `TranscriptDocument` (v0.1 baseline, not deferred)        |
| G9  | repo root                        | `CHANGELOG.md` and `README.md` missing                               | Create both                                                                                                           |
| G10 | `core/pipeline.py`               | Chunking config exists but unused                                    | Implement minimal chunk-aware flow OR explicitly route chunking to ASR backend; must not be dead config               |

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
  └─ validate_wav() [G1]         Stage 2 guard: assert 16 kHz, mono
  └─ SileroVADBackend            Stage 3: speech_intervals + silence_spans
  └─ FasterWhisperBackend        Stage 4: raw_segments + detected_language
  └─ WhisperXAlignmentBackend    Stage 5: word-level timestamps
  └─ _build_segments()           assemble Segment objects
  └─ _attach_silence_gaps() [G4] populate silence_before_ms / silence_after_ms
  └─ SpeechBrainEmotionBackend   Stage 8: EmotionScore per segment
  └─ _attach_provenance() [G8]   attach backend + pipeline provenance
  └─ Exporters                   Stage 9
  └─ TranscriptDocument returned
  └─ Job.__exit__()              cleanup temp + release resources
```

Pipeline rules:

- Required stages must not silently skip
- Optional stages may fail soft
- All failures recorded in `ProcessingReport.errors`
- No hard crash beyond preflight

---

## 5. Test Design (TDD)

Tests are written before implementation.

### 5.1 Unit tests

Same structure as original plan (audio_normalize, job, pipeline_orchestration, cli), with these enforced rules:

- No ML dependencies in unit tests
- All pipeline logic tested via mocks
- Deterministic outputs only
- No network/model downloads

### 5.2 Mock backends

Same as original, with strict interface adherence.

### 5.3 Integration fixture

- Deterministic sine wave (3s, 16kHz mono)
- Stored in repo
- Used across tests

### 5.4 Integration smoke test

- Runs only with ML installed
- Marked `@pytest.mark.slow`

---

## 6. Implementation Order

```
Step 1.  Create mock_backends.py
Step 2.  Write test_audio_normalize.py → run → fail
Step 3.  Write test_job.py → run → fail
Step 4.  Write test_pipeline_orchestration.py → run → fail
Step 5.  Write test_cli.py → run → fail

Step 6.  Fix G1: validate_wav wiring
Step 7.  Fix G4: silence fields
Step 8.  Fix G2: remove CLI device resolution
Step 9.  Fix G3: make CLI import-safe (typer optional)
Step 10. Fix G8: wire provenance minimally
Step 11. Fix G10: implement or route chunking
Step 12. Add fixture WAV
Step 13. Add README + CHANGELOG

Step 14. pytest → all pass
Step 15. ruff
Step 16. black
Step 17. mypy
Step 18. fix issues
Step 19. update changelog
Step 20. commit
```

---

## 7. Acceptance Criteria

Same as original, plus:

| #    | Criterion                                       |
| ---- | ----------------------------------------------- |
| AC14 | Provenance present in TranscriptDocument        |
| AC15 | `device="auto"` passes through without error    |
| AC16 | CLI works with and without typer installed      |
| AC17 | Chunking config is not dead (used or delegated) |

---

## 8. Out of Scope

Unchanged, with one clarification:

- Chunking is NOT out of scope — minimal support required in v0.1
- Advanced chunk orchestration is deferred

Still out:

- Diarization
- Prosody
- Streaming
- Alternative backends
- Full provenance enrichment (only minimal required now)

---

## Final Notes

- Keep MIT-compatible dependency strategy intact
- Do not introduce GPL dependencies into required path
- Keep GPU handling delegated to backend layer (`device="auto"`)
- Keep implementation minimal and test-driven
- Avoid adding new abstractions unless required by tests

This plan fully aligns with:

- Architecture spec
- Developer reference
- License policy
- AI agent playbook

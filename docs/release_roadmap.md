# speechtelemetry v0.1 — Public Release Roadmap

**Generated:** 2026-05-18
**Last updated:** 2026-05-18 (Phase 1 + B5 complete)
**Branch:** master
**Status:** Pre-release. Coverage 73.71% ✅ (threshold 70% passed). Phase 1 and blocker B5 resolved. Phase 2 (B6, B7) in progress.

---

## 1. Current State

### What is done

| Area | Status |
|------|--------|
| Core library architecture (types, config, interfaces, registry) | ✅ complete |
| Pipeline orchestration (`core/pipeline.py`) | ✅ complete |
| All 6 default backend implementations (faster-whisper, silero, whisperx, pyannote, parselmouth, speechbrain) | ✅ complete |
| All 4 exporters (JSON, SRT, VTT, TextGrid) | ✅ complete |
| CLI (`speechtelemetry transcribe`, `speechtelemetry info`) | ✅ complete |
| 189 unit tests — all passing | ✅ |
| Backend contract unit tests — all 6 backends (import guard, ABC, signatures) | ✅ |
| API stage-level function tests (`test_api.py`) | ✅ |
| FFmpeg error path tests (`test_ffmpeg.py`) | ✅ |
| Pipeline preflight + export path tests (extended `test_pipeline_orchestration.py`) | ✅ |
| Registry GPL warning + instantiation tests (extended `test_registry.py`) | ✅ |
| `ruff check` — clean | ✅ |
| `ruff format --check` — clean | ✅ |
| `mypy src/` — clean (34 source files) | ✅ |
| GitHub Actions CI workflow (`.github/workflows/ci.yml`) | ✅ |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md` (`black` → `ruff format` fixed) | ✅ |
| `tests/fixtures/sample_16k_mono.wav` — deterministic 3s WAV | ✅ |
| `tests/fixtures/mock_backends.py` — deterministic ML-free mocks | ✅ |
| Integration tests (real pipeline + diarization) | ✅ written; require media files |

### What is blocking release

| ID | Blocker | Severity |
|----|---------|----------|
| **B1** | ~~Coverage 53% — fails `--cov-fail-under=70` in CI~~ → 73.71% | ✅ Resolved |
| **B2** | ~~No backend contract unit tests (0% coverage on all 6 backends)~~ | ✅ Resolved |
| **B3** | ~~`api.py` stage-level functions untested (29% coverage)~~ → 79% | ✅ Resolved |
| **B4** | ~~`io/ffmpeg.py` error paths untested (50% coverage)~~ → 100% | ✅ Resolved |
| **B5** | ~~`CONTRIBUTING.md` references `black` instead of `ruff format`~~ | ✅ Resolved |
| **B6** | Registry stubs for 9 unimplemented backends could mislead users | **Medium** |
| **B7** | No committed golden output structural test for `sample_16k_mono.wav` | **Medium** |
| **B8** | Package not published on PyPI | **Must-have for public release** |
| **B9** | GitHub repository not confirmed public | **Must-have for public release** |

---

## 2. Architecture Review

No structural changes are required. The 5-layer architecture is correct and complete:

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

All layer import rules are respected. No violations found.

---

## 3. Implementation Plan

Follow the agent playbook order: tests first, then implementation, then quality gates.

---

### Phase 1 — Unblock CI (Coverage to ≥70%)

**Goal:** `python -m pytest tests/unit/ --cov=src/speechtelemetry --cov-fail-under=70` exits 0.

**Coverage gap analysis:**

| Module | Stmts | Miss | Cover | Gap source |
|--------|-------|------|-------|-----------|
| `backends/vad/silero.py` | 35 | 35 | 0% | No contract test |
| `backends/asr/faster_whisper.py` | 38 | 38 | 0% | No contract test |
| `backends/alignment/whisperx.py` | 53 | 53 | 0% | No contract test |
| `backends/diarization/pyannote.py` | 121 | 121 | 0% | No contract test |
| `backends/prosody/parselmouth.py` | 58 | 58 | 0% | No contract test |
| `backends/emotion/speechbrain.py` | 50 | 50 | 0% | No contract test |
| `api.py` | 52 | 37 | 29% | Stage-level functions not tested |
| `io/ffmpeg.py` | 18 | 9 | 50% | Error paths not tested |
| `registry.py` | 60 | 25 | 58% | resolve_backend error paths not tested |
| `core/pipeline.py` | 296 | 103 | 65% | Preflight check, export paths |

**Total:** 1198 statements, 634 covered (53%). Need 839 covered (70%).

---

#### Step 1.1 — Backend contract unit tests

**Location:** `tests/unit/backends/<stage>/test_<name>.py`

**Rule from playbook:** "A new backend requires a unit test that exercises the class contract without loading the model."

**Acceptance criteria for each backend:**
- The backend class is importable from its module path.
- The class inherits from the correct ABC in `interfaces.py`.
- When the required optional dependency is not installed, the constructor raises `BackendNotAvailableError` with an install hint (test by mocking the import to raise `ImportError`).
- `_check_available()` can be called without loading a model.
- The method signatures (method names, parameter names) match the ABC contract.

**Files to create:**

```
tests/unit/backends/
├── __init__.py
├── vad/
│   ├── __init__.py
│   └── test_silero.py           — SileroVADBackend contract
├── asr/
│   ├── __init__.py
│   └── test_faster_whisper.py   — FasterWhisperBackend contract
├── alignment/
│   ├── __init__.py
│   └── test_whisperx.py         — WhisperXAlignmentBackend contract
├── diarization/
│   ├── __init__.py
│   └── test_pyannote.py         — PyannoteBackend contract
├── prosody/
│   ├── __init__.py
│   └── test_parselmouth.py      — ParselmouthBackend contract
└── emotion/
    ├── __init__.py
    └── test_speechbrain.py      — SpeechBrainEmotionBackend contract
```

**Test template (apply to each backend):**

```python
"""Contract tests for <BackendClass> — no ML loading, no I/O."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_class_importable():
    from speechtelemetry.backends.<stage>.<module> import <ClassName>
    assert <ClassName> is not None


def test_inherits_from_abc():
    from speechtelemetry.backends.<stage>.<module> import <ClassName>
    from speechtelemetry.interfaces import <ABCClass>
    assert issubclass(<ClassName>, <ABCClass>)


def test_raises_backend_not_available_when_dep_missing():
    """When the required dependency is not installed, __init__ must raise BackendNotAvailableError."""
    from speechtelemetry.exceptions import BackendNotAvailableError

    # Block the dependency import
    with patch.dict(sys.modules, {"<dep_name>": None}):
        # Re-import the backend module so the guarded import runs fresh
        import importlib
        import speechtelemetry.backends.<stage>.<module> as mod
        importlib.reload(mod)
        cls = getattr(mod, "<ClassName>")

        with pytest.raises(BackendNotAvailableError):
            cls()
```

**Estimated coverage gain:** ~175 statements (across all 6 backends, covering the import guard, class body, and basic method structure).

---

#### Step 1.2 — API stage-level function unit tests

**Location:** `tests/unit/test_api.py`

**Acceptance criteria:**
- `run_vad()` calls the registry's VAD backend with correct kwargs and returns the result.
- `run_asr()` calls the ASR backend and returns `(segments, info)`.
- `run_alignment()` calls the alignment backend correctly.
- `run_diarization()` returns `[]` when `config.diarization_backend is None`.
- `run_prosody()` and `run_emotion()` delegate to the pipeline helpers.
- `enrich_media()` raises `FileNotFoundError` when the input path does not exist.
- `enrich_audio()` raises `FileNotFoundError` when the WAV path does not exist.

All tests use `unittest.mock.patch` on `speechtelemetry.registry.get_backend` — no real backends.

**Estimated coverage gain:** ~35 statements.

---

#### Step 1.3 — Registry error path tests

**Location:** `tests/unit/test_registry.py` (extend existing file)

**Acceptance criteria:**
- `resolve_backend("asr", "nonexistent")` raises `BackendNotFoundError` with a helpful message listing available backends.
- `get_backend()` correctly instantiates and returns a backend instance.
- `list_backends("asr")` returns the expected list of names.
- `list_stages()` returns all expected stage names.
- A GPL-licensed backend emits a `UserWarning` when resolved.
- `register()` adds a custom class and it is subsequently returned by `get_backend()`.

**Estimated coverage gain:** ~20 statements.

---

#### Step 1.4 — FFmpeg error path tests

**Location:** `tests/unit/test_ffmpeg.py`

**Acceptance criteria:**
- `normalize_to_wav()` raises a clear exception when FFmpeg is not found on PATH (mock `subprocess.run`).
- `normalize_to_wav()` raises when FFmpeg exits with a non-zero return code.
- The function passes the canonical normalization flags (`-ac 1 -ar 16000 -acodec pcm_s16le`).

**Estimated coverage gain:** ~9 statements.

---

#### Step 1.5 — Pipeline preflight and export path tests

**Location:** `tests/unit/test_pipeline_orchestration.py` (extend existing file)

**New tests needed:**
- `preflight_check()` raises `EnvironmentCheckError` when FFmpeg is missing.
- `preflight_check()` raises when `device="cuda"` but `torch.cuda.is_available()` is `False`.
- `preflight_check()` raises when `diarization_backend="pyannote"` but `HF_TOKEN` is unset.
- `run_pipeline()` calls the exporter when `output_dir` is set.
- Export failure is recorded as a `StageError` and does not crash the pipeline.

**Estimated coverage gain:** ~60 statements in `core/pipeline.py` (lines 86–142, 398–407).

---

#### Coverage summary after Phase 1

| Module | Before | After (actual, 2026-05-18) |
|--------|--------|---------------------------|
| `backends/vad/silero.py` | 0% | 60% |
| `backends/asr/faster_whisper.py` | 0% | 53% |
| `backends/alignment/whisperx.py` | 0% | 40% |
| `backends/diarization/pyannote.py` | 0% | 50% |
| `backends/prosody/parselmouth.py` | 0% | 40% |
| `backends/emotion/speechbrain.py` | 0% | 48% |
| `api.py` | 29% | 79% |
| `io/ffmpeg.py` | 50% | 100% |
| `registry.py` | 58% | 77% |
| `core/pipeline.py` | 65% | 76% |
| **Total** | **53%** | **73.71%** ✅ |

---

### Phase 2 — Correctness Gaps

#### Step 2.1 — Fix CONTRIBUTING.md (`black` → `ruff format`)

**Location:** `CONTRIBUTING.md`, line ~74

**Change:** Replace `black src/ tests/` with `ruff format --check src/ tests/` in the code style section.

**Acceptance criterion:** `grep -n "black" CONTRIBUTING.md` returns no matches.

---

#### Step 2.2 — Document registry stubs clearly

The registry currently lists 9 backends that have no implementation file:

```
asr:         whisperx, whisper.cpp, sensevoice
vad:         funasr
alignment:   forcealign
diarization: funasr
prosody:     copasul, librosa, audioflux
emotion:     emotion2vec, emobox
```

**Options (choose one per stub):**
- **Remove** entries that have no planned implementation timeline.
- **Keep** entries but add a `# future: not yet implemented` comment in `registry.py`.

**Decision:** Keep all stubs with inline comments documenting their status. Users who try a stub get `ImportError` from the registry (`module not found`) which is a clear signal. Add a "Planned backends" section to `docs/developer_reference.md`.

**Acceptance criterion:** `docs/developer_reference.md` lists all planned-but-not-yet-implemented backends with install commands (as placeholders).

---

#### Step 2.3 — Golden output test for `sample_16k_mono.wav`

**What:** A fixture test that runs `enrich_audio()` on the committed 3s sine-wave WAV with mock backends and asserts that the output JSON is structurally correct.

This is different from the existing integration smoke test in `test_pipeline_smoke.py`, which requires real ML backends.

**Location:** `tests/unit/test_golden_output.py`

**Acceptance criteria:**
- Runs with zero ML dependencies (mocks the backends).
- Asserts `doc.duration_s` is within 5% of 3.0 seconds.
- Asserts `doc.segments` is a list.
- Asserts `doc.processing_report.errors == []`.
- Asserts the JSON exported by `JsonExporter` round-trips cleanly through `json.loads`.

**Note:** The playbook's committed golden output pattern (golden file checked into repo, test diffs against it) is deferred until the pipeline is stabilized and we have a deterministic fixture that exercises real backends. The structural assertion above is the v0.1 form.

---

### Phase 3 — Pre-release Polish

#### Step 3.1 — CI coverage gate verification

**Action:** After Phase 1 is complete, run the exact CI command locally to confirm:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/unit/ -q --cov=src/speechtelemetry --cov-report=term-missing --cov-fail-under=70
```

Must exit 0.

---

#### Step 3.2 — Full quality gate run

Run all four quality gates (playbook step 8 + step 9):

```powershell
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pre-commit run --all-files
```

All must exit 0. Fix any regressions before proceeding.

---

#### Step 3.3 — GitHub repository setup

1. Create the public repository at `https://github.com/speechtelemetry/speechtelemetry`.
2. Push the `master` branch.
3. Confirm GitHub Actions CI passes on push (lint, unit tests, secret scan).
4. Enable branch protection on `master`: require CI to pass before merge.
5. Add `CODEOWNERS` file if desired.

**CI badge in README will light up green once this is done.**

---

#### Step 3.4 — PyPI publishing (v0.1.0)

**Pre-conditions:**
- All Phase 1 and Phase 2 steps complete.
- CI passing on GitHub.
- `CHANGELOG.md [Unreleased]` section finalized and version bumped.

**Steps:**

```powershell
# 1. Bump version in pyproject.toml from 0.1.0 to 0.1.0 (already correct)
# 2. Update CHANGELOG.md: move [Unreleased] items to [0.1.0] — <date>
# 3. Build the distribution
pip install build twine
python -m build

# 4. Verify the build is clean
twine check dist/*

# 5. Upload to TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ speechtelemetry

# 6. Verify install and basic import
python -c "import speechtelemetry; print(speechtelemetry.__version__)"

# 7. Upload to PyPI
twine upload dist/*
```

**Acceptance criteria:**
- `pip install speechtelemetry` installs without errors.
- `pip install speechtelemetry[default]` installs the full CLI stack.
- `speechtelemetry --help` shows the Typer help output.
- `speechtelemetry info` lists backends.

---

### Phase 4 — Post-release Backlog (v0.1.x)

These are not blocking v0.1.0 but should follow shortly after.

#### 4.1 — Real integration test fixture in CI

Add a 5-second public-domain WAV to `tests/fixtures/` and run the real pipeline in CI against it with `asr_model_size="tiny"` for speed. Committed `golden_output.json` lets the test detect output regressions.

**Prerequisite:** Must find a WAV with a permissive license (CC0 or public domain) that is ≤500 KB and produces deterministic output across OS/hardware.

#### 4.2 — Windows CI matrix

The CI workflow currently runs on `ubuntu-latest` only. Add a `windows-latest` job for unit tests to catch Windows-specific path/subprocess issues before they reach users.

#### 4.3 — Implement missing backend stubs

Priority order (by user demand and license compatibility):

| Backend | Stage | License | Effort |
|---------|-------|---------|--------|
| `librosa` | prosody | MIT | Medium — well-documented API |
| `whisper.cpp` | ASR | MIT | Medium — ctranslate2 alternative |
| `forcealign` | alignment | MIT | High — different API shape |
| `copasul` | prosody | MIT | High — complex setup |
| `emotion2vec` | emotion | Apache 2.0 | Medium |

For each: create `backends/<stage>/<name>.py`, write unit contract test, register in `registry.py`, add to `pyproject.toml` optional-dependencies, document in `developer_reference.md` and `license_policy.md`.

#### 4.4 — Reduce torchaudio deprecation warnings

The current install emits `UserWarning: torchaudio._backend.list_audio_backends has been deprecated`. This comes from SpeechBrain importing torchaudio internals. Track upstream fix; pin SpeechBrain if a version resolves it.

---

## 4. Acceptance Criteria for v0.1.0 Release

All of the following must be true:

- [x] `python -m pytest tests/unit/ --cov=src/speechtelemetry --cov-fail-under=70 -q` exits 0 — 73.71% ✅
- [x] `ruff check src/ tests/` exits 0 ✅
- [x] `ruff format --check src/ tests/` exits 0 ✅
- [x] `mypy src/` exits 0 ✅
- [x] `pre-commit run --all-files` exits 0 ✅
- [x] `CONTRIBUTING.md` contains no reference to `black` ✅
- [ ] `CHANGELOG.md` has a `[0.1.0]` entry with a release date
- [ ] `pip install speechtelemetry[default]` succeeds on a clean environment
- [ ] `speechtelemetry transcribe --help` prints usage
- [ ] `speechtelemetry info` lists all registered backends
- [ ] `python -c "from speechtelemetry import enrich_media, PipelineConfig; print('ok')"` prints `ok`
- [ ] GitHub Actions CI passes (lint + unit + secret scan) on `master`
- [ ] Package published to PyPI as `speechtelemetry==0.1.0`

---

## 5. Work Sequencing

```
Phase 1 (coverage)
  └─ 1.1  Backend contract tests (6 files)          ← highest coverage impact
  └─ 1.2  API stage-level function tests             ← second highest
  └─ 1.3  Registry error path tests
  └─ 1.4  FFmpeg error path tests
  └─ 1.5  Pipeline preflight + export path tests
  └─ Verify: --cov-fail-under=70 exits 0

Phase 2 (correctness)
  └─ 2.1  Fix CONTRIBUTING.md (black → ruff format)
  └─ 2.2  Document registry stubs in developer_reference.md
  └─ 2.3  Golden output structural test

Phase 3 (release)
  └─ 3.1  Final CI gate run (local)
  └─ 3.2  Full quality gate run
  └─ 3.3  Create public GitHub repository + push
  └─ 3.4  PyPI publish + verify install

Phase 4 (post-release, v0.1.x)
  └─ 4.1  Real integration fixture in CI
  └─ 4.2  Windows CI matrix
  └─ 4.3  Implement missing backend stubs (librosa first)
  └─ 4.4  Torchaudio deprecation fix
```

---

## 6. Notes and Decisions

**Why backend contract tests and not full backend integration tests in CI?**
Loading ML models (faster-whisper, SpeechBrain, WhisperX) in CI adds 5–15 GB of model downloads and 10–60 minutes of runtime per run. The integration tests (`tests/integration/`) are correctly gate-kept with `@pytest.mark.slow` and skip conditions. The unit suite must stay fast (< 5s). Contract tests that mock the import achieve coverage without model downloads.

**Why coverage matters as a blocking gate?**
`--cov-fail-under=70` is already in `.github/workflows/ci.yml`. It currently causes CI to fail on every push to master. This means the CI badge (referenced in README) is currently red. Fixing coverage unblocks the green badge before the public announcement.

**Why keep registry stubs for unimplemented backends?**
The registry stubs signal intent to contributors and make the extension model visible. Users who specify an unimplemented backend name get a clear `BackendNotFoundError` or `ImportError` with a module path, which is sufficient. The alternative (deleting stubs) would require a CHANGELOG entry and re-adding them later. Documenting them is the cheaper path.

**faster-whisper and CUDA compatibility note:**
faster-whisper v1.1.0 requires CUDA 12 + cuDNN 9. Users on CUDA 11 must pin `ctranslate2==3.24.0`. This should be documented in `docs/developer_reference.md` section 3.3 as a known caveat, not as a code change.

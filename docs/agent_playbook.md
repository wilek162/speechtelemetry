# Developer & Agent Playbook

Authoritative workflow for executing changes in speechtelemetry — human contributors and AI agents follow the same steps.

---

## Document hierarchy

Consult in this order before making any change:

1. `docs/spec.md` — product intent, scope, non-goals
2. `docs/architecture.md` — layer boundaries, dependency rules, extension model
3. `docs/developer_reference.md` — backend APIs, install commands, caveats
4. **This document** — workflow for executing changes
5. Code, tests, and fixtures — the implementation

When documents conflict, lower numbers win.

---

## Development flow

Every change follows these steps in order. Do not skip steps.

### 1. Read relevant docs

Read the docs listed above that apply to your change. Understand the layer a file belongs to before touching it.

### 2. Confirm the task outcome

State in one sentence what the change produces. If you cannot write it in one sentence, the scope is too large — split it.

### 3. Define acceptance criteria

List the conditions that must be true when the task is done. Each criterion maps to at least one test. No criterion = no test = no acceptance.

### 4. Verify setup

```powershell
# Confirm the package is installed editable
python -c "import speechtelemetry; print(speechtelemetry.__version__)"

# Run the unit suite — must be 100% green before you start
python -m pytest tests/unit/ -q
```

If any unit test fails before you begin, fix it first.

### 5. Write tests first

Write the test that fails, then write the code that makes it pass. Tests live in:

| Test type | Location | Purpose |
|-----------|----------|---------|
| Unit | `tests/unit/` | Pure logic and data models; no ML deps, no I/O |
| Integration | `tests/integration/` | Real pipeline flows; requires fixtures and ML backends |
| Fixture/golden | `tests/fixtures/` + test file | Canonical input → expected output; assert exact output |
| Setup/preflight | `tests/unit/test_preflight.py` | Package importable, public API intact, env sane |

Rules:
- Unit tests must run in under 1 second total with no external deps.
- Integration tests are gated with `@pytest.mark.slow` and `skipif` on missing fixtures.
- Never use `unittest.mock` to mock the database or a real backend in integration tests.
- A new backend requires a unit test that exercises the class contract without loading the model.

### 6. Implement

Write the smallest change that satisfies the tests. No extras.

Layer rules (in dependency order, inner → outer):

```
types.py          ← zero imports from any speechtelemetry module
interfaces.py     ← imports types only
exceptions.py     ← no imports
config.py         ← imports pydantic; no speechtelemetry imports
registry.py       ← imports exceptions; lazy-imports backends at call time
backends/         ← imports interfaces, types, exceptions
core/pipeline.py  ← imports all layers; never imports a concrete backend directly
api.py            ← thin wrapper over core/pipeline.py
cli/main.py       ← imports api; requires [cli] extras
```

Never import a lower-numbered layer from a higher-numbered module. The pipeline imports from the registry, not from `backends/` directly.

### 7. Run tests and fix

```powershell
# Unit only
python -m pytest tests/unit/ -q

# All tests (integration requires ML deps and fixtures)
python -m pytest -q

# With coverage
python -m pytest tests/unit/ --cov=src/speechtelemetry --cov-report=term-missing -q
```

Fix all failures before proceeding. Do not bypass pre-commit hooks.

### 8. Lint and type-check

```powershell
ruff check src/ tests/
black src/ tests/
mypy src/
```

All three must exit 0.

### 9. Prepare for main

- Update `CHANGELOG.md` under `[Unreleased]`.
- Confirm the PR checklist below.
- Open a PR against `master`.

---

## Adding a new backend

1. Create `src/speechtelemetry/backends/<stage>/my_backend.py`.
2. Inherit from the correct ABC in `interfaces.py`.
3. Implement the exact method signature for the stage — do not add extra public methods.
4. Guard the import:
   ```python
   try:
       import my_dep
   except ImportError as exc:
       raise BackendNotAvailableError(
           "my_dep is not installed. pip install my_dep"
       ) from exc
   ```
5. Override `_check_available` on the class if the import test is insufficient.
6. Add one line to `registry.py` `_REGISTRY`.
7. Add to `pyproject.toml` optional-dependencies.
8. Write a unit test in `tests/unit/backends/<stage>/test_my_backend.py` that instantiates a stub and verifies the contract without loading the model.
9. Document the license in `registry.py` `_LICENSE_FLAGS` (if non-permissive) and `docs/license_policy.md`.

---

## License rules

| Category | Policy |
|----------|--------|
| Core library | MIT only. No non-MIT dependency in `dependencies`. |
| Optional adapters | Listed under `[project.optional-dependencies]` only. |
| GPL components | Optional adapters only. Never a required default. Warn at `resolve_backend` time via `_LICENSE_FLAGS`. |
| Model weights | Never bundled. Downloaded at runtime by the backend. |
| HuggingFace models | Require explicit HF_TOKEN + model license acceptance. Document in `docs/license_policy.md`. |

If you are unsure about a license, do not merge. Check `docs/license_policy.md` first.

---

## Code quality rules

- **No comments by default.** Add one only when the *why* is non-obvious: a hidden constraint, a subtle invariant, a known upstream bug workaround.
- **No over-engineering.** Three similar lines is better than a premature abstraction. No helper functions for one-time operations.
- **No error handling for impossible cases.** Trust internal invariants and framework guarantees. Validate only at system boundaries.
- **No feature flags, no backwards-compatibility shims.** Change the code directly.
- **Names: clear, consistent, package-aligned, boring.** `FasterWhisperBackend`, not `FWB` or `FastWhisper`.
- **Prefer editing existing files** to creating new ones.

---

## PR checklist

- [ ] No imports from outer layers into inner layers (see layer rules above)
- [ ] New backend has guarded import and `BackendNotAvailableError` with pip install hint
- [ ] New dependency documented with SPDX license in `pyproject.toml` and `docs/license_policy.md`
- [ ] All four test types covered as appropriate
- [ ] `python -m pytest tests/unit/ -q` exits 0
- [ ] `ruff check`, `black --check`, `mypy` all exit 0
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No hardcoded tokens, secrets, or absolute paths
- [ ] No bundled model weights

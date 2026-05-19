# Contributing to speechtelemetry

Thank you for contributing. This document covers everything needed to make a correct, reviewable change.

## Source-of-truth order

Before making any change, consult documents in this priority order:

1. **`docs/spec.md`** — product intent, scope, non-goals
2. **`docs/architecture.md`** — layer boundaries, dependency rules, extension model
3. **`docs/developer_reference.md`** — exact backend APIs, install commands, caveats
4. **`docs/agent_playbook.md`** — workflow for executing changes (humans and AI agents)
5. **Code, tests, and fixtures** — the implementation

## Setup

```powershell
git clone https://github.com/wilek162/speechtelemetry.git
cd speechtelemetry
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # Linux/macOS
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

For GPU setup or a full Windows 11 walkthrough, see [docs/setup_windows.md](docs/setup_windows.md).

## Running tests

```bash
# Unit tests only (fast, no GPU needed)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests
pytest
```

## Adding a new backend

1. Create `src/speechtelemetry/backends/<stage>/my_backend.py`
2. Inherit from the correct ABC in `interfaces.py`
3. Implement the exact method signature for the stage
4. Guard imports with `try/except ImportError` and raise `BackendNotAvailableError` with a clear install hint
5. Register in `registry.py` with a one-line entry
6. Add to `pyproject.toml` optional-dependencies
7. Write a unit test in `tests/unit/backends/<stage>/test_my_backend.py`
8. Document the license in `registry.py` and `docs/license_policy.md`

See `docs/agent_playbook.md` for the full step-by-step workflow.

## PR checklist

- [ ] No imports from outer layers into inner layers
- [ ] New backend has guarded import + clear `BackendNotAvailableError`
- [ ] New dependency documented with SPDX license in `pyproject.toml` and `docs/license_policy.md`
- [ ] Tests added and passing (`python -m pytest tests/unit/ -q` exits 0)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No hardcoded tokens, secrets, or absolute paths
- [ ] No bundled model weights

## Code style

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

All three are enforced in CI and by pre-commit hooks.

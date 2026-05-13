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

```bash
git clone https://github.com/speechtelemetry/speechtelemetry.git
cd speechtelemetry
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

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
4. Guard imports with `try/except ImportError` and raise `BackendNotAvailableError`
5. Register in `registry.py` with a one-line entry
6. Add a unit test in `tests/unit/backends/<stage>/test_my_backend.py`
7. Document the license in `registry.py` and `docs/developer_reference.md`

## PR checklist

- [ ] No imports from outer layers into inner layers
- [ ] New backend has guarded import + clear availability error
- [ ] New dependency documented with license in `docs/developer_reference.md`
- [ ] Tests added and passing
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No hardcoded tokens, secrets, or absolute paths

## Code style

```bash
ruff check src/ tests/
black src/ tests/
mypy src/
```

All three are enforced in CI and by pre-commit hooks.

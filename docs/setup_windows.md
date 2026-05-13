# Windows 11 Setup Guide

Complete PowerShell instructions to go from a clean Windows 11 machine to a fully operational speechtelemetry development environment.

Run all commands in a **PowerShell terminal** (not CMD, not PowerShell ISE). Open as Administrator where noted.

---

## Prerequisites check

Open PowerShell and check what is already installed:

```powershell
python --version          # need 3.10 or 3.11
ffmpeg -version           # need any recent version
git --version             # need any recent version
nvidia-smi                # optional — confirms NVIDIA GPU driver
```

---

## Step 1 — Install Python 3.11

Python 3.11 is recommended. Use `winget` (built into Windows 11):

```powershell
winget install --id=Python.Python.3.11 -e
```

Close and reopen PowerShell after installation. Verify:

```powershell
python --version     # should print Python 3.11.x
```

---

## Step 2 — Install FFmpeg

FFmpeg is mandatory. The `Gyan.FFmpeg` build via winget defaults to the LGPL variant (recommended for license hygiene):

```powershell
winget install --id=Gyan.FFmpeg -e
```

Close and reopen PowerShell. Verify:

```powershell
ffmpeg -version    # should print ffmpeg version x.x.x
```

---

## Step 3 — Install Git

```powershell
winget install --id=Git.Git -e
```

---

## Step 4 — Clone the repository

```powershell
git clone https://github.com/speechtelemetry/speechtelemetry.git
cd speechtelemetry
```

---

## Step 5 — Create and activate virtual environment

Always use a virtual environment. Never install speechtelemetry into the system Python.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you see a script execution error, allow local scripts:

```powershell
# Run once as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Verify the venv is active (the prompt should show `(.venv)`):

```powershell
python -c "import sys; print(sys.prefix)"   # should show path to .venv
```

---

## Step 6 — Install PyTorch

PyTorch must be installed **before** any ML backends. Choose one:

**CPU-only (works on any machine, no GPU required):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1 (NVIDIA GPU — most common current CUDA version):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (older NVIDIA GPU):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4 (latest NVIDIA GPU):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```powershell
# CPU
python -c "import torch; print(torch.__version__)"

# CUDA — should print True and a CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Step 7 — Install speechtelemetry

```powershell
pip install -e ".[dev]"
```

For a non-development install:
```powershell
pip install speechtelemetry[default]    # CPU
pip install speechtelemetry[cuda]       # GPU (after CUDA torch install)
pip install speechtelemetry[all]        # Full stack including diarization + prosody
```

---

## Step 8 — Set up pre-commit hooks

```powershell
pre-commit install
```

This installs black, ruff, and a token-detection hook. Hooks run automatically on `git commit`.

---

## Step 9 — Verify the full installation

```powershell
# Package importable and version correct
python -c "import speechtelemetry; print(speechtelemetry.__version__)"

# FFmpeg accessible
ffmpeg -version

# All unit tests pass
python -m pytest tests/unit/ -q

# Lint and type checks
ruff check src/ tests/
mypy src/
```

All should exit cleanly.

---

## Step 10 — HuggingFace token (diarization only)

Required only if you use `diarization_backend="pyannote"`.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model license at [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
3. Generate a read token at [hf.co/settings/tokens](https://hf.co/settings/tokens)
4. Set the token in PowerShell:

```powershell
# Current session only
$env:HF_TOKEN = "hf_YOUR_TOKEN_HERE"

# Persist across sessions
[System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_YOUR_TOKEN_HERE", "User")
```

---

## Step 11 — Optional: disable pyannote telemetry

pyannote.audio sends anonymous usage analytics by default. To disable:

```powershell
$env:PYANNOTE_NO_ANALYTICS = "1"
# Or persist:
[System.Environment]::SetEnvironmentVariable("PYANNOTE_NO_ANALYTICS", "1", "User")
```

---

## CPU-only quickstart (copy-paste)

Complete setup on a clean Windows 11 machine for CPU-only development:

```powershell
# Install prerequisites (run as Administrator if winget prompts)
winget install --id=Python.Python.3.11 -e
winget install --id=Gyan.FFmpeg -e
winget install --id=Git.Git -e

# Reopen PowerShell, then:
git clone https://github.com/speechtelemetry/speechtelemetry.git
cd speechtelemetry
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install

# Verify
python -c "import speechtelemetry; print(speechtelemetry.__version__)"
python -m pytest tests/unit/ -q
```

---

## GPU quickstart (copy-paste)

Complete CUDA setup (replace `cu121` with your CUDA version):

```powershell
# Prerequisites same as CPU setup above, then:
git clone https://github.com/speechtelemetry/speechtelemetry.git
cd speechtelemetry
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CUDA-enabled torch BEFORE speechtelemetry
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Install speechtelemetry with cuda extras
pip install -e ".[dev]"
pre-commit install

# Verify
python -c "import speechtelemetry; print(speechtelemetry.__version__)"
python -m pytest tests/unit/ -q
```

Then use `device="cuda"` in `PipelineConfig`:

```python
from speechtelemetry import enrich_media, PipelineConfig

doc = enrich_media(
    "interview.mp4",
    config=PipelineConfig(device="cuda"),
)
print(f"RTF: {doc.processing_report.real_time_factor:.2f}")
```

---

## Troubleshooting

**`torch.cuda.is_available()` returns `False`:**
- Confirm NVIDIA driver is installed: `nvidia-smi`
- Confirm CUDA Toolkit version: `nvcc --version`
- Confirm torch was installed with the matching `--index-url` (see Step 6)
- Reinstall torch with the correct CUDA variant

**`ffmpeg` not found:**
- Close and reopen PowerShell after `winget install`
- Verify: `Get-Command ffmpeg`
- If still missing, add FFmpeg `bin/` folder to your `PATH` manually

**`pre-commit install` fails:**
- Run `pip install pre-commit` then retry
- Ensure you are inside the repository root directory

**`python-m pytest` import errors:**
- Confirm venv is active (check prompt shows `(.venv)`)
- Run `pip install -e ".[dev]"` again

**`HF_TOKEN` errors during diarization:**
- Verify the token is set: `echo $env:HF_TOKEN`
- Ensure the model license is accepted at hf.co (the accept button must be clicked while logged in)

**speechtelemetry**

Developer & AI Agent Reference

Open-source transcript, subtitle, prosody, and emotion enrichment
library

Version 0.1 \| May 2026

+-----------------------------------------------------------------------+
| This document is the authoritative reference for all packages used in |
| the speechtelemetry project. It covers exact API usage, installation, |
| configuration, integration patterns, licensing constraints, and known |
| pitfalls. It is designed for developers and AI agents to ensure a     |
| smooth, consistent implementation process with zero guesswork.        |
|                                                                       |
| **Pipeline order: Decode → Normalize → VAD → ASR → Align → Diarize →  |
| Prosody → Emotion → Export**                                          |
+-----------------------------------------------------------------------+

**Table of Contents**

**1. Project Overview**

speechtelemetry is a library-first, local-first speech intelligence
engine. The library is the source of truth. CLI, UI, and API wrappers
are secondary and derived from it. The project targets Windows first,
then Linux. CPU-only operation is always required; CUDA GPU acceleration
is optional but first-class when available.

**1.1 Pipeline Stages**

  --------------------------------------------------------------------------------------
  **Stage**     **Module**                          **Description**
  ------------- ----------------------------------- ------------------------------------
  1\. Decode    io/ffmpeg.py                        Convert any input file to canonical
                                                    mono 16 kHz PCM WAV using FFmpeg.

  2\. Normalize io/audio_normalize.py               Validate sample rate, channel count,
                                                    and bit depth. Rechunk for memory
                                                    efficiency.

  3\. VAD       backends/vad/silero.py              Detect speech/silence boundaries.
                                                    Output SilenceSpan list.

  4\. ASR       backends/asr/faster_whisper.py      Transcribe speech segments. Output
                                                    Segment list with text and rough
                                                    timestamps.

  5\. Alignment backends/alignment/whisperx.py      Refine word-level timestamps via
                                                    forced alignment (wav2vec2).

  6\.           backends/diarization/pyannote.py    Optional. Assign speaker labels to
  Diarization                                       words/segments.

  7\. Prosody   backends/prosody/parselmouth.py +   Extract F0 mean/variance, intensity,
                copasul.py                          voice quality, rhythm features per
                                                    segment/window.

  8\. Emotion   backends/emotion/speechbrain.py     Estimate utterance-level emotion
                                                    probability distributions.

  9\. Export    exporters/json.py (+                Serialize canonical
                srt/vtt/textgrid)                   TranscriptDocument to JSON
                                                    (authoritative) and subtitle
                                                    formats.
  --------------------------------------------------------------------------------------

**1.2 Core Data Model (types.py)**

All pipeline outputs are stored in a typed canonical object tree. The
full schema is defined in src/speechtelemetry/types.py using Python
dataclasses or Pydantic models.

  ---------------------------------------------------------------------------
  **Type**             **Key Fields**
  -------------------- ------------------------------------------------------
  TranscriptDocument   source_path, language, duration_s, segments:
                       List\[Segment\], silence_spans: List\[SilenceSpan\],
                       processing_report: ProcessingReport

  Segment              start, end, text, speaker, confidence,
                       silence_before_ms, silence_after_ms, words:
                       List\[Word\], prosody: ProsodyWindow, emotion:
                       EmotionScore

  Word                 text, start, end, confidence, token_id,
                       alignment_backend

  SilenceSpan          start, end, duration_ms, reason (speech_gap \|
                       non_speech \| overlap)

  ProsodyWindow        f0_mean, f0_variance, energy_mean, energy_variance,
                       speech_rate_sps, pause_density, voice_quality_hnr

  EmotionScore         label_distribution: Dict\[str,float\], valence,
                       arousal, confidence, backend_name

  ProcessingReport     stage_timings, peak_ram_mb, peak_vram_mb,
                       real_time_factor, errors: List\[StageError\]
  ---------------------------------------------------------------------------

**1.3 Primary Public API (api.py)**

+-----------------------------------------------------------------------+
| from speechtelemetry import enrich_media, PipelineConfig              |
|                                                                       |
| result = enrich_media(                                                |
|                                                                       |
| input_path=\"sample.mp4\",                                            |
|                                                                       |
| config=PipelineConfig(                                                |
|                                                                       |
| asr_backend=\"faster-whisper\",                                       |
|                                                                       |
| vad_backend=\"silero\",                                               |
|                                                                       |
| diarization_backend=\"pyannote\", \# set None to skip                 |
|                                                                       |
| prosody_backend=\[\"parselmouth\", \"copasul\"\],                     |
|                                                                       |
| emotion_backend=\"speechbrain\",                                      |
|                                                                       |
| device=\"auto\", \# auto \| cpu \| cuda                               |
|                                                                       |
| ),                                                                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# result is a TranscriptDocument                                     |
+-----------------------------------------------------------------------+

**2. Installation & Environment Setup**

**2.1 Python Version & Virtual Environment**

Python 3.10 or 3.11 is recommended. Python 3.12 is supported but some
audio backends may have wheel gaps on Windows. Always use a virtual
environment.

+-----------------------------------------------------------------------+
| python -m venv .venv                                                  |
|                                                                       |
| \# Windows                                                            |
|                                                                       |
| .venv\\Scripts\\activate                                              |
|                                                                       |
| \# Linux / macOS                                                      |
|                                                                       |
| source .venv/bin/activate                                             |
+-----------------------------------------------------------------------+

**2.2 CUDA Setup (Optional but First-Class)**

GPU acceleration requires the NVIDIA CUDA Toolkit 12.x. Install PyTorch
with CUDA before installing any ML backends.

+-----------------------------------------------------------------------+
| \# PyTorch with CUDA 12.4 (adjust for your CUDA version)              |
|                                                                       |
| pip install torch torchaudio \--index-url                             |
| https://download.pytorch.org/whl/cu124                                |
|                                                                       |
| \# CPU-only (always works, required baseline)                         |
|                                                                       |
| pip install torch torchaudio \--index-url                             |
| https://download.pytorch.org/whl/cpu                                  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **CUDA Libraries Required by GPU Backends**                           |
|                                                                       |
| cuBLAS, cuDNN --- required by faster-whisper and SpeechBrain GPU      |
| paths.                                                                |
|                                                                       |
| Install the NVIDIA CUDA Toolkit 12.x from                             |
| developer.nvidia.com/cuda-downloads.                                  |
|                                                                       |
| On Windows: add C:\\Program Files\\NVIDIA GPU Computing               |
| Toolkit\\CUDA\\v12.x\\bin to PATH.                                    |
|                                                                       |
| Verify: python -c \"import torch;                                     |
| print(torch.cuda.get_device_name(0))\"                                |
+-----------------------------------------------------------------------+

**2.3 FFmpeg System Dependency**

FFmpeg must be installed system-wide and available on PATH. It is the
universal decode/transcode layer for all media input.

+-----------------------------------------------------------------------+
| \# Windows --- download static build from                             |
| https://ffmpeg.org/download.html                                      |
|                                                                       |
| \# and add the bin/ folder to PATH, OR use winget:                    |
|                                                                       |
| winget install \--id=Gyan.FFmpeg -e                                   |
|                                                                       |
| \# Ubuntu / Debian                                                    |
|                                                                       |
| sudo apt-get install ffmpeg                                           |
|                                                                       |
| \# Verify                                                             |
|                                                                       |
| ffmpeg -version                                                       |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **WARNING: FFmpeg is mandatory**                                      |
|                                                                       |
| Every media file (MP3, MP4, MKV, FLAC, etc.) must be decoded through  |
| FFmpeg before any backend sees it.                                    |
|                                                                       |
| If FFmpeg is missing, the library MUST raise a clear EnvironmentError |
| at startup (check in core/pipeline.py before job start).              |
|                                                                       |
| Do NOT rely on PyAV or torchaudio alone for media ingest --- FFmpeg   |
| gives the widest codec coverage.                                      |
+-----------------------------------------------------------------------+

**2.4 Full Dependency Install Matrix**

  ----------------------------------------------------------------------------
  **Package**         **pip install         **Notes**
                      command**
  ------------------- --------------------- ----------------------------------
  faster-whisper      pip install           Also pulls CTranslate2. CUDA libs
                      faster-whisper        must be separate.

  whisperx            pip install whisperx  Requires torch+CUDA 12.4+ for GPU.
                                            CPU works with compute_type=int8.

  silero-vad          pip install           Pure Python + PyTorch/ONNX. MIT
                      silero-vad            license.

  pyannote.audio      pip install           Requires HuggingFace token. Accept
                      pyannote.audio        model license at hf.co.

  praat-parselmouth   pip install           Binary wheels for Win/Linux/macOS.
                      praat-parselmouth     No Praat install needed.

  copasul             pip install copasul   Config-file driven. Python 3 only.

  speechbrain         pip install           Apache 2.0. Models auto-download
                      speechbrain           from HuggingFace Hub.

  ffmpeg-normalize    pip install           Thin Python wrapper around FFmpeg
                      ffmpeg-normalize      CLI.

  pydantic            pip install           For typed schema validation in
                      pydantic\>=2          types.py.

  soundfile           pip install soundfile For WAV I/O used by several
                                            backends.
  ----------------------------------------------------------------------------

**3. Backend Reference**

**3.1 FFmpeg --- Media Decode & Audio Normalization**

**Role in Pipeline**

FFmpeg is the first stage. Every input file --- regardless of format ---
is decoded into a canonical mono 16 kHz 16-bit PCM WAV before any Python
backend touches it. This guarantees all downstream components receive a
predictable audio format.

**Canonical Normalization Command**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/io/ffmpeg.py                                   |
|                                                                       |
| import subprocess                                                     |
|                                                                       |
| import shutil                                                         |
|                                                                       |
| import os                                                             |
|                                                                       |
| def normalize_to_wav(input_path: str, output_path: str) -\> None:     |
|                                                                       |
| \"\"\"Decode any media to mono 16 kHz PCM WAV. Raises                 |
| EnvironmentError if ffmpeg missing.\"\"\"                             |
|                                                                       |
| if not shutil.which(\'ffmpeg\'):                                      |
|                                                                       |
| raise EnvironmentError(                                               |
|                                                                       |
| \'FFmpeg not found on PATH. Install from                              |
| https://ffmpeg.org/download.html\'                                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| cmd = \[                                                              |
|                                                                       |
| \'ffmpeg\', \'-y\',                                                   |
|                                                                       |
| \'-i\', input_path,                                                   |
|                                                                       |
| \'-vn\', \# strip video                                               |
|                                                                       |
| \'-acodec\', \'pcm_s16le\',                                           |
|                                                                       |
| \'-ac\', \'1\', \# mono                                               |
|                                                                       |
| \'-ar\', \'16000\', \# 16 kHz                                         |
|                                                                       |
| output_path,                                                          |
|                                                                       |
| \'-loglevel\', \'error\'                                              |
|                                                                       |
| \]                                                                    |
|                                                                       |
| result = subprocess.run(cmd, capture_output=True)                     |
|                                                                       |
| if result.returncode != 0:                                            |
|                                                                       |
| raise RuntimeError(f\'FFmpeg failed: {result.stderr.decode()}\')      |
+-----------------------------------------------------------------------+

**Key Parameters**

  -----------------------------------------------------------------------
  **Flag**         **Value**     **Purpose**
  ---------------- ------------- ----------------------------------------
  -vn                            Strip all video streams. Audio only.

  -acodec                        16-bit little-endian PCM --- maximum
  pcm_s16le                      compatibility.

  -ac 1                          Downmix to mono (averages all channels).

  -ar 16000                      Resample to 16 kHz --- the universal
                                 rate for all speech models.

  -y                             Overwrite output without prompting.
                                 Required for temp files.

  -loglevel error                Suppress verbose output; only print
                                 errors.
  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Implementation Notes**                                              |
|                                                                       |
| Always write to a tempfile.NamedTemporaryFile(suffix=\'.wav\',        |
| delete=False). Clean up in a finally block.                           |
|                                                                       |
| Check ffmpeg on PATH at library init (core/pipeline.py), not on first |
| use, for early clear error messages.                                  |
|                                                                       |
| Do not use PyAV or torchaudio as primary ingest --- use FFmpeg for    |
| maximum codec coverage.                                               |
|                                                                       |
| For very large files (\>1 hour), consider piping to a chunked reader  |
| rather than materializing the full WAV on disk.                       |
+-----------------------------------------------------------------------+

**3.2 Silero VAD --- Voice Activity Detection**

**Role in Pipeline**

Silero VAD runs immediately after audio normalization. Its output --- a
list of speech timestamp intervals --- is used to (a) skip silence
before ASR, reducing compute waste and model hallucination, and (b)
populate SilenceSpan objects in the canonical output.

**Installation & Licensing**

-   pip install silero-vad

-   License: MIT --- zero restrictions, no keys, no telemetry.

-   Model size: \~2 MB JIT model. Downloads from HuggingFace Hub on
    first use.

-   Supported sample rates: 8 kHz and 16 kHz only.

**Exact API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/vad/silero.py                         |
|                                                                       |
| from silero_vad import load_silero_vad, read_audio,                   |
| get_speech_timestamps                                                 |
|                                                                       |
| import torch                                                          |
|                                                                       |
| torch.set_num_threads(1) \# Important for CPU reproducibility         |
|                                                                       |
| class SileroVADBackend:                                               |
|                                                                       |
| def \_\_init\_\_(self):                                               |
|                                                                       |
| self.model = load_silero_vad()                                        |
|                                                                       |
| def get_speech_intervals(self, wav_path: str) -\> list\[dict\]:       |
|                                                                       |
| \"\"\"Returns list of {\'start\': float, \'end\': float} in           |
| seconds.\"\"\"                                                        |
|                                                                       |
| wav = read_audio(wav_path) \# returns torch.Tensor, mono 16 kHz       |
|                                                                       |
| timestamps = get_speech_timestamps(                                   |
|                                                                       |
| wav,                                                                  |
|                                                                       |
| self.model,                                                           |
|                                                                       |
| return_seconds=True, \# Always True --- we work in seconds            |
|                                                                       |
| threshold=0.5, \# Speech prob threshold (0.5 = default, tune per      |
| corpus)                                                               |
|                                                                       |
| min_silence_duration_ms=300, \# Merge short silences \< 300ms         |
|                                                                       |
| min_speech_duration_ms=250, \# Discard speech bursts \< 250ms         |
|                                                                       |
| speech_pad_ms=30, \# Add 30ms padding around each speech region       |
|                                                                       |
| )                                                                     |
|                                                                       |
| return timestamps \# \[{\'start\': 0.5, \'end\': 3.2}, \...\]         |
|                                                                       |
| def get_silence_spans(self, wav_path: str, total_duration_s: float)   |
| -\> list\[dict\]:                                                     |
|                                                                       |
| \"\"\"Invert speech intervals to produce silence spans.\"\"\"         |
|                                                                       |
| speech = self.get_speech_intervals(wav_path)                          |
|                                                                       |
| silence = \[\]                                                        |
|                                                                       |
| prev_end = 0.0                                                        |
|                                                                       |
| for s in speech:                                                      |
|                                                                       |
| if s\[\'start\'\] - prev_end \> 0.01: \# \>10ms gap                   |
|                                                                       |
| silence.append({\'start\': prev_end, \'end\': s\[\'start\'\]})        |
|                                                                       |
| prev_end = s\[\'end\'\]                                               |
|                                                                       |
| if total_duration_s - prev_end \> 0.01:                               |
|                                                                       |
| silence.append({\'start\': prev_end, \'end\': total_duration_s})      |
|                                                                       |
| return silence                                                        |
+-----------------------------------------------------------------------+

**Key Parameters & Tuning**

  ---------------------------------------------------------------------------------
  **Parameter**             **Default**   **Notes**
  ------------------------- ------------- -----------------------------------------
  threshold                 0.5           Probability above which a chunk is
                                          speech. Tune per corpus; 0.5 is safe for
                                          most audio.

  min_silence_duration_ms   300           Silences shorter than this are bridged
                                          (not split). Prevents sentence
                                          fragmentation.

  min_speech_duration_ms    250           Discard isolated speech bursts shorter
                                          than this.

  speech_pad_ms             30            Padding added to both sides of detected
                                          speech. Prevents clipping first/last
                                          phoneme.

  return_seconds            True          Always set True in this library. All
                                          timestamps are in seconds throughout.
  ---------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Implementation Notes**                                              |
|                                                                       |
| Silero VAD supports 8 kHz and 16 kHz ONLY. Always normalize to 16 kHz |
| first.                                                                |
|                                                                       |
| torch.set_num_threads(1) is important for deterministic CPU results.  |
|                                                                       |
| The model auto-downloads (\~2 MB) on first call. Cache lives in       |
| \~/.cache/torch/hub/.                                                 |
|                                                                       |
| ONNX runtime mode (onnxruntime package) gives \~4-5x speedup on CPU   |
| --- consider as optional fast path.                                   |
|                                                                       |
| For streaming (future): use silero_vad.VADIterator for chunk-by-chunk |
| processing.                                                           |
+-----------------------------------------------------------------------+

**3.3 faster-whisper --- ASR (Default Backend)**

**Role in Pipeline**

faster-whisper is the default ASR backend. It transcribes VAD-filtered
audio segments into text with segment-level timestamps. It does not
produce word-level timestamps by itself --- that is the job of the
alignment stage.

**Installation & Licensing**

-   pip install faster-whisper

-   Depends on: CTranslate2 (auto-installed), sentencepiece, tokenizers.

-   For GPU: requires cuBLAS and cuDNN matching your CUDA version.

-   License: MIT. Models on HuggingFace Hub under OpenAI license
    (permissive).

**How It Works**

faster-whisper reimplements OpenAI Whisper using CTranslate2, a
high-performance inference engine for Transformer models. It is up to 4x
faster than the original implementation with identical accuracy. It
supports int8 quantization on both CPU and GPU, halving memory usage
with minimal quality loss.

**Exact API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/asr/faster_whisper.py                 |
|                                                                       |
| from faster_whisper import WhisperModel                               |
|                                                                       |
| class FasterWhisperBackend:                                           |
|                                                                       |
| def \_\_init\_\_(self, model_size: str = \'large-v3\', device: str =  |
| \'auto\',                                                             |
|                                                                       |
| compute_type: str \| None = None):                                    |
|                                                                       |
| if device == \'auto\':                                                |
|                                                                       |
| import torch                                                          |
|                                                                       |
| device = \'cuda\' if torch.cuda.is_available() else \'cpu\'           |
|                                                                       |
| \# compute_type: \'float16\' on GPU, \'int8\' on CPU (best            |
| perf/quality tradeoff)                                                |
|                                                                       |
| if compute_type is None:                                              |
|                                                                       |
| compute_type = \'float16\' if device == \'cuda\' else \'int8\'        |
|                                                                       |
| self.model = WhisperModel(model_size, device=device,                  |
| compute_type=compute_type)                                            |
|                                                                       |
| def transcribe(self, wav_path: str, language: str \| None = None,     |
|                                                                       |
| beam_size: int = 5, vad_filter: bool = False) -\> tuple:              |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Returns (segments_generator, TranscriptionInfo).                      |
|                                                                       |
| segments is a generator --- consume it before calling again.          |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| segments, info = self.model.transcribe(                               |
|                                                                       |
| wav_path,                                                             |
|                                                                       |
| language=language, \# None = auto-detect                              |
|                                                                       |
| beam_size=beam_size,                                                  |
|                                                                       |
| word_timestamps=False, \# Use whisperx alignment instead              |
|                                                                       |
| vad_filter=vad_filter, \# Optional built-in VAD (we prefer Silero)    |
|                                                                       |
| temperature=\[0.0, 0.2, 0.4, 0.6, 0.8, 1.0\], \# Fallback chain       |
|                                                                       |
| )                                                                     |
|                                                                       |
| return list(segments), info \# Materialise generator                  |
|                                                                       |
| def transcribe_chunked(self, wav_path: str, chunk_speech_intervals:   |
| list,                                                                 |
|                                                                       |
| language: str \| None = None) -\> list:                               |
|                                                                       |
| \"\"\"Transcribe only pre-detected speech chunks to avoid silence     |
| hallucination.\"\"\"                                                  |
|                                                                       |
| from faster_whisper.audio import decode_audio                         |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| audio = decode_audio(wav_path, sampling_rate=16000)                   |
|                                                                       |
| all_segments = \[\]                                                   |
|                                                                       |
| for interval in chunk_speech_intervals:                               |
|                                                                       |
| start_sample = int(interval\[\'start\'\] \* 16000)                    |
|                                                                       |
| end_sample = int(interval\[\'end\'\] \* 16000)                        |
|                                                                       |
| chunk = audio\[start_sample:end_sample\]                              |
|                                                                       |
| segs, \_ = self.model.transcribe(chunk, language=language)            |
|                                                                       |
| for s in segs:                                                        |
|                                                                       |
| \# offset timestamps back to global position                          |
|                                                                       |
| s = s.\_replace(start=s.start + interval\[\'start\'\],                |
|                                                                       |
| end=s.end + interval\[\'start\'\])                                    |
|                                                                       |
| all_segments.append(s)                                                |
|                                                                       |
| return all_segments                                                   |
+-----------------------------------------------------------------------+

**Model Size Selection**

  ---------------------------------------------------------------------------
  **Model**     **VRAM       **CPU RAM**  **Accuracy**   **Speed**
                (GPU)**
  ------------- ------------ ------------ -------------- --------------------
  tiny          \~1 GB       \~500 MB     Lowest         Fastest

  base          \~1 GB       \~600 MB     Low            Very fast

  small         \~2 GB       \~1 GB       Moderate       Fast

  medium        \~5 GB       \~2.5 GB     Good           Moderate

  large-v3      \~10 GB      \~5 GB       Best           Slow on CPU

  large-v3      \~6 GB       \~3 GB       Best (tiny     Recommended GPU
  (int8)                                  loss)          default
  ---------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Critical Implementation Notes**                                     |
|                                                                       |
| Model download: Auto-downloads from HuggingFace Hub on first          |
| instantiation. Cache at \~/.cache/huggingface/hub/.                   |
|                                                                       |
| Generator consumption: model.transcribe() returns a lazy generator.   |
| Always materialise it with list() before accessing info.              |
|                                                                       |
| word_timestamps=False: Do NOT use faster-whisper\'s built-in word     |
| timestamps. They are less accurate than whisperx forced alignment.    |
| Always follow with the alignment stage.                               |
|                                                                       |
| compute_type on CPU: Use \'int8\' or \'int8_float32\' for best CPU    |
| performance. \'float32\' is accurate but \~3x slower.                 |
|                                                                       |
| Chunked inference: For long files, split by VAD intervals and         |
| transcribe each chunk. This reduces hallucinations in silence regions |
| and peak memory.                                                      |
|                                                                       |
| temperature fallback: Pass a list of temperatures. Whisper will retry |
| with higher temperature if repetition is detected.                    |
+-----------------------------------------------------------------------+

**3.4 WhisperX --- Word Alignment & Optional Combined ASR+Diarization**

**Role in Pipeline**

WhisperX serves two roles: (1) as the word-level alignment backend,
refining segment timestamps to word-level precision using wav2vec2
forced alignment, and (2) as an optional combined ASR+align+diarize
backend when tight integration is preferred. Use faster-whisper for
standalone ASR and WhisperX only for alignment, unless WhisperX\'s
combined mode is explicitly configured.

**Installation**

+-----------------------------------------------------------------------+
| \# Requires CUDA Toolkit 12.8 for GPU mode                            |
|                                                                       |
| pip install whisperx                                                  |
|                                                                       |
| \# With GPU (install PyTorch CUDA first):                             |
|                                                                       |
| pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 \--index-url   |
| https://download.pytorch.org/whl/cu124                                |
|                                                                       |
| pip install whisperx                                                  |
+-----------------------------------------------------------------------+

-   License: BSD-4-Clause

-   HuggingFace token required for diarization sub-pipeline
    (pyannote.audio).

**Usage --- Alignment Only (Primary Use Case)**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/alignment/whisperx.py                 |
|                                                                       |
| import whisperx                                                       |
|                                                                       |
| import gc, torch                                                      |
|                                                                       |
| class WhisperXAlignmentBackend:                                       |
|                                                                       |
| def \_\_init\_\_(self, device: str = \'auto\'):                       |
|                                                                       |
| if device == \'auto\':                                                |
|                                                                       |
| self.device = \'cuda\' if torch.cuda.is_available() else \'cpu\'      |
|                                                                       |
| else:                                                                 |
|                                                                       |
| self.device = device                                                  |
|                                                                       |
| self.\_align_model = None                                             |
|                                                                       |
| self.\_metadata = None                                                |
|                                                                       |
| self.\_loaded_lang = None                                             |
|                                                                       |
| def align(self, segments: list, audio_path: str, language: str) -\>   |
| list:                                                                 |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Refine segment timestamps to word level.                              |
|                                                                       |
| segments: list of dicts with \'start\', \'end\', \'text\' from        |
| faster-whisper.                                                       |
|                                                                       |
| Returns: updated segments with \'words\' key on each segment.         |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| if self.\_loaded_lang != language:                                    |
|                                                                       |
| if self.\_align_model is not None:                                    |
|                                                                       |
| del self.\_align_model                                                |
|                                                                       |
| gc.collect()                                                          |
|                                                                       |
| if self.device == \'cuda\': torch.cuda.empty_cache()                  |
|                                                                       |
| self.\_align_model, self.\_metadata = whisperx.load_align_model(      |
|                                                                       |
| language_code=language,                                               |
|                                                                       |
| device=self.device                                                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| self.\_loaded_lang = language                                         |
|                                                                       |
| audio = whisperx.load_audio(audio_path)                               |
|                                                                       |
| result = whisperx.align(                                              |
|                                                                       |
| segments,                                                             |
|                                                                       |
| self.\_align_model,                                                   |
|                                                                       |
| self.\_metadata,                                                      |
|                                                                       |
| audio,                                                                |
|                                                                       |
| self.device,                                                          |
|                                                                       |
| return_char_alignments=False \# Word level only                       |
|                                                                       |
| )                                                                     |
|                                                                       |
| return result\[\'segments\'\] \# Each segment now has a \'words\'     |
| list                                                                  |
+-----------------------------------------------------------------------+

**Usage --- Combined Mode (Optional, when diarization is needed
inline)**

+-----------------------------------------------------------------------+
| \# Only use this mode when PipelineConfig.asr_backend == \'whisperx\' |
|                                                                       |
| import whisperx, gc, torch                                            |
|                                                                       |
| def run_whisperx_full(audio_path, hf_token, device=\'cuda\',          |
|                                                                       |
| compute_type=\'float16\', batch_size=16):                             |
|                                                                       |
| \# Step 1: Transcribe                                                 |
|                                                                       |
| model = whisperx.load_model(\'large-v3-turbo\', device,               |
| compute_type=compute_type)                                            |
|                                                                       |
| audio = whisperx.load_audio(audio_path)                               |
|                                                                       |
| result = model.transcribe(audio, batch_size=batch_size)               |
|                                                                       |
| del model; gc.collect();                                              |
|                                                                       |
| if device == \'cuda\': torch.cuda.empty_cache()                       |
|                                                                       |
| \# Step 2: Align                                                      |
|                                                                       |
| model_a, meta = whisperx.load_align_model(result\[\'language\'\],     |
| device=device)                                                        |
|                                                                       |
| result = whisperx.align(result\[\'segments\'\], model_a, meta, audio, |
| device,                                                               |
|                                                                       |
| return_char_alignments=False)                                         |
|                                                                       |
| del model_a; gc.collect();                                            |
|                                                                       |
| if device == \'cuda\': torch.cuda.empty_cache()                       |
|                                                                       |
| \# Step 3: Diarize (requires HF token)                                |
|                                                                       |
| if hf_token:                                                          |
|                                                                       |
| diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, |
|                                                                       |
| device=device)                                                        |
|                                                                       |
| diarize_segs = diarize_model(audio, min_speakers=1, max_speakers=10)  |
|                                                                       |
| result = whisperx.assign_word_speakers(diarize_segs, result)          |
|                                                                       |
| return result\[\'segments\'\]                                         |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Critical Notes for WhisperX**                                       |
|                                                                       |
| Memory management: Each model (ASR, align, diarize) is large. Always  |
| del + gc.collect() + torch.cuda.empty_cache() between stages on GPU.  |
|                                                                       |
| batch_size tuning: Start at 16 for 16 GB VRAM, reduce to 4-8 for 12   |
| GB, raise to 32 for 24 GB.                                            |
|                                                                       |
| Language-specific align model: load_align_model() downloads a         |
| language-specific wav2vec2 model. Supported natively: en, fr, de, es, |
| it, ja, zh, nl, uk, pt. Others require a custom HuggingFace model.    |
|                                                                       |
| CPU mode: Use compute_type=\'int8\' and device=\'cpu\'. Batch size    |
| should be 4-8.                                                        |
|                                                                       |
| HF token: Set HF_TOKEN env var or pass use_auth_token explicitly.     |
| Required for diarization pipeline (pyannote models).                  |
|                                                                       |
| Speaker labels: Output uses \'SPEAKER_00\', \'SPEAKER_01\', etc. Map  |
| these to your Speaker type in the canonical model.                    |
+-----------------------------------------------------------------------+

**3.5 pyannote.audio --- Speaker Diarization**

**Role in Pipeline**

pyannote.audio is the default diarization backend. It runs after
alignment and assigns speaker labels to time intervals in the
transcript. Diarization is optional --- if it fails or is disabled, the
pipeline returns a transcript without speaker labels rather than
failing.

**Installation & Prerequisites**

+-----------------------------------------------------------------------+
| pip install pyannote.audio                                            |
|                                                                       |
| \# Required before first use:                                         |
|                                                                       |
| \# 1. Create HuggingFace account at hf.co                             |
|                                                                       |
| \# 2. Accept model license at                                         |
| hf.co/pyannote/speaker-diarization-community-1                        |
|                                                                       |
| \# 3. Generate read token at hf.co/settings/tokens                    |
|                                                                       |
| \# 4. Store as environment variable:                                  |
|                                                                       |
| \# Windows: set HF_TOKEN=hf\_\...                                     |
|                                                                       |
| \# Linux: export HF_TOKEN=hf\_\...                                    |
+-----------------------------------------------------------------------+

-   License: pyannote.audio --- MIT. Model weights --- CC-BY-4.0. ALWAYS
    check model card licensing.

-   Recommended model: pyannote/speaker-diarization-community-1 (better
    than 3.1 for speaker counting accuracy).

-   Fallback model: pyannote/speaker-diarization-3.1 (more stable,
    widely tested).

**Exact API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/diarization/pyannote.py               |
|                                                                       |
| import torch, os                                                      |
|                                                                       |
| from pyannote.audio import Pipeline                                   |
|                                                                       |
| from pyannote.audio.pipelines.utils.hook import ProgressHook          |
|                                                                       |
| class PyannoteBackend:                                                |
|                                                                       |
| def \_\_init\_\_(self, device: str = \'auto\',                        |
|                                                                       |
| model_id: str = \'pyannote/speaker-diarization-community-1\'):        |
|                                                                       |
| hf_token = os.environ.get(\'HF_TOKEN\') or                            |
| os.environ.get(\'HUGGINGFACE_TOKEN\')                                 |
|                                                                       |
| if not hf_token:                                                      |
|                                                                       |
| raise EnvironmentError(                                               |
|                                                                       |
| \'HF_TOKEN environment variable not set. \'                           |
|                                                                       |
| \'Required for pyannote.audio. See hf.co/settings/tokens\'            |
|                                                                       |
| )                                                                     |
|                                                                       |
| self.pipeline = Pipeline.from_pretrained(model_id, token=hf_token)    |
|                                                                       |
| if device == \'auto\':                                                |
|                                                                       |
| device = \'cuda\' if torch.cuda.is_available() else \'cpu\'           |
|                                                                       |
| self.pipeline.to(torch.device(device))                                |
|                                                                       |
| def diarize(self, wav_path: str,                                      |
|                                                                       |
| min_speakers: int \| None = None,                                     |
|                                                                       |
| max_speakers: int \| None = None) -\> list\[dict\]:                   |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Returns list of {\'start\': float, \'end\': float, \'speaker\': str}. |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| kwargs = {}                                                           |
|                                                                       |
| if min_speakers: kwargs\[\'min_speakers\'\] = min_speakers            |
|                                                                       |
| if max_speakers: kwargs\[\'max_speakers\'\] = max_speakers            |
|                                                                       |
| with ProgressHook() as hook:                                          |
|                                                                       |
| output = self.pipeline(wav_path, hook=hook, \*\*kwargs)               |
|                                                                       |
| turns = \[\]                                                          |
|                                                                       |
| for turn, \_, speaker in output.itertracks(yield_label=True):         |
|                                                                       |
| turns.append({\'start\': turn.start, \'end\': turn.end, \'speaker\':  |
| speaker})                                                             |
|                                                                       |
| return turns                                                          |
|                                                                       |
| def assign_to_words(self, word_list: list, diarize_output: list) -\>  |
| list:                                                                 |
|                                                                       |
| \"\"\"Map speaker labels onto word-level timestamps using overlap     |
| heuristic.\"\"\"                                                      |
|                                                                       |
| for word in word_list:                                                |
|                                                                       |
| word_mid = (word\[\'start\'\] + word\[\'end\'\]) / 2                  |
|                                                                       |
| best_speaker = None                                                   |
|                                                                       |
| for turn in diarize_output:                                           |
|                                                                       |
| if turn\[\'start\'\] \<= word_mid \<= turn\[\'end\'\]:                |
|                                                                       |
| best_speaker = turn\[\'speaker\'\]                                    |
|                                                                       |
| break                                                                 |
|                                                                       |
| word\[\'speaker\'\] = best_speaker \# None if no match                |
|                                                                       |
| return word_list                                                      |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Important Notes for Diarization**                                   |
|                                                                       |
| HF token is mandatory --- store in environment, never hardcode in     |
| source.                                                               |
|                                                                       |
| RTTM format: diarization output also supports write_rttm() for        |
| standard output format.                                               |
|                                                                       |
| Offline mode: Download model with git lfs clone to a local path and   |
| use Pipeline.from_pretrained(\'/local/path\') for air-gapped          |
| environments.                                                         |
|                                                                       |
| Failure policy: Wrap diarize() in try/except. On any failure, log to  |
| ProcessingReport and return transcript without speakers.              |
|                                                                       |
| Known limitations: Quality degrades with overlapping speech,          |
| far-field microphones, short utterances (\<1s), and heavy noise.      |
|                                                                       |
| num_speakers hint: If speaker count is known, pass num_speakers for   |
| better accuracy. Otherwise use min/max bounds.                        |
|                                                                       |
| Telemetry: pyannote.audio sends anonymous usage metrics by default.   |
| Disable with: export PYANNOTE_METRICS_ENABLED=0                       |
+-----------------------------------------------------------------------+

**3.6 Parselmouth (praat-parselmouth) --- Prosody Feature Extraction**

**Role in Pipeline**

Parselmouth provides Praat\'s phonetics algorithms as a native Python
library --- no Praat installation required. It extracts the core prosody
features per segment: fundamental frequency (F0/pitch), intensity
(energy), jitter, shimmer, and harmonics-to-noise ratio (HNR).

**Installation**

+-----------------------------------------------------------------------+
| pip install praat-parselmouth                                         |
|                                                                       |
| \# Binary wheels available for Windows, Linux (manylinux), macOS.     |
|                                                                       |
| \# No external Praat installation needed.                             |
+-----------------------------------------------------------------------+

-   License: GPL-3.0 --- important: this means any code that imports and
    distributes parselmouth must also be GPL-3.0 or compatible. Use
    behind an optional adapter that is clearly documented.

-   Current version: 0.4.7 (November 2025).

-   Supports Python 3.7+, including PyPy.

**Exact API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/prosody/parselmouth.py                |
|                                                                       |
| import parselmouth                                                    |
|                                                                       |
| from parselmouth.praat import call                                    |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| class ParselmouthBackend:                                             |
|                                                                       |
| def extract_segment(self, wav_path: str,                              |
|                                                                       |
| start_s: float, end_s: float) -\> dict:                               |
|                                                                       |
| \"\"\"Extract prosody features for a single segment \[start_s,        |
| end_s\].\"\"\"                                                        |
|                                                                       |
| snd = parselmouth.Sound(wav_path)                                     |
|                                                                       |
| \# Trim to segment                                                    |
|                                                                       |
| snd = snd.extract_part(                                               |
|                                                                       |
| from_time=start_s, to_time=end_s,                                     |
|                                                                       |
| window_shape=parselmouth.WindowShape.RECTANGULAR,                     |
|                                                                       |
| relative_width=1.0, preserve_times=False                              |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# ── Pitch (F0) ────────────────────────────────────────────         |
|                                                                       |
| pitch = snd.to_pitch(time_step=0.01, pitch_floor=75,                  |
| pitch_ceiling=600)                                                    |
|                                                                       |
| pitch_values = pitch.selected_array\[\'frequency\'\]                  |
|                                                                       |
| pitch_values = pitch_values\[pitch_values \> 0\] \# voiced frames     |
| only                                                                  |
|                                                                       |
| f0_mean = float(np.mean(pitch_values)) if len(pitch_values) else 0.0  |
|                                                                       |
| f0_variance = float(np.var(pitch_values)) if len(pitch_values) else   |
| 0.0                                                                   |
|                                                                       |
| \# ── Intensity (Energy) ────────────────────────────────────         |
|                                                                       |
| intensity = snd.to_intensity(minimum_pitch=75, time_step=0.01)        |
|                                                                       |
| energy_mean = float(call(intensity, \'Get mean\', 0, 0, \'energy\'))  |
|                                                                       |
| energy_variance = float(call(intensity, \'Get standard deviation\',   |
| 0, 0)) \*\* 2                                                         |
|                                                                       |
| \# ── Voice Quality ─────────────────────────────────────────         |
|                                                                       |
| point_process = call(snd, \'To PointProcess (periodic, cc)\', 75,     |
| 600)                                                                  |
|                                                                       |
| jitter = call(point_process, \'Get jitter (local)\', 0, 0, 0.0001,    |
| 0.02, 1.3)                                                            |
|                                                                       |
| shimmer = call(\[snd, point_process\], \'Get shimmer (local)\',       |
|                                                                       |
| 0, 0, 0.0001, 0.02, 1.3, 1.6)                                         |
|                                                                       |
| \# HNR: Harmonics-to-Noise Ratio                                      |
|                                                                       |
| harmonicity = call(snd, \'To Harmonicity (cc)\', 0.01, 75, 0.1, 1.0)  |
|                                                                       |
| hnr = float(call(harmonicity, \'Get mean\', 0, 0))                    |
|                                                                       |
| return {                                                              |
|                                                                       |
| \'f0_mean\': f0_mean,                                                 |
|                                                                       |
| \'f0_variance\': f0_variance,                                         |
|                                                                       |
| \'energy_mean\': energy_mean,                                         |
|                                                                       |
| \'energy_variance\': energy_variance,                                 |
|                                                                       |
| \'jitter\': jitter,                                                   |
|                                                                       |
| \'shimmer\': shimmer,                                                 |
|                                                                       |
| \'hnr\': hnr,                                                         |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**Key Parameters**

  ------------------------------------------------------------------------
  **Parameter**      **Typical     **Description**
                     Value**
  ------------------ ------------- ---------------------------------------
  pitch_floor        75 Hz         Minimum expected F0. Use 75 Hz for
                                   adult speech. Lower for bass voices.

  pitch_ceiling      600 Hz        Maximum expected F0. 600 Hz covers most
                                   adult + child speech.

  time_step          0.01 s        Analysis frame step size (10 ms).
                                   Smaller = higher resolution but slower.

  minimum_pitch      75 Hz         Pitch floor used for intensity
  (intensity)                      extraction. Keep consistent with F0
                                   floor.
  ------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Parselmouth Gotchas**                                               |
|                                                                       |
| GPL License: parselmouth is GPL-3.0. It MUST be in an optional        |
| adapter (not a required import in core/). Document clearly.           |
|                                                                       |
| Case sensitivity: Praat action names are case-sensitive. Use \'To     |
| Intensity\' not \'to_intensity\' in praat.call(). The Pythonic object |
| methods (snd.to_pitch()) do exist for common operations.              |
|                                                                       |
| Unvoiced frames: pitch.selected_array\[\'frequency\'\] contains 0.0   |
| for unvoiced frames. Always filter these out before computing         |
| statistics.                                                           |
|                                                                       |
| Short segments: Parselmouth will raise errors on segments \< 0.04s.   |
| Guard against very short segments.                                    |
|                                                                       |
| Thread safety: parselmouth is NOT thread-safe. Use process-level      |
| parallelism (multiprocessing) not thread-level.                       |
+-----------------------------------------------------------------------+

**3.7 CoPaSul --- Advanced Prosody Stylization**

**Role in Pipeline**

CoPaSul (Contour-based, Parametric, Superpositional intonation
stylization) provides deeper prosody analysis beyond basic F0/energy
statistics. It models intonation as a superposition of global and local
polynomial contours, extracts register features, prosodic boundary
indicators, rhythm measures, and bottom-up contour classes. It is a
config-file-driven batch processor.

**Installation**

+-----------------------------------------------------------------------+
| pip install copasul                                                   |
|                                                                       |
| \# Requires Python 3. Configuration via JSON files.                   |
+-----------------------------------------------------------------------+

-   License: MIT

-   Input requirements: WAV files + TextGrid or custom annotation files
    marking segment boundaries.

-   Output: Python dict / pickle file with nested feature
    subdictionaries.

**API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/prosody/copasul.py                    |
|                                                                       |
| import json, pickle, tempfile, os                                     |
|                                                                       |
| from copasul import copasul                                           |
|                                                                       |
| class CoPaSulBackend:                                                 |
|                                                                       |
| def extract_features(self, config_dict: dict) -\> dict:               |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Run CoPaSul feature extraction.                                       |
|                                                                       |
| config_dict: CoPaSul JSON config. Must specify:                       |
|                                                                       |
| \- fsys.\*.dir : paths to audio, annotation, output directories       |
|                                                                       |
| \- prosody.\* : which feature sets to compute                         |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| fex = copasul.Copasul()                                               |
|                                                                       |
| copa = fex.process(config=config_dict)                                |
|                                                                       |
| return copa                                                           |
|                                                                       |
| def build_config(self, audio_dir: str, annot_dir: str,                |
|                                                                       |
| output_dir: str) -\> dict:                                            |
|                                                                       |
| \"\"\"Minimal CoPaSul config for speechtelemetry pipeline.\"\"\"      |
|                                                                       |
| return {                                                              |
|                                                                       |
| \"navigate\": {                                                       |
|                                                                       |
| \"from_scratch\": True,                                               |
|                                                                       |
| \"overwrite_config\": True                                            |
|                                                                       |
| },                                                                    |
|                                                                       |
| \"fsys\": {                                                           |
|                                                                       |
| \"audio\": {\"dir\": audio_dir, \"ext\": \"wav\"},                    |
|                                                                       |
| \"annot\": {\"dir\": annot_dir, \"ext\": \"TextGrid\"},               |
|                                                                       |
| \"export\": {\"dir\": output_dir}                                     |
|                                                                       |
| },                                                                    |
|                                                                       |
| \"prosody\": {                                                        |
|                                                                       |
| \"features\": \[\"f0\", \"energy\", \"rhythm\"\],                     |
|                                                                       |
| \"f0\": {\"min\": 75, \"max\": 600}                                   |
|                                                                       |
| }                                                                     |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **CoPaSul Integration Notes**                                         |
|                                                                       |
| CoPaSul is annotation-driven: it requires TextGrid or custom          |
| annotation files marking segment/syllable boundaries. Generate these  |
| from WhisperX word timestamps using a TextGrid helper.                |
|                                                                       |
| Warm start: Set config\[\'navigate\'\]\[\'from_scratch\'\] = False to |
| resume from an existing COPA pickle, avoiding recomputation.          |
|                                                                       |
| Feature output is a deeply nested Python dict. Access paths           |
| documented at github.com/reichelu/copasul.                            |
|                                                                       |
| CoPaSul is best suited for detailed linguistic prosody research. For  |
| production speed, use Parselmouth\'s direct extraction and reserve    |
| CoPaSul for detailed contour and rhythm features.                     |
|                                                                       |
| Run CoPaSul in its own virtual environment if dependency conflicts    |
| arise (it pulls scipy, numpy).                                        |
+-----------------------------------------------------------------------+

**3.8 SpeechBrain --- Speech Emotion Recognition**

**Role in Pipeline**

SpeechBrain provides the default emotion recognition backend. It runs on
each segment (utterance level) and returns a probability distribution
over emotion classes, not a single predicted label. The library MUST
store full probability distributions and confidence scores --- never
collapse to a single predicted label in the canonical output.

**Installation**

+-----------------------------------------------------------------------+
| pip install speechbrain                                               |
|                                                                       |
| \# Also required:                                                     |
|                                                                       |
| pip install transformers\>=4.30.0 huggingface_hub\>=0.8.0 soundfile   |
+-----------------------------------------------------------------------+

-   License: Apache 2.0 --- fully permissive.

-   Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP (default,
    auto-downloads from HuggingFace).

-   Classes (IEMOCAP): neutral, happy, sad, anger. Accuracy: \~75.3% on
    test set.

-   GPU: pass run_opts={\'device\': \'cuda\'} to from_hparams or
    foreign_class.

**Exact API Usage**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/backends/emotion/speechbrain.py                |
|                                                                       |
| from speechbrain.inference.interfaces import foreign_class            |
|                                                                       |
| import torch, tempfile, soundfile as sf                               |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| class SpeechBrainEmotionBackend:                                      |
|                                                                       |
| MODEL_HF = \'speechbrain/emotion-recognition-wav2vec2-IEMOCAP\'       |
|                                                                       |
| def \_\_init\_\_(self, device: str = \'auto\', savedir: str =         |
| \'.models/speechbrain\'):                                             |
|                                                                       |
| if device == \'auto\':                                                |
|                                                                       |
| device = \'cuda\' if torch.cuda.is_available() else \'cpu\'           |
|                                                                       |
| self.device = device                                                  |
|                                                                       |
| run_opts = {\'device\': device}                                       |
|                                                                       |
| self.classifier = foreign_class(                                      |
|                                                                       |
| source=self.MODEL_HF,                                                 |
|                                                                       |
| pymodule_file=\'custom_interface.py\',                                |
|                                                                       |
| classname=\'CustomEncoderWav2vec2Classifier\',                        |
|                                                                       |
| savedir=savedir,                                                      |
|                                                                       |
| run_opts=run_opts,                                                    |
|                                                                       |
| )                                                                     |
|                                                                       |
| def predict_segment(self, wav_path: str,                              |
|                                                                       |
| start_s: float, end_s: float) -\> dict:                               |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| Classify emotion for an audio segment.                                |
|                                                                       |
| Returns EmotionScore-compatible dict with full probability            |
| distribution.                                                         |
|                                                                       |
| \"\"\"                                                                |
|                                                                       |
| \# Write segment to temp WAV for classify_file()                      |
|                                                                       |
| data, sr = sf.read(wav_path)                                          |
|                                                                       |
| seg = data\[int(start_s \* sr):int(end_s \* sr)\]                     |
|                                                                       |
| if seg.ndim \> 1:                                                     |
|                                                                       |
| seg = seg.mean(axis=1) \# ensure mono                                 |
|                                                                       |
| with tempfile.NamedTemporaryFile(suffix=\'.wav\', delete=False) as f: |
|                                                                       |
| seg_path = f.name                                                     |
|                                                                       |
| try:                                                                  |
|                                                                       |
| sf.write(seg_path, seg, sr)                                           |
|                                                                       |
| out_prob, score, index, text_lab =                                    |
| self.classifier.classify_file(seg_path)                               |
|                                                                       |
| finally:                                                              |
|                                                                       |
| import os; os.unlink(seg_path)                                        |
|                                                                       |
| \# out_prob: tensor of shape (1, n_classes)                           |
|                                                                       |
| probs = out_prob.squeeze(0).tolist()                                  |
|                                                                       |
| labels = self.classifier.hparams.label_encoder.decode_ndim(           |
|                                                                       |
| torch.arange(len(probs)))                                             |
|                                                                       |
| \# Build probability distribution dict                                |
|                                                                       |
| label_dist = dict(zip(labels, probs))                                 |
|                                                                       |
| return {                                                              |
|                                                                       |
| \'label_distribution\': label_dist,                                   |
|                                                                       |
| \'top_label\': text_lab\[0\],                                         |
|                                                                       |
| \'confidence\': float(score.squeeze()),                               |
|                                                                       |
| \'backend\': \'speechbrain/emotion-recognition-wav2vec2-IEMOCAP\',    |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Emotion Recognition --- Critical Caveats**                          |
|                                                                       |
| NEVER output a single emotion label as authoritative. Always store    |
| and expose the full probability distribution.                         |
|                                                                       |
| Cross-corpus transfer: The IEMOCAP-trained model\'s benchmark scores  |
| do NOT transfer to other domains. Treat outputs as probabilistic      |
| estimates, not ground truth.                                          |
|                                                                       |
| Segment minimum length: SpeechBrain auto-resamples but very short     |
| segments (\<0.5s) produce unreliable results. Log a warning and skip  |
| classification for segments shorter than this.                        |
|                                                                       |
| Model caching: Models download to savedir on first run. Set savedir   |
| to a stable project-level path to avoid re-downloading.               |
|                                                                       |
| Labels are IEMOCAP-specific: neutral, happy, sad, anger. Other        |
| emotion models (emotion2vec, EmoBox) may return different label sets  |
| --- always include backend_name in EmotionScore.                      |
|                                                                       |
| Output probabilities do NOT sum to exactly 1.0 due to floating point  |
| --- normalize if needed.                                              |
+-----------------------------------------------------------------------+

**4. Fallback Backends**

All backends listed here are secondary options. They must be implemented
in the appropriate backends/ subdirectory with the same interface
contract as the default backend. The registry (registry.py) resolves
backend names to classes at runtime.

  ----------------------------------------------------------------------------------------
  **Stage**     **Fallback**      **When to Use**              **Installation**
  ------------- ----------------- ---------------------------- ---------------------------
  ASR           whisper.cpp       Maximum portability; no      pip install pywhispercpp OR
                                  Python ML deps; Windows      use prebuilt binary
                                  native.

  ASR           WhisperX          When integrated              pip install whisperx
                (combined)        word-align+ASR is needed in
                                  one pass.

  ASR           SenseVoice        When emotion+event detection pip install funasr; Docker
                                  alongside ASR is needed.     recommended

  Alignment     forcealign        Lightweight alternative to   pip install forcealign
                                  whisperx aligner.

  VAD           FunASR VAD        When already using FunASR    pip install funasr
                                  stack.

  Diarization   FunASR            CPU-only diarization without pip install funasr
                                  HuggingFace dependency.

  Prosody       librosa           Fast, dependency-light       pip install librosa
                                  F0+energy extraction.

  Prosody       pyAudioAnalysis   Feature sets compatible with pip install pyAudioAnalysis
                                  ML classification tasks.

  Prosody       audioFlux         High-performance C-core;     pip install audioflux
                                  GPU-accelerated spectral
                                  features.

  Emotion       emotion2vec       Multilingual SER via         pip install modelscope
                                  self-supervised              funasr
                                  pre-training.

  Emotion       EmoBox            Multi-corpus evaluation      See
                                  toolkit; useful for          github.com/emo-box/EmoBox
                                  benchmarking.

  Emotion       GigaAM-Emo        Russian-language SER; strong See Sber AI Hub
                                  performance on CIS speech.
  ----------------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Fallback Backend Contract**                                         |
|                                                                       |
| Every fallback backend module must implement the same method          |
| signature as the default backend for its stage.                       |
|                                                                       |
| Fallback backends that require non-permissive licenses or             |
| non-commercial-only use MUST live in an optional adapter and be       |
| clearly flagged in the repo README.                                   |
|                                                                       |
| Any backend requiring a HuggingFace token, model download acceptance, |
| or specific CUDA runtime MUST be documented in docs/ and checked for  |
| presence before the job starts.                                       |
|                                                                       |
| Use registry.py to map backend name strings to classes. Never         |
| hardwire imports in core/pipeline.py.                                 |
+-----------------------------------------------------------------------+

**5. License Policy & Compliance**

  -----------------------------------------------------------------------------
  **Package**         **License**    **Distribution   **Notes**
                                     Status**
  ------------------- -------------- ---------------- -------------------------
  faster-whisper      MIT            Required         Model weights under
                                     dependency ---   OpenAI license
                                     OK               (permissive)

  WhisperX            BSD-4-Clause   Required         HuggingFace token needed
                                     dependency ---   for diarization models
                                     OK

  silero-vad          MIT            Required         Zero restrictions
                                     dependency ---
                                     OK

  pyannote.audio      MIT (lib) +    Optional --- OK  Token required; accept
                      CC-BY-4.0      with attribution model license at hf.co
                      (models)

  praat-parselmouth   GPL-3.0        Optional adapter GPL is viral; must NOT be
                                     only --- REVIEW  a required default

  copasul             MIT            Optional --- OK

  speechbrain         Apache 2.0     Required /       Fully permissive
                                     Optional --- OK

  FFmpeg              LGPL 2.1 / GPL System           Use LGPL build (no
                      2 (depends on  dependency ---   \--enable-gpl) for clean
                      build)         OK (LGPL build)  distribution

  emotion2vec         Apache 2.0     Optional --- OK

  EmoBox              Apache 2.0     Optional --- OK

  librosa             ISC            Optional --- OK

  torch / torchaudio  BSD-style      Transitive ---
                                     OK
  -----------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **License Rules**                                                     |
|                                                                       |
| Core library defaults MUST be permissive (MIT, Apache, BSD, ISC).     |
|                                                                       |
| GPL-licensed components (praat-parselmouth) are ONLY permitted behind |
| optional adapters with explicit documentation.                        |
|                                                                       |
| Non-commercial-only components are NEVER permitted as required        |
| defaults.                                                             |
|                                                                       |
| Model weights on HuggingFace may have separate license terms from the |
| library code --- check the model card.                                |
|                                                                       |
| FFmpeg: Use a build without \--enable-gpl (i.e., LGPL 2.1) to avoid   |
| GPL contamination. Gyan.dev Windows builds offer both variants.       |
+-----------------------------------------------------------------------+

**6. Configuration Reference**

**6.1 PipelineConfig (config.py)**

+-----------------------------------------------------------------------+
| \# src/speechtelemetry/config.py                                      |
|                                                                       |
| from pydantic import BaseModel, Field                                 |
|                                                                       |
| from typing import Literal, Optional                                  |
|                                                                       |
| class PipelineConfig(BaseModel):                                      |
|                                                                       |
| \# Device                                                             |
|                                                                       |
| device: Literal\[\'auto\', \'cpu\', \'cuda\'\] = \'auto\'             |
|                                                                       |
| \# ASR                                                                |
|                                                                       |
| asr_backend: Literal\[\'faster-whisper\', \'whisperx\',               |
| \'whisper.cpp\', \'sensevoice\'\] = \'faster-whisper\'                |
|                                                                       |
| asr_model_size: str = \'large-v3\'                                    |
|                                                                       |
| asr_compute_type: Optional\[str\] = None \# None = auto               |
| (float16/int8)                                                        |
|                                                                       |
| asr_language: Optional\[str\] = None \# None = auto-detect            |
|                                                                       |
| asr_beam_size: int = 5                                                |
|                                                                       |
| \# VAD                                                                |
|                                                                       |
| vad_backend: Literal\[\'silero\', \'funasr\'\] = \'silero\'           |
|                                                                       |
| vad_threshold: float = 0.5                                            |
|                                                                       |
| vad_min_silence_ms: int = 300                                         |
|                                                                       |
| vad_min_speech_ms: int = 250                                          |
|                                                                       |
| \# Alignment                                                          |
|                                                                       |
| alignment_backend: Literal\[\'whisperx\', \'forcealign\'\] =          |
| \'whisperx\'                                                          |
|                                                                       |
| \# Diarization                                                        |
|                                                                       |
| diarization_backend: Optional\[Literal\[\'pyannote\', \'funasr\'\]\]  |
| = None \# None = disabled                                             |
|                                                                       |
| diarization_min_speakers: Optional\[int\] = None                      |
|                                                                       |
| diarization_max_speakers: Optional\[int\] = None                      |
|                                                                       |
| \# Prosody                                                            |
|                                                                       |
| prosody_backend: list\[Literal\[\'parselmouth\', \'copasul\',         |
| \'librosa\', \'audioflux\'\]\] = \[\'parselmouth\'\]                  |
|                                                                       |
| \# Emotion                                                            |
|                                                                       |
| emotion_backend: Optional\[Literal\[\'speechbrain\', \'emotion2vec\', |
| \'emobox\'\]\] = \'speechbrain\'                                      |
|                                                                       |
| \# Export                                                             |
|                                                                       |
| export_formats: list\[Literal\[\'json\', \'srt\', \'vtt\',            |
| \'textgrid\'\]\] = \[\'json\'\]                                       |
|                                                                       |
| \# Performance                                                        |
|                                                                       |
| chunk_audio: bool = True                                              |
|                                                                       |
| max_chunk_duration_s: float = 30.0                                    |
+-----------------------------------------------------------------------+

**6.2 Environment Variables**

  ---------------------------------------------------------------------------------
  **Variable**                **Purpose**              **Required For**
  --------------------------- ------------------------ ----------------------------
  HF_TOKEN                    HuggingFace read token   pyannote.audio, WhisperX
                                                       diarization, auto model
                                                       downloads

  PYANNOTE_METRICS_ENABLED    Disable telemetry (set   Privacy-conscious
                              to 0)                    deployments

  SPEECHTELEMETRY_CACHE_DIR   Override model cache     All backends (default:
                              directory                \~/.cache/speechtelemetry)

  CUDA_VISIBLE_DEVICES        Select GPU device(s)     Multi-GPU machines

  OMP_NUM_THREADS             Limit OpenMP threads     CPU-only deployments on
                              (CPU)                    shared machines
  ---------------------------------------------------------------------------------

**7. Error Handling & Fail-Soft Policy**

The library must implement fail-soft behavior: partial output is always
better than a hard crash. Errors at any stage are recorded in the
ProcessingReport and the pipeline continues with degraded output.

  -------------------------------------------------------------------------
  **Failure Mode**    **Policy**                    **ProcessingReport
                                                    Field**
  ------------------- ----------------------------- -----------------------
  FFmpeg not found    Raise EnvironmentError        N/A --- hard fail
                      immediately at startup. Do
                      not proceed.

  Model download      Raise BackendError with URL   N/A --- hard fail
  fails               and cache path. Do not
                      proceed.

  HF_TOKEN missing    Raise EnvironmentError at job N/A --- hard fail
  when diarization    start.
  enabled

  VAD returns no      Log warning, pass full audio  StageError with
  speech regions      to ASR, continue.             stage=\'vad\'

  ASR transcription   Return TranscriptDocument     StageError with
  empty               with empty segments list.     stage=\'asr\'
                      Continue.

  Alignment fails for Keep segment with original    StageError with
  a segment           timestamps, mark              stage=\'alignment\'
                      alignment_backend=\'none\'.

  Diarization fails   Return transcript without     StageError with
  entirely            speaker labels. Set all       stage=\'diarization\'
                      speakers=None.

  Prosody fails for a Return segment with all       StageError with
  segment             prosody fields=None.          stage=\'prosody\'

  Emotion fails for a Return segment with           StageError with
  segment             emotion=None.                 stage=\'emotion\'

  Parselmouth segment Skip, log warning, return     StageError with
  too short (\<40ms)  None prosody for that         stage=\'prosody\'
                      segment.
  -------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| \# Error recording pattern in core/pipeline.py                        |
|                                                                       |
| from dataclasses import dataclass, field                              |
|                                                                       |
| from typing import Optional                                           |
|                                                                       |
| \@dataclass                                                           |
|                                                                       |
| class StageError:                                                     |
|                                                                       |
| stage: str                                                            |
|                                                                       |
| message: str                                                          |
|                                                                       |
| exception_type: str                                                   |
|                                                                       |
| segment_index: Optional\[int\] = None                                 |
|                                                                       |
| \@dataclass                                                           |
|                                                                       |
| class ProcessingReport:                                               |
|                                                                       |
| real_time_factor: float = 0.0                                         |
|                                                                       |
| peak_ram_mb: float = 0.0                                              |
|                                                                       |
| peak_vram_mb: float = 0.0                                             |
|                                                                       |
| stage_timings: dict = field(default_factory=dict)                     |
|                                                                       |
| errors: list\[StageError\] = field(default_factory=list)              |
|                                                                       |
| \# Usage in pipeline:                                                 |
|                                                                       |
| try:                                                                  |
|                                                                       |
| prosody = prosody_backend.extract_segment(wav_path, seg.start,        |
| seg.end)                                                              |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| report.errors.append(StageError(                                      |
|                                                                       |
| stage=\'prosody\',                                                    |
|                                                                       |
| message=str(e),                                                       |
|                                                                       |
| exception_type=type(e).\_\_name\_\_,                                  |
|                                                                       |
| segment_index=i                                                       |
|                                                                       |
| ))                                                                    |
|                                                                       |
| prosody = None                                                        |
+-----------------------------------------------------------------------+

**8. Performance Targets & Telemetry**

**8.1 Required Performance Measurements**

Every job MUST record and include in ProcessingReport:

-   Real-time factor (RTF): wall_time_s / audio_duration_s. Target: RTF
    \< 0.2 on GPU (large-v3), RTF \< 2.0 on CPU (large-v3 int8).

-   Peak RAM usage: measure with tracemalloc or resource.getrusage().
    Target: \< 8 GB CPU-only.

-   Peak VRAM usage: torch.cuda.max_memory_allocated() if CUDA
    available.

-   Per-stage timing: use time.perf_counter() around each stage call.

**8.2 Memory Management Rules**

1.  Always delete large model objects (del model) and call gc.collect()
    when switching between ML stages on GPU.

2.  Call torch.cuda.empty_cache() after clearing GPU models.

3.  Process audio in chunks (max_chunk_duration_s, default 30s). Never
    load a full multi-hour WAV as a numpy array.

4.  Do NOT hold full feature tensors across the entire audio duration.
    Process segment-by-segment.

5.  For Parselmouth, load the full WAV once as parselmouth.Sound, then
    extract_part() per segment. Do NOT re-open the file per segment.

**8.3 Quality Gates (v0.1)**

  ------------------------------------------------------------------------
  **Metric**            **Target**         **Method**
  --------------------- ------------------ -------------------------------
  Transcript WER        \< 10% on clean    Evaluate on LibriSpeech
                        speech (en)        test-clean subset

  Word timestamp MAE    \< 80 ms mean      Compare to MFA-aligned fixtures
                        absolute error

  VAD precision/recall  precision \> 0.90, Evaluate on annotated silence
                        recall \> 0.85     fixtures

  Diarization error     \< 20% on          Evaluate on AMI corpus subset
  rate                  2-speaker clean
                        audio

  Emotion macro-F1      \> 0.60 on IEMOCAP SpeechBrain model benchmark
                        test set

  Runtime regression    \< 10% slowdown    Automated benchmark suite
                        between releases
  ------------------------------------------------------------------------

**9. Canonical Package Layout**

+-----------------------------------------------------------------------+
| src/speechtelemetry/                                                  |
|                                                                       |
| api.py \# Public enrich_media() entry point                           |
|                                                                       |
| config.py \# PipelineConfig (Pydantic model)                          |
|                                                                       |
| types.py \# All canonical dataclasses                                 |
|                                                                       |
| registry.py \# Backend name -\> class resolver                        |
|                                                                       |
| core/                                                                 |
|                                                                       |
| pipeline.py \# Orchestrates all stages; fail-soft logic               |
|                                                                       |
| job.py \# Job lifecycle; temp file management                         |
|                                                                       |
| provenance.py \# Backend provenance tracking                          |
|                                                                       |
| io/                                                                   |
|                                                                       |
| ffmpeg.py \# normalize_to_wav() using subprocess+FFmpeg               |
|                                                                       |
| audio_normalize.py \# Post-FFmpeg validation and chunking             |
|                                                                       |
| backends/                                                             |
|                                                                       |
| asr/                                                                  |
|                                                                       |
| faster_whisper.py \# DEFAULT ASR backend                              |
|                                                                       |
| whisperx.py \# Combined ASR+align fallback                            |
|                                                                       |
| whisper_cpp.py \# whisper.cpp fallback                                |
|                                                                       |
| sensevoice.py \# SenseVoice fallback                                  |
|                                                                       |
| alignment/                                                            |
|                                                                       |
| whisperx.py \# DEFAULT alignment backend                              |
|                                                                       |
| forcealign.py \# forcealign fallback                                  |
|                                                                       |
| vad/                                                                  |
|                                                                       |
| silero.py \# DEFAULT VAD backend                                      |
|                                                                       |
| funasr.py \# FunASR fallback                                          |
|                                                                       |
| diarization/                                                          |
|                                                                       |
| pyannote.py \# DEFAULT diarization backend                            |
|                                                                       |
| funasr.py \# FunASR fallback                                          |
|                                                                       |
| prosody/                                                              |
|                                                                       |
| parselmouth.py \# DEFAULT prosody backend (GPL adapter --- optional)  |
|                                                                       |
| copasul.py \# CoPaSul advanced prosody adapter                        |
|                                                                       |
| librosa.py \# librosa fallback                                        |
|                                                                       |
| audioflux.py \# audioFlux fallback                                    |
|                                                                       |
| emotion/                                                              |
|                                                                       |
| speechbrain.py \# DEFAULT emotion backend                             |
|                                                                       |
| emotion2vec.py \# emotion2vec fallback                                |
|                                                                       |
| emobox.py \# EmoBox fallback                                          |
|                                                                       |
| exporters/                                                            |
|                                                                       |
| json.py \# Authoritative canonical export                             |
|                                                                       |
| srt.py \# SRT subtitle export (lossy)                                 |
|                                                                       |
| vtt.py \# WebVTT export (lossy)                                       |
|                                                                       |
| textgrid.py \# Praat TextGrid export                                  |
|                                                                       |
| benchmarking/                                                         |
|                                                                       |
| fixtures.py \# Golden reference audio/transcripts                     |
|                                                                       |
| metrics.py \# WER, MAE, DER, F1 calculations                          |
|                                                                       |
| compare.py \# Backend comparison harness                              |
|                                                                       |
| cli/                                                                  |
|                                                                       |
| main.py \# Click/Typer CLI entry point                                |
|                                                                       |
| commands.py \# CLI subcommands                                        |
|                                                                       |
| tests/                                                                |
|                                                                       |
| examples/                                                             |
|                                                                       |
| docs/                                                                 |
+-----------------------------------------------------------------------+

**10. Developer & AI Agent Checklist**

Use this checklist before merging any implementation of a new backend or
pipeline stage.

**10.1 Environment**

6.  FFmpeg is installed and on PATH. Verified with ffmpeg -version.

7.  Python 3.10 or 3.11 virtual environment created and activated.

8.  PyTorch installed with correct CUDA version OR CPU-only build.

9.  HF_TOKEN set as environment variable (for pyannote, WhisperX
    diarization).

10. pyannote model license accepted at
    hf.co/pyannote/speaker-diarization-community-1.

**10.2 Backend Implementation**

11. New backend placed in correct backends/\*/ subdirectory.

12. Method signature matches the default backend for that stage.

13. Backend name registered in registry.py.

14. License checked and documented in registry.py comment.

15. GPL-licensed backends (parselmouth) placed in optional adapter only.

16. Backend availability check implemented (import error caught
    gracefully).

**10.3 Data Model**

17. All outputs mapped to types.py canonical types (Segment, Word,
    etc.).

18. EmotionScore always stores full probability distribution --- never a
    single label.

19. ProcessingReport populated with stage timings and any errors.

20. Backend provenance stored in every relevant output field.

**10.4 Error Handling**

21. All backends wrapped in try/except inside core/pipeline.py.

22. Failures append to ProcessingReport.errors and return None/empty for
    that field.

23. Short audio segments (\<40ms for prosody, \<500ms for emotion)
    handled gracefully.

24. Missing model weights, missing FFmpeg, missing HF_TOKEN all raise
    clear EnvironmentError at job start.

**10.5 Testing**

25. Unit test for each backend using a short (5s) canonical WAV fixture.

26. Integration test running full enrich_media() on a 30s sample.

27. All benchmarking metrics calculated against golden fixtures.

28. No VRAM / GPU left allocated after test teardown.

**11. Quick Reference --- Backend API Signatures**

+-----------------------------------------------------------------------+
| \# ── VAD ─────────────────────────────────────────────────────────   |
|                                                                       |
| SileroVADBackend.get_speech_intervals(wav_path: str) -\> list\[dict\] |
|                                                                       |
| \# Returns: \[{\'start\': float, \'end\': float}, \...\]              |
|                                                                       |
| \# ── ASR ─────────────────────────────────────────────────────────   |
|                                                                       |
| FasterWhisperBackend.transcribe(wav_path, language=None, beam_size=5) |
| -\> tuple                                                             |
|                                                                       |
| \# Returns: (list\[Segment\], TranscriptionInfo)                      |
|                                                                       |
| \# Segment has: .start, .end, .text, .words=None                      |
|                                                                       |
| \# ── ALIGNMENT ───────────────────────────────────────────────────   |
|                                                                       |
| WhisperXAlignmentBackend.align(segments, audio_path, language) -\>    |
| list\[dict\]                                                          |
|                                                                       |
| \# Returns: segments with \'words\' key per segment                   |
|                                                                       |
| \# Each word: {\'word\': str, \'start\': float, \'end\': float,       |
| \'score\': float}                                                     |
|                                                                       |
| \# ── DIARIZATION ─────────────────────────────────────────────────   |
|                                                                       |
| PyannoteBackend.diarize(wav_path, min_speakers=None,                  |
| max_speakers=None) -\> list\[dict\]                                   |
|                                                                       |
| \# Returns: \[{\'start\': float, \'end\': float, \'speaker\': str},   |
| \...\]                                                                |
|                                                                       |
| \# speaker format: \'SPEAKER_00\', \'SPEAKER_01\', etc.               |
|                                                                       |
| \# ── PROSODY ─────────────────────────────────────────────────────   |
|                                                                       |
| ParselmouthBackend.extract_segment(wav_path, start_s, end_s) -\> dict |
|                                                                       |
| \# Returns: {\'f0_mean\', \'f0_variance\', \'energy_mean\',           |
| \'energy_variance\',                                                  |
|                                                                       |
| \# \'jitter\', \'shimmer\', \'hnr\'}                                  |
|                                                                       |
| \# ── EMOTION ─────────────────────────────────────────────────────   |
|                                                                       |
| SpeechBrainEmotionBackend.predict_segment(wav_path, start_s, end_s)   |
| -\> dict                                                              |
|                                                                       |
| \# Returns: {\'label_distribution\': dict\[str,float\],               |
| \'top_label\': str,                                                   |
|                                                                       |
| \# \'confidence\': float, \'backend\': str}                           |
|                                                                       |
| \# ── EXPORT ──────────────────────────────────────────────────────   |
|                                                                       |
| \# All exporters accept a TranscriptDocument and output_path: str     |
|                                                                       |
| export_json(doc, output_path) \# Authoritative; preserves all fields  |
|                                                                       |
| export_srt(doc, output_path) \# Lossy; text+timestamps only           |
|                                                                       |
| export_vtt(doc, output_path) \# Lossy; WebVTT format                  |
|                                                                       |
| export_textgrid(doc, output_path) \# Praat TextGrid format            |
+-----------------------------------------------------------------------+

*Document version 0.1. Generated from spec v0.1. Last updated May 2026.*

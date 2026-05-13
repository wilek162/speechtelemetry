**Speech Telemetry Library Spec\
***Open-source transcript, subtitle, prosody, and emotion enrichment
library*

**Purpose.** Define the first production-grade open-source library
architecture, defaults, and developer-facing behavior for a local-first
speech intelligence engine that produces transcripts, subtitles, word
timing, silence spans, speaker turns, prosody features, and emotion
estimates.

# 1. Product definition

**Working model.** A library-first speech telemetry engine. The library
is the source of truth. CLI, UI, and API wrappers are secondary.

**Target platforms.** Windows first, then Linux. CPU-only must work.
GPU/CUDA is optional but first-class when available.

**Core output.** A canonical rich transcript object tree containing
document metadata, segments, words, speakers, silence spans, prosody
windows, emotion scores, confidence, and backend provenance.

# 2. Product principles

-   Local-first and offline-capable by default.

-   Deterministic, typed, inspectable outputs.

-   Probabilistic emotion, not fake certainty.

-   Pluggable backends behind one stable schema.

-   Fail soft: partial output is better than hard failure.

-   Prefer proven tooling over novelty for the default path.

# 3. Non-goals for v0.1

-   Streaming transcription.

-   Realtime conversational agent features.

-   Automatic summarization, chaptering, translation, or entity
    extraction.

-   Human-level emotion claims.

-   Mandatory diarization.

-   Web SaaS dependency.

-   A monolithic app with hardcoded model choices.

# 4. Backend selection and why

  ---------------------------------------------------------------------------------------------------------
  **Layer**     **Default         **GPU       **Why default**    **Fallbacks**      **Notes**
                backend**         support**
  ------------- ----------------- ----------- ------------------ ------------------ -----------------------
  ASR           faster-whisper    Yes, CUDA   Best speed/quality whisper.cpp,       Use WhisperX only when
                                              tradeoff for local WhisperX,          integrated
                                              batch              SenseVoice         alignment/diarization
                                              transcription                         is needed.

  Word timing / WhisperX or       Yes         Word-level         WhisperX aligner,  easytranscriber is
  alignment     easytranscriber               timestamps are     MFA-style aligner  young; keep as
                                              core;                                 alternate backend.
                                              easytranscriber is
                                              newer and faster
                                              in its own claims

  VAD / silence Silero VAD        Optional,   Very fast,         FunASR VAD         Use for silence spans
                                  mostly      lightweight,                          and speech
                                  CPU-first   accurate enough                       segmentation.
                                              for dead-air spans

  Diarization   pyannote.audio    Yes,        Best OSS           FunASR,            Make optional due to
                                  PyTorch     diarization        VibeVoice-ASR      setup friction.
                                  CUDA        ecosystem          structured output

  Prosody       Parselmouth +     Mostly CPU  Best fit for       librosa,           Parselmouth maps to
                CoPaSul                       pitch, intensity,  pyAudioAnalysis,   Praat; CoPaSul is
                                              rhythm, voice      audioFlux          prosody-focused.
                                              quality

  Emotion       SpeechBrain       Yes,        Strong existing    emotion2vec,       Output probabilities
                                  PyTorch     SER model          SenseVoice,        and confidence, not
                                  CUDA        ecosystem and      EmoBox, GigaAM-Emo absolute labels.
                                              explicit inference
                                              support

  Media ingest  FFmpeg            N/A         Universal          None               Always normalize to
                                              decode/transcode                      canonical audio first.
                                              layer
  ---------------------------------------------------------------------------------------------------------

# 5. Architecture

**Pipeline order.** Decode -\> normalize -\> VAD -\> ASR -\> alignment
-\> diarization -\> prosody -\> emotion -\> export.

1.  Decode media with FFmpeg into a canonical mono 16 kHz WAV.

2.  Run VAD before ASR to reduce wasted compute and improve gap
    detection.

3.  Transcribe with a fast ASR backend.

4.  Align words to audio time boundaries.

5.  Attach speakers when diarization is enabled.

6.  Compute prosody features on segments or sliding windows.

7.  Estimate emotion using probabilities and store backend provenance.

8.  Export JSON first, then SRT/VTT/TextGrid.

# 6. Public API

**Primary call pattern.** One-shot high level API plus step-by-step
stage APIs.

Example API shape:

result = enrich_media(\
input_path=\"sample.mp4\",\
config=PipelineConfig(\
asr_backend=\"faster-whisper\",\
vad_backend=\"silero\",\
diarization_backend=\"pyannote\",\
prosody_backend=\[\"parselmouth\", \"copasul\"\],\
emotion_backend=\"speechbrain\",\
device=\"auto\",\
),\
)

# 7. Canonical data model

-   TranscriptDocument: source metadata, language, duration, processing
    report, backend provenance.

-   Segment: start, end, text, speaker, confidence, silence_before_ms,
    silence_after_ms, prosody summary, emotion summary.

-   Word: text, start, end, confidence, token id if available, alignment
    provenance.

-   SilenceSpan: start, end, duration_ms, reason (speech gap /
    non-speech / overlap).

-   ProsodyWindow: f0 mean, f0 variance, energy mean, energy variance,
    speech rate, pause density, voice quality summary.

-   EmotionScore: label distribution, valence/arousal dimensions if
    supported, confidence, backend name.

# 8. Defaults and behavior decisions

  -----------------------------------------------------------------------
  **Default**                         **Decision**
  ----------------------------------- -----------------------------------
  Input handling                      Accept local audio/video files only
                                      in v0.1. Remote URLs and streaming
                                      are out.

  Audio normalization                 Always convert to mono 16 kHz WAV
                                      internally unless a backend has a
                                      strict alternative requirement.

  Device selection                    Auto-select CUDA when available,
                                      otherwise CPU. Never require GPU.

  Chunking                            Prefer chunked/batched inference
                                      for long media; avoid holding full
                                      feature tensors in memory when
                                      unnecessary.

  Silence detection                   Silero VAD is the default
                                      speech/silence boundary source.

  Speaker turns                       Optional. If diarization fails,
                                      return transcript without speakers
                                      rather than failing the job.

  Emotion output                      Probabilities only; store
                                      confidence and backend provenance.
                                      No absolute emotion claims.

  Prosody granularity                 Segment or short-window level by
                                      default. Do not pretend per-word
                                      prosody is robust.

  Exports                             JSON is authoritative. SRT/VTT are
                                      derived views and may lose
                                      metadata.

  Failure mode                        Partial results are acceptable;
                                      errors should be recorded in
                                      ProcessingReport.
  -----------------------------------------------------------------------

# 9. Caveats and constraints

-   Emotion recognition is probabilistic and domain-dependent. It must
    be framed as estimation, not truth.

-   Diarization quality degrades on overlap, noise, far-field
    microphones, and short utterances.

-   Prosody feature quality depends heavily on sampling rate, pitch
    tracking quality, and language/prosody norms.

-   Cross-corpus SER is still a hard research problem; benchmark scores
    do not transfer cleanly.

-   A single backend should never be hardwired into the public API.

-   Any backend with non-permissive licensing must not become a required
    dependency.

# 10. Dependency and license policy

-   Core library dependencies should be permissively licensed and
    redistributable.

-   Research-only or non-commercial components are allowed only behind
    optional adapters and must not ship as required defaults.

-   Optional adapters must fail gracefully when the dependency or model
    is absent.

-   The repository must document whether a backend requires a Hugging
    Face token, model download acceptance, or CUDA runtime.

# 11. Package layout

**Recommended package tree.**

src/speechtelemetry/

api.py

config.py

types.py

registry.py

core/pipeline.py

core/job.py

core/provenance.py

io/ffmpeg.py

io/audio_normalize.py

backends/asr/{whisperx.py,faster_whisper.py,whisper_cpp.py,sensevoice.py,easytranscriber.py}

backends/alignment/{whisperx.py,forcealign.py,mfa.py}

backends/vad/{silero.py,funasr.py}

backends/diarization/{pyannote.py,funasr.py,vibevoice.py}

backends/prosody/{parselmouth.py,copasul.py,librosa.py,audioflux.py,pyaudioanalysis.py}

backends/emotion/{speechbrain.py,emotion2vec.py,emobox.py,sensevoice.py,gigaam.py}

exporters/{json.py,srt.py,vtt.py,textgrid.py}

benchmarking/{fixtures.py,metrics.py,compare.py}

cli/main.py

cli/commands.py

tests/

examples/

docs/

# 12. Performance targets

-   CPU-only must be usable on consumer Windows machines.

-   GPU path should provide materially faster throughput, especially for
    long files.

-   All long-audio jobs should stream through the pipeline in chunks,
    not require full-memory processing.

-   The library should report real-time factor, peak RAM, peak VRAM, and
    backend timings for every job.

-   The default ASR engine should be the best balance of accuracy and
    throughput, not the most experimental model.

Practical quality gates for v0.1:

9.  Transcript WER/CER on the test corpus.

10. Word timestamp mean absolute error on forced-alignment fixtures.

11. VAD precision/recall and dead-air interval coverage.

12. Diarization error rate where speaker labels exist.

13. Emotion macro-F1 or top-k accuracy on labeled fixtures.

14. Runtime and memory regressions between releases.

# 13. Developer experience requirements

-   One install path for normal users; no manual model graph surgery.

-   Clear configuration defaults with a single config file and
    environment-variable overrides.

-   Backend availability checks before job start.

-   Helpful structured errors that state missing CUDA, missing FFmpeg,
    missing model weights, or unsupported file type.

-   Reproducible fixtures and golden outputs in the repo.

-   No hidden behavior based on model internals; everything meaningful
    should be explicit in the result object.

# 14. MVP scope

-   Local file ingest.

-   Canonical audio normalization.

-   Transcription with word-level timestamps.

-   Dead-air and silence detection.

-   Optional diarization.

-   Basic prosody feature extraction.

-   Emotion probabilities.

-   Structured JSON export plus subtitle export.

-   CLI and importable Python library.

Not in MVP:

15. Realtime streaming.

16. Summaries or chaptering.

17. Translation.

18. Web UI.

19. Cloud services.

20. Training pipelines.

# 15. Source map for implementation review

-   WhisperX: fast ASR with word-level timestamps, speaker diarization,
    and VAD preprocessing.

-   faster-whisper: faster Whisper reimplementation with CUDA support
    and quantization.

-   whisper.cpp: portable local inference with multiple acceleration
    backends and Windows/Linux support.

-   Silero VAD: lightweight speech activity detection with very fast CPU
    performance.

-   pyannote.audio: PyTorch speaker diarization toolkit with pretrained
    pipelines.

-   Parselmouth: Praat in Python, installable on Linux, macOS, and
    Windows.

-   CoPaSul: prosody-focused feature extraction for intonation, energy,
    rhythm, and voice quality.

-   SpeechBrain emotion recognition model: utterance-level SER with
    reported benchmark accuracy.

-   easytranscriber: newer forced-alignment/transcription backend with
    GPU-accelerated alignment.

-   VibeVoice-ASR: long-form structured transcription with
    Who/When/What.

-   SenseVoice: ASR + emotion recognition + audio event detection with
    Docker/GPU support.

-   EmoBox: multilingual multi-corpus SER toolkit and benchmark.

*Version 0.1 draft*

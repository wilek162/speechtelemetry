# License Policy

## Project License

speechtelemetry core is MIT licensed. All users may use, modify, and redistribute the library under MIT terms.

## Backend License Considerations

| Backend | License | Notes |
|---------|---------|-------|
| faster-whisper | MIT | Safe default for any project |
| whisperx | BSD-4-Clause | Generally compatible; check with legal for commercial use |
| silero-vad | MIT | Safe default |
| pyannote.audio | MIT (library) + CC-BY-4.0 (models) | Model weights require license acceptance at hf.co |
| praat-parselmouth | GPL-3.0 | **Viral license** — see below |
| copasul | MIT | Safe |
| speechbrain | Apache-2.0 | Safe for commercial use |
| soundfile | BSD | Safe |

## GPL-3.0 Warning: praat-parselmouth

`praat-parselmouth` is licensed under **GPL-3.0**, which is a copyleft (viral) license.

If your project links or bundles GPL-licensed code, your project may need to be GPL-licensed too.

**Recommended approach:**
- Do not include parselmouth in the core `dependencies` list in `pyproject.toml`.
- It is listed only in the optional `prosody` extra.
- If you are building a proprietary product, use the `librosa` or `audioflux` prosody backends instead (both MIT/BSD licensed).
- Always check with your legal team before using GPL-licensed code in commercial projects.

## HuggingFace Model Licenses

Some backends download models from HuggingFace Hub:

- `pyannote/speaker-diarization-community-1` — requires license acceptance at hf.co before use.
- `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` — Apache 2.0; auto-downloads, no extra steps.

Set `HF_TOKEN` environment variable to authenticate with HuggingFace.

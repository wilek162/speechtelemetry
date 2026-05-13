"""Script to generate tests/fixtures/sample_16k_mono.wav.

Run once: python tests/fixtures/generate_wav.py
Produces a deterministic 3s 440 Hz sine wave at 16 kHz mono 16-bit PCM.
"""

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION_S = 3.0
FREQUENCY = 440.0
AMPLITUDE = 0.5


def generate(output: Path) -> None:
    n_frames = int(SAMPLE_RATE * DURATION_S)
    samples = [
        int(AMPLITUDE * 32767 * math.sin(2 * math.pi * FREQUENCY * i / SAMPLE_RATE))
        for i in range(n_frames)
    ]
    with wave.open(str(output), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{n_frames}h", *samples))
    print(f"Written: {output} ({n_frames} frames, {DURATION_S}s @ {SAMPLE_RATE} Hz mono)")


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_16k_mono.wav"
    generate(out)

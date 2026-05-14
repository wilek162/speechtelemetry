"""SpeechBrain emotion recognition backend — default emotion stage.

License: Apache 2.0
Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
Classes: neutral, happy, sad, anger (~75.3% accuracy on IEMOCAP test set)

CRITICAL: Always store full probability distributions. Never collapse to a single label.
Cross-corpus transfer is limited — treat outputs as estimates, not ground truth.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import EmotionBackend

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    import torch
    from speechbrain.inference.interfaces import foreign_class

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class SpeechBrainEmotionBackend(EmotionBackend):
    """SpeechBrain wav2vec2 SER — Apache 2.0, auto-downloads from HuggingFace."""

    STAGE: ClassVar[str] = "emotion"
    NAME: ClassVar[str] = "speechbrain"
    MODEL_HF: ClassVar[str] = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def __init__(
        self,
        device: str = "auto",
        savedir: str = ".models/speechbrain",
    ) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "speechbrain is not installed.\nRun: pip install speechbrain transformers soundfile"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Prefer SPEECHTELEMETRY_CACHE_DIR if set
        cache_root = os.environ.get(
            "SPEECHTELEMETRY_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "speechtelemetry"),
        )
        savedir = os.path.join(cache_root, "speechbrain")
        os.makedirs(savedir, exist_ok=True)

        logger.info("Loading SpeechBrain emotion model on %s", device)
        self.classifier = foreign_class(
            source=self.MODEL_HF,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=savedir,
            run_opts={"device": device},
        )

    @classmethod
    def _check_available(cls) -> None:
        if not _AVAILABLE:
            raise BackendNotAvailableError(
                "speechbrain is not installed.\nRun: pip install speechbrain transformers soundfile"
            )

    def predict_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict[str, object]:
        """Classify emotion for an audio segment.

        Returns a dict with full probability distribution — never a single label.
        """
        data, sr = sf.read(wav_path)
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        segment = data[start_sample:end_sample]

        if segment.ndim > 1:
            segment = segment.mean(axis=1)

        # Use classify_batch() with a tensor directly — avoids the temp-file
        # path that torchaudio's classify_file() fails to resolve on Windows.
        # The pipeline guarantees wav_path is already mono 16 kHz PCM.
        wavs = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        out_prob, score, index, text_lab = self.classifier.classify_batch(wavs)

        probs = out_prob.squeeze(0).tolist()
        n_classes = len(probs)
        labels = [
            self.classifier.hparams.label_encoder.decode_ndim(torch.tensor([i]))[0]
            for i in range(n_classes)
        ]
        label_distribution = dict(zip(labels, probs, strict=False))
        top = (
            text_lab[0] if text_lab else max(label_distribution, key=label_distribution.__getitem__)
        )
        conf = float(score.squeeze())

        logger.debug(
            "SpeechBrain: [%.2f-%.2f] top=%s (%.3f) dist=%s",
            start_s,
            end_s,
            top,
            conf,
            {k: round(v, 3) for k, v in label_distribution.items()},
        )

        return {
            "label_distribution": label_distribution,
            "top_label": top,
            "confidence": conf,
            "backend_name": self.MODEL_HF,
        }

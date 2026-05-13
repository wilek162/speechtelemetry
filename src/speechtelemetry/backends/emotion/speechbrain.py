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
import tempfile
from typing import ClassVar

from speechtelemetry.exceptions import BackendNotAvailableError
from speechtelemetry.interfaces import EmotionBackend

logger = logging.getLogger(__name__)

try:
    from speechbrain.inference.interfaces import foreign_class
    import torch
    import soundfile as sf
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
                "speechbrain is not installed.\n"
                "Run: pip install speechbrain transformers soundfile"
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
                "speechbrain is not installed.\n"
                "Run: pip install speechbrain transformers soundfile"
            )

    def predict_segment(
        self,
        wav_path: str,
        start_s: float,
        end_s: float,
    ) -> dict:
        """Classify emotion for an audio segment.

        Returns a dict with full probability distribution — never a single label.
        """
        import numpy as np  # noqa: PLC0415

        data, sr = sf.read(wav_path)
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        segment = data[start_sample:end_sample]

        if segment.ndim > 1:
            segment = segment.mean(axis=1)  # ensure mono

        # Write segment to temp file for classify_file()
        seg_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                seg_path = f.name
            sf.write(seg_path, segment, sr)

            out_prob, score, index, text_lab = self.classifier.classify_file(seg_path)
        finally:
            if seg_path and os.path.exists(seg_path):
                os.unlink(seg_path)

        # out_prob: tensor of shape (1, n_classes)
        probs = out_prob.squeeze(0).tolist()

        # Get label names from the encoder
        n_classes = len(probs)
        labels = [
            self.classifier.hparams.label_encoder.decode_ndim(
                torch.tensor([i])
            )[0]
            for i in range(n_classes)
        ]

        label_distribution = dict(zip(labels, probs))

        return {
            "label_distribution": label_distribution,
            "top_label": text_lab[0] if text_lab else max(label_distribution, key=label_distribution.__getitem__),
            "confidence": float(score.squeeze()),
            "backend_name": self.MODEL_HF,
        }

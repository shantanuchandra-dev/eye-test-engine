"""Silero VAD via ONNX Runtime.

Matches silero_vad's OnnxWrapper: each step feeds concat(previous 64 samples, current 512)
→ 576 samples @ 16 kHz, plus LSTM state (see snakers4/silero-vad).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

import onnxruntime as ort

_CONTEXT_16K = 64
_CHUNK_16K = 512


class SileroOnnxVAD:
    """Stateful streaming VAD; call once per 512-sample float32 chunk in [-1, 1]."""

    def __init__(self, onnx_path: Union[str, Path]):
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Silero ONNX not found: {path}. "
                "From Eye_test_engine_v2 run: python -m voice.download_models"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_16K), dtype=np.float32)

    def speech_probability(self, audio_float32_512: np.ndarray) -> float:
        """audio_float32_512: shape (512,) float32 in [-1, 1]."""
        chunk = np.asarray(audio_float32_512, dtype=np.float32).reshape(1, _CHUNK_16K)
        x = np.concatenate([self._context, chunk], axis=1)
        out, self._state = self._session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._context = np.array(x[:, -_CONTEXT_16K:], dtype=np.float32, copy=True)
        return float(np.asarray(out, dtype=np.float64).ravel()[0])

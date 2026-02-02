from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class OnnxRuntimeBackendConfig:
    """
    Configuration for ONNX Runtime inference.

    - providers: ORT execution providers (e.g., ["CUDAExecutionProvider", "CPUExecutionProvider"])
    - input_name/output_name: override auto-selected I/O names if needed
    """

    providers: Optional[Sequence[str]] = None
    input_name: Optional[str] = None
    output_name: Optional[str] = None


class OnnxRuntimeBackend:
    """
    Minimal ONNX Runtime backend.

    Expects an NCHW float32 blob, typically shaped (1, 3, H, W).
    Returns the primary output as a NumPy array.
    """

    def __init__(self, model_path: PathLike, cfg: OnnxRuntimeBackendConfig = OnnxRuntimeBackendConfig()):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "onnxruntime is required for the ONNX backend. Install it with `pip install onnxruntime` "
                "(or `onnxruntime-gpu`)."
            ) from e

        self._ort = ort
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))

        sess_opts = ort.SessionOptions()
        providers = list(cfg.providers) if cfg.providers is not None else None
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)

        self.input_name = cfg.input_name or self.session.get_inputs()[0].name
        # If output_name not provided, pick first output.
        self.output_name = cfg.output_name or self.session.get_outputs()[0].name

    @property
    def providers_in_use(self) -> Sequence[str]:
        # ORT returns providers in priority order for this session.
        return tuple(self.session.get_providers())

    @property
    def available_providers(self) -> Sequence[str]:
        return tuple(self._ort.get_available_providers())

    def infer(self, blob: np.ndarray, extra_inputs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        inputs: Dict[str, Any] = {self.input_name: blob}
        if extra_inputs:
            inputs.update(extra_inputs)
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0]

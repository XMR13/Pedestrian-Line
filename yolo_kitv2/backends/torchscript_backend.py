from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TorchScriptBackendConfig:
    """
    Configuration for TorchScript inference.

    - device: "cpu" or "cuda" (if available)
    - half: cast input to float16 (only if the model expects it)
    - output_index: if the model returns multiple outputs, select this index
    """

    device: str = "cpu"
    half: bool = False
    output_index: int = 0


class TorchScriptBackend:
    """
    Minimal TorchScript backend using `torch.jit.load`.

    This is the most "plug-and-play" Torch option because it doesn't require model
    class code (unlike many raw .pt weight checkpoints).
    """

    def __init__(self, model_path: PathLike, cfg: TorchScriptBackendConfig = TorchScriptBackendConfig()):
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("torch is required for the TorchScript backend. Install with `pip install torch`.") from e

        self._torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))

        self.device = torch.device(cfg.device)
        self.half = cfg.half
        self.output_index = cfg.output_index

        model = torch.jit.load(str(self.model_path), map_location=self.device)
        model.eval()
        self.model = model

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.as_tensor(blob, device=self.device)
        if self.half:
            x = x.half()
        else:
            x = x.float()
        x = x.contiguous()

        with torch.no_grad():
            y = self.model(x)

        if isinstance(y, (tuple, list)):
            y = y[self.output_index]

        if hasattr(y, "detach"):
            y = y.detach()
        return y.to("cpu").numpy()


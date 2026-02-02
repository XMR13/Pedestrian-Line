from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TensorRTBackendConfig:
    """
    Configuration for TensorRT engine inference.

    Notes:
    - TensorRT engines require a CUDA-capable environment.
    - This backend uses Torch CUDA tensors for device buffers (no PyCUDA).
    """

    device: str = "cuda"
    input_name: Optional[str] = None
    output_name: Optional[str] = None
    output_index: int = 0


def _torch_dtype_from_trt(trt_dtype) -> "object":
    import torch  # type: ignore

    # Avoid importing tensorrt types at module import time; compare by name.
    name = getattr(trt_dtype, "name", str(trt_dtype)).lower()
    if "float16" in name or "fp16" in name:
        return torch.float16
    if "float32" in name or "fp32" in name:
        return torch.float32
    if "int8" in name:
        return torch.int8
    if "int32" in name:
        return torch.int32
    if "bool" in name:
        return torch.bool
    # Fallback
    return torch.float32


def _torch_device(device: str) -> "object":
    import torch  # type: ignore

    return torch.device(device)


class TensorRTBackend:
    """
    Minimal TensorRT engine runner.

    Supports both classic binding-based execution (execute_async_v2) and the newer
    tensor-name API (set_tensor_address + execute_async_v3) depending on what the
    installed TensorRT version exposes.
    """

    def __init__(self, engine_path: PathLike, cfg: TensorRTBackendConfig = TensorRTBackendConfig()):
        try:
            import tensorrt as trt  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "tensorrt is required for the TensorRT backend. Install NVIDIA TensorRT Python bindings."
            ) from e

        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("torch is required for the TensorRT backend buffers. Install with `pip install torch`.") from e

        self._trt = trt
        self._torch = torch

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(str(self.engine_path))

        self.device = _torch_device(cfg.device)
        if self.device.type != "cuda":
            raise ValueError("TensorRTBackend requires a CUDA device (device='cuda').")
        if not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError("CUDA is not available in this torch install, but TensorRT requires CUDA.")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = self.engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._use_io_tensors = hasattr(engine, "num_io_tensors")
        self.input_name, self.output_names = self._discover_io(cfg.input_name)
        if cfg.output_name is not None:
            self.primary_output = cfg.output_name
        else:
            if cfg.output_index < 0 or cfg.output_index >= len(self.output_names):
                raise IndexError(f"output_index {cfg.output_index} out of range (num outputs={len(self.output_names)}).")
            self.primary_output = self.output_names[cfg.output_index]

    def _discover_io(self, preferred_input: Optional[str]) -> Tuple[str, List[str]]:
        trt = self._trt
        engine = self.engine

        if self._use_io_tensors:
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if not inputs:
                raise RuntimeError("TensorRT engine has no inputs.")
            if not outputs:
                raise RuntimeError("TensorRT engine has no outputs.")
            input_name = preferred_input or inputs[0]
            if input_name not in inputs:
                raise ValueError(f"Input name {input_name!r} not found. Available: {inputs}")
            return input_name, outputs

        # Older binding API
        inputs: List[str] = []
        outputs: List[str] = []
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                inputs.append(name)
            else:
                outputs.append(name)
        if not inputs:
            raise RuntimeError("TensorRT engine has no inputs.")
        if not outputs:
            raise RuntimeError("TensorRT engine has no outputs.")
        input_name = preferred_input or inputs[0]
        if input_name not in inputs:
            raise ValueError(f"Input name {input_name!r} not found. Available: {inputs}")
        return input_name, outputs

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self._torch
        trt = self._trt

        if blob is None:
            raise TypeError("blob must be a NumPy array.")

        # Ensure input on CUDA with expected dtype
        input_shape = tuple(int(x) for x in np.asarray(blob).shape)
        if self._use_io_tensors:
            expected_dtype = self.engine.get_tensor_dtype(self.input_name)
        else:
            idx = self.engine.get_binding_index(self.input_name)
            expected_dtype = self.engine.get_binding_dtype(idx)
        torch_in_dtype = _torch_dtype_from_trt(expected_dtype)

        x = torch.as_tensor(blob, device=self.device).to(dtype=torch_in_dtype).contiguous()

        stream = torch.cuda.current_stream(device=self.device)
        stream_handle = int(stream.cuda_stream)

        if self._use_io_tensors:
            return self._infer_io_tensors(x, input_shape, stream_handle)
        return self._infer_bindings(x, input_shape, stream_handle)

    def _infer_io_tensors(self, x: "object", input_shape: Tuple[int, ...], stream_handle: int) -> np.ndarray:
        torch = self._torch
        ctx = self.context

        # Set shape (dynamic engines)
        if hasattr(ctx, "set_input_shape"):
            ctx.set_input_shape(self.input_name, input_shape)
        else:  # pragma: no cover
            # Fallback to binding-style shape setting if exposed.
            bidx = self.engine.get_binding_index(self.input_name)
            ctx.set_binding_shape(bidx, input_shape)

        # Allocate outputs on CUDA
        outputs: Dict[str, "object"] = {}
        for name in self.output_names:
            shape = tuple(int(s) for s in ctx.get_tensor_shape(name))
            dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(size=shape, dtype=dtype, device=self.device)

        # Bind addresses
        ctx.set_tensor_address(self.input_name, int(x.data_ptr()))
        for name, t in outputs.items():
            ctx.set_tensor_address(name, int(t.data_ptr()))

        if not hasattr(ctx, "execute_async_v3"):
            raise RuntimeError("TensorRT context does not support execute_async_v3 with IO tensors.")

        ok = ctx.execute_async_v3(stream_handle)
        if not ok:  # pragma: no cover
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        y = outputs[self.primary_output]
        return y.detach().to("cpu").numpy()

    def _infer_bindings(self, x: "object", input_shape: Tuple[int, ...], stream_handle: int) -> np.ndarray:
        torch = self._torch
        ctx = self.context
        engine = self.engine

        input_idx = engine.get_binding_index(self.input_name)
        ctx.set_binding_shape(input_idx, input_shape)

        bindings: List[int] = [0] * engine.num_bindings
        bindings[input_idx] = int(x.data_ptr())

        outputs: Dict[str, "object"] = {}
        for name in self.output_names:
            out_idx = engine.get_binding_index(name)
            shape = tuple(int(s) for s in ctx.get_binding_shape(out_idx))
            dtype = _torch_dtype_from_trt(engine.get_binding_dtype(out_idx))
            t = torch.empty(size=shape, dtype=dtype, device=self.device)
            outputs[name] = t
            bindings[out_idx] = int(t.data_ptr())

        if not hasattr(ctx, "execute_async_v2"):
            raise RuntimeError("TensorRT context does not support execute_async_v2.")

        ok = ctx.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
        if not ok:  # pragma: no cover
            raise RuntimeError("TensorRT execute_async_v2 failed.")

        y = outputs[self.primary_output]
        return y.detach().to("cpu").numpy()


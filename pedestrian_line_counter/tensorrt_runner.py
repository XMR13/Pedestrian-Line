from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _try_import_tensorrt():
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "TensorRT backend requires the 'tensorrt' Python package (Jetson: installed via JetPack / apt)."
        ) from exc
    return trt


def _try_import_cudart():
    try:
        from cuda.bindings import runtime as cudart  # type: ignore
        return cudart
    except Exception:
        try:
            from cuda import cudart  # type: ignore
            return cudart
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "TensorRT backend requires CUDA runtime bindings. "
                "Install a compatible 'cuda-python' package on the target device."
            ) from exc


def _cuda_call(cudart, result):
    """
    Helper kecil untuk cuda-python runtime 

    Terkadang runtime mengembalikan tuple (cudaError_t, value?) dimana elemen pertama
    merupakan error code
.
    """

    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected CUDA runtime return type: {type(result)}")
    if len(result) == 0:
        raise TypeError("Unexpected CUDA runtime return tuple: empty")
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        name = getattr(err, "name", None) or str(err)
        raise RuntimeError(f"CUDA runtime call failed: {name}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


@dataclass(frozen=True)
class _Binding:
    name: str
    index: int  # Binding index (for execute_v2) when available; else 0..N-1
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    device_ptr: int


class TensorRTRunner:
    """
    Minimal TensorRT engine runner untuk model seperti YOLO

    Runner ini tidak terlalu bergantung pada dependensi
    - Memerulkan `tensorrt` (biasanya disediakan oleh Jetpack/Hardware NVIDIA)
    - Menggunakan `cuda-python` untuk cuda runtime
    """

    def __init__(
        self,
        engine_path: Path,
        *,
        device_id: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 1,
    ) -> None:
        self._engine_path = Path(engine_path)
        trt = _try_import_tensorrt()
        cudart = _try_import_cudart()

        self._trt = trt
        self._cudart = cudart
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._input_size = input_size
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if not self._engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self._engine_path}")

        _cuda_call(cudart, cudart.cudaSetDevice(int(device_id)))

        runtime = trt.Runtime(self._logger)
        with self._engine_path.open("rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self._engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._runtime = runtime
        self._engine = engine
        self._context = context

        self._stream = _cuda_call(cudart, cudart.cudaStreamCreate())

        # Lebih mempertimabngkan profile 0 jia tersedia .
        if hasattr(self._context, "set_optimization_profile_async"):
            try:
                self._context.set_optimization_profile_async(0, self._stream)
            except Exception:
                pass

        self._bindings: List[_Binding] = []
        self._inputs: List[_Binding] = []
        self._outputs: List[_Binding] = []
        self._host_outputs: Dict[str, np.ndarray] = {}

        self._use_v3 = hasattr(self._context, "execute_async_v3") and hasattr(self._context, "set_tensor_address")
        self._bindings_meta = self._get_bindings_meta()
        self._tensor_names = [name for _, name in self._bindings_meta]

        self._init_bindings()
        self._bindings_ptrs: Optional[List[int]] = self._build_bindings_ptrs()

        self.input_name = self._inputs[0].name if self._inputs else ""
        self.output_names = [b.name for b in self._outputs]
        self.input_hw = self._infer_input_hw(self._inputs[0].shape if self._inputs else ())
        self.input_dtype = self._inputs[0].dtype if self._inputs else np.dtype(np.float32)

    def _get_bindings_meta(self) -> Sequence[Tuple[int, str]]:
        """
        Return (binding_index, name) pairs:

        Important : Untuk API execute_v2/execute_async_v2, pointer array di indexkan dengan
        binding index (0..num_binding-1). Beberpaa versi TensorRT menampilkan 
        enumerasi kedua IO-Tensor, untuk v2 kita harus menggunakan indeks binding
        Return (binding_index, name) pairs.
        """

        if hasattr(self._engine, "num_bindings") and hasattr(self._engine, "get_binding_name"):
            n = int(self._engine.num_bindings)
            return [(i, str(self._engine.get_binding_name(i))) for i in range(n)]

        if hasattr(self._engine, "num_io_tensors") and hasattr(self._engine, "get_tensor_name"):
            n = int(self._engine.num_io_tensors)
            names = [str(self._engine.get_tensor_name(i)) for i in range(n)]
            if hasattr(self._engine, "get_binding_index"):
                try:
                    return [(int(self._engine.get_binding_index(name)), name) for name in names]
                except Exception:
                    pass
            # jika kita tidka memiliki binding dan v2 diperlukan, tidka bisa secara safely bergerak
            if not self._use_v3:
                raise RuntimeError(
                    "TensorRT engine exposes IO tensors but no binding indices, and v3 execution is unavailable."
                )
            return list(enumerate(names))

        raise RuntimeError("Unsupported TensorRT engine API (missing IO/binding enumeration)")

    def _build_bindings_ptrs(self) -> Optional[List[int]]:
        if self._use_v3:
            return None
        if hasattr(self._engine, "num_bindings"):
            n = int(self._engine.num_bindings)
            ptrs = [0] * n
            for binding in self._bindings:
                if 0 <= binding.index < n:
                    ptrs[binding.index] = int(binding.device_ptr)
            return ptrs
        # Best-effort fallback if binding indices are 0..N-1.
        return [int(b.device_ptr) for b in sorted(self._bindings, key=lambda x: x.index)]

    def _tensor_is_input(self, name: str, index: int) -> bool:
        trt = self._trt
        if hasattr(self._engine, "get_tensor_mode") and hasattr(trt, "TensorIOMode"):
            return self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if hasattr(self._engine, "binding_is_input"):
            return bool(self._engine.binding_is_input(index))
        raise RuntimeError("Unsupported TensorRT engine API (cannot query input/output)")

    def _tensor_dtype(self, name: str, index: int) -> np.dtype:
        trt = self._trt
        if hasattr(self._engine, "get_tensor_dtype") and hasattr(trt, "nptype"):
            return np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
        if hasattr(self._engine, "get_binding_dtype") and hasattr(trt, "nptype"):
            return np.dtype(trt.nptype(self._engine.get_binding_dtype(index)))
        raise RuntimeError("Unsupported TensorRT engine API (cannot query tensor dtype)")

    def _tensor_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if hasattr(self._engine, "get_tensor_shape"):
            shape = tuple(int(x) for x in self._engine.get_tensor_shape(name))
        elif hasattr(self._engine, "get_binding_shape"):
            shape = tuple(int(x) for x in self._engine.get_binding_shape(index))
        else:
            raise RuntimeError("Unsupported TensorRT engine API (cannot query tensor shape)")
        return shape

    def _runtime_tensor_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if hasattr(self._context, "get_tensor_shape"):
            shape = tuple(int(x) for x in self._context.get_tensor_shape(name))
        elif hasattr(self._context, "get_binding_shape"):
            shape = tuple(int(x) for x in self._context.get_binding_shape(index))
        else:
            shape = self._tensor_shape(name, index)
        return shape

    def _set_input_shape(self, name: str, index: int, shape: Tuple[int, ...]) -> None:
        if hasattr(self._context, "set_input_shape"):
            self._context.set_input_shape(name, shape)
            return
        if hasattr(self._context, "set_binding_shape"):
            self._context.set_binding_shape(index, shape)
            return
        raise RuntimeError("Unsupported TensorRT context API (cannot set dynamic input shapes)")

    def _desired_input_shape(self, engine_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Build a concrete input shape for dynamic engines based on configured input_size.
        """

        if self._input_size is None:
            raise RuntimeError(
                "Dynamic TensorRT engine detected, but no input_size was provided. "
                "Pass input_size=(w,h) from ModelConfig.input_size."
            )
        input_w, input_h = self._input_size

        if len(engine_shape) == 4:
            # Common NCHW. Preserve known channel count if present.
            c = 3 if engine_shape[1] in (-1, 0, 3) else int(engine_shape[1])
            return (self._batch_size, c, int(input_h), int(input_w))
        if len(engine_shape) == 3:
            # CHW (rare in TRT engines, but handle).
            c = 3 if engine_shape[0] in (-1, 0, 3) else int(engine_shape[0])
            return (c, int(input_h), int(input_w))
        raise RuntimeError(f"Unsupported TensorRT input rank for dynamic shape: {engine_shape}")

    def _init_bindings(self) -> None:
        cudart = self._cudart

        # If any input is dynamic, set concrete shapes first.
        for index, name in self._bindings_meta:
            is_input = self._tensor_is_input(name, index)
            if not is_input:
                continue
            engine_shape = self._tensor_shape(name, index)
            if any(dim <= 0 for dim in engine_shape):
                self._set_input_shape(name, index, self._desired_input_shape(engine_shape))

        if hasattr(self._context, "all_binding_shapes_specified"):
            try:
                if not self._context.all_binding_shapes_specified:
                    raise RuntimeError("TensorRT context shapes are not fully specified after setting input shapes")
            except Exception:
                pass

        for index, name in self._bindings_meta:
            is_input = self._tensor_is_input(name, index)
            dtype = self._tensor_dtype(name, index)
            shape = self._runtime_tensor_shape(name, index)
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT tensor '{name}' has unresolved shape {shape}. "
                    "Build a fixed-shape engine or ensure the runner sets input shapes correctly."
                )

            nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
            device_ptr = int(_cuda_call(cudart, cudart.cudaMalloc(nbytes)))
            binding = _Binding(
                name=str(name),
                index=int(index),
                is_input=bool(is_input),
                dtype=dtype,
                shape=shape,
                nbytes=nbytes,
                device_ptr=device_ptr,
            )
            self._bindings.append(binding)

            if is_input:
                self._inputs.append(binding)
            else:
                self._outputs.append(binding)
                self._host_outputs[binding.name] = np.empty(binding.shape, dtype=binding.dtype)

        if not self._inputs:
            raise RuntimeError("TensorRT engine has no inputs")
        if len(self._inputs) != 1:
            names = [binding.name for binding in self._inputs]
            raise RuntimeError(
                "TensorRT backend currently supports only single-input engines. "
                f"Engine inputs: {names}"
            )
        if not self._outputs:
            raise RuntimeError("TensorRT engine has no outputs")

    @staticmethod
    def _infer_input_hw(shape: Tuple[int, ...]) -> Tuple[int, int]:
        # Common cases: (1,3,H,W) or (3,H,W)
        if len(shape) == 4:
            return int(shape[3]), int(shape[2])
        if len(shape) == 3:
            return int(shape[2]), int(shape[1])
        raise RuntimeError(f"Unsupported TensorRT input shape: {shape}")

    def infer(self, input_array: np.ndarray) -> List[np.ndarray]:
        """
        Menjalankan inference

        Mengembalikan view ke output buffer, yang akan di overwrite 
        di panggilan selanjutnya.
        """

        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be a numpy array")
        inp = self._inputs[0]
        if input_array.dtype != inp.dtype:
            # Common case: preprocessing generates float32 but FP16 engines expect float16.
            try:
                input_array = input_array.astype(inp.dtype, copy=False)
            except Exception as exc:
                raise TypeError(f"Expected {inp.dtype} input, got {input_array.dtype}") from exc
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

        if tuple(int(x) for x in input_array.shape) != inp.shape:
            raise ValueError(f"Input shape mismatch: got {input_array.shape}, expected {inp.shape}")

        cudart = self._cudart
        stream = self._stream

        _cuda_call(
            cudart,
            cudart.cudaMemcpyAsync(
                inp.device_ptr,
                int(input_array.ctypes.data),
                int(inp.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
        )

        if self._use_v3:
            for binding in self._bindings:
                self._context.set_tensor_address(binding.name, int(binding.device_ptr))
            ok = bool(self._context.execute_async_v3(stream))
        elif hasattr(self._context, "execute_async_v2"):
            assert self._bindings_ptrs is not None
            ok = bool(self._context.execute_async_v2(self._bindings_ptrs, stream))
        elif hasattr(self._context, "execute_v2"):
            assert self._bindings_ptrs is not None
            ok = bool(self._context.execute_v2(self._bindings_ptrs))
        else:  # pragma: no cover - defensive
            raise RuntimeError("Unsupported TensorRT execution context API (missing execute methods)")

        if not ok:
            raise RuntimeError("TensorRT execution failed")

        for out in self._outputs:
            host = self._host_outputs[out.name]
            _cuda_call(
                cudart,
                cudart.cudaMemcpyAsync(
                    int(host.ctypes.data),
                    out.device_ptr,
                    int(out.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
            )

        _cuda_call(cudart, cudart.cudaStreamSynchronize(stream))
        return [self._host_outputs[name] for name in self.output_names]

    def close(self) -> None:
        cudart = self._cudart
        try:
            if self._stream:
                _cuda_call(cudart, cudart.cudaStreamDestroy(self._stream))
        except Exception:
            pass
        for binding in self._bindings:
            try:
                _cuda_call(cudart, cudart.cudaFree(int(binding.device_ptr)))
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

from __future__ import annotations

import builtins
from pathlib import Path
import types

import numpy as np
import pytest

from pedestrian_line_counter.config import ModelConfig
from pedestrian_line_counter.detector import Detector
from pedestrian_line_counter.tensorrt_runner import TensorRTRunner, _Binding, _select_default_primary_output_name, _try_import_cudart


def test_tensorrt_backend_reports_missing_deps(monkeypatch, tmp_path: Path) -> None:
    # Create a fake engine file so initialization gets to the import step.
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"not-an-engine")

    cfg = ModelConfig(backend="tensorrt")
    cfg.model_path = engine_path
    cfg.allow_all_classes = True

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"tensorrt", "cuda", "cuda.bindings", "cuda.bindings.runtime"}:
            raise ImportError("forced-missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        Detector(cfg)


def test_tensorrt_backend_rejects_onnx_path(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not-onnx")

    cfg = ModelConfig(backend="tensorrt")
    cfg.model_path = onnx_path
    cfg.allow_all_classes = True

    with pytest.raises(RuntimeError):
        Detector(cfg)


def test_tensorrt_infer_fn_rejects_ambiguous_multi_output() -> None:
    class _FakeRunner:
        def infer(self, _blob):
            return [
                np.zeros((1, 84, 8400), dtype=np.float32),
                np.zeros((1, 32, 160), dtype=np.float32),
            ]

    infer_fn = Detector._build_trt_infer_fn(_FakeRunner())

    with pytest.raises(RuntimeError, match="could not identify a unique YOLO-style output tensor"):
        infer_fn(np.zeros((1, 3, 640, 640), dtype=np.float32))


def test_tensorrt_infer_fn_prefers_primary_output_fast_path() -> None:
    class _FakeRunner:
        primary_output_name = "output0"

        def infer_primary(self, _blob):
            return np.zeros((1, 84, 8400), dtype=np.float32)

        def infer(self, _blob):
            raise AssertionError("infer() should not be called when infer_primary() is available")

    infer_fn = Detector._build_trt_infer_fn(_FakeRunner())
    output = infer_fn(np.zeros((1, 3, 640, 640), dtype=np.float32))

    assert output.shape == (1, 84, 8400)


def test_select_default_primary_output_name_prefers_single_output() -> None:
    outputs = [
        _Binding(
            name="output0",
            index=1,
            is_input=False,
            dtype=np.dtype(np.float32),
            shape=(1, 84, 8400),
            nbytes=1,
            device_ptr=1,
        )
    ]

    assert _select_default_primary_output_name(outputs) == "output0"


def test_select_default_primary_output_name_returns_none_for_ambiguous_outputs() -> None:
    outputs = [
        _Binding(
            name="boxes",
            index=1,
            is_input=False,
            dtype=np.dtype(np.float32),
            shape=(1, 84, 8400),
            nbytes=1,
            device_ptr=1,
        ),
        _Binding(
            name="proto",
            index=2,
            is_input=False,
            dtype=np.dtype(np.float32),
            shape=(1, 32, 160),
            nbytes=1,
            device_ptr=2,
        ),
    ]

    assert _select_default_primary_output_name(outputs) is None


def test_try_import_cudart_falls_back_to_legacy_cuda_python_import(monkeypatch) -> None:
    legacy_cudart = types.SimpleNamespace(cudaError_t=types.SimpleNamespace(cudaSuccess=0))
    cuda_pkg = types.SimpleNamespace(cudart=legacy_cudart)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cuda.bindings":
            raise ImportError("no modern cuda.bindings")
        if name == "cuda":
            return cuda_pkg
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _try_import_cudart() is legacy_cudart


def test_tensorrt_execute_reuses_host_input_buffer_for_dtype_conversion() -> None:
    success = object()

    class _FakeCudaMemcpyKind:
        cudaMemcpyHostToDevice = "h2d"

    class _FakeCudart:
        cudaError_t = types.SimpleNamespace(cudaSuccess=success)
        cudaMemcpyKind = _FakeCudaMemcpyKind

        @staticmethod
        def cudaMemcpyAsync(dst, src, nbytes, kind, stream):
            _ = dst
            _ = src
            _ = nbytes
            _ = kind
            _ = stream
            return (success,)

    class _FakeContext:
        def execute_async_v2(self, _bindings_ptrs, _stream):
            return True

    runner = object.__new__(TensorRTRunner)
    runner._inputs = [
        _Binding(
            name="images",
            index=0,
            is_input=True,
            dtype=np.dtype(np.float16),
            shape=(1, 3, 2, 2),
            nbytes=int(np.dtype(np.float16).itemsize * 12),
            device_ptr=123,
        )
    ]
    runner._cudart = _FakeCudart()
    runner._stream = 7
    runner._use_v3 = False
    runner._bindings_ptrs = [123]
    runner._context = _FakeContext()
    runner._input_host_buffer = None
    runner._registered_host_ptrs = set()

    blob = np.ones((1, 3, 2, 2), dtype=np.float32)

    runner._execute(blob)
    first_buffer = runner._input_host_buffer
    runner._execute(blob * 2.0)

    assert first_buffer is runner._input_host_buffer
    assert runner._input_host_buffer.dtype == np.float16
    assert np.allclose(runner._input_host_buffer, np.full((1, 3, 2, 2), 2.0, dtype=np.float16))


def test_tensorrt_host_buffer_registration_is_best_effort() -> None:
    success = object()
    calls = []

    class _FakeCudart:
        cudaError_t = types.SimpleNamespace(cudaSuccess=success)

        @staticmethod
        def cudaHostRegister(ptr, nbytes, flags):
            calls.append(("register", ptr, nbytes, flags))
            return (success,)

        @staticmethod
        def cudaHostUnregister(ptr):
            calls.append(("unregister", ptr))
            return (success,)

    runner = object.__new__(TensorRTRunner)
    runner._cudart = _FakeCudart()
    runner._registered_host_ptrs = set()

    arr = np.empty((1, 3, 2, 2), dtype=np.float16)
    runner._try_register_host_buffer(arr)
    runner._try_register_host_buffer(arr)
    runner._try_unregister_host_buffer(arr)

    assert len([c for c in calls if c[0] == "register"]) == 1
    assert len([c for c in calls if c[0] == "unregister"]) == 1


def test_tensorrt_host_buffer_registration_falls_back_when_unsupported() -> None:
    runner = object.__new__(TensorRTRunner)
    runner._cudart = types.SimpleNamespace()
    runner._registered_host_ptrs = set()

    arr = np.empty((1, 3, 2, 2), dtype=np.float16)
    runner._try_register_host_buffer(arr)

    assert runner._registered_host_ptrs == set()

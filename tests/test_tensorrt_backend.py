from __future__ import annotations

import builtins
from pathlib import Path
import types

import numpy as np
import pytest

from pedestrian_line_counter.config import ModelConfig
from pedestrian_line_counter.detector import Detector
from pedestrian_line_counter.tensorrt_runner import _try_import_cudart


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

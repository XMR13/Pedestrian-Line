from pathlib import Path

import pytest

from pedestrian_line_counter.config import ModelConfig
from pedestrian_line_counter.detector import Detector


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

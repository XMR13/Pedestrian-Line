# TensorRT `.engine` Bring-Up Summary

This document summarizes, step by step, how we got the TensorRT `.engine` path running on the Jetson environment.

## 1. Start from the existing TensorRT backend

The project already had a TensorRT path in place:

- `--backend tensorrt`
- `.engine` loading in the detector
- a custom TensorRT runner

But it had not yet been proven with a real engine on the target device.

## 2. Audit the actual Python compatibility

The first concern was the Python version mismatch between the project baseline and the TensorRT environment.

Instead of relying only on packaging metadata, we checked the actual source code and found:

- most runtime modules were already close to Python 3.8-compatible
- the main real blocker was `argparse.BooleanOptionalAction`
- some test files also needed postponed annotation evaluation

## 3. Make the CLI compatible with older Python

We added a small compatibility helper for boolean flags so the runtime still supports:

- `--foo`
- `--no-foo`

even when `argparse.BooleanOptionalAction` is not available.

This was applied to:

- `pedestrian_line_counter/main.py`
- `pedestrian_line_counter/portal_uploader.py`

## 4. Clean up test import compatibility

We added `from __future__ import annotations` to the remaining test files that used `X | Y` annotation syntax directly.

This reduced the chance of Python 3.8 failing during test import.

## 5. Review the TensorRT backend itself

We inspected the TensorRT path and found several risks:

- multi-input engines were not truly supported
- multi-output engines could be interpreted incorrectly
- relative `.engine` paths were less robust than the ONNX path

## 6. Harden the TensorRT backend before using it

We added guardrails so unsupported cases fail clearly instead of failing silently.

Changes made:

- reject unsupported multi-input engines explicitly
- reject ambiguous multi-output layouts instead of guessing the wrong output
- resolve relative `.engine` paths more predictably from the project root

## 7. Hit the real environment issue

The first real TensorRT run failed because the runner could not import CUDA runtime bindings:

- `ModuleNotFoundError: No module named 'cuda'`

That showed the next blocker was the Jetson environment, not the detector logic.

## 8. Identify the target environment correctly

The target device is on:

- JetPack 5.1
- TensorRT `8.5.2.2`

That matters because newer generic `cuda-python` package paths are not always compatible with this environment.

## 9. Add support for the older CUDA Python import layout

Originally, the runner only tried the modern import:

```python
from cuda.bindings import runtime as cudart
```

That failed on the Jetson.

So the runner was patched to try both:

1. modern layout:
   `from cuda.bindings import runtime as cudart`
2. legacy layout:
   `from cuda import cudart`

This made the runner compatible with the CUDA Python package layout actually available on the device.

## 10. Verify the environment directly on-device

We verified the required imports manually:

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "from cuda import cudart; print('legacy cudart ok')"
python3 -c "from cuda.bindings import runtime as cudart; print('modern cudart ok')"
```

Observed result:

- `tensorrt` import worked
- legacy `from cuda import cudart` worked
- modern `cuda.bindings` did not exist

That was enough for the patched runner to proceed.

## 11. Run the real `.engine`

After the fallback patch and import verification, the TensorRT backend was run with the real engine.

It worked successfully.

This was the key milestone: real inference through the `.engine`, not just code review or synthetic tests.

## 12. Benchmark the working engine

### Stride 3, no draw, no write

```text
[bench] done frames=7635 processed=2545 fps=31.77 read=11.6ms det=58.2ms track=0.1ms count=0.2ms draw=0.0ms write=0.0ms
[main] Done. A->B: 10, B->A: 2
```

### Stride 2 with `--show`

```text
[bench] done frames=7635 processed=3818 fps=14.15 read=12.6ms det=62.9ms track=0.1ms count=0.2ms draw=5.1ms write=0.0ms
[main] Done. A->B: 10, B->A: 3
```

### Stride 2, no draw, no write

```text
[bench] done frames=7635 processed=3818 fps=26.08 read=10.4ms det=54.8ms track=0.1ms count=0.2ms draw=0.0ms write=0.0ms
```

## 13. Interpret the result

From those runs, we concluded:

- TensorRT is actually working on the target machine
- the detector is still the main bottleneck
- `--show` reduces throughput significantly
- `stride 3` is less reliable for counting than `stride 2`
- `stride 2` is the safer operating point

## 14. Final working scope

At the end of the session, the TensorRT path was working for this narrower supported case:

- JetPack 5.1
- TensorRT `8.5.2.2`
- legacy `cuda-python` import path
- single-input YOLO-style `.engine`
- real inference confirmed by benchmark

## 15. One-sentence summary

We got the TensorRT `.engine` running by combining:

- a minimal Python compatibility patch
- safer TensorRT backend guardrails
- a fallback for the older CUDA Python import layout
- direct on-device validation against the real Jetson environment

## 16. Recommended operating mode

For current deployment, the practical direction is:

- `--backend tensorrt`
- `--frame-stride 2`
- no `--show` in production
- no output writing unless needed
- RTSP validation as the next operational step

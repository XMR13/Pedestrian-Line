# TensorRT `.engine` Debugging Journey on Jetson Xavier NX

This note documents the troubleshooting path from the first `.engine` / TensorRT backend failure to the final working setup and benchmark results.

---

## Context

Project:
- `~/Development/Line_counting`

Target hardware / platform:
- Jetson device with JetPack-provided TensorRT
- System TensorRT Python package available under system Python 3.8

Project situation:
- Main project environment was using **Python 3.10**
- TensorRT Python bindings installed by JetPack were available only for **Python 3.8**

Goal:
- Run the project with:
  - `--backend tensorrt`
  - TensorRT `.engine` model

---

## 1. First TensorRT problem: `.engine` backend failed in the Python 3.10 environment

When running the project in the existing project environment, the TensorRT backend crashed with:

```text
ModuleNotFoundError: No module named 'tensorrt'
```

Full failure path looked like this:

```text
File "/home/iks-ai3/Development/Line_counting/pedestrian_line_counter/tensorrt_runner.py", line 12, in _try_import_tensorrt
    import tensorrt as trt
ModuleNotFoundError: No module named 'tensorrt'
...
RuntimeError: TensorRT backend requires the 'tensorrt' Python package (Jetson: installed via JetPack / apt).
```

### What this proved

The issue was **not** the `.engine` file itself yet.
The program failed before model execution because the Python environment could not import `tensorrt`.

### Root cause

TensorRT from JetPack was installed for the system Python stack:

- `/usr/bin/python3`
- Python **3.8**
- TensorRT module present in `/usr/lib/python3.8/dist-packages/tensorrt/...`

But the project environment used **Python 3.10**, so it could not see or use the JetPack TensorRT bindings.

---

## 2. Wrong path that was ruled out

A tempting but incorrect conclusion would have been:
- downgrade the whole project to Python 3.8

That was **not acceptable** because the project relied on a Python 3.10-based environment / workflow.

So the actual constraint became:

- keep project compatibility with Python 3.10 where needed
- still find a practical way to run TensorRT on Jetson

---

## 3. Environment recovery after breaking the existing environment

During troubleshooting, the environment became unusable.
Fortunately, there was a backup visible in the project directory:

```text
.venv_backup
```

### Recovery command

The environment was restored by replacing the broken `.venv` with the backup:

```bash
cd ~/Development/Line_counting

deactivate 2>/dev/null || true
mv .venv .venv_broken_$(date +%Y%m%d_%H%M%S)
mv .venv_backup .venv

source .venv/bin/activate
python -V
which python
```

This recovered the original environment state.

---

## 4. Trying Python 3.8 directly for the TensorRT path

Since TensorRT bindings existed in Python 3.8, the next practical test was to try the TensorRT path using Python 3.8.

This moved the failure further forward, which was progress.

New error:

```text
File "/home/iks-ai3/Development/Line_counting/pedestrian_line_counter/tensorrt_runner.py", line 22, in _try_import_cudart
    from cuda.bindings import runtime as cudart
ModuleNotFoundError: No module named 'cuda'
...
RuntimeError: TensorRT backend requires CUDA runtime bindings. Install 'cuda-python' (pip) on the target device.
```

### What this proved

At this point:
- `tensorrt` import was no longer the blocker
- the next blocker was the CUDA runtime Python binding expected by the custom TensorRT runner

So the TensorRT runner had another dependency:

```python
from cuda.bindings import runtime as cudart
```

---

## 5. Attempt to install `cuda-python`

The obvious next step was to install `cuda-python`.

Attempt:

```bash
pip install cuda-python
```

But it failed with:

```text
ERROR: Could not find a version that satisfies the requirement cuda-bindings~=13.0.3 (from cuda-python)
ERROR: No matching distribution found for cuda-bindings~=13.0.3 (from cuda-python)
```

### Interpretation

This happened because pip tried to install a **new CUDA Python package family** that did not match the Jetson environment / available wheels.

In other words:
- the custom runner expected `cuda-python`
- but the latest pip package set was not a clean fit for the current Jetson stack

This showed that the issue was no longer just "TensorRT missing".
It was now a **runner dependency / packaging compatibility** issue.

---

## 6. Final outcome: TensorRT path was made to work

After continuing the setup and dependency work, the TensorRT `.engine` backend was eventually made to run successfully.

The exact final working state was confirmed by real benchmark runs using:

- TensorRT backend
- `.engine` model
- line counting pipeline

So the troubleshooting sequence successfully moved from:

1. `tensorrt` import failure
2. CUDA runtime binding failure
3. packaging friction around `cuda-python`
4. final working TensorRT execution

---

## 7. Benchmark results after the fix

### A. Stride 3, `--no-draw`, `--no-write`

Command mode:
- `--frame-stride 3`
- `--no-draw`
- `--no-write`

Result:

```text
[bench] done frames=7635 processed=2545 fps=31.77 read=11.6ms det=58.2ms track=0.1ms count=0.2ms draw=0.0ms write=0.0ms
[main] Done. A->B: 10, B->A: 2
```

### B. Stride 2 with display (`--show`)

Command mode:
- `--frame-stride 2`
- `--show`

Result:

```text
[bench] done frames=7635 processed=3818 fps=14.15 read=12.6ms det=62.9ms track=0.1ms count=0.2ms draw=5.1ms write=0.0ms
[main] Done. A->B: 10, B->A: 3
```

### C. Stride 2, no draw / no write

Command mode:
- `--frame-stride 2`
- `--no-write`
- `--no-draw`

Result:

```text
[bench] done frames=7635 processed=3818 fps=26.08 read=10.4ms det=54.8ms track=0.1ms count=0.2ms draw=0.0ms write=0.0ms
```

---

## 8. Performance interpretation

### Main bottleneck

The benchmark clearly showed that the detector remained the dominant cost:

- detection: ~55–63 ms
- tracking: ~0.1 ms
- counting: ~0.2 ms
- draw/write: negligible when disabled

So the TensorRT pipeline worked, but the runtime was still overwhelmingly dominated by detection.

### Why `--show` is much slower

With `--show`, FPS dropped sharply:

- ~26 FPS without drawing / display
- ~14 FPS with display

This means the UI / display path is expensive on the Jetson and should be treated as a debugging-only mode.

---

## 9. Accuracy vs speed tradeoff: stride 2 vs stride 3

During a 5-minute pilot test:

- **stride 2 detected one more case than stride 3**
- the difference was small, but real

### Meaning

This strongly suggests:
- stride 3 is faster
- stride 2 preserves better temporal coverage
- line crossings near the boundary can be missed when too many frames are skipped

### Practical deployment decision

Even though stride 3 gave the highest raw FPS, **stride 2 was the better real deployment setting** because:

- it still achieved around **25–26 FPS** without display/write overhead
- it preserved count quality better
- line counting values recall / stable crossing detection more than raw benchmark speed

---

## 10. Final recommendation

### Recommended production mode

Use:

```bash
--backend tensorrt --frame-stride 2 --no-write --no-draw
```

### Why

Because this mode gave the best balance between:
- real-time performance
- stable counting
- acceptable detection coverage

### Avoid for production

Avoid using `--show` in production because it cuts FPS significantly.
Use it only for debugging / visual verification.

---

## 11. Key lessons learned

1. A TensorRT `.engine` failure in Python may actually be an **environment / binding mismatch**, not an engine problem.
2. On Jetson, JetPack TensorRT Python bindings are often tied to the **system Python version**.
3. A custom TensorRT runner may require **extra CUDA Python bindings**, which can become a second compatibility problem.
4. Fixing import issues does not guarantee the whole stack works; the next dependency may fail immediately after.
5. For this line-counting pipeline, **stride 2** is the practical sweet spot.

---

## 12. Final status

Status at the end of debugging:

- TensorRT `.engine` backend: **working**
- Benchmark achieved: **~26 FPS practical mode**
- Best deployment setting found so far: **stride 2, no draw, no write**
- Accuracy note: **stride 2 detected one more case than stride 3 in pilot testing**

---

## 13. Suggested follow-up tests

For a stronger deployment decision, test the same scene with:

- stride 1
- stride 2
- stride 3

and compare:

- total count
- missed crossings
- false crossings
- FPS

That will give a proper evidence-backed deployment choice.


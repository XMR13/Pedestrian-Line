from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Protocol

import cv2
import numpy as np


class VideoFrameWriter(Protocol):
    def write(self, frame: np.ndarray) -> None:
        ...

    def release(self) -> None:
        ...


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass
class OutputWriterConfig:
    path: Path
    fps: float
    width: int
    height: int
    encoder: str = "auto"  # auto | ffmpeg | opencv
    ffmpeg_codec: str = "libx264"
    ffmpeg_preset: str = "slow"
    ffmpeg_crf: int = 28
    ffmpeg_pix_fmt: str = "yuv420p"
    opencv_fourcc: str = "mp4v"


class OpenCvVideoWriter:
    def __init__(self, path: Path, fourcc: str, fps: float, width: int, height: int) -> None:
        code = str(fourcc or "mp4v").strip() or "mp4v"
        if len(code) != 4:
            raise ValueError("OpenCV fourcc must be exactly 4 characters.")
        self.path = Path(path)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*code),
            float(fps),
            (int(width), int(height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open OpenCV VideoWriter for output: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


class FFmpegVideoWriter:
    def __init__(self, cfg: OutputWriterConfig) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg executable not found in PATH.")

        self.path = Path(cfg.path)
        self._proc = subprocess.Popen(
            build_ffmpeg_command(cfg, ffmpeg_bin=ffmpeg_bin),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            self._proc.kill()
            raise RuntimeError("Could not open ffmpeg stdin pipe.")

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe is not available.")
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(_read_ffmpeg_error(self._proc)) from exc

    def release(self) -> None:
        stderr_text = ""
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        try:
            stderr_bytes = self._proc.stderr.read() if self._proc.stderr is not None else b""
        except Exception:
            stderr_bytes = b""
        try:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        except Exception:
            stderr_text = ""
        return_code = self._proc.wait()
        if return_code != 0:
            msg = stderr_text.strip() or f"ffmpeg exited with code {return_code}"
            raise RuntimeError(msg)


def create_video_writer(cfg: OutputWriterConfig) -> VideoFrameWriter:
    mode = str(cfg.encoder or "auto").strip().lower()
    if mode not in {"auto", "ffmpeg", "opencv"}:
        raise ValueError("output encoder must be 'auto', 'ffmpeg', or 'opencv'")

    if mode in {"auto", "ffmpeg"}:
        try:
            return FFmpegVideoWriter(cfg)
        except Exception:
            if mode == "ffmpeg":
                raise

    return OpenCvVideoWriter(
        cfg.path,
        cfg.opencv_fourcc,
        cfg.fps,
        cfg.width,
        cfg.height,
    )


def build_ffmpeg_command(cfg: OutputWriterConfig, *, ffmpeg_bin: str = "ffmpeg") -> List[str]:
    width = int(cfg.width)
    height = int(cfg.height)
    fps = max(float(cfg.fps), 1e-3)
    crf = max(0, min(int(cfg.ffmpeg_crf), 51))
    codec = str(cfg.ffmpeg_codec or "libx264").strip() or "libx264"
    preset = str(cfg.ffmpeg_preset or "slow").strip() or "slow"
    pix_fmt = str(cfg.ffmpeg_pix_fmt or "yuv420p").strip() or "yuv420p"

    return [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
        str(Path(cfg.path)),
    ]


def _read_ffmpeg_error(proc: subprocess.Popen[bytes]) -> str:
    try:
        if proc.stderr is None:
            return "ffmpeg pipe failed."
        raw = proc.stderr.read()
        return raw.decode("utf-8", errors="replace").strip() or "ffmpeg pipe failed."
    except Exception:
        return "ffmpeg pipe failed."


__all__ = [
    "FFmpegVideoWriter",
    "OpenCvVideoWriter",
    "OutputWriterConfig",
    "VideoFrameWriter",
    "build_ffmpeg_command",
    "create_video_writer",
    "is_ffmpeg_available",
]

from __future__ import annotations

import io
from pathlib import Path
import subprocess

import pytest

import pedestrian_line_counter.output_writer as output_writer
from pedestrian_line_counter.output_writer import FFmpegVideoWriter, OutputWriterConfig, build_ffmpeg_command, create_video_writer


def test_build_ffmpeg_command_uses_h264_crf_and_faststart(tmp_path) -> None:
    cfg = OutputWriterConfig(
        path=tmp_path / "out.mp4",
        fps=25.0,
        width=1280,
        height=720,
        encoder="ffmpeg",
        ffmpeg_codec="libx264",
        ffmpeg_preset="slow",
        ffmpeg_crf=28,
        ffmpeg_pix_fmt="yuv420p",
    )

    cmd = build_ffmpeg_command(cfg, ffmpeg_bin="ffmpeg")

    assert cmd[:4] == ["ffmpeg", "-y", "-f", "rawvideo"]
    assert "libx264" in cmd
    assert "slow" in cmd
    assert "28" in cmd
    assert "+faststart" in cmd
    assert str(tmp_path / "out.mp4") == cmd[-1]


def test_create_video_writer_ffmpeg_mode_requires_binary(monkeypatch, tmp_path) -> None:
    cfg = OutputWriterConfig(
        path=tmp_path / "out.mp4",
        fps=25.0,
        width=320,
        height=240,
        encoder="ffmpeg",
    )

    monkeypatch.setattr(output_writer.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg executable not found"):
        create_video_writer(cfg)


class _FakePipe:
    def close(self) -> None:
        return None


class _FakeProc:
    def __init__(
        self,
        *,
        return_code: int,
        stderr_text: str,
        communicate_timeout_count: int = 0,
    ) -> None:
        self.stdin = _FakePipe()
        self.stderr = io.BytesIO(stderr_text.encode("utf-8"))
        self._return_code = return_code
        self._communicate_timeout_count = communicate_timeout_count
        self.terminated = False
        self.killed = False
        self.returncode = None

    def wait(self) -> int:
        self.returncode = self._return_code
        return self._return_code

    def communicate(self, timeout=None):
        _ = timeout
        if self._communicate_timeout_count > 0:
            self._communicate_timeout_count -= 1
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
        self.returncode = self._return_code
        return b"", self.stderr.read()

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_ffmpeg_release_allows_expected_interrupt_exit() -> None:
    writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
    writer._proc = _FakeProc(
        return_code=255,
        stderr_text="Exiting normally, received signal 2.",
    )

    writer.release(interrupted=True)


def test_ffmpeg_release_still_raises_without_interrupt_flag() -> None:
    writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
    writer._proc = _FakeProc(
        return_code=255,
        stderr_text="Exiting normally, received signal 2.",
    )

    with pytest.raises(RuntimeError, match="received signal 2"):
        writer.release(interrupted=False)


def test_ffmpeg_release_forces_stop_when_interrupted_finalize_takes_too_long() -> None:
    writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
    fake_proc = _FakeProc(
        return_code=255,
        stderr_text="",
        communicate_timeout_count=1,
    )
    writer._proc = fake_proc

    writer.release(interrupted=True)

    assert fake_proc.terminated is True

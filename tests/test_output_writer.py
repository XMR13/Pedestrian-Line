from __future__ import annotations

from pathlib import Path

import pytest

from pedestrian_line_counter.output_writer import OutputWriterConfig, build_ffmpeg_command, create_video_writer


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


def test_create_video_writer_ffmpeg_mode_requires_binary(tmp_path) -> None:
    cfg = OutputWriterConfig(
        path=tmp_path / "out.mp4",
        fps=25.0,
        width=320,
        height=240,
        encoder="ffmpeg",
    )

    with pytest.raises(RuntimeError, match="ffmpeg executable not found"):
        create_video_writer(cfg)

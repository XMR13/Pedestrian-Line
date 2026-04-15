"""
Quick RTSP connectivity probe for the AI box.

Examples:
  python3 scripts/check_rtsp.py --url rtsp://192.168.1.25:8554/live/ezviz_cam01
  python3 scripts/check_rtsp.py --url rtsp://192.168.1.25:8554/live/ezviz_cam01 --backend gstreamer
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from pedestrian_line_counter.videoio_utils import open_video_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an RTSP stream and decode a few frames.")
    parser.add_argument("--url", required=True, help="RTSP URL to open.")
    parser.add_argument(
        "--backend",
        choices=["opencv", "gstreamer"],
        default="opencv",
        help="Capture backend to try first.",
    )
    parser.add_argument(
        "--transport",
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport used when backend=gstreamer.",
    )
    parser.add_argument(
        "--codec",
        choices=["h264", "h265"],
        default="h264",
        help="Expected RTSP codec used when backend=gstreamer.",
    )
    parser.add_argument("--latency-ms", type=int, default=120, help="GStreamer RTSP latency.")
    parser.add_argument("--open-timeout-ms", type=int, default=5000, help="Open timeout in milliseconds.")
    parser.add_argument("--read-timeout-ms", type=int, default=5000, help="Read timeout in milliseconds.")
    parser.add_argument("--frames", type=int, default=10, help="How many frames to decode before succeeding.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()

    cap = open_video_capture(
        args.url,
        open_timeout_ms=args.open_timeout_ms,
        read_timeout_ms=args.read_timeout_ms,
        rtsp_capture_backend=args.backend,
        rtsp_transport=args.transport,
        rtsp_latency_ms=args.latency_ms,
        rtsp_codec=args.codec,
    )

    print("url:", args.url)
    print("opened:", cap.isOpened())
    if hasattr(cap, "getBackendName"):
        try:
            print("backend:", cap.getBackendName())
        except Exception:
            pass

    if not cap.isOpened():
        cap.release()
        print("result: FAIL (could not open stream)")
        return 2

    decoded = 0
    first_shape = None
    for _ in range(max(args.frames, 1)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        if first_shape is None:
            first_shape = frame.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    elapsed = time.time() - started_at
    print("reported_resolution:", f"{reported_w}x{reported_h}")
    print("reported_fps:", fps)
    print("decoded_frames:", decoded)
    if first_shape is not None:
        print("first_frame_shape:", f"{first_shape[1]}x{first_shape[0]}")
    print("elapsed_s:", f"{elapsed:.2f}")

    if decoded <= 0:
        print("result: FAIL (opened stream but could not decode frames)")
        return 3

    print("result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

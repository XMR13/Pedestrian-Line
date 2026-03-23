from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class MockDeliveryBackend:
    def __init__(self, *, host: str, port: int, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((host, port), self._make_handler())

    @property
    def address(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def serve_forever(self) -> None:
        print(f"[mock-delivery] listening on {self.address}")
        self._server.serve_forever()

    def _make_handler(self):
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length)

                if self.path == "/api/runs/upsert":
                    payload = json.loads(body.decode("utf-8"))
                    run_uid = str(payload.get("run_uid") or "unknown")
                    out_path = backend.state_dir / "runs" / f"{run_uid}.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    self._send_json(200, {"ok": True, "run_uid": run_uid})
                    return

                if self.path == "/api/events/upsert":
                    payload = json.loads(body.decode("utf-8"))
                    accepted = []
                    for event in payload.get("events", []):
                        event_uid = str(event.get("event_uid") or "unknown")
                        out_path = backend.state_dir / "events" / f"{event_uid}.json"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
                        accepted.append(event_uid)
                    self._send_json(200, {"ok": True, "accepted_event_uids": accepted})
                    return

                if self.path.startswith("/api/events/") and self.path.endswith("/thumbnail"):
                    event_uid = self.path.split("/")[3]
                    kind = str(self.headers.get("X-Evidence-Kind", "object"))
                    out_path = backend.state_dir / "evidence" / kind / f"{event_uid}.bin"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(body)
                    self._send_json(200, {"ok": True, "event_uid": event_uid, "kind": kind})
                    return

                self._send_json(404, {"error": "not found"})

            def log_message(self, fmt, *args):
                return

            def _send_json(self, status: int, payload) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local mock backend for edge delivery testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--state-dir", default="tmp/mock_delivery_backend")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = MockDeliveryBackend(host=str(args.host), port=int(args.port), state_dir=Path(args.state_dir))
    try:
        backend.serve_forever()
    except KeyboardInterrupt:
        print("[mock-delivery] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

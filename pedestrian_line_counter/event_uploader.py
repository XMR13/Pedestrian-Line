from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib.error
import urllib.request
import uuid

from .event_contract import (
    EVENT_CONTRACT_VERSION,
    EventContractError,
    build_event_payload,
    build_events_batch_payload,
    build_run_payload,
    load_event_records,
    load_run_metadata,
    split_batches,
)


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class UploadError(RuntimeError):
    """Raised for non-retryable upload failures."""


class RetryableUploadError(UploadError):
    """Raised for transient failures that should be retried."""


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    dest: str,
    default: Optional[bool],
    help: Optional[str] = None,
) -> None:
    """Add a --foo/--no-foo flag pair with a Python 3.8-compatible fallback."""

    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            option,
            dest=dest,
            action=argparse.BooleanOptionalAction,
            default=default,
            help=help,
        )
        return

    if not option.startswith("--"):
        raise ValueError(f"Expected long option starting with '--', got: {option}")

    parser.set_defaults(**{dest: default})
    parser.add_argument(option, dest=dest, action="store_true", help=help)
    parser.add_argument(
        f"--no-{option[2:]}",
        dest=dest,
        action="store_false",
        help=argparse.SUPPRESS,
    )


@dataclass
class RetryConfig:
    max_attempts: int = 8
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    backoff_factor: float = 2.0


@dataclass
class UploaderConfig:
    spool_dir: Path
    api_base_url: str
    api_key: str
    timeout_s: float = 20.0
    # Kept as legacy default so existing deployed runs keep one state file name.
    state_filename: str = ".portal_upload_state.json"
    events_batch_size: int = 200
    upload_thumbnails: bool = True
    upload_scene_thumbnails: bool = False
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class SyncSummary:
    discovered_runs: int = 0
    completed_runs: int = 0
    skipped_runs: int = 0
    failed_runs: int = 0


class DeliveryApiClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 20.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout_s = max(float(timeout_s), 0.1)

    def upsert_run(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post_json("/api/runs/upsert", dict(payload))

    def upsert_events(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post_json("/api/events/upsert", dict(payload))

    def upload_thumbnail(self, event_uid: str, *, filename: str, content: bytes, kind: str = "object") -> Dict[str, Any]:
        field_name = "file"
        body, content_type = _encode_multipart(
            field_name=field_name,
            filename=filename,
            content=content,
            mime_type="image/jpeg",
        )
        headers = {
            "Content-Type": content_type,
            "X-API-Key": self.api_key,
            "X-Evidence-Kind": str(kind),
        }
        return self._request_json(
            path=f"/api/events/{event_uid}/thumbnail",
            method="POST",
            body=body,
            headers=headers,
        )

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        return self._request_json(
            path=path,
            method="POST",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            },
        )

    def _request_json(self, *, path: str, method: str, body: bytes, headers: Mapping[str, str]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url=url, data=body, method=method)
        for k, v in headers.items():
            req.add_header(str(k), str(v))

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    return {"raw": raw.decode("utf-8", errors="replace")}
                if isinstance(obj, dict):
                    return obj
                return {"data": obj}
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            msg = f"HTTP {exc.code} on {url}: {body_text}".strip()
            if int(exc.code) in RETRYABLE_STATUS_CODES:
                raise RetryableUploadError(msg) from exc
            raise UploadError(msg) from exc
        except urllib.error.URLError as exc:
            raise RetryableUploadError(f"network error on {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RetryableUploadError(f"timeout on {url}") from exc


def resolve_api_key(
    direct_api_key: Optional[str],
    *,
    api_key_env: str = "PORTAL_API_KEY",
    appsettings_local_path: Optional[str] = None,
) -> str:
    """
    Resolve API key with this precedence:
    1) direct CLI value
    2) environment variable
    3) local settings JSON
    """
    key = (str(direct_api_key).strip() if direct_api_key is not None else "")
    if key:
        return key

    env_name = str(api_key_env).strip()
    if env_name:
        key = (os.getenv(env_name, "") or "").strip()
        if key:
            return key

    for path in _candidate_local_settings_paths(appsettings_local_path):
        key = _load_api_key_from_local_settings(path)
        if key:
            return key

    return ""


def process_pending_runs(
    cfg: UploaderConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    max_runs: Optional[int] = None,
) -> SyncSummary:
    client = DeliveryApiClient(base_url=cfg.api_base_url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)
    summary = SyncSummary()

    for i, run_dir in enumerate(iter_spool_runs(cfg.spool_dir), start=1):
        if max_runs is not None and i > int(max_runs):
            break
        summary.discovered_runs += 1
        try:
            status = process_single_run(run_dir, cfg=cfg, client=client, force=force, dry_run=dry_run)
        except Exception as exc:
            summary.failed_runs += 1
            print(f"[uploader] failed {run_dir}: {exc}")
            continue

        if status == "completed":
            summary.completed_runs += 1
        elif status == "skipped":
            summary.skipped_runs += 1

    return summary


def process_single_run(
    run_dir: Path,
    *,
    cfg: UploaderConfig,
    client: DeliveryApiClient,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    run_dir = Path(run_dir)
    run_meta = load_run_metadata(run_dir)
    events = load_event_records(run_dir)
    state = _load_state(run_dir, cfg.state_filename)

    if state.get("completed_at_utc") and not force:
        return "skipped"

    run_payload = build_run_payload(run_meta)
    event_payloads = [build_event_payload(ev) for ev in events]
    run_is_closed = bool(str(run_payload.get("ended_at_utc") or "").strip())
    run_meta_updated_at = str(run_meta.get("updated_at_utc") or "").strip() or None
    prev_run_meta_updated_at = str(state.get("run_meta_updated_at_utc") or "").strip() or None

    if dry_run:
        print(
            f"[uploader] dry-run run_uid={run_payload['run_uid']} closed={run_is_closed} "
            f"events={len(event_payloads)} "
            f"thumbs={_count_evidence(event_payloads, 'thumb_relpath')}"
        )
        return "completed"

    try:
        should_upsert_run = bool(force or not state.get("run_upserted_at_utc"))
        if run_is_closed and not state.get("run_finalized_at_utc"):
            should_upsert_run = True
        if (not run_is_closed) and run_meta_updated_at and (run_meta_updated_at != prev_run_meta_updated_at):
            should_upsert_run = True

        if should_upsert_run:
            what = (
                f"upsert_run_finalize({run_payload['run_uid']})"
                if run_is_closed and state.get("run_upserted_at_utc")
                else f"upsert_run({run_payload['run_uid']})"
            )
            _retry_with_backoff(
                lambda: client.upsert_run(run_payload),
                cfg.retry,
                what=what,
            )
            state["run_upserted_at_utc"] = _utcnow_iso()
            state["run_meta_updated_at_utc"] = run_meta_updated_at
            if run_is_closed:
                state["run_finalized_at_utc"] = _utcnow_iso()
            else:
                state.pop("run_finalized_at_utc", None)
            _save_state(run_dir, cfg.state_filename, state)

        prev_event_count = _state_int(state, "events_uploaded_count")
        if prev_event_count > len(event_payloads):
            prev_event_count = 0
        should_sync_events = force or (len(event_payloads) > prev_event_count) or (not state.get("events_upserted_at_utc"))
        if should_sync_events:
            start_idx = 0 if force else prev_event_count
            events_delta = event_payloads[start_idx:]
            for batch in split_batches(events_delta, int(cfg.events_batch_size)):
                payload = build_events_batch_payload(batch)
                _retry_with_backoff(
                    lambda p=payload: client.upsert_events(p),
                    cfg.retry,
                    what=f"upsert_events({run_payload['run_uid']})",
                )
            state["events_upserted_at_utc"] = _utcnow_iso()
            state["events_uploaded_count"] = len(event_payloads)
            _save_state(run_dir, cfg.state_filename, state)

        prev_thumb_seen = _state_int(state, "thumbs_seen_event_count")
        if prev_thumb_seen > len(event_payloads):
            prev_thumb_seen = 0
        if cfg.upload_thumbnails and (force or (len(event_payloads) > prev_thumb_seen) or (not state.get("thumbs_upserted_at_utc"))):
            start_idx = 0 if force else prev_thumb_seen
            uploaded = _upload_event_thumbnails(
                run_dir,
                event_payloads[start_idx:],
                cfg,
                client,
                relpath_key="thumb_relpath",
                evidence_kind="object",
            )
            state["thumbs_upserted_at_utc"] = _utcnow_iso()
            if force:
                state["thumbs_uploaded_count"] = uploaded
            else:
                state["thumbs_uploaded_count"] = _state_int(state, "thumbs_uploaded_count") + uploaded
            state["thumbs_seen_event_count"] = len(event_payloads)
            _save_state(run_dir, cfg.state_filename, state)

        prev_scene_seen = _state_int(state, "scene_seen_event_count")
        if prev_scene_seen > len(event_payloads):
            prev_scene_seen = 0
        if cfg.upload_scene_thumbnails and (force or (len(event_payloads) > prev_scene_seen) or (not state.get("scene_upserted_at_utc"))):
            start_idx = 0 if force else prev_scene_seen
            uploaded_scene = _upload_event_thumbnails(
                run_dir,
                event_payloads[start_idx:],
                cfg,
                client,
                relpath_key="scene_relpath",
                evidence_kind="scene",
            )
            state["scene_upserted_at_utc"] = _utcnow_iso()
            if force:
                state["scene_uploaded_count"] = uploaded_scene
            else:
                state["scene_uploaded_count"] = _state_int(state, "scene_uploaded_count") + uploaded_scene
            state["scene_seen_event_count"] = len(event_payloads)
            _save_state(run_dir, cfg.state_filename, state)

        state["contract_version"] = EVENT_CONTRACT_VERSION
        state["run_uid"] = run_payload["run_uid"]
        state["last_error"] = None
        if run_is_closed:
            state["completed_at_utc"] = _utcnow_iso()
            state["in_progress_last_sync_at_utc"] = None
        else:
            state.pop("completed_at_utc", None)
            state["in_progress_last_sync_at_utc"] = _utcnow_iso()
        _save_state(run_dir, cfg.state_filename, state)
        if run_is_closed:
            print(
                f"[uploader] completed run_uid={run_payload['run_uid']} "
                f"events={len(event_payloads)} thumbs={state.get('thumbs_uploaded_count', 0)}"
            )
        else:
            print(
                f"[uploader] synced(in-progress) run_uid={run_payload['run_uid']} "
                f"events={len(event_payloads)} thumbs={state.get('thumbs_uploaded_count', 0)}"
            )
        return "completed"
    except Exception as exc:
        state["last_error"] = str(exc)
        state["last_error_at_utc"] = _utcnow_iso()
        _save_state(run_dir, cfg.state_filename, state)
        raise


def iter_spool_runs(root_dir: Path) -> Iterator[Path]:
    root = Path(root_dir)
    if not root.exists():
        return iter(())

    run_jsons = sorted(root.rglob("run.json"))
    run_dirs: List[Path] = []
    for path in run_jsons:
        run_dir = path.parent
        if (run_dir / "events.jsonl").exists():
            run_dirs.append(run_dir)

    return iter(run_dirs)


def _retry_with_backoff(fn, retry_cfg: RetryConfig, *, what: str) -> Any:
    attempts = 0
    delay = max(float(retry_cfg.initial_delay_s), 0.0)
    max_attempts = int(retry_cfg.max_attempts)
    backoff_factor = max(float(retry_cfg.backoff_factor), 1.0)
    max_delay = max(float(retry_cfg.max_delay_s), 0.0)

    while True:
        attempts += 1
        try:
            return fn()
        except RetryableUploadError as exc:
            if max_attempts > 0 and attempts >= max_attempts:
                raise UploadError(f"{what} failed after {attempts} attempts: {exc}") from exc
            sleep_s = min(delay, max_delay) if max_delay > 0 else delay
            if sleep_s > 0:
                time.sleep(sleep_s)
            delay = delay * backoff_factor if delay > 0 else 0.0


def _upload_event_thumbnails(
    run_dir: Path,
    events: Sequence[Mapping[str, Any]],
    cfg: UploaderConfig,
    client: DeliveryApiClient,
    *,
    relpath_key: str,
    evidence_kind: str,
) -> int:
    uploaded = 0
    for ev in events:
        rel = ev.get(relpath_key)
        if not rel:
            continue

        event_uid = ev.get("event_uid")
        if not event_uid:
            continue

        file_path = run_dir / str(rel)
        if not file_path.exists():
            # Evidence is optional; do not fail the full run for missing file.
            print(f"[uploader] skip missing evidence: {file_path}")
            continue

        content = file_path.read_bytes()
        _retry_with_backoff(
            lambda uid=str(event_uid), name=file_path.name, data=content: client.upload_thumbnail(
                uid,
                filename=name,
                content=data,
                kind=evidence_kind,
            ),
            cfg.retry,
            what=f"upload_thumbnail({event_uid})",
        )
        uploaded += 1
    return uploaded


def _load_state(run_dir: Path, state_filename: str) -> Dict[str, Any]:
    state_path = run_dir / state_filename
    if not state_path.exists():
        return {}
    try:
        obj = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def _state_int(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key, 0)
    try:
        return max(int(value), 0)
    except Exception:
        return 0


def _save_state(run_dir: Path, state_filename: str, state: Mapping[str, Any]) -> None:
    state_path = run_dir / state_filename
    payload = dict(state)
    payload["updated_at_utc"] = _utcnow_iso()
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _count_evidence(events: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for ev in events if ev.get(key))


def _encode_multipart(*, field_name: str, filename: str, content: bytes, mime_type: str) -> Tuple[bytes, str]:
    boundary = f"----plc-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _candidate_local_settings_paths(path_arg: Optional[str]) -> List[Path]:
    if path_arg is not None and str(path_arg).strip():
        return [Path(str(path_arg).strip())]

    module_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / "edge_service" / "appsettings.Local.json",
        Path.cwd() / "portal" / "appsettings.Local.json",
        module_root / "edge_service" / "appsettings.Local.json",
        module_root / "portal" / "appsettings.Local.json",
    ]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _load_api_key_from_local_settings(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""

    for section_name in ("Delivery", "EdgeDelivery", "Portal"):
        section = obj.get(section_name)
        if not isinstance(section, dict):
            continue
        raw = section.get("ApiKey")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload edge spool runs to delivery API (idempotent + retry/backoff).")
    parser.add_argument("--spool-dir", type=str, required=True, help="Root spool directory containing YYYY-MM-DD/<run_uid> runs.")
    parser.add_argument("--api-base-url", type=str, required=True, help="Delivery API base URL, e.g. http://it-backend.local:5000")
    parser.add_argument("--api-key", type=str, default=None, help="API key. If omitted, --api-key-env is used.")
    parser.add_argument("--api-key-env", type=str, default="PORTAL_API_KEY", help="Environment variable containing API key.")
    parser.add_argument(
        "--api-key-json-path",
        type=str,
        default=None,
        help=(
            "Optional path to local settings JSON containing ApiKey. "
            "If omitted, uploader tries ./edge_service/appsettings.Local.json then ./portal/appsettings.Local.json."
        ),
    )

    parser.add_argument("--watch", action="store_true", help="Run forever and poll for new runs.")
    parser.add_argument("--poll-interval-s", type=float, default=10.0, help="Watch mode polling interval in seconds.")
    parser.add_argument("--max-runs-per-pass", type=int, default=None, help="Limit how many runs are processed per pass.")

    parser.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--events-batch-size", type=int, default=200, help="Batch size for /api/events/upsert.")
    parser.add_argument("--state-filename", type=str, default=".portal_upload_state.json", help="Per-run state marker file.")

    _add_bool_arg(
        parser,
        "--upload-thumbnails",
        dest="upload_thumbnails",
        default=True,
    )
    _add_bool_arg(
        parser,
        "--upload-scene-thumbnails",
        dest="upload_scene_thumbnails",
        default=False,
        help="Upload scene thumbnails (scene/*.jpg) as extra evidence.",
    )

    parser.add_argument("--retry-max-attempts", type=int, default=8, help="Retry attempts per request (0 = unlimited).")
    parser.add_argument("--retry-initial-delay-s", type=float, default=1.0, help="Initial retry delay seconds.")
    parser.add_argument("--retry-max-delay-s", type=float, default=30.0, help="Max retry delay seconds.")
    parser.add_argument("--retry-backoff-factor", type=float, default=2.0, help="Exponential backoff factor >= 1.0.")

    parser.add_argument("--dry-run", action="store_true", help="Validate + print what would be uploaded, without API calls.")
    parser.add_argument("--force", action="store_true", help="Ignore per-run state markers and re-upload idempotently.")
    return parser.parse_args()


def _build_cfg(args: argparse.Namespace) -> UploaderConfig:
    api_key = resolve_api_key(
        args.api_key,
        api_key_env=str(args.api_key_env),
        appsettings_local_path=args.api_key_json_path,
    )
    if not args.dry_run and not api_key:
        raise SystemExit(
            "Missing API key. Provide --api-key, set "
            f"{args.api_key_env}, or add ApiKey in local settings JSON."
        )

    retry = RetryConfig(
        max_attempts=int(args.retry_max_attempts),
        initial_delay_s=float(args.retry_initial_delay_s),
        max_delay_s=float(args.retry_max_delay_s),
        backoff_factor=float(args.retry_backoff_factor),
    )
    return UploaderConfig(
        spool_dir=Path(args.spool_dir),
        api_base_url=str(args.api_base_url),
        api_key=str(api_key),
        timeout_s=float(args.timeout_s),
        state_filename=str(args.state_filename),
        events_batch_size=int(args.events_batch_size),
        upload_thumbnails=bool(args.upload_thumbnails),
        upload_scene_thumbnails=bool(args.upload_scene_thumbnails),
        retry=retry,
    )


def main() -> int:
    args = _parse_args()
    cfg = _build_cfg(args)

    try:
        if args.watch:
            print("[uploader] watch mode enabled")
            while True:
                summary = process_pending_runs(
                    cfg,
                    force=bool(args.force),
                    dry_run=bool(args.dry_run),
                    max_runs=args.max_runs_per_pass,
                )
                print(
                    f"[uploader] pass discovered={summary.discovered_runs} "
                    f"completed={summary.completed_runs} skipped={summary.skipped_runs} failed={summary.failed_runs}"
                )
                time.sleep(max(float(args.poll_interval_s), 0.5))
        else:
            summary = process_pending_runs(
                cfg,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
                max_runs=args.max_runs_per_pass,
            )
            print(
                f"[uploader] done discovered={summary.discovered_runs} "
                f"completed={summary.completed_runs} skipped={summary.skipped_runs} failed={summary.failed_runs}"
            )
            if summary.failed_runs > 0:
                return 1
            return 0
    except EventContractError as exc:
        print(f"[uploader] contract error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("[uploader] interrupted")
        return 130


# Backward-compatible portal naming.
PortalApiClient = DeliveryApiClient
DeliveryConfig = UploaderConfig
resolve_portal_api_key = resolve_api_key


__all__ = [
    "DeliveryApiClient",
    "DeliveryConfig",
    "PortalApiClient",
    "RetryConfig",
    "RetryableUploadError",
    "SyncSummary",
    "UploaderConfig",
    "UploadError",
    "iter_spool_runs",
    "main",
    "process_pending_runs",
    "process_single_run",
    "resolve_api_key",
    "resolve_portal_api_key",
]


if __name__ == "__main__":
    raise SystemExit(main())

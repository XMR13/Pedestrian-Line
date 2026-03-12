from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import uvicorn

from .api import create_app
from .config import ROOT_DIR, get_default_config
from .config_io import apply_config_overrides, load_config_overrides, split_overrides
from .event_uploader import RetryConfig, UploaderConfig, resolve_api_key


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    dest: str,
    default: Optional[bool],
    help: Optional[str] = None,
) -> None:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FastAPI service for spool visibility and delivery control.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host for the local service.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port for the local service.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override file (same shape as main.py).")
    parser.add_argument("--spool-dir", type=str, default=None, help="Spool root directory. Overrides config app.spool.root_dir.")
    parser.add_argument("--title", type=str, default="Pedestrian Line Edge Service", help="OpenAPI title.")

    parser.add_argument("--api-base-url", type=str, default=None, help="Optional delivery API base URL for sync endpoints.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional delivery API key.")
    parser.add_argument("--api-key-env", type=str, default="PORTAL_API_KEY", help="Environment variable containing the delivery API key.")
    parser.add_argument(
        "--api-key-json-path",
        type=str,
        default=None,
        help="Optional path to local settings JSON containing Delivery/EdgeDelivery/Portal ApiKey.",
    )
    parser.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout for sync calls.")
    parser.add_argument("--events-batch-size", type=int, default=200, help="Events batch size for sync calls.")
    parser.add_argument("--state-filename", type=str, default=".portal_upload_state.json", help="Per-run uploader state marker file.")
    _add_bool_arg(
        parser,
        "--upload-thumbnails",
        dest="upload_thumbnails",
        default=True,
        help="Upload event thumbnails during sync calls.",
    )
    _add_bool_arg(
        parser,
        "--upload-scene-thumbnails",
        dest="upload_scene_thumbnails",
        default=False,
        help="Upload scene thumbnails during sync calls.",
    )
    parser.add_argument("--retry-max-attempts", type=int, default=8, help="Retry attempts per request (0 = unlimited).")
    parser.add_argument("--retry-initial-delay-s", type=float, default=1.0, help="Initial retry delay seconds.")
    parser.add_argument("--retry-max-delay-s", type=float, default=30.0, help="Max retry delay seconds.")
    parser.add_argument("--retry-backoff-factor", type=float, default=2.0, help="Retry backoff factor.")
    return parser.parse_args()


def _normalize_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _resolve_spool_dir(args: argparse.Namespace) -> Path:
    cfg = get_default_config()
    if args.config:
        cfg_path = _normalize_cli_path(str(args.config))
        overrides_all = load_config_overrides(cfg_path)
        app_overrides, _extra = split_overrides(overrides_all)
        apply_config_overrides(cfg, app_overrides, path_base_dir=cfg_path.parent)

    if args.spool_dir:
        return _normalize_cli_path(str(args.spool_dir))
    if cfg.spool.root_dir is not None:
        return Path(cfg.spool.root_dir)
    raise SystemExit("FastAPI service requires --spool-dir or app.spool.root_dir in --config.")


def _build_uploader_cfg(args: argparse.Namespace, *, spool_dir: Path) -> Optional[UploaderConfig]:
    api_base_url = str(args.api_base_url or "").strip()
    if api_base_url == "":
        return None

    api_key = resolve_api_key(
        args.api_key,
        api_key_env=str(args.api_key_env),
        appsettings_local_path=args.api_key_json_path,
    )
    if api_key == "":
        raise SystemExit(
            "Missing API key. Provide --api-key, set the configured --api-key-env, "
            "or add ApiKey in local settings JSON."
        )

    return UploaderConfig(
        spool_dir=Path(spool_dir),
        api_base_url=api_base_url,
        api_key=str(api_key),
        timeout_s=float(args.timeout_s),
        state_filename=str(args.state_filename),
        events_batch_size=int(args.events_batch_size),
        upload_thumbnails=bool(args.upload_thumbnails),
        upload_scene_thumbnails=bool(args.upload_scene_thumbnails),
        retry=RetryConfig(
            max_attempts=int(args.retry_max_attempts),
            initial_delay_s=float(args.retry_initial_delay_s),
            max_delay_s=float(args.retry_max_delay_s),
            backoff_factor=float(args.retry_backoff_factor),
        ),
    )


def main() -> int:
    args = _parse_args()
    spool_dir = _resolve_spool_dir(args)
    uploader_cfg = _build_uploader_cfg(args, spool_dir=spool_dir)
    app = create_app(
        spool_dir=spool_dir,
        uploader_cfg=uploader_cfg,
        title=str(args.title),
    )

    uvicorn.run(
        app,
        host=str(args.host),
        port=int(args.port),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

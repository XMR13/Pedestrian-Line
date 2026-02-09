# Config Files

This folder contains versioned configuration data (JSON) and local-only overrides.

Recommended workflow:

- Keep `pedestrian_line_counter/config.py` as the *schema + sane defaults* (tracked in Git).
- Put production-ready, shareable configs under `config/` (tracked in Git).
- Put experimental configs under `config/local/` (ignored by Git).

## Local Experiments (Ignored By Git)

Create a local override file like `config/local/exp.json` and run:

```bash
uv run python main.py --config config/local/exp.json ...
```

CLI flags still override values from `--config`.

Recommended for RTSP: keep credentials in `config/local/*.local.json` so they are never committed.
Prefer using environment variables inside configs, e.g.:

```json
{ "app": { "io": { "rtsp_url": "env:PLC_RTSP_URL" } } }
```

Then run with `PLC_RTSP_URL=...` in your shell environment.

You can also avoid passing the URL directly in CLI history:

```bash
PLC_RTSP_URL='rtsp://user:pass@host:554/stream' uv run python main.py --config config/local/exp.json --rtsp-url-env PLC_RTSP_URL
```

## Dumping A Production Config Snapshot

Once you have a setup you like, dump the resolved config to a tracked file:

```bash
uv run python main.py --config config/local/exp.json --dump-config config/cameras/camera_prod.json
```

Then use that JSON as the baseline for future runs (and keep iterating locally in `config/local/`).

## Example

See `config/overrides.example.json` for the supported shape.
See `config/subang.example.json` for a concrete camera-style example (includes line coords, RTSP placeholder, and spool metadata).

## Overridable Keys

Top-level structure for `--config`:

- `app.model`
- `app.tracker`
- `app.line`
- `app.io`
- `app.spool`

Main keys you can override:

- `app.io.input_path`
- `app.io.output_path`
- `app.io.rtsp_url` (recommended as `env:...`)
- `app.model.backend`
- `app.model.model_path`
- `app.model.class_names_path`
- `app.model.track_class_ids`
- `app.model.confidence_threshold`
- `app.model.nms_iou_threshold`
- `app.model.pre_nms_topk`
- `app.model.min_box_area_ratio`
- `app.model.ignore_regions`
- `app.model.allow_all_classes`
- `app.tracker.max_distance`
- `app.tracker.max_lost`
- `app.tracker.max_distance_scale_cap`
- `app.line.line_json_path` (use a line JSON file directly)
- `app.line.camera_name` (resolve by name from `config/cameras/<name>.json` or `config/<name>.json`)
- `app.line.start_norm`
- `app.line.end_norm`
- `app.spool.root_dir`
- `app.spool.site_id`
- `app.spool.camera_id`
- `app.spool.write_thumbnails`
- `app.spool.thumb_pad`
- `app.spool.thumb_max_side`

Line source precedence:

1. CLI `--line-json` / `--camera`
2. Config `app.line.line_json_path` / `app.line.camera_name`
3. Config/default `app.line.start_norm` + `app.line.end_norm`
